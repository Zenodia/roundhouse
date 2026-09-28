// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real, token-level log-probability scoring of pre-written option
//! continuations against the local Dynamo worker.
//!
//! Closes the gap `use-cases/openjev-demo/INTEGRATION.md` named as Gap 1:
//! `/v1/responses` carries text turns, not log-probabilities, so a Jev-style
//! decision layer built on it could only *approximate* `open-jev`'s own
//! scoring with a prompted free-text answer that has to be parsed and can
//! fail to parse. This module is the real thing — the exact sum of
//! `log p(option token | context, prior option tokens)` for each option,
//! computed the same way `open-jev`'s own `OptionScorer._score_with_prefix`
//! does (a shared context, no decoding, a softmax over the sums), just over
//! HTTP against Dynamo's `prompt_logprobs` sampling param instead of an
//! in-process batched forward pass.
//!
//! **Getting `prompt_logprobs` out of `/v1/completions` took three tries.**
//! Recorded because none of the first two are discoverable from the OpenAI
//! spec, and the working shape is not what either the OpenAI or the vLLM
//! native docs would suggest:
//! 1. `echo: true`: `NvCreateCompletionRequest`
//!    (`dynamo/lib/llm/src/protocols/openai/completions.rs`) does not
//!    implement `echo` at all — the field is commented out in its own source
//!    — so it is silently dropped rather than refused.
//! 2. `prompt_logprobs` set at the request's top level alone: accepted by
//!    request validation, but the completions route never places it back
//!    onto the top-level response.
//! 3. **The one that works**: `prompt_logprobs` at the top level *and*
//!    `"nvext": {"extra_fields": ["prompt_logprobs"]}` — Dynamo's own
//!    response-field-selection mechanism
//!    (`NvExtResponseFieldSelection::from_nvext`,
//!    `dynamo/lib/llm/src/protocols/common/extensions.rs`). The worker
//!    computes `prompt_logprobs` regardless, but only serializes it back
//!    (under `response.nvext.prompt_logprobs`, not the top-level field the
//!    request named) when the request explicitly opts in to that field
//!    appearing. Confirmed against a live Dynamo/vLLM 0.26.0 worker,
//!    2026-09-28.
//!
//! (`/inference/v1/generate`, Dynamo's experimental token-in/token-out
//! surface, was tried first and ruled out: its Rust-side plumbing for
//! `prompt_logprobs` is real, but no code path in the pinned rev's shipped
//! `dynamo.vllm` Python launcher ever advertises the
//! `vllm_inference_v1_generate` runtime capability a worker needs for the
//! frontend to route to it — `grep -rn vllm_inference_v1_generate` across
//! both the Rust and Python trees turns up the capability's *definition* and
//! its *check*, never anything that *sets* it. Every request timed out at
//! `"no generate-capable model is registered"` regardless of frontend flags.)
//!
//! Response shape: `nvext.prompt_logprobs` is a JSON array aligned with the
//! prompt token sequence, `null` at position 0 (no left context for the
//! first token), and elsewhere an object whose keys are the *candidate*
//! token ids at that position (as strings) — not only the one actually in
//! the prompt; vLLM includes the top-ranked alternative too when it differs
//! — mapping to `{"logprob": f64, "rank": ...}`. This module looks up by the
//! exact token id it already knows it put in the prompt at that position,
//! never by position alone or by assuming which key is present.
//!
//! **Off by default, and a separate opt-in from the rest of the local
//! fleet.** [`ENABLE_SCORE_VAR`] gates whether `main.rs` mounts
//! `local_score_router` at all — see [`enabled`]. A deployment that wires a
//! local fleet purely for ordinary turn dispatch (`cache-aware-routing`, for
//! instance) has no use for this route: every call it makes is an extra
//! per-option HTTP round trip against the worker (`score`'s own doc comment
//! on why — one request per option, not one batched pass), and mounting an
//! unused route is a surface an operator has to reason about for nothing.
//! Absence is silence, not a refusal: unset, `/v1/local/score` simply does
//! not exist (404), the same as when no local fleet is configured at all.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use roundhouse_core::context::Tokenizer;
use roundhouse_core::now_ms;

use crate::control_config::{AuthError, PlaneSource};
use crate::local_fleet::RuntimeTokenizer;

/// Scores option continuations against one local worker's `/v1/completions`
/// endpoint, via the `nvext.prompt_logprobs` mechanism this module's doc
/// comment works through.
pub struct LocalScorer {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    tokenizer: RuntimeTokenizer,
}

impl LocalScorer {
    pub fn new(endpoint: String, model: String, tokenizer: RuntimeTokenizer) -> Self {
        Self {
            // A generous but finite timeout, the same posture
            // `HttpLocalExecutor` takes and for the same reason: a worker
            // that never answers must surface as a failure, not a hang.
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("reqwest client with only a timeout must build"),
            endpoint,
            model,
            tokenizer,
        }
    }

    /// Score each of `options` as a continuation of `context`.
    ///
    /// `probability` is a real softmax over each option's mean log-probability
    /// (`open-jev`'s `--norm mean`, the default there for the same reason it is
    /// here: options of different lengths need normalizing by token count to
    /// be comparable) — not a parsed guess at what the model meant.
    pub async fn score(
        &self,
        context: &str,
        options: &[String],
    ) -> Result<Vec<OptionScore>, ScoreError> {
        if options.len() < 2 {
            return Err(ScoreError::TooFewOptions);
        }
        let context_ids = self.tokenizer.encode(context);
        let mut sums = Vec::with_capacity(options.len());
        let mut token_counts = Vec::with_capacity(options.len());
        for option in options {
            let option_ids = self.tokenizer.encode(option);
            if option_ids.is_empty() {
                return Err(ScoreError::EmptyOption(option.clone()));
            }
            let mut prompt_ids = context_ids.clone();
            prompt_ids.extend_from_slice(&option_ids);
            let sum = self.option_logprob_sum(&prompt_ids, &option_ids).await?;
            sums.push(sum);
            token_counts.push(option_ids.len());
        }
        let means: Vec<f64> = sums
            .iter()
            .zip(&token_counts)
            .map(|(s, n)| s / *n as f64)
            .collect();
        let probabilities = softmax(&means);
        Ok((0..options.len())
            .map(|i| OptionScore {
                option: options[i].clone(),
                n_tokens: token_counts[i],
                logprob_sum: sums[i],
                logprob_mean: means[i],
                probability: probabilities[i],
            })
            .collect())
    }

    /// `log p(option token | context + prior option tokens)`, summed over
    /// `option_ids`. One request per option — `open-jev`'s own scorer shares
    /// one prefill across a whole batch of options in a single forward pass;
    /// this makes one HTTP round trip per option instead, which is the real
    /// cost of doing this over a network boundary rather than in-process. See
    /// `INTEGRATION.md` Gap 1's Option A writeup for why that trade was made
    /// anyway (a new local-tier-only endpoint, not a rewrite of the shared
    /// turn-dispatch path).
    async fn option_logprob_sum(
        &self,
        prompt_ids: &[u32],
        option_ids: &[u32],
    ) -> Result<f64, ScoreError> {
        let url = format!("{}/v1/completions", self.endpoint.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "nvext": {"extra_fields": ["prompt_logprobs"]},
            "stream": false,
        });
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|err| ScoreError::Request(format!("{url}: request failed: {err}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();
            return Err(ScoreError::Request(format!(
                "{url} answered {status}: {detail}"
            )));
        }
        let parsed: CompletionsResponse = response
            .json()
            .await
            .map_err(|err| ScoreError::Request(format!("{url}: bad response body: {err}")))?;
        let prompt_logprobs = parsed
            .nvext
            .and_then(|nvext| nvext.prompt_logprobs)
            .ok_or_else(|| {
                ScoreError::Request(format!(
                    "{url}: response carried no nvext.prompt_logprobs -- expected it under \
                     `nvext` because the request set `nvext.extra_fields: [\"prompt_logprobs\"]`"
                ))
            })?;
        // **Aligned from the end, not from `context_len` — this is load-bearing,
        // not a style choice.** With prefix caching on (this deployment always
        // has it on; it is vLLM's default and the whole reason a local tier is
        // interesting here), a cached block is never recomputed, so the worker
        // never has a fresh logprob to report for it — `prompt_logprobs` comes
        // back *shorter than the prompt*, covering only the uncached tail.
        // Confirmed empirically (2026-09-28): an identical prompt returns a
        // full-length array cold and a truncated one once its prefix is warm —
        // for a 75-token prompt with a 64-token cached prefix, exactly 11
        // entries came back, `75 - 64`. The option's own tokens are always
        // freshly appended and therefore always uncached, so they are always
        // exactly the last `option_ids.len()` entries of whatever came back,
        // regardless of how much of the context ahead of them was skipped.
        if prompt_logprobs.len() < option_ids.len() {
            return Err(ScoreError::Request(format!(
                "{url}: prompt_logprobs carried {} entries, fewer than the {} option tokens \
                 requested -- expected at least the option's own tokens uncached",
                prompt_logprobs.len(),
                option_ids.len()
            )));
        }
        let tail = &prompt_logprobs[prompt_logprobs.len() - option_ids.len()..];
        let mut sum = 0.0;
        for (entry, &token_id) in tail.iter().zip(option_ids) {
            let logprob = entry
                .as_object()
                .and_then(|obj| obj.get(&token_id.to_string()))
                .and_then(|v| v.get("logprob"))
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    ScoreError::Request(format!(
                        "prompt_logprobs tail entry missing or has no logprob for option token \
                         {token_id}"
                    ))
                })?;
            sum += logprob;
        }
        Ok(sum)
    }
}

fn softmax(xs: &[f64]) -> Vec<f64> {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct OptionScore {
    pub option: String,
    pub n_tokens: usize,
    pub logprob_sum: f64,
    pub logprob_mean: f64,
    pub probability: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ScoreError {
    #[error("need at least two options")]
    TooFewOptions,
    #[error("option tokenizes to nothing: {0:?}")]
    EmptyOption(String),
    #[error("{0}")]
    Request(String),
}

#[derive(Deserialize)]
struct CompletionsResponse {
    #[serde(default)]
    nvext: Option<NvExtResponse>,
}

#[derive(Deserialize)]
struct NvExtResponse {
    #[serde(default)]
    prompt_logprobs: Option<Vec<serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// HTTP surface
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ScoreState {
    scorer: Arc<LocalScorer>,
    planes: Arc<dyn PlaneSource>,
}

/// Opt-in gate for mounting `/v1/local/score` at all — see this module's own
/// doc comment for why it is a separate switch from the rest of the local
/// fleet rather than following automatically from `ROUNDHOUSE_LOCAL_ENDPOINT`
/// being set.
///
/// Presence (any non-empty value) enables it; absence — the default — leaves
/// `main.rs` mounting no route at all. Follows the same presence/absence
/// convention `local_fleet.rs`'s own `env()` helper uses for every other
/// `ROUNDHOUSE_LOCAL_*` variable, rather than inventing a `"true"`/`"1"`
/// truthy parse this codebase has no other example of.
pub const ENABLE_SCORE_VAR: &str = "ROUNDHOUSE_LOCAL_ENABLE_SCORE";

/// Whether [`ENABLE_SCORE_VAR`] is set.
pub fn enabled() -> bool {
    std::env::var(ENABLE_SCORE_VAR).is_ok_and(|v| !v.is_empty())
}

/// Mount `POST /v1/local/score`, gated by a control plane.
///
/// Callers wire this in only when a local fleet is actually configured
/// (`local_fleet::from_env` returned `Some`) *and* [`enabled`] — see
/// `main.rs` — rather than mounting it always and refusing inside the
/// handler, so a deployment that didn't ask for scoring simply has no such
/// route (404) instead of a route that always errors. "Local-tier only, and
/// opt-in even then," the way `INTEGRATION.md`'s Gap 1 Option A describes it.
pub fn local_score_router<P: PlaneSource>(planes: Arc<P>, scorer: Arc<LocalScorer>) -> Router {
    let planes: Arc<dyn PlaneSource> = planes;
    Router::new()
        .route("/v1/local/score", post(score_handler))
        .with_state(ScoreState { scorer, planes })
}

/// Shaped to match `open-jev`'s own `ScoreRequest`/`ScoreResponse`
/// (`openjev/server.py`) field-for-field where the concepts line up, so the
/// two are directly comparable rather than merely similar.
#[derive(Deserialize)]
struct ScoreRequestBody {
    context: String,
    options: Vec<String>,
}

#[derive(Serialize)]
struct ScoreResponseBody {
    best: String,
    best_index: usize,
    options: Vec<OptionScore>,
}

enum HandlerError {
    Auth(AuthError),
    Score(ScoreError),
}

impl IntoResponse for HandlerError {
    fn into_response(self) -> Response {
        match self {
            HandlerError::Auth(err) => err.into_response(),
            HandlerError::Score(err) => (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {"code": "score_failed", "message": err.to_string()}
                })),
            )
                .into_response(),
        }
    }
}

impl From<AuthError> for HandlerError {
    fn from(err: AuthError) -> Self {
        HandlerError::Auth(err)
    }
}

async fn score_handler(
    State(state): State<ScoreState>,
    headers: HeaderMap,
    Json(req): Json<ScoreRequestBody>,
) -> Result<Json<ScoreResponseBody>, HandlerError> {
    // A valid turn key, the same gate a turn-serving surface uses elsewhere
    // (`ControlPlane::turn_principal` — see `metrics_api.rs` for the sibling
    // pattern on a read-only surface). An admin key is refused here for the
    // same reason it is refused on every turn surface: an admin acts on the
    // deployment, not as a spending membership.
    let plane = state.planes.plane(now_ms()).await;
    let _ = plane.turn_principal(&headers)?;

    let scored = state
        .scorer
        .score(&req.context, &req.options)
        .await
        .map_err(HandlerError::Score)?;
    let best_index = scored
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.probability.partial_cmp(&b.probability).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(Json(ScoreResponseBody {
        best: scored[best_index].option.clone(),
        best_index,
        options: scored,
    }))
}
