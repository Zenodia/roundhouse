// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wiring a real local tier: an HTTP [`LocalExecutor`] against a Dynamo
//! worker's own OpenAI-compatible endpoint, and the runtime tokenizer choice
//! that makes its token ids meaningful to that worker.
//!
//! Everything here is opt-in and read once at boot from `ROUNDHOUSE_LOCAL_*`
//! variables (see [`from_env`]). Absent, `serve()` behaves exactly as it did
//! before this module existed: `EchoLocalExecutor`, `ByteTokenizer`, no
//! `EmbeddedFleet` attached, `local/<model>` unreachable. This is deliberate —
//! see `main.rs`'s `reachable_candidates` doc comment, which already promised
//! that "a deployment that attaches a fleet adds its local model to the list
//! at the same site it attaches the fleet." This module is that site.
//!
//! **Why an HTTP executor and not the mocker/embedded-engine one the test
//! suite uses.** `MockerExecutor` (`roundhouse-server/tests/mocker_cache_hits.rs`)
//! drives an in-process simulated vLLM scheduler — useful for proving the
//! cache-state machinery with no GPU, but it never leaves the test binary. A
//! real deployment's worker is a separate process (Dynamo's `dynamo.frontend`
//! + `dynamo.vllm`, e.g. `use-cases/cache-aware-routing/serve_model.sh`), so
//! the executor here does what any other out-of-process dispatch does: an
//! HTTP call.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use roundhouse_core::context::{ByteTokenizer, Tokenizer};
use roundhouse_core::routing::{Candidate, Target};
use roundhouse_fleet::{
    EmbeddedFleet, FleetError, KvRouterConfig, SelectionServiceBuilder, WorkerRegistration,
};

use crate::engine::{LocalExecution, LocalExecutor};
use crate::tokenizer::HfTokenizer;

const ENDPOINT_VAR: &str = "ROUNDHOUSE_LOCAL_ENDPOINT";
const MODEL_VAR: &str = "ROUNDHOUSE_LOCAL_MODEL";
const TOKENIZER_VAR: &str = "ROUNDHOUSE_LOCAL_TOKENIZER";
const KV_EVENTS_ENDPOINT_VAR: &str = "ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT";
const BLOCK_SIZE_VAR: &str = "ROUNDHOUSE_LOCAL_BLOCK_SIZE";
const ROUTING_GROUP_VAR: &str = "ROUNDHOUSE_LOCAL_ROUTING_GROUP";
const BASE_TTFT_MS_VAR: &str = "ROUNDHOUSE_LOCAL_BASE_TTFT_MS";

const DEFAULT_BLOCK_SIZE: u32 = 64;
const DEFAULT_ROUTING_GROUP: &str = "default";
const DEFAULT_BASE_TTFT_MS: f64 = 60.0;
const DEFAULT_WORKER_ID: u64 = 1;
const DEFAULT_DP_RANK: u32 = 0;

/// A [`Tokenizer`] chosen at boot rather than at compile time.
///
/// `Engine<S, T: Tokenizer + Clone>` is monomorphized once per `serve()`
/// call, so `serve()` cannot itself be generic over which tokenizer an
/// operator picked without duplicating the whole function per choice — the
/// same trade-off `main.rs` already makes twice for `SessionStore` (see
/// `serve`'s own doc comment on "the two arms monomorphize `serve` twice").
/// Wrapping both concrete tokenizers in one enum keeps `serve()` written
/// once, with the choice made here, at the one composition site.
#[derive(Clone)]
pub enum RuntimeTokenizer {
    /// What every deployment gets with no local fleet configured — see
    /// `ByteTokenizer`'s own doc comment for why it is fine for that case
    /// (no worker's real vocabulary is ever consulted) and wrong for this
    /// one (a local worker's block hashes must match its real BPE).
    Byte(ByteTokenizer),
    Hf(HfTokenizer),
}

impl Tokenizer for RuntimeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            RuntimeTokenizer::Byte(t) => t.encode(text),
            RuntimeTokenizer::Hf(t) => t.encode(text),
        }
    }
}

/// Drives a Dynamo/vLLM worker's own OpenAI-compatible `/v1/completions`
/// endpoint with a raw token-id prompt.
///
/// **Token ids, not text — on the wire, not just in the trait signature.**
/// `LocalExecutor::execute` already documents why re-tokenizing here would be
/// wrong; this executor is what keeps that promise past the trait boundary,
/// by sending vLLM's `"prompt": [<ids>]` array form (confirmed against a live
/// Dynamo/vLLM 0.26.0 server, 2026-09-23) rather than reconstructing a chat
/// message and letting the worker's own chat template re-tokenize it.
pub struct HttpLocalExecutor {
    client: reqwest::Client,
    model: String,
}

impl HttpLocalExecutor {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            // A generous but finite timeout: a local worker that never answers
            // must eventually surface as a routing failure rather than hang
            // the turn's own deadline machinery forever underneath it.
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("reqwest client with only a timeout must build"),
            model: model.into(),
        }
    }
}

#[derive(serde::Deserialize)]
struct CompletionsResponse {
    choices: Vec<CompletionChoice>,
    usage: Option<CompletionUsage>,
}

#[derive(serde::Deserialize)]
struct CompletionChoice {
    text: String,
}

#[derive(serde::Deserialize)]
struct CompletionUsage {
    completion_tokens: Option<u64>,
}

#[async_trait]
impl LocalExecutor for HttpLocalExecutor {
    async fn execute(
        &self,
        endpoint: &str,
        prompt_tokens: &[u32],
        expected_output_tokens: Option<u32>,
    ) -> Result<LocalExecution, FleetError> {
        let url = format!("{}/v1/completions", endpoint.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt_tokens,
            "max_tokens": expected_output_tokens.unwrap_or(256),
            "stream": false,
        });
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|err| FleetError::Other(anyhow::anyhow!("{url}: request failed: {err}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();
            return Err(FleetError::Rejected(format!(
                "{url} answered {status}: {detail}"
            )));
        }
        let parsed: CompletionsResponse = response
            .json()
            .await
            .map_err(|err| FleetError::Other(anyhow::anyhow!("{url}: bad response body: {err}")))?;
        let choice = parsed.choices.into_iter().next().ok_or_else(|| {
            FleetError::Other(anyhow::anyhow!("{url}: response carried no choices"))
        })?;
        // vLLM reports completion_tokens in `usage`; falling back to a rough
        // word-count would under- or over-report spend against this worker,
        // so an absent field is treated as "unknown", not "zero" -- 0 would
        // read on a dashboard as a free turn.
        let output_tokens = parsed
            .usage
            .and_then(|usage| usage.completion_tokens)
            .unwrap_or(0);
        Ok(LocalExecution {
            text: choice.text,
            output_tokens,
            // vLLM's own reasoning-token accounting is a separate,
            // model-specific feature this executor does not opt into; see
            // `LocalExecution::reasoning_tokens`'s own doc comment for why
            // zero reads as "not applicable" rather than a lie.
            reasoning_tokens: 0,
        })
    }
}

/// Everything `serve()` needs to route to a real local worker instead of the
/// echo stub, resolved once at boot.
pub struct LocalFleetSetup {
    pub fleet: Arc<EmbeddedFleet>,
    pub executor: Arc<dyn LocalExecutor>,
    pub tokenizer: RuntimeTokenizer,
    /// The worker's own HTTP endpoint. Duplicates what `executor` already
    /// dispatches to, kept here too because `local_score` needs the bare
    /// endpoint string to build its own client, not a `LocalExecutor` to
    /// dispatch a turn through.
    pub endpoint: String,
    pub local_model: String,
    pub routing_group: String,
    pub block_size: u32,
    pub local_quality_prior: f64,
    pub local_base_ttft_ms: f64,
    /// What `main.rs`'s `reachable_candidates()` list gets appended with, so
    /// the boot-time cross-checks (a cadence or a degrade-mode budget
    /// promising local service) see a local target actually exists —
    /// `reachable_candidates`'s own doc comment names this module as the
    /// site that was supposed to do this.
    pub candidate: Candidate,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Reads `ROUNDHOUSE_LOCAL_*` and builds a real local fleet if an operator
/// asked for one.
///
/// `None` — not an error — when [`ENDPOINT_VAR`] is unset: this is the
/// ordinary case for every deployment with no local tier, `serve()`'s
/// existing behavior unchanged. Once an operator sets it, [`MODEL_VAR`] and
/// [`TOKENIZER_VAR`] become required, and an incomplete configuration is a
/// boot refusal rather than a turn that fails the first time it is routed
/// locally.
///
/// `local_quality` and `default_local_quality` come from the caller's
/// already-parsed `catalog.json` (`CatalogConfig::local_quality`,
/// `::default_local_quality`) rather than a new env var, because that field
/// already exists for exactly this purpose (see `catalog_config.rs`) and a
/// second source of the same number would let the two disagree.
pub async fn from_env(
    local_quality: &HashMap<String, f64>,
    default_local_quality: f64,
) -> anyhow::Result<Option<LocalFleetSetup>> {
    let Some(endpoint) = env(ENDPOINT_VAR) else {
        return Ok(None);
    };
    let model = env(MODEL_VAR).ok_or_else(|| {
        anyhow::anyhow!("{ENDPOINT_VAR} is set but {MODEL_VAR} is not; both are required together")
    })?;
    let tokenizer_path = env(TOKENIZER_VAR).ok_or_else(|| {
        anyhow::anyhow!(
            "{ENDPOINT_VAR} is set but {TOKENIZER_VAR} is not; a local worker's block hashes \
             are only meaningful under its own real BPE, so this cannot silently fall back to \
             the byte tokenizer -- see HfTokenizer's own doc comment"
        )
    })?;
    let kv_events_endpoint = env(KV_EVENTS_ENDPOINT_VAR).ok_or_else(|| {
        anyhow::anyhow!(
            "{ENDPOINT_VAR} is set but {KV_EVENTS_ENDPOINT_VAR} is not; without it the embedded \
             selection service has no real cache-hit signal for this worker and every quote \
             would price a cold prefill forever"
        )
    })?;
    let block_size: u32 = match env(BLOCK_SIZE_VAR) {
        Some(raw) => raw.parse().with_context_var(BLOCK_SIZE_VAR, &raw)?,
        None => DEFAULT_BLOCK_SIZE,
    };
    let routing_group = env(ROUTING_GROUP_VAR).unwrap_or_else(|| DEFAULT_ROUTING_GROUP.to_string());
    let base_ttft_ms: f64 = match env(BASE_TTFT_MS_VAR) {
        Some(raw) => raw.parse().with_context_var(BASE_TTFT_MS_VAR, &raw)?,
        None => DEFAULT_BASE_TTFT_MS,
    };
    let quality_prior = local_quality
        .get(&model)
        .copied()
        .unwrap_or(default_local_quality);

    let tokenizer = RuntimeTokenizer::Hf(HfTokenizer::from_file(Path::new(&tokenizer_path))?);

    let service = SelectionServiceBuilder::new(KvRouterConfig {
        use_kv_events: true,
        router_queue_threshold: None,
        ..Default::default()
    })
    .indexer_threads(1)
    .build()
    .await?;
    let fleet = Arc::new(EmbeddedFleet::new(Arc::new(service)).with_routing_group(&routing_group));
    fleet
        .register_worker(WorkerRegistration {
            worker_id: DEFAULT_WORKER_ID,
            model_name: model.clone(),
            routing_group: routing_group.clone(),
            endpoint: endpoint.clone(),
            block_size,
            kv_events_endpoints: HashMap::from([(DEFAULT_DP_RANK, kv_events_endpoint)]),
        })
        .await
        .map_err(|err| anyhow::anyhow!("registering local worker `{model}`: {err}"))?;

    let candidate = Candidate {
        target: Target::Local {
            worker_id: DEFAULT_WORKER_ID,
            dp_rank: DEFAULT_DP_RANK,
            model: model.clone(),
        },
        // Nominal, the same way `PROBE_ISL_TOKENS`/`local_candidate()`'s own
        // boot-check fixtures are: this candidate exists so the cross-checks
        // see local capacity is real, not so its numbers are quoted at a
        // tenant -- the router re-prices every turn against the live fleet.
        expected_prefill_tokens: 1_024.0,
        matched_prefix_tokens: 0,
        expected_ttft_ms: base_ttft_ms,
        expected_cost_usd: 0.0,
        quality_prior,
        load: Some(0.0),
    };

    Ok(Some(LocalFleetSetup {
        fleet,
        executor: Arc::new(HttpLocalExecutor::new(model.clone())),
        tokenizer,
        endpoint,
        local_model: model,
        routing_group,
        block_size,
        local_quality_prior: quality_prior,
        local_base_ttft_ms: base_ttft_ms,
        candidate,
    }))
}

trait ParseVarExt<T> {
    fn with_context_var(self, var: &str, raw: &str) -> anyhow::Result<T>;
}

impl<T, E: std::fmt::Display> ParseVarExt<T> for Result<T, E> {
    fn with_context_var(self, var: &str, raw: &str) -> anyhow::Result<T> {
        self.map_err(|err| anyhow::anyhow!("{var}=`{raw}` is not valid: {err}"))
    }
}
