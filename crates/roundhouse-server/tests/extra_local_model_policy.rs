// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A policy that admits only a non-primary local model must still route to it.
//!
//! The cross-model KV demo gives each project one model of a pair
//! (`source-only` admits the primary, `target-only` admits the second). The
//! engine decided whether to quote the fleet from the *primary's* identity
//! alone, so a policy that named only an `extra_local_models` entry skipped
//! every local quote, left no candidate, and refused the turn with
//! `PolicyRefused` — although the second model was exactly what the policy
//! allowed. The tool arm of the skip is shared by every local model; the
//! policy arm is per model identity, and the quoting loop already applied it
//! per extra model. Only the gate in front of the loop did not.
//!
//! No other suite builds a fleet with two models, which is how this stayed
//! hidden: the single-model rigs cannot tell "the primary is excluded" from
//! "local is excluded".

use std::collections::HashMap;
use std::sync::Arc;

use roundhouse_core::context::ByteTokenizer;
use roundhouse_core::control::{Principal, TargetFilter, TurnPolicy};
use roundhouse_core::ids::{SessionId, TurnId};
use roundhouse_core::item::Item;
use roundhouse_core::routing::{AffinityPolicy, Target};
use roundhouse_core::store::MemoryStore;
use roundhouse_fleet::{
    EchoFrontierClient, EmbeddedFleet, FrontierClient, FrontierClients, KvRouterConfig, LocalFleet,
    SelectionServiceBuilder, WorkerRegistration,
};
use roundhouse_server::engine::LocalModelConfig;
use roundhouse_server::{
    Admission, EchoLocalExecutor, Engine, EngineConfig, EngineError, LocalExecutor, TurnHistory,
    TurnInput,
};

mod common;

const PRIMARY: &str = "local";
const EXTRA: &str = "local-large";

async fn two_model_engine() -> Engine<MemoryStore, ByteTokenizer> {
    let service = SelectionServiceBuilder::new(KvRouterConfig {
        use_kv_events: false,
        router_queue_threshold: None,
        ..Default::default()
    })
    .indexer_threads(1)
    .build()
    .await
    .expect("selection service should start");
    let fleet = Arc::new(EmbeddedFleet::new(Arc::new(service)));
    for (worker_id, model) in [(1, PRIMARY), (2, EXTRA)] {
        fleet
            .register_worker(WorkerRegistration {
                worker_id,
                model_name: model.to_string(),
                routing_group: "default".to_string(),
                endpoint: format!("http://worker-{worker_id}:8000"),
                block_size: common::BLOCK_SIZE,
                kv_events_endpoints: HashMap::new(),
            })
            .await
            .expect("the worker must register");
    }
    let clients = FrontierClients::keyed(
        [(
            "anthropic".to_string(),
            Arc::new(EchoFrontierClient::new("frontier answer")) as Arc<dyn FrontierClient>,
        )]
        .into_iter()
        .collect(),
    );
    Engine::with_provider_clients(
        Arc::new(MemoryStore::new()),
        ByteTokenizer,
        Arc::new(EchoLocalExecutor::new("local answer")) as Arc<dyn LocalExecutor>,
        common::frontier_catalog(),
        Arc::new(clients),
        Arc::new(AffinityPolicy::new()),
        EngineConfig {
            turn_deadline_ms: 5_000,
            local_model: PRIMARY.to_string(),
            extra_local_models: vec![LocalModelConfig {
                model: EXTRA.to_string(),
                quality_prior: 0.72,
            }],
            ..common::config()
        },
    )
    .with_fleet(fleet as Arc<dyn LocalFleet>)
}

/// An admission whose policy names exactly one local model.
fn admitting_only(model: &str) -> Admission {
    let identity = Target::local_policy_identity(model);
    Admission {
        principal: Principal::new("proj", "user"),
        policy: Arc::new(TurnPolicy {
            allow: TargetFilter::parse([identity.as_str()]).expect("the identity parses"),
            ..TurnPolicy::unrestricted()
        }),
        ..Admission::open()
    }
}

fn input() -> TurnInput {
    TurnInput {
        history: TurnHistory::Reference,
        items: vec![Item::user_text("hi")],
        declared_baseline: None,
        output_token_cap: None,
        tools: None,
        tool_choice: None,
        tools_dialect: None,
    }
}

/// The model a turn under `admission` was routed to, or the engine's refusal.
async fn routed_model(admission: &Admission) -> Result<String, EngineError> {
    let engine = two_model_engine().await;
    let session_id = SessionId::generate();
    engine
        .create_session(&session_id)
        .await
        .expect("the session opens");
    let result = engine
        .run_turn(&session_id, TurnId::new("t1"), input(), admission)
        .await?;
    match result.decision.expect("a routed turn carries a decision").target {
        Target::Local { model, .. } => Ok(model),
        other => panic!("expected a local target, got {other:?}"),
    }
}

/// The control: the primary is admitted, so this passes before and after the
/// fix. Without it the next test could fail for a broken rig and not for the
/// defect.
#[tokio::test]
async fn a_policy_admitting_only_the_primary_routes_to_the_primary() {
    assert_eq!(routed_model(&admitting_only(PRIMARY)).await.unwrap(), PRIMARY);
}

/// The defect: the primary is excluded, the extra model is the one the policy
/// allows, and the turn was refused instead of routed to it.
#[tokio::test]
async fn a_policy_admitting_only_an_extra_model_routes_to_the_extra_model() {
    match routed_model(&admitting_only(EXTRA)).await {
        Ok(model) => assert_eq!(model, EXTRA),
        Err(err) => panic!("the policy admits `{EXTRA}` and the turn was refused: {err:?}"),
    }
}
