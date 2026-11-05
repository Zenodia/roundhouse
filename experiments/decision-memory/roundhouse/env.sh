# Roundhouse environment for the decision-memory runs. Source, then start the binary.
#   source experiments/decision-memory/roundhouse/env.sh
#   ./target/release/roundhouse
# Frontier tier needs INFERENCE_API_KEY exported. Local tier needs serve_pair.sh
# from experiments/cross-model-kv/serving running (Qwen3-8B + Qwen3-14B on Dynamo);
# leave ROUNDHOUSE_LOCAL_ENDPOINT unset to run frontier-only.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROUNDHOUSE_CATALOG="${ROUNDHOUSE_CATALOG:-$HERE/catalog.json}"
export ROUNDHOUSE_CONTROL_PLANE="${ROUNDHOUSE_CONTROL_PLANE:-$HERE/control-plane.json}"
export ROUNDHOUSE_ADDR="${ROUNDHOUSE_ADDR:-127.0.0.1:8080}"
export ROUNDHOUSE_FRONTIER_UPSTREAM="${ROUNDHOUSE_FRONTIER_UPSTREAM:-openai_responses}"
export RUST_LOG="${RUST_LOG:-info}"
if [ -n "${ROUNDHOUSE_LOCAL_ENDPOINT:-}" ]; then
  # Same worker wiring as the KV experiment: worker 1 = 8B, worker 2 = 14B.
  SOURCE="${SOURCE:-Qwen/Qwen3-8B}" TARGET="${TARGET:-Qwen/Qwen3-14B}" source "$HERE/../../cross-model-kv/serving/env.sh"
  export ROUNDHOUSE_CATALOG="$HERE/catalog.json" ROUNDHOUSE_CONTROL_PLANE="$HERE/control-plane.json"
fi
echo "roundhouse: catalog=$ROUNDHOUSE_CATALOG control-plane=$ROUNDHOUSE_CONTROL_PLANE addr=$ROUNDHOUSE_ADDR"
