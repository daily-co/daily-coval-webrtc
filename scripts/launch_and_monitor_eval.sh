#! /bin/bash

set -euo pipefail

EVAL_CONFIG="$1"
COVAL_API_KEY="$2"

COVAL_API_EVAL_ENDPOINT="https://api.coval.dev/eval"
COVAL_API_RUN_ENDPOINT="${COVAL_API_EVAL_ENDPOINT}/run"

function fetch_metrics_results() {
  local run_id=$1
  echo "Fetching results for run $run_id..."
  results=$(curl -sfSL \
    -H "x-api-key: ${COVAL_API_KEY}" \
    "${COVAL_API_RUN_ENDPOINT}?type=metrics&run_id=$run_id")
  echo "Results:"
  echo "$results" | jq '.'
  exit 0
}

response=$(curl -sfSL \
  -H "x-api-key: ${COVAL_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-raw "${EVAL_CONFIG}" \
  "${COVAL_API_EVAL_ENDPOINT}")
echo "response"
echo $response

eval_id=$(jq -r '.run_id // empty' <<<"$response")

if [[ -z "$eval_id" ]]; then
  echo "::error:: API did not return an evaluation id"; exit 1
fi

timeout_s=600
start=$(date +%s)

while :; do
  status=$(curl -sfSL \
    -H "x-api-key: ${COVAL_API_KEY}" \
    "${COVAL_API_RUN_ENDPOINT}?run_id=$eval_id" | jq -r '.status')

  case $status in
    COMPLETED)
      echo "✅ Evaluation completed"
      fetch_metrics_results "$eval_id"
  exit 0
      ;;
    FAILED)    
      echo "❌ Evaluation failed"
      exit 1
      ;;
  esac

  echo $status

  (( $(date +%s) - start > timeout_s )) && {
    echo "::error:: Timed out after $timeout_s s"; exit 1; }
  echo "⏳ status: $status"; sleep 10
done