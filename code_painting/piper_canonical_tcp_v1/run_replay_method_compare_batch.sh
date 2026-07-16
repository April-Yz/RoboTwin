#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/batch_manifest.tsv"
CANONICAL_ROOT="$SCRIPT_DIR/outputs_canonical_20260715"
OUTPUT_ROOT="$SCRIPT_DIR/outputs_replay_method_compare_20260716"
GPU=0
LEGACY_TARGET_RETREAT_M=""

while (($#)); do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --canonical-root) CANONICAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --legacy-target-retreat-m) LEGACY_TARGET_RETREAT_M="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --legacy-target-retreat-m M [--gpu N] [--manifest TSV] [--canonical-root PATH] [--output-root PATH]"
      exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$LEGACY_TARGET_RETREAT_M" ]] || { echo "[error] --legacy-target-retreat-m is required" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "[error] missing manifest: $MANIFEST" >&2; exit 3; }

mkdir -p "$OUTPUT_ROOT/_batch_logs"
STATUS="$OUTPUT_ROOT/_batch_logs/status.tsv"
FAILURES="$OUTPUT_ROOT/_batch_logs/failures.tsv"
printf 'state\ttask\tid\tarm\tutc\n' >"$STATUS"
: >"$FAILURES"

while IFS=$'\t' read -r TASK ID ARM; do
  [[ -z "$TASK" || "$TASK" == \#* ]] && continue
  printf 'started\t%s\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" "$(date -u +%FT%TZ)" >>"$STATUS"
  if "$SCRIPT_DIR/run_replay_method_compare.sh" \
    --task "$TASK" --id "$ID" --arm "$ARM" --gpu "$GPU" \
    --canonical-root "$CANONICAL_ROOT" --output-root "$OUTPUT_ROOT" \
    --legacy-target-retreat-m "$LEGACY_TARGET_RETREAT_M"; then
    printf 'complete\t%s\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" "$(date -u +%FT%TZ)" >>"$STATUS"
  else
    status=$?
    printf 'failed\t%s\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" "$(date -u +%FT%TZ)" >>"$STATUS"
    printf '%s\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" "$status" >>"$FAILURES"
  fi
done <"$MANIFEST"

if [[ -s "$FAILURES" ]]; then
  echo "[batch-finished-with-failures] $FAILURES"
  exit 1
fi
touch "$OUTPUT_ROOT/_batch_logs/SUCCESS"
echo "[batch-success] $OUTPUT_ROOT"
