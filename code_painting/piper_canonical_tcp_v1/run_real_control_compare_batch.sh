#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/real_control_manifest.tsv"
GPU=0
OUTPUT_ROOT="$SCRIPT_DIR/outputs_real_control_compare_20260716"

while (($#)); do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG_ROOT="$OUTPUT_ROOT/_batch_logs"
STATUS="$LOG_ROOT/status.tsv"
mkdir -p "$LOG_ROOT"
if [[ ! -e "$STATUS" ]]; then
  printf 'timestamp\ttask\tepisode\tslot\tstatus\texit_code\n' >"$STATUS"
fi
failed=0
while IFS=$'\t' read -r task episode slot; do
  [[ -n "$task" ]] || continue
  echo "[batch] slot=$slot task=$task episode=$episode"
  printf '%s\t%s\t%s\t%s\tstarted\t\n' \
    "$(date -Iseconds)" "$task" "$episode" "$slot" >>"$STATUS"
  if "$SCRIPT_DIR/run_real_control_compare.sh" \
      --task "$task" --episode "$episode" --gpu "$GPU" --output-root "$OUTPUT_ROOT"; then
    printf '%s\t%s\t%s\t%s\tcomplete\t0\n' \
      "$(date -Iseconds)" "$task" "$episode" "$slot" >>"$STATUS"
  else
    status=$?
    failed=1
    printf '%s\t%s\t%s\t%s\tfailed\t%s\n' \
      "$(date -Iseconds)" "$task" "$episode" "$slot" "$status" >>"$STATUS"
  fi
done < <(tail -n +2 "$MANIFEST")
exit "$failed"
