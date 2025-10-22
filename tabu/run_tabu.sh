#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Uso: $0 <instancia> <target> [csv]"
  exit 1
fi

INSTANCE="$1"
TARGET="$2"
CSV="${3:-results/ttt_TABU_$(basename "${INSTANCE%.*}")_target${TARGET}.csv}"

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"

for SEED in $(seq 0 49); do
  echo "[seed=$SEED] rodando..."
  python3 tabu_adjusted_ttt.py 20 1000 "$INSTANCE" \
    seed="$SEED" \
    target="$TARGET" \
    output="$CSV" \
    timeout=600 \
    quiet
done

echo "Feito: $CSV"
