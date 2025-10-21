#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Uso: $0 <instancia> <target> [csv]"
  exit 1
fi

INSTANCE="$1"
TARGET="$2"
CSV="${3:-results/ttt_GA_$(basename "${INSTANCE%.*}")_target${TARGET}.csv}"

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"

for SEED in $(seq 0 49); do
  echo "[seed=$SEED] rodando..."
  python3 ga_adjusted_ttt.py "$INSTANCE" 600 \
    --seed "$SEED" \
    --target "$TARGET" \
    --csv "$CSV" \
    --quiet
done

echo "Feito: $CSV"
