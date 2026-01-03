#!/usr/bin/env bash
set -Eeuo pipefail

: "${TPU_NAME:?set TPU_NAME}"
: "${ZONE:?set ZONE}"
: "${REPO_URL:?set REPO_URL}"

REPO_DIR="${REPO_DIR:-$HOME/qwen3_grpo}"
VENV_DIR="${VENV_DIR:-$HOME/venv-qwen3-grpo}"

worker_ips_raw="$(
  gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
    --format="value(networkEndpoints.ipAddress)" 2>/dev/null || true
)"

if [[ -z "${worker_ips_raw}" ]]; then
  worker_count=1
else
  worker_count="$(printf "%s" "$worker_ips_raw" | tr ';' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"
fi

echo "TPU_NAME=$TPU_NAME ZONE=$ZONE worker_count=$worker_count"
for ((w=0; w<worker_count; w++)); do
  echo "→ bootstrap worker=$w"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker="$w" --command \
    "bash -lc 'set -euo pipefail; mkdir -p \"$REPO_DIR/logs\"; \
      if [ ! -d \"$REPO_DIR/.git\" ]; then git clone \"$REPO_URL\" \"$REPO_DIR\"; else cd \"$REPO_DIR\" && git fetch && git reset --hard origin/main; fi; \
      nohup bash -lc \"python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && python -m pip install -U pip && python -m pip install -r $REPO_DIR/requirements.txt && touch $REPO_DIR/.deps_done\" \
        > \"$REPO_DIR/logs/bootstrap_worker${w}.log\" 2>&1 &'"
done

echo "Bootstrap started. Tail example:"
echo "gcloud compute tpus tpu-vm ssh \"$TPU_NAME\" --zone \"$ZONE\" --worker=0 --command \"tail -n 200 $REPO_DIR/logs/bootstrap_worker0.log\""
