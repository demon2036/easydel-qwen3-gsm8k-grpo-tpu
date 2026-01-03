#!/usr/bin/env bash
set -Eeuo pipefail

: "${TPU_NAME:?set TPU_NAME}"
: "${ZONE:?set ZONE}"
: "${REPO_URL:?set REPO_URL}"

REPO_DIR="${REPO_DIR:-$HOME/qwen3_grpo}"
VENV_DIR="${VENV_DIR:-$HOME/venv-qwen3-grpo}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
COORD_PORT="${COORD_PORT:-8476}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"

worker_ips_raw="$(
  gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
    --format="value(networkEndpoints.ipAddress)" 2>/dev/null || true
)"

if [[ -z "${worker_ips_raw}" ]]; then
  worker_count=1
  coord_addr="127.0.0.1"
else
  mapfile -t worker_ips < <(printf "%s\n" "$worker_ips_raw" | tr ';' '\n' | sed '/^$/d')
  worker_count="${#worker_ips[@]}"
  coord_addr="${worker_ips[0]}"
fi

echo "TPU_NAME=$TPU_NAME ZONE=$ZONE worker_count=$worker_count"
echo "JAX_COORDINATOR_ADDRESS=$coord_addr JAX_COORDINATOR_PORT=$COORD_PORT"

for ((w=0; w<worker_count; w++)); do
  echo "→ launch worker=$w"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker="$w" --command \
    "bash -lc 'set -euo pipefail; mkdir -p \"$LOG_DIR\"; \
      if [ ! -d \"$REPO_DIR/.git\" ]; then \
        if [ -e \"$REPO_DIR\" ]; then mv \"$REPO_DIR\" \"${REPO_DIR}.bak.$(date +%s)\"; fi; \
        git clone \"$REPO_URL\" \"$REPO_DIR\"; \
      else \
        cd \"$REPO_DIR\" && git fetch && git reset --hard origin/main; \
      fi; \
      test -f \"$REPO_DIR/.deps_done\" || echo \"WARN: deps not bootstrapped; run tpu/bootstrap_workers.sh\"; \
      nohup bash -lc \"export JAX_COORDINATOR_ADDRESS=$coord_addr; export JAX_COORDINATOR_PORT=$COORD_PORT; export JAX_PROCESS_COUNT=$worker_count; export JAX_PROCESS_INDEX=$w; \
        (source ~/.bash_profile 2>/dev/null || true); (source ~/.profile 2>/dev/null || true); (source ~/.bashrc 2>/dev/null || true); \
        WANDB_ARGS=\\\"\\\"; \
        if [ -n \\\"\\${WANDB_API_KEY:-}\\\" ]; then \
          export WANDB_MODE=online; \
          WANDB_ARGS=\\\"--use_wandb\\\"; \
          if [ -n \\\"\\${WANDB_ENTITY:-}\\\" ]; then WANDB_ARGS=\\\"$WANDB_ARGS --wandb_entity \\${WANDB_ENTITY}\\\"; fi; \
        else \
          export WANDB_MODE=disabled; \
        fi; \
        source $VENV_DIR/bin/activate; \
        python -m experiments.qwen3_8b_gsm8k_grpo.train \
          --model_id $MODEL_ID \
          --max_train_samples 64 \
          --max_eval_samples 64 \
          --max_prompt_length 256 \
          --max_completion_length 128 \
          --num_return_sequences 2 \
          --total_batch_size 2 \
          --log_steps 1 \
          --report_steps 1 \
          --dp 1 --tp 1 \
          ${TRAIN_ARGS} $WANDB_ARGS\" \
        > \"$LOG_DIR/train_worker${w}.log\" 2>&1 &'"
done

echo "Training started. Tail example:"
echo "gcloud compute tpus tpu-vm ssh \"$TPU_NAME\" --zone \"$ZONE\" --worker=0 --command \"tail -n 200 $LOG_DIR/train_worker0.log\""
