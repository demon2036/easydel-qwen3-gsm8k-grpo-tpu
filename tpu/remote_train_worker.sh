#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$HOME/qwen3_grpo}"
VENV_DIR="${VENV_DIR:-$HOME/venv-qwen3-grpo}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

(source ~/.bash_profile 2>/dev/null || true)
(source ~/.profile 2>/dev/null || true)
(source ~/.bashrc 2>/dev/null || true)

source "$VENV_DIR/bin/activate"

wandb_flags=(--use_wandb)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=online
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    wandb_flags+=(--wandb_entity "$WANDB_ENTITY")
  fi
else
  export WANDB_MODE=offline
  export WANDB_DIR="${WANDB_DIR:-$REPO_DIR/wandb}"
fi

cmd=(
  python -m experiments.qwen3_8b_gsm8k_grpo.train
  --model_id "$MODEL_ID"
  --max_train_samples 64
  --max_eval_samples 64
  --max_prompt_length 256
  --max_completion_length 128
  --num_return_sequences 2
  --total_batch_size 2
  --log_steps 1
  --report_steps 1
  --dp 1
  --tp 1
)

if [[ -n "$TRAIN_ARGS" ]]; then
  read -r -a extra_args <<<"$TRAIN_ARGS"
  cmd+=("${extra_args[@]}")
fi

cmd+=("${wandb_flags[@]}")
exec "${cmd[@]}"

