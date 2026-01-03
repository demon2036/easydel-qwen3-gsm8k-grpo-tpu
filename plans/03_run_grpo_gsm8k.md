# Run GRPO (Qwen3-8B) on GSM8K

This runbook uses:

- `experiments/qwen3_8b_gsm8k_grpo/train.py` as the entrypoint
- A **callable reward** that checks GSM8K final numeric correctness
- **nohup** for long tasks + `tail` for observability

## 1) Install deps on TPU worker0 (repeat per worker)

Recommended: run the bootstrap helper from your laptop/workstation so installs happen on all workers via `nohup`.

```bash
export TPU_NAME="your-tpu"
export ZONE="europe-west4-a"
export REPO_URL="https://github.com/<you>/<repo>.git"

bash tpu/bootstrap_workers.sh
```

## 2) Launch multi-host training

Use the helper script from your laptop/workstation:

```bash
export PROJECT_ID="your-project"
export ZONE="europe-west4-a"
export TPU_NAME="your-tpu"
export REPO_URL="https://github.com/<you>/<repo>.git"

bash tpu/launch_grpo_multihost.sh
```

## 3) Watch logs (worker0)

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 \
  --command "tail -n 200 ~/qwen3_grpo/logs/train_worker0.log"
```

## 4) Expected “it’s running” signals

- `python` process exists on all workers (`pgrep -af train.py`)
- `jax.devices()` shows TPU devices
- logs show compilation + step metrics (reward mean/std, loss, etc.)
