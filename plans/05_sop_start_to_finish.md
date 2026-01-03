# SOP: start → finish on your TPU

This SOP assumes **exactly one TPU VM exists** and it is:

- Project: `civil-rarity-482610-s5`
- Zone: `europe-west4-a`
- TPU name: `qwen3-1-7b-grpo-20260103-124738`
- Accelerator: `v6e-8` (single worker: `--worker=0`)

Repo to run:

- `https://github.com/demon2036/easydel-qwen3-gsm8k-grpo-tpu`

## 0) Set variables (on your laptop/workstation)

```bash
export PROJECT_ID="civil-rarity-482610-s5"
export ZONE="europe-west4-a"
export TPU_NAME="qwen3-1-7b-grpo-20260103-124738"
export REPO_URL="https://github.com/demon2036/easydel-qwen3-gsm8k-grpo-tpu"
export REPO_DIR="~/qwen3_grpo"
```

## 1) Confirm TPU is READY

```bash
gcloud config set project "$PROJECT_ID"
gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
  --format="value(state,health,acceleratorType)"
```

## 2) Bootstrap deps on TPU (background install)

```bash
git clone "$REPO_URL" /tmp/qwen3_grpo_local
cd /tmp/qwen3_grpo_local
bash tpu/bootstrap_workers.sh
```

Watch install log:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 \
  --command "tail -n 200 $REPO_DIR/logs/bootstrap_worker0.log"
```

## 3) (Optional) Enable HF + W&B (on TPU worker0)

If you want W&B online logging and the model/dataset requires auth, set tokens **on the TPU VM** (do not commit them).

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 --command "bash -lc '
  echo \"export HF_TOKEN=...\" >> ~/.bashrc
  echo \"export WANDB_API_KEY=...\" >> ~/.bashrc
  echo \"export WANDB_ENTITY=...\" >> ~/.bashrc
  echo \"export WANDB_PROJECT=qwen3-gsm8k-grpo\" >> ~/.bashrc
  source ~/.bashrc
'"
```

## 4) Launch GRPO training (background)

```bash
cd /tmp/qwen3_grpo_local
export MODEL_ID="Qwen/Qwen3-8B-Instruct"   # requires HF_TOKEN if gated
# If you want a public smoke test first:
# export MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
bash tpu/launch_grpo_multihost.sh
```

Tail training log:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 \
  --command "tail -n 200 $REPO_DIR/logs/train_worker0.log"
```

## 5) “Done” signal

This smoke run is configured to be small (`max_train_samples=64`, short seq lens). It should eventually stop emitting new steps.

Quick checks:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 \
  --command "pgrep -af 'experiments.qwen3_8b_gsm8k_grpo.train' || true"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 \
  --command "ls -la $REPO_DIR/outputs || true"
```
