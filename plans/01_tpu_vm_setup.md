# TPU VM setup (GCloud)

This doc is written assuming **TPU VM** (not GKE) and you will run commands from your laptop/workstation.

## 1) Choose zone + TPU type

Example:

```bash
export PROJECT_ID="your-project"
export ZONE="europe-west4-a"
export TPU_NAME="qwen3-grpo-v6e-16"
export TPU_TYPE="v6e-16"   # or v4-8 / v5litepod-8 / ...
```

## 2) Create TPU VM

Pick a runtime version suitable for your TPU generation. Example (you must adjust):

```bash
gcloud config set project "$PROJECT_ID"
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone "$ZONE" \
  --accelerator-type "$TPU_TYPE" \
  --version "v6e-ubuntu-2404"
```

## 3) Confirm readiness

```bash
gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
  --format="value(state,health,acceleratorType)"
```

## 4) Sanity-check JAX on worker0

Short command (should return quickly):

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 \
  --command "python3 -c 'import jax; print(jax.devices())'"
```

If JAX is missing or CPU-only, install TPU JAX *on the TPU VM* before installing this repo’s deps.

