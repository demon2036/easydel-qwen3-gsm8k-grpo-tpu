# Clone this repo onto TPU VM

## From your laptop/workstation

```bash
export PROJECT_ID="your-project"
export ZONE="europe-west4-a"
export TPU_NAME="your-tpu"
export REPO_URL="https://github.com/<you>/<repo>.git"
export REPO_DIR="~/qwen3_grpo"
```

Clone on each worker (short command):

```bash
for w in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker="$w" \
    --command "bash -lc 'set -euo pipefail; test -d $REPO_DIR/.git || git clone $REPO_URL $REPO_DIR; echo ok'"
done
```

If your TPU has a different worker count, see `tpu/launch_grpo_multihost.sh` which auto-detects workers.

