# Overview

Goal: run **GRPO** training for **Qwen3-8B** on **GSM8K** using **EasyDeL**, on **Google Cloud TPU VM**.

Design principles:

1. **Do not modify EasyDeL**: we install EasyDeL from upstream (`requirements.txt`) and keep all custom logic in this repo.
2. **TPU-first runbook**: every long task uses `nohup ... &` and is observable via `tail`.
3. **Reproducible knobs**: model id, TPU mesh (`dp/tp`), batch sizes, dataset split, and output dirs are CLI args.

Repo layout:

- `plans/`: step-by-step operations docs (TPU VM creation → clone → install → launch → debug).
- `tpu/`: `gcloud compute tpus tpu-vm ssh ...` wrappers (multi-host safe; no `--worker=all`).
- `experiments/qwen3_8b_gsm8k_grpo/`: training entrypoint + GSM8K reward.

