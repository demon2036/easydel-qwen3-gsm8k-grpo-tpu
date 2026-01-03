# EasyDeL GRPO on GSM8K (Qwen3-8B) — TPU runbook

This repo is a **non-invasive wrapper project** to run **EasyDeL GRPO** training on **GSM8K** for **Qwen3-8B** on **Google Cloud TPU VM**.

- Training code lives under `experiments/qwen3_8b_gsm8k_grpo/`.
- TPU operational runbooks live under `plans/`.
- `tpu/` contains helper scripts you run from your laptop/workstation via `gcloud`.

## Quick links

- `plans/03_run_grpo_gsm8k.md`
- `tpu/launch_grpo_multihost.sh`
- `experiments/qwen3_8b_gsm8k_grpo/train.py`

## Notes

- This repo is designed to be **cloned and executed on TPU VM** via `gcloud compute tpus tpu-vm ssh ...`.
- Run `tpu/bootstrap_workers.sh` first (installs deps in background), then `tpu/launch_grpo_multihost.sh`.
