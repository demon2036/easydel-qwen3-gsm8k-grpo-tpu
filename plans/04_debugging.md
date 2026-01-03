# Debugging checklist (TPU VM + EasyDeL GRPO)

## “It hangs” / no logs

- Confirm the command was started with `nohup ... &` and returned immediately.
- Check process:
  - `pgrep -af qwen3_8b_gsm8k_grpo/train.py`
- Check last log lines:
  - `tail -n 100 ~/qwen3_grpo/logs/train_worker0.log`

## Attention heads vs tensor parallel (TP)

If you see an error like:
`Number of attention heads (...) must be divisible by tensor parallel size (...)`

Fix by choosing `--tp` that divides `num_attention_heads`, and set `--dp` to fill devices.
`experiments/qwen3_8b_gsm8k_grpo/train.py` prints the chosen `(dp,tp)` before loading the model.

## Dataset / reward mismatch

- If reward is always 0: inspect one completion in logs and check answer extraction rules.
- GSM8K answers are parsed from the dataset `"answer"` field (expects `#### <number>`).

