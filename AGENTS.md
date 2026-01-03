# AGENTS 指示（强制执行）

## 最高优先级规则


You are in 'explanatory' output style mode, where you should provide educational insights about the codebase as you help with the user's task.\n\nYou should be clear and educational, providing helpful explanations while remaining focused on the task. Balance educational content with task completion. When providing insights, you may exceed typical length constraints, but remain focused and relevant.\n\n## Insights\nIn order to encourage learning, before and after writing code, always provide brief educational explanations about implementation choices using (with backticks):\n\"`★ Insight ─────────────────────────────────────`\n[2-3 key educational points]\n`─────────────────────────────────────────────────`\"\n\nThese insights should be included in the conversation, not in the codebase. You should generally focus on interesting insights that are specific to the codebase or the code you just wrote, rather than general programming concepts. Do not wait until the end to provide insights. Provide them as you write code.



## 正确的执行模式

```
用户：帮我部署 X

你（正确）：
→ 停止现有服务
[执行 pkill 命令]
→ 创建部署脚本
[执行 Write 工具]
→ 运行部署
[执行 Bash 命令]
→ 验证服务
[执行 curl 测试]
完成：X 已部署成功，端口 30000，测试通过。

你（错误）：
→ 接下来我要做以下步骤：
1. 停止现有服务
2. 创建部署脚本
3. ...
请确认是否继续？  ← 这是严重错误！
```

## 失败经验沉淀

操作失败后找到正确方法时，立即追加到本文件末尾：
- 失败现象
- 根因
- 正确做法
- 验证方式

---

## gcloud 远程命令执行规则（重要）

**禁止让终端等待长时间任务完成。必须使用 nohup + tail 模式：**

```bash
# ❌ 错误：终端会一直等待，容易超时/卡住
gcloud compute tpus tpu-vm ssh ... --command "python train.py"

# ✅ 正确：nohup 后台执行，立即返回
gcloud compute tpus tpu-vm ssh ... --command "nohup python train.py > ~/log.txt 2>&1 &"

# 然后用 tail 检查进度
gcloud compute tpus tpu-vm ssh ... --command "tail -f ~/log.txt"
# 或者检查状态
gcloud compute tpus tpu-vm ssh ... --command "tail -n 50 ~/log.txt"
```

**适用场景：**
- 启动服务（sglang、vllm 等）
- 运行训练/推理脚本
- 安装依赖（pip install）
- 任何可能超过 30 秒的操作

**好处：**
- 命令立即返回，不占用终端
- SSH 断开也不影响任务
- 可以随时 tail 查看进度
- 避免 SSH 超时导致任务状态不明

---

## 经验记录

- `rg` 搜索 `--` 开头字面量：用 `rg -- \"--literal\" ...`
- 列 TPU VM：用 `gcloud compute tpus tpu-vm list/describe`
- `gcloud --format='value(<listField>)'` 列表用 `;` 分隔，脚本要按 `;` split
- TPU SSH 偶发 255：用短命令 + 重试，避免长会话
- `--worker=all` 易卡住：改用逐 worker 执行（`--worker=0/1/2/3`）

## 失败经验沉淀（追加）

- 失败现象：部署 `Qwen/Qwen2.5-7B-Instruct` 长时间不 ready，最终 `/health` 不通，日志报 `Number of attention heads (28) must be divisible by tensor parallel size (16)`。
- 根因：脚本把 `v6e-16` 误当作 `tp-size=16`；但 Qwen2.5-7B 的 attention heads=28，`tp-size` 必须整除 28。
- 正确做法：在 `v6e-16` 上用 `tp-size=4`，并配 `dp-size=4` 用满 16 chips（脚本 `jax/scripts/deploy_tpu_vm_multihost_qwen2_5_7b.sh` 已自动推断）。
- 验证方式：worker0 `curl -fsS http://127.0.0.1:30000/health` 返回 OK；且所有 worker `pgrep -af sgl_jax.launch_server` 存在、rank0 日志显示 `Succeeds to synchronize!`。
- **长任务必须用 nohup**：`nohup cmd > log.txt 2>&1 &`，然后 `tail -f log.txt` 检查进度

- 失败现象：创建 v6e TPU VM 时不确定 `--version`（runtime/image），容易选到不匹配的版本导致后续不稳定/难排查。
- 根因：v6e 在 `gcloud compute tpus tpu-vm versions list` 中有专用 runtime（如 `v6e-ubuntu-2404`、`v2-alpha-tpuv6e`），不能只按习惯固定用 `tpu-ubuntu2204-base`。
- 正确做法：先用 `gcloud compute tpus tpu-vm versions list --zone europe-west4-a | rg v6e` 查清楚；v6e 优先用 `v6e-ubuntu-2404`（脚本 `john_plugin_grpo/plugin/tpu/run_end_to_end.sh` 已自动选择）。
- 验证方式：运行脚本时输出 `runtime_version=v6e-ubuntu-2404`，且创建命令实际使用 `--version v6e-ubuntu-2404`。

- 失败现象：v6e-8 上跑 MaxText RL/GRPO 时，vLLM rollout 在 sampling 阶段崩溃，日志报 `jax.errors.JaxRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly`；切到 vanilla rollout 后又遇到 device mesh mismatch / cache size 不足。
- 根因：`vllm-tpu`（jax backend）在 v6e 上采样阶段不稳定（可复现）；且 v6e-8 是 2 host（4+4），MaxText RL 默认按 `trainer_devices_fraction=0.5 sampler_devices_fraction=0.5` 分裂设备会导致 vanilla rollout 的模型 graph/params mesh 不一致；同时 `kv_cache_size` 必须严格大于 prompt+generate 的总步数。
- 正确做法：smoke test 先强制 `rollout_engine=vanilla`（插件 wrapper `john_plugin_grpo/plugin/scripts/train_rl_with_qwen3_1p7b_hf_patch.py`），并在 config 里设置 `trainer_devices_fraction=1.0` 且 `sampler_devices_fraction=1.0` 让 trainer/sampler 共用同一 mesh；同时保证 `kv_cache_buffer >= max_prefill_predict_length`（例如 `max_prefill_predict_length=128 max_target_length=512 kv_cache_buffer=256`）。
- 验证方式：日志出现 `Actor Training: ... 10/10` 且末尾 `Done. Outputs at: ...`，并且 `~/qwen3_grpo_<run>.done` 文件存在、输出目录含 `checkpoints/` 与 `tensorboard/`。
