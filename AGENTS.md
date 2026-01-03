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

- 失败现象：`gcloud compute tpus tpu-vm describe --format="value(networkEndpoints[].ipAddress)"` 得到空，导致脚本算出来 `worker_count=0`，bootstrap/launch 实际没跑任何 worker。
- 根因：TPU VM 的字段路径是 `networkEndpoints.ipAddress`；且 `wc -l` 统计行数时如果字符串不带换行会得到 0。
- 正确做法：用 `--format="value(networkEndpoints.ipAddress)"`，并在统计前 `printf "%s\n" "$worker_ips_raw"` 确保有换行。
- 验证方式：脚本输出 `worker_count=1`（或期望值），且 `--worker=0` 的 bootstrap/launch 命令确实执行。

- 失败现象：TPU VM 上 `python3 -m venv ...` 报 `ensurepip is not available` / 提示安装 `python3.10-venv`。
- 根因：Ubuntu 默认不带 `python3-venv` 包。
- 正确做法：后台安装 `sudo apt-get update && sudo apt-get install -y python3-venv`（长任务用 `nohup`），再创建 venv。
- 验证方式：`python -c "import venv"` 正常、venv 可创建，pip 可用。

- 失败现象：安装 EasyDeL 时报 `Package 'easydel' requires a different Python: 3.10.x not in '<3.14,>=3.11'`。
- 根因：EasyDeL 需要 Python>=3.11，但 TPU VM 镜像默认 Python3.10。
- 正确做法：安装 `python3.11`/`python3.11-venv` 并用 `python3.11 -m venv ...` 创建环境。
- 验证方式：venv 内 `python -V` 为 3.11.x，`python -c "import easydel"` 成功。

- 失败现象：`pip install torch` 在 TPU VM 上拉取巨大的 CUDA 依赖（`nvidia_*_cu12`），浪费磁盘和时间。
- 根因：PyPI 上的 `torch` 默认 wheel 会带 GPU 依赖；TPU 场景只需要 CPU torch 用于 checkpoint 转换。
- 正确做法：用 PyTorch CPU index：`pip install --index-url https://download.pytorch.org/whl/cpu torch`。
- 验证方式：`python -c "import torch; print(torch.__version__)"` 输出带 `+cpu` 后缀，且不再下载 `nvidia_*_cu12`。

- 失败现象：训练脚本传 `wandb_project` 给 `GRPOConfig` 报 `unexpected keyword argument`。
- 根因：EasyDeL 的 WandB project 名由内部 `wandb.init(project=f\"EasyDeL-...-{model_name}\")` 自动生成；Config 只接受 `wandb_entity`/`wandb_name` 等字段。
- 正确做法：只设置 `use_wandb=True` + `wandb_entity`/`wandb_name`，并依赖 `WANDB_API_KEY` 环境变量。
- 验证方式：启动后能正常 `wandb.init(...)` 且日志无 TypeError。

- 失败现象：`GRPOTrainer` 初始化时报 `AttributeError: 'GRPOTrainer' object has no attribute 'padding_value'`。
- 根因：当前 EasyDeL 版本的 GRPO trainer 里 `create_grain_collect_function` 访问了未初始化的 `padding_value`。
- 正确做法：在 `GRPOConfig` 设置 `use_data_collactor=False`，并让 trainer 走 purify/stack 路径（我们的数据已固定长度 tokenized）。
- 验证方式：trainer 能完成配置并进入 generation / metrics 步骤。

- 失败现象：使用 `LINEAR` scheduler 时抛 `ValueError: Linear scheduler requires learning_rate_end`。
- 根因：eformer 的线性 scheduler 配置要求显式的 `learning_rate_end`。
- 正确做法：给 `GRPOConfig` 补 `learning_rate_end`（例如 `learning_rate * 0.1`）。
- 验证方式：`configure Model, Optimizer, Scheduler` 阶段通过，训练开始打 `TrainerMetrics`。

- 正确做法（SOP）：TPU VM 上跑训练必须分成两步：
  1) `bootstrap`：在 TPU 上 clone repo + 装依赖（Python3.11 + venv + `pip install -r requirements.txt` + CPU torch），后台 `nohup`，产物是 `~/qwen3_grpo/.deps_done` 与 `~/qwen3_grpo/logs/bootstrap_worker0.log`。
  2) `launch`：再启动训练（后台 `nohup`），产物是 `~/qwen3_grpo/logs/train_worker0.log` 与 `~/qwen3_grpo/outputs/.../run-XX/`。
  验证方式：`tail -n 200 ~/qwen3_grpo/logs/bootstrap_worker0.log` 看到 `Successfully built easydel` / 无报错；训练日志出现 `TrainerMetrics` 且最终 `Saving checkpoint at step ...`。

- 正确做法（SOP）：所有长任务（pip install / 启动训练）都用 `nohup ... > log 2>&1 &`，并且只用 `tail -n`（不要 `tail -f`）检查进度，避免终端卡住/超时。
- 验证方式：`pgrep -af experiments.qwen3_8b_gsm8k_grpo.train` 能看到进程；`tail -n 50 train_worker0.log` 持续增长。

- 失败现象：训练日志报 HuggingFace 401/RepositoryNotFound（例如 `Qwen/Qwen3-8B-Instruct` 拉 tokenizer/config 失败）。
- 根因：模型可能是 gated/private，需要在 TPU VM 上配置 `HF_TOKEN`（环境变量）。
- 正确做法：**只在 TPU VM 的环境里**设置 `HF_TOKEN`（不要写进仓库/不要写进 AGENTS.md），然后重新启动训练；或者先用公开模型（例如 `Qwen/Qwen2.5-7B-Instruct`）做 smoke test。
- 验证方式：`python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('...')"` 在 TPU 上不再 401；训练继续进入 `Converting Model` / `TrainerMetrics`。

- 失败现象：希望打 WandB 但没有 run / 没有任何 wandb 日志。
- 根因：TPU VM 没有设置 `WANDB_API_KEY`，或脚本没有启用 `use_wandb=True`；同时 EasyDeL 的 project 名由内部自动生成（不是 `wandb_project` 参数）。
- 正确做法：**只在 TPU VM 的环境里**设置 `WANDB_API_KEY`（以及可选 `WANDB_ENTITY`），启动脚本检测到 key 后自动加 `--use_wandb`；run name 用 `wandb_name`（我们用 run_name 注入）。
- 验证方式：训练日志不再出现 wandb 初始化异常；wandb 后台出现新 run，并持续上报 `TrainerMetrics`。

- 失败现象：模型 ID 直接“拍脑袋”写成 `Qwen/Qwen3-8B-Instruct`，导致 `transformers`/`huggingface_hub` 拉取时报 `401` 或 `not a valid model identifier`。
- 根因：没有先确认 HF 上的**真实仓库名**；Qwen3 8B 的官方公开模型是 `Qwen/Qwen3-8B`（另有 `Qwen/Qwen3-8B-Base` 等），并不存在（至少公开不可见）`Qwen/Qwen3-8B-Instruct` 这个 id。
- 正确做法：在写训练脚本/启动参数前，先用 HF API 查证 modelId（比 Google 搜索更稳）：
  - `curl -s 'https://huggingface.co/api/models?search=Qwen3-8B&limit=20' | head`
  - 或本地 python：`python -c "import json,urllib.request; print([m['modelId'] for m in json.load(urllib.request.urlopen('https://huggingface.co/api/models?search=Qwen3-8B&limit=20'))][:5])"`
- 验证方式：TPU VM 上 `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')"` 成功，不再 401；训练能进入 `Converting Model`/`TrainerMetrics`。

- 失败现象：为了“自动化”，把 `HF_TOKEN`/`WANDB_API_KEY` 直接写进脚本、命令行、仓库文件，导致凭据在 git history、终端记录、CI 日志里泄露风险极高。
- 根因：没有区分“可复制 SOP”与“敏感配置”；且 `gcloud ... --command "<包含 token 的字符串>"` 往往会出现在本机 shell history / 日志 / 工具调用记录中。
- 正确做法：token **只放在 TPU VM 的环境变量/配置文件中**（例如 `~/.bashrc` 或 `wandb login`/`hf auth login`），仓库只写占位符和验证命令；启动脚本通过检测 `WANDB_API_KEY`/`HF_TOKEN` 是否存在来决定是否启用对应功能。
- 验证方式：`env | rg 'HF_TOKEN|WANDB_API_KEY'` 在 TPU 上可见，但仓库 `git grep -n 'hf_|wandb|token'` 不包含真实 token；W&B 正常出现 run、HF 可拉取模型。

- 失败现象：即使用户声明 token “用完就换”，也把 token 直接出现在 agent 的工具调用（`gcloud ... --command ...`）或聊天内容里。
- 根因：工具调用参数/日志会被系统记录，可被回放；“短期 token”依然属于敏感信息。
- 正确做法：让用户在 TPU VM 上用交互方式登录（`wandb login --relogin` / `huggingface-cli login`），或在 TPU 上本地写入环境变量文件，然后脚本只负责读取环境（我们已在 `tpu/*.sh` 里 `source ~/.bashrc`/`~/.profile`）。
- 验证方式：仓库与工具调用中不出现明文 token；W&B/HF 认证生效且训练 run 正常。

- 失败现象：为了下载/预取模型，在 `gcloud ... --command` 里直接塞复杂的 `python -c "..."`，经常因为引号嵌套导致远端脚本语法错误（例如 `SyntaxError: invalid decimal literal`）。
- 根因：`gcloud --command` + `bash -lc` + `nohup` 的多层引号很容易被本地 shell/远端 shell分词。
- 正确做法：尽量把复杂逻辑下沉到仓库脚本；如果必须用 `python -c`，采用稳妥的引号嵌套模式：
  - 外层 `--command 'bash -lc \"...\"'`
  - 内层 `python -c '\'' ... '\''`（用 `'\''` 包住 python 代码，python 代码里用 `\"` 只包字符串）
- 验证方式：远端 log 不再出现 python 语法错误，并能看到 `Fetching ... files` + `done`。

- 正确做法（W&B）：EasyDeL 的 W&B `project` 名由 `trainer_prefix` + `model_name` 决定（不是 `WANDB_PROJECT`），建议在 `GRPOConfig` 里显式设置：
  - `trainer_prefix="grpo"`
  - `model_name="qwen3_8b_gsm8k_grpo"`
- 验证方式：wandb 上出现 project 名类似 `EasyDeL-grpo-qwen3_8b_gsm8k_grpo`，run name 为 `wandb_name`（我们用 timestamp run_name）。

- 失败现象：`gcloud compute tpus tpu-vm ssh ... --command "bash -lc '...'"` 里包含花括号/引号时命令被 gcloud 错误解析（例如把 `%s`/`;` 当成 gcloud 参数）。
- 根因：外层双引号/单引号嵌套不当，导致 `--command` 的字符串被 shell 或 gcloud 提前分词。
- 正确做法：优先把复杂逻辑放到仓库脚本（如 `tpu/*.sh`），gcloud 只执行 `bash -lc '<simple command>'`；必要时避免在 `--command` 字符串里出现未转义的 `%`、`$()`、`;`。
- 验证方式：gcloud 不再报 `unrecognized arguments`，且远端脚本确实执行并产生 log。

- 失败现象：在 `gcloud ... --command` 中尝试用 heredoc（`python - <<'PY' ... PY`）做小检查，远端经常报 `here-document delimited by end-of-file` / `syntax error near unexpected token`。
- 根因：`--command` 的字符串在本地 shell + gcloud + 远端 shell 多层解析后，换行/引号很容易被破坏，导致 heredoc 无法正确闭合。
- 正确做法：对“检查/探测”类命令优先用纯 shell（`test -s file`、`printenv`）或单行 `python -c`；复杂逻辑一律下沉到仓库脚本再 `nohup` 执行。
- 验证方式：远端命令稳定返回（退出码 0），且不会出现 heredoc/引号相关错误。

## SOP（交付导向，TPU 上跑通）

- ✅ 正确：所有长任务一律 `nohup ... > log 2>&1 &`，然后用 `tail -n 200 log` 观察；脚本放在仓库 `tpu/*.sh`，`gcloud --command` 只跑“很短的入口命令”。
- ✅ 正确：凭据（HF/W&B token）只在 TPU VM 本地通过交互登录或本地环境配置，不出现在仓库、命令行、tool logs、shell history。
- ✅ 正确：先用小规模 smoke（样本数/长度/return seq 都小）验证能进入 `TrainerMetrics` 和能保存 checkpoint，再逐步放大。
- ❌ 错误：把 token 直接写入仓库/脚本/`gcloud --command` 参数里；即使“临时 token”，也会进入命令记录与回放日志。
- ❌ 错误：在 `gcloud ssh --command` 里写多行复杂逻辑（heredoc / 大段 python），容易被引号解析破坏导致不可重复。

- 失败现象：`tpu/launch_grpo_multihost.sh` 启动时本地直接报 `WANDB_ARGS: unbound variable`，训练根本没在 TPU 上启动。
- 根因：外层脚本 `set -u`，但 `gcloud --command "<双引号字符串>"` 里包含 `$WANDB_ARGS`，导致 **本地 shell** 在构造命令时尝试展开未定义变量（而不是留给 TPU 远端展开）。
- 正确做法：避免在 `--command` 内拼复杂字符串/变量；把远端训练逻辑下沉到 TPU 端脚本 `tpu/remote_train_worker.sh`，`--command` 只负责设置少量 env + `nohup bash ... &` 调用该脚本。
- 验证方式：本地执行 `bash tpu/launch_grpo_multihost.sh` 能返回 `Training started...`；TPU 上 `~/qwen3_grpo/logs/train_worker0.log` 持续增长并出现 `Loading checkpoint shards`/`TrainerMetrics`。
