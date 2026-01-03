from __future__ import annotations

import argparse
import os
import time
import typing as tp


def _build_prompt(question: str) -> str:
    return (
        "You are a careful math tutor.\n"
        "Solve the following problem step by step.\n"
        "At the end, output the final integer answer on its own line in the form:\n"
        "#### <answer>\n"
        "\n"
        f"Problem: {question.strip()}\n"
        "\n"
        "Solution:\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--dataset_id", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--total_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--save_dir", default=os.path.expanduser("~/qwen3_grpo/outputs"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="qwen3-gsm8k-grpo")
    parser.add_argument("--wandb_entity", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import jax
    import jax.numpy as jnp
    from datasets import load_dataset
    from transformers import AutoTokenizer

    import easydel as ed

    from .gsm8k_reward import gsm8k_correctness_reward, gsm8k_format_reward, parse_gsm8k_gold

    logger = ed.utils.get_logger("qwen3_8b_gsm8k_grpo")
    logger.info(f"jax_backend={jax.default_backend()}")
    logger.info(f"jax_process_index={int(os.environ.get('JAX_PROCESS_INDEX', '0'))}")
    logger.info(f"jax_process_count={int(os.environ.get('JAX_PROCESS_COUNT', '1'))}")

    run_name = args.run_name or time.strftime("qwen3_8b_gsm8k_grpo_%Y%m%d_%H%M%S")
    save_directory = os.path.join(args.save_dir, run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_train = load_dataset(args.dataset_id, args.dataset_config, split=args.train_split)
    raw_eval = load_dataset(args.dataset_id, args.dataset_config, split=args.eval_split)

    def preprocess(example: dict[str, tp.Any]) -> dict[str, tp.Any]:
        gold = parse_gsm8k_gold(example.get("answer", ""))
        prompt = _build_prompt(example["question"])
        tokenized = tokenizer(
            prompt,
            padding="max_length",
            max_length=args.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        tokenized["answer_value"] = int(gold) if gold is not None else -10**18
        return tokenized

    remove_train_cols = list(raw_train.column_names)
    remove_eval_cols = list(raw_eval.column_names)
    train_dataset = raw_train.map(preprocess, remove_columns=remove_train_cols)
    eval_dataset = raw_eval.map(preprocess, remove_columns=remove_eval_cols)

    max_sequence_length = args.max_prompt_length + args.max_completion_length
    sharding_axis_dims = (args.dp, -1, 1, args.tp, 1)
    logger.info(f"sharding_axis_dims={sharding_axis_dims}")
    logger.info(f"max_sequence_length={max_sequence_length}")

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        args.model_id,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        backend=ed.EasyDeLBackends.TPU,
        platform=ed.EasyDeLPlatforms.PALLAS,
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=jnp.bfloat16,
            attn_softmax_dtype=jnp.bfloat16,
            kvdtype=jnp.bfloat16,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
    )

    grpo_config = ed.GRPOConfig(
        total_batch_size=args.total_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        save_directory=save_directory,
        save_steps=500,
        report_steps=10,
        log_steps=5,
        progress_bar_type="json",
        beta=args.beta,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project if args.use_wandb else None,
        wandb_entity=args.wandb_entity if args.use_wandb else None,
        track_memory=False,
        do_last_save=True,
        save_optimizer_state=False,
        save_total_limit=0,
    )

    trainer = ed.GRPOTrainer(
        model=model,
        arguments=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[gsm8k_format_reward, gsm8k_correctness_reward],
        reward_processing_classes=[None, None],
    )

    trainer.train()


if __name__ == "__main__":
    main()

