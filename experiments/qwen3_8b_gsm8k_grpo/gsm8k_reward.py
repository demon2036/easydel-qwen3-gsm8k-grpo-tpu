from __future__ import annotations

import re
import typing as tp


_GSM8K_GOLD_RE = re.compile(r"####\\s*([-+]?\\d[\\d,]*)")
_ANY_NUMBER_RE = re.compile(r"([-+]?\\d[\\d,]*)")


def parse_gsm8k_gold(answer_text: str) -> int | None:
    match = _GSM8K_GOLD_RE.search(answer_text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def extract_final_int(text: str) -> int | None:
    candidates = _ANY_NUMBER_RE.findall(text)
    if not candidates:
        return None
    raw = candidates[-1].replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def _as_text_completion(completion: tp.Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        content = completion[0].get("content")
        return content if isinstance(content, str) else ""
    return ""


def _repeat_to_match(values: list[int | None], target_len: int) -> list[int | None]:
    if not values:
        return [None] * target_len
    if target_len % len(values) != 0:
        return (values + [None] * max(0, target_len - len(values)))[:target_len]
    factor = target_len // len(values)
    return [v for v in values for _ in range(factor)]


def gsm8k_correctness_reward(
    prompts: tp.Any,
    completions: list[tp.Any],
    batch: dict,
    **_: tp.Any,
) -> list[float]:
    gold = batch.get("answer_value")
    if gold is None:
        return [0.0] * len(completions)

    gold_list: list[int | None]
    try:
        gold_list = [int(x) for x in list(gold)]
    except Exception:
        try:
            gold_list = [int(gold)]
        except Exception:
            gold_list = [None]

    gold_expanded = _repeat_to_match(gold_list, len(completions))
    rewards: list[float] = []
    for completion, g in zip(completions, gold_expanded, strict=False):
        pred = extract_final_int(_as_text_completion(completion))
        rewards.append(1.0 if (g is not None and pred == g) else 0.0)
    return rewards


def gsm8k_format_reward(
    prompts: tp.Any,
    completions: list[tp.Any],
    **_: tp.Any,
) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = _as_text_completion(completion)
        rewards.append(1.0 if "####" in text else 0.0)
    return rewards

