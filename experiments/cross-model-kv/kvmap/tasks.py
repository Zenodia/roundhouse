# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmarks in the shape the scorer needs: a context string, candidate
continuations, and the gold index. Prompt formats follow lm-evaluation-
harness defaults (the paper's reference), reproduced here because the
harness has no seam for swapping the prefix KV cache.

Two log-likelihood tasks cover the paper's selection criterion cheaply
(HellaSwag, ARC-Challenge) and one perplexity protocol isolates prefix-KV
quality (WikiText-2, appendix D): score the second 1,024 tokens of each
2,048-token chunk conditioned on a 1,024-token prefix cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterator


@dataclass
class MultipleChoice:
    context: str
    choices: list[str]  # each already carries its leading delimiter
    gold: int


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def hellaswag(limit: int | None = None) -> Iterator[MultipleChoice]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    for i, doc in enumerate(ds):
        if limit is not None and i >= limit:
            return
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        query = _hellaswag_preprocess(doc["activity_label"] + ": " + ctx)
        choices = [" " + _hellaswag_preprocess(e) for e in doc["endings"]]
        yield MultipleChoice(query, choices, int(doc["label"]))


def arc_challenge(limit: int | None = None) -> Iterator[MultipleChoice]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for i, doc in enumerate(ds):
        if limit is not None and i >= limit:
            return
        query = "Question: " + doc["question"] + "\nAnswer:"
        labels = doc["choices"]["label"]
        choices = [" " + t for t in doc["choices"]["text"]]
        yield MultipleChoice(query, choices, labels.index(doc["answerKey"]))


TASKS = {
    "hellaswag": (hellaswag, 0.25),
    "arc_challenge": (arc_challenge, 0.25),
}


def wikitext_chunks(tokenizer, prefix_len: int = 1024, cont_len: int = 1024, limit: int | None = None) -> Iterator[tuple[list[int], list[int]]]:
    """Non-overlapping (prefix, continuation) token windows over the
    WikiText-2 raw test split, joined as one document."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    step = prefix_len + cont_len
    n = len(ids) // step
    for i in range(n):
        if limit is not None and i >= limit:
            return
        chunk = ids[i * step : (i + 1) * step]
        yield chunk[:prefix_len], chunk[prefix_len:]
