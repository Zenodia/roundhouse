# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration sequences: FineWeb-Edu documents, tokenized, cut to a fixed
length. The paper uses 500 sequences of 1,024 tokens from FineWeb-Edu and
keeps every 4th token position for the fit (stride 4).

Streamed, so nothing downloads but what is read. Documents shorter than the
sequence length are skipped rather than padded, because a padded position
would be a token the models never attend over in practice.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

FINEWEB_EDU = ("HuggingFaceFW/fineweb-edu", "sample-10BT")


def calibration_sequences(
    tokenizer,
    num_seqs: int,
    seq_len: int,
    dataset: tuple[str, str] = FINEWEB_EDU,
    split: str = "train",
    seed: int = 0,
    shuffle_buffer: int = 10_000,
) -> Iterator[torch.Tensor]:
    """Yields `num_seqs` LongTensors of shape (seq_len,)."""
    from datasets import load_dataset

    path, config = dataset
    stream = load_dataset(path, name=config, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    produced = 0
    for doc in stream:
        ids = tokenizer(doc["text"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if len(ids) < seq_len:
            continue
        yield torch.tensor(ids[:seq_len], dtype=torch.long)
        produced += 1
        if produced >= num_seqs:
            return
    if produced < num_seqs:
        raise RuntimeError(f"only {produced} of {num_seqs} documents reached {seq_len} tokens")


def local_text_sequences(path: str, tokenizer, num_seqs: int, seq_len: int) -> Iterator[torch.Tensor]:
    """Offline fallback: one plain-text file, chunked into consecutive
    `seq_len`-token windows. For machines without Hub access; the paper's
    domain ablation shows Wikipedia-register text costs under 1 pp, code
    about 4 pp, against FineWeb-Edu."""
    text = open(path, encoding="utf-8").read()
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    n = min(num_seqs, len(ids) // seq_len)
    if n < num_seqs:
        raise RuntimeError(f"{path}: {len(ids)} tokens only yield {n} sequences of {seq_len}")
    for i in range(n):
        yield torch.tensor(ids[i * seq_len : (i + 1) * seq_len], dtype=torch.long)
