# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from kvpress import KVSquaredPress
from kvpress.presses.keydiff_press import KeyDiffPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_kvsquared_nested_press_runs(unit_test_model):  # noqa: F811
    inner_press = KVSquaredPress(compression_ratio=0.5, chunk_size=16, inner_press=KeyDiffPress())
    press = KVSquaredPress(compression_ratio=0.5, chunk_size=32, inner_press=inner_press)

    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    for layer in unit_test_model.model.layers:
        assert hasattr(layer.self_attn, "masked_key_indices")


def test_kvsquared_selects_at_least_one_query():
    press = KVSquaredPress(compression_ratio=0.9, top_ratio=0.0)
    scores = torch.tensor([[0.1, 0.7, 0.3]])

    positions = press._select_query_positions(scores, chunk_start=4)

    assert positions.tolist() == [5]
