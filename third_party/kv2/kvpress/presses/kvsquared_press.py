# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class KVSquaredPress(KVzipPress):
    """
    KV² (KVSquared): selective-reconstruction KV cache compression.

    For each chunk, KV² first uses a lightweight proxy scorer to identify informative
    positions, then reprocesses only those selected tokens to compute final KV eviction
    scores. This preserves the benefits of reconstruction-based scoring while avoiding
    the cost of replaying the entire context.

    The implementation also supports iterative self-refinement by using another
    `KVSquaredPress` as the `inner_press`.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during final compression.
    chunk_size : int, default=4096
        Size of chunks for processing the reusable context.
    inner_press : ScorerPress or KVSquaredPress, default=KeyDiffPress()
        Proxy scorer used during Stage 1 to select reconstruction queries.
    top_ratio : float or None, default=None
        Fraction of tokens selected per chunk as reconstruction queries. If None,
        defaults to `1 - compression_ratio`.
    """

    chunk_size: int = 4096
    inner_press: ScorerPress | KVSquaredPress = field(default_factory=KeyDiffPress)
    top_ratio: float | None = None

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert hasattr(self.inner_press, "score") or hasattr(
            self.inner_press, "compute_chunk_scores"
        ), "inner_press must have `.score()` or `.compute_chunk_scores()`"
        self._reset_internal_parameters()

    @property
    def _effective_top_ratio(self) -> float:
        return self.top_ratio if self.top_ratio is not None else 1 - self.compression_ratio

    def _compute_chunk_scores(self, model: PreTrainedModel, chunk_start: int, chunk_end: int) -> torch.Tensor:
        """
        Stage 1: score chunk tokens using the proxy `inner_press`.

        Nested KVSquared presses expose `compute_chunk_scores()`, while simpler
        score-based presses implement `score()`.
        """
        if hasattr(self.inner_press, "compute_chunk_scores"):
            self.inner_press.compression_ratio = self.compression_ratio
            return self.inner_press.compute_chunk_scores(
                model,
                self._cache,
                self._context_ids,
                chunk_start,
                chunk_end,
                self.prefix_length,
                self.context_length,
            )

        all_scores = []
        for layer_idx, layer in enumerate(model.model.layers):
            keys, values = extract_keys_and_values(self._cache, layer_idx)
            scores = self.inner_press.score(
                module=layer.self_attn,
                hidden_states=None,
                keys=keys[:, :, chunk_start:chunk_end, :],
                values=values[:, :, chunk_start:chunk_end, :],
                attentions=None,
                kwargs={},
            )
            all_scores.append(scores.mean(dim=1))

        return torch.stack(all_scores).mean(dim=0)

    def _select_query_positions(self, scores: torch.Tensor, chunk_start: int) -> torch.Tensor:
        """Select the top-scoring positions from a chunk as reconstruction queries."""
        n_selected = max(int(scores.shape[-1] * self._effective_top_ratio), 1)
        return (scores.topk(n_selected, dim=-1).indices + chunk_start).squeeze(0).sort().values

    def _run_chunk_reconstruction(self, model: PreTrainedModel, chunk_start: int, chunk_end: int):
        """
        Process a single chunk through both KV² stages.

        Stage 1: score tokens with `inner_press` and select the best query positions.
        Stage 2: replay only the selected positions so `KVzipPress.forward_hook`
        can aggregate final importance scores.
        """
        self.start_idx, self.end_idx = chunk_start, chunk_end

        scores = self._compute_chunk_scores(model, chunk_start, chunk_end)
        positions = self._select_query_positions(scores, chunk_start)
        selected_ids = self._context_ids.index_select(1, positions.to(self._context_ids.device))
        logger.debug("[KV²] Chunk [%s:%s] -> %s reconstruction queries", chunk_start, chunk_end, len(positions))

        with torch.inference_mode():
            model(input_ids=selected_ids, past_key_values=self._cache, use_cache=True, num_logits_to_keep=1)

    def _with_scoring_hooks(self, model: PreTrainedModel, fn):
        """Execute `fn` with KVzip scoring hooks registered on every attention layer."""
        hooks = [
            layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True) for layer in model.model.layers
        ]
        try:
            return fn()
        finally:
            for hook in hooks:
                hook.remove()

    def compute_chunk_scores(
        self,
        model: PreTrainedModel,
        cache,
        context_ids: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        prefix_length: int,
        context_length: int,
    ) -> torch.Tensor:
        """
        Compute chunk scores by running KV²'s full selective-reconstruction pipeline.

        This is used when a `KVSquaredPress` is itself the `inner_press` of another
        `KVSquaredPress`, enabling iterative self-refinement.
        """
        self._cache = cache
        self._context_ids = context_ids
        self.context_length = context_length
        self.prefix_length = prefix_length
        self._init_score_val(model)

        self._with_scoring_hooks(model, lambda: self._run_chunk_reconstruction(model, chunk_start, chunk_end))
        return self.score_val[..., chunk_start:chunk_end].mean(dim=(0, 2))

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Run KV²'s two-stage selective reconstruction instead of text replay."""
        del tokenizer

        self.context_length = self._context_ids.shape[1]
        self._context_ids = self._context_ids.to(model.device)
        self._init_score_val(model)

        pos = self.prefix_length
        chunks = self._chunk_fn(self._context_ids[:, self.prefix_length :].cpu(), chunk_size=self.chunk_size)
        for chunk in chunks:
            self._run_chunk_reconstruction(model, pos, pos + chunk.shape[1])
            pos += chunk.shape[1]

        self.compress_post(model)
