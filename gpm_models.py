from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union

import torch
from torch import Tensor
import torch.nn as nn


class GateModelBase(nn.Module, ABC):
    """Abstract base class for the *gate* p_theta deciding delete vs. substitute.

    The gate outputs, for each example in the batch, a scalar probability p_delete in [0, 1]
    for choosing a *deletion* at timestep t given the current state x_t.

    Inputs (batched):
        token_ids: LongTensor of shape (B, L_max). Padded with pad_id where necessary.
        lengths:   LongTensor or Tensor of shape (B,) with the true lengths |x_t| per example.
        t:         Either an int or Tensor of shape (B,) with the current timestep index.
        T:         Either an int or Tensor of shape (B,) with the total nominal horizon.
        L:         Either an int or Tensor of shape (B,) with the target terminal length.
        mask:      Optional BoolTensor of shape (B, L_max) marking valid (True) positions.
        kwargs:    For model-specific extras (e.g., conditioning features).

    Returns:
        p_delete:  Tensor of shape (B,) with unbarriered probabilities in [0, 1].
                   Barriering (enforcing feasibility constraints) is handled by the
                   combined model wrapper, not necessarily by the gate itself.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        token_ids: Tensor,
        lengths: Tensor,
        t: Union[int, Tensor],
        T: Union[int, Tensor],
        L: Union[int, Tensor],
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError


class ProposalModelBase(nn.Module, ABC):
    """Abstract base class for the *proposal* over per-position edits.

    The proposal produces a per-position logit tensor Z from which both the deletion
    and substitution distributions are derived (shared head with a dedicated null
    channel representing deletion).

    Inputs (batched):
        token_ids: LongTensor (B, L_max)
        lengths:   Tensor (B,)
        t:         int or Tensor (B,)
        T:         int or Tensor (B,)
        L:         int or Tensor (B,)
        mask:      Optional BoolTensor (B, L_max)
        kwargs:    Model-specific conditioning (e.g., time/length features).

    Returns:
        Z:         Tensor (B, L_max, V+1) of per-position logits.
                   The last channel is conventionally reserved for the null (∅) token
                   indicating deletion; subclasses should document their convention.
    """

    def __init__(self, vocab_size: int, use_null_last: bool = True) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.use_null_last = bool(use_null_last)

    @abstractmethod
    def forward(
        self,
        token_ids: Tensor,
        lengths: Tensor,
        t: Union[int, Tensor],
        T: Union[int, Tensor],
        L: Union[int, Tensor],
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError


class GatedEditProposal(nn.Module):
    """Combines a GateModelBase and a ProposalModelBase into a single module.

    This wrapper is responsible for:
      1) Calling the gate and proposal models with consistent inputs.
      2) Applying *barriers* to the gate output based on the feasibility constraints
         described in the draft (gt = ℓ_t - L, rt = T - t):

            - If g_t == 0  => must substitute  => p_delete := 0
            - If g_t == r_t => must delete     => p_delete := 1

         For 0 < g_t < r_t, p_delete is left unchanged (model-determined).

    Forward returns the barriered gate probabilities and the proposal logits Z.
    It does not compute distributions q_del or q_sub; those are derived downstream
    (e.g., inside the alignment/loss module) to keep concerns separated.
    """

    def __init__(self, gate: GateModelBase, proposal: ProposalModelBase) -> None:
        super().__init__()
        self.gate = gate
        self.proposal = proposal

    @staticmethod
    def _to_tensor(x: Union[int, Tensor], like: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return torch.full((like.shape[0],), float(x), dtype=like.dtype, device=like.device)

    @staticmethod
    def apply_barriers(
        p_delete: Tensor,
        lengths: Tensor,
        t: Union[int, Tensor],
        T: Union[int, Tensor],
        L: Union[int, Tensor],
    ) -> Tensor:
        """Apply feasibility barriers to p_delete in a vectorized manner.

        Args:
            p_delete: (B,) unbarriered probabilities in [0, 1].
            lengths:  (B,) current sequence lengths ℓ_t.
            t, T, L:  int or (B,) tensors.
        Returns:
            (B,) barriered probabilities satisfying constraints.
        """
        # Convert to tensors compatible with p_delete
        t = GatedEditProposal._to_tensor(t, p_delete)
        T = GatedEditProposal._to_tensor(T, p_delete)
        L = GatedEditProposal._to_tensor(L, p_delete)
        lengths = lengths.to(dtype=p_delete.dtype, device=p_delete.device)

        g = lengths - L            # gap: remaining deletions needed
        r = T - t                  # remaining steps

        # Start with the original probabilities
        out = p_delete.clone()

        # Case 1: g == 0  => must substitute => p = 0
        must_sub = (g == 0)
        out = torch.where(must_sub, torch.zeros_like(out), out)

        # Case 2: g == r  => must delete    => p = 1
        must_del = (g == r)
        out = torch.where(must_del, torch.ones_like(out), out)

        # For 0 < g < r, leave as is.
        return out.clamp(0.0, 1.0)

    def forward(
        self,
        token_ids: Tensor,
        lengths: Tensor,
        t: Union[int, Tensor],
        T: Union[int, Tensor],
        L: Union[int, Tensor],
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Compute (barriered_p_delete, Z_logits).

        Args:
            token_ids: LongTensor (B, L_max)
            lengths:   Tensor (B,)
            t, T, L:   int or Tensor (B,)
            mask:      Optional BoolTensor (B, L_max)
            kwargs:    Passed through to gate/proposal.
        Returns:
            p_delete_barriered: Tensor (B,) in [0, 1]
            Z:                   Tensor (B, L_max, V+1)
        """
        # Proposal logits
        Z = self.proposal(
            token_ids=token_ids,
            lengths=lengths,
            t=t,
            T=T,
            L=L,
            mask=mask,
            **kwargs,
        )

        # Gate probabilities (unbarriered)
        p_delete = self.gate(
            token_ids=token_ids,
            lengths=lengths,
            t=t,
            T=T,
            L=L,
            mask=mask,
            **kwargs,
        )

        if p_delete.dim() != 1:
            raise ValueError("Gate must return shape (B,) probabilities.")

        # Apply feasibility barriers
        p_delete = self.apply_barriers(p_delete, lengths=lengths, t=t, T=T, L=L)
        return p_delete, Z

class NeuralGateModel(GateModelBase):
    """A concrete gated neural model that outputs P(delete | x_t, t, T, L).

    Design goals:
      • Sequence-aware: encodes the current sequence x_t via token + positional embeddings
        and masked mean pooling.
      • Time/length-aware: concatenates scalar features capturing progress and feasibility,
        e.g., gap g = |x_t| - L, remaining steps r = T - t, normalized ratios, etc.
      • Lightweight: a small MLP head maps the fused representation to a single logit.

    Inputs follow GateModelBase. The output is shape (B,) with values in [0, 1].
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        mlp_hidden: int = 256,
        max_len: int = 4096,
        pad_id: int = 0,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        import math
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.use_positional_encoding = bool(use_positional_encoding)

        self.token_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_emb = None
        if self.use_positional_encoding:
            # Learned positional embedding (simpler than sinusoidal; easy to swap later)
            self.pos_emb = nn.Embedding(self.max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Feature projection for scalar/time/length features
        # We will concatenate these scalars to the pooled token representation.
        self.feat_norm = nn.LayerNorm(d_model + 8)  # 8 engineered scalars below

        self.mlp = nn.Sequential(
            nn.Linear(d_model + 8, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        """Masked mean over sequence length dimension.
        x: (B, L, D), mask: (B, L) with True for valid tokens.
        Returns: (B, D)
        """
        mask = mask.to(dtype=x.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return x

    @staticmethod
    def _as_tensor(val: Union[int, Tensor], ref: Tensor) -> Tensor:
        if isinstance(val, Tensor):
            return val.to(device=ref.device, dtype=ref.dtype)
        return torch.full((ref.shape[0],), float(val), device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        token_ids: Tensor,
        lengths: Tensor,
        t: Union[int, Tensor],
        T: Union[int, Tensor],
        L: Union[int, Tensor],
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        B, L_max = token_ids.shape
        device = token_ids.device
        dtype = torch.float32

        if mask is None:
            mask = token_ids != self.pad_id
        mask = mask.to(device=device)

        # Token + positional embeddings
        x = self.token_emb(token_ids)
        if self.use_positional_encoding:
            pos_ids = torch.arange(L_max, device=device).unsqueeze(0).expand(B, L_max)
            x = x + self.pos_emb(pos_ids)
        x = self.dropout(x)

        # Pooled sequence representation
        seq_repr = self._masked_mean(x, mask)  # (B, d_model)

        # Scalar features (all float32 tensors shape (B,))
        lengths = lengths.to(device=device, dtype=dtype)
        t = self._as_tensor(t, lengths)
        T = self._as_tensor(T, lengths)
        L = self._as_tensor(L, lengths)

        gap = lengths - L               # g = |x_t| - L
        rem = T - t                     # r = T - t
        prog = t / (T.clamp_min(1.0))   # normalized time in [0, +)
        need = gap / (rem.clamp_min(1.0))  # how many deletions per remaining step (soft)
        len_ratio = lengths / (L.clamp_min(1.0))
        gap_ratio = gap / (lengths.clamp_min(1.0))
        rem_ratio = rem / (T.clamp_min(1.0))
        tgt_ratio = L / (T.clamp_min(1.0))

        feats = torch.stack([
            gap,
            rem,
            prog,
            need,
            len_ratio,
            gap_ratio,
            rem_ratio,
            tgt_ratio,
        ], dim=1)

        # Concatenate and predict
        fused = torch.cat([seq_repr, feats], dim=1)
        fused = self.feat_norm(fused)
        logit = self.mlp(fused).squeeze(-1)  # (B,)
        p_delete = torch.sigmoid(logit)
        return p_delete

