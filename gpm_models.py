from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union

import torch
from torch import Tensor
import torch.nn as nn
import esm
from typing import Optional


class GateModelBase(nn.Module, ABC):
    """Abstract base class for the gate model."""
    
    @abstractmethod
    def forward(
        self,
        token_ids: Optional[Tensor] = None,  # Made optional
        embeddings: Optional[Tensor] = None,  # New parameter
        lengths: Tensor = None,
        t: Union[int, Tensor] = None,
        T: Union[int, Tensor] = None,
        L: Union[int, Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError

class ProposalModelBase(nn.Module, ABC):
    """Abstract base class for the proposal model."""
    
    @abstractmethod
    def forward(
        self,
        token_ids: Optional[Tensor] = None,  # Made optional
        embeddings: Optional[Tensor] = None,  # New parameter
        lengths: Tensor = None,
        t: Union[int, Tensor] = None,
        T: Union[int, Tensor] = None,
        L: Union[int, Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError


class ESMGatedEditProposal(nn.Module):
    """GatedEditProposal with ESM-2 backbone for protein sequence embedding.
    
    This version embeds protein sequences using ESM-2 before passing them to
    the gate and proposal models.
    """
    
    def __init__(
        self, 
        gate: GateModelBase, 
        proposal: ProposalModelBase,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        freeze_esm: bool = True,
        esm_layer: int = 33,  # Which ESM layer to extract embeddings from
    ) -> None:
        super().__init__()
        self.gate = gate
        self.proposal = proposal
        self.freeze_esm = freeze_esm
        self.esm_layer = esm_layer
        
        # Load ESM-2 model
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        
        # Get ESM embedding dimension
        self.esm_embed_dim = self.esm_model.embed_tokens.embedding_dim
        
        # Freeze ESM parameters if requested
        if self.freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            self.esm_model.eval()
        
        # Projection layers to adapt ESM embeddings to gate/proposal input dimensions
        self.gate_projection = nn.Linear(self.esm_embed_dim, gate.d_model if hasattr(gate, 'd_model') else 128)
        self.proposal_projection = nn.Linear(self.esm_embed_dim, proposal.vocab_size if hasattr(proposal, 'vocab_size') else 26)
    
    def _embed_sequences(self, sequences: list) -> Tensor:
        """Convert protein sequences to ESM embeddings.
        
        Args:
            sequences: List of protein sequence strings
            
        Returns:
            Tensor of shape (B, L_max, esm_embed_dim)
        """
        # Format sequences for ESM
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        
        # Tokenize with ESM
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
        
        # Get embeddings
        with torch.no_grad() if self.freeze_esm else torch.enable_grad():
            results = self.esm_model(
                batch_tokens, 
                repr_layers=[self.esm_layer], 
                return_contacts=False
            )
        
        # Extract embeddings from specified layer
        embeddings = results["representations"][self.esm_layer]
        return embeddings
    
    def forward(
        self,
        sequences: list,  # Changed from token_ids to sequences
        lengths: Tensor,
        t: Union[int, Tensor],
        T: Union[int, Tensor],
        L: Union[int, Tensor],
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Compute (barriered_p_delete, Z_logits) using ESM embeddings.
        
        Args:
            sequences: List of protein sequence strings
            lengths: Tensor (B,) current sequence lengths
            t, T, L: int or Tensor (B,)
            mask: Optional BoolTensor (B, L_max)
            kwargs: Passed through to gate/proposal
            
        Returns:
            p_delete_barriered: Tensor (B,) in [0, 1]
            Z: Tensor (B, L_max, V+1)
        """
        # Get ESM embeddings
        esm_embeddings = self._embed_sequences(sequences)  # (B, L_max, esm_embed_dim)
        
        # Project embeddings for gate model
        gate_embeddings = self.gate_projection(esm_embeddings)  # (B, L_max, gate_d_model)
        
        # Project embeddings for proposal model  
        proposal_embeddings = self.proposal_projection(esm_embeddings)  # (B, L_max, vocab_size)
        
        # Call gate model with ESM embeddings
        p_delete = self.gate(
            token_ids=None,  # Gate will use embeddings instead
            embeddings=gate_embeddings,  # New parameter
            lengths=lengths,
            t=t,
            T=T,
            L=L,
            mask=mask,
            **kwargs,
        )
        
        # Call proposal model with ESM embeddings
        Z = self.proposal(
            token_ids=None,  # Proposal will use embeddings instead
            embeddings=proposal_embeddings,  # New parameter
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
    
    # Copy the static methods from original GatedEditProposal
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
        """Apply feasibility barriers to p_delete in a vectorized manner."""
        # Convert to tensors compatible with p_delete
        t = ESMGatedEditProposal._to_tensor(t, p_delete)
        T = ESMGatedEditProposal._to_tensor(T, p_delete)
        L = ESMGatedEditProposal._to_tensor(L, p_delete)
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


class ESMNeuralGateModel(GateModelBase):
    """Neural gate model that can work with ESM embeddings."""
    
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
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.use_positional_encoding = bool(use_positional_encoding)

        # Only create token embedding if we're not using ESM embeddings
        self.token_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_emb = None
        if self.use_positional_encoding:
            self.pos_emb = nn.Embedding(self.max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.feat_norm = nn.LayerNorm(d_model + 8)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 8, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        token_ids: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        lengths: Tensor = None,
        t: Union[int, Tensor] = None,
        T: Union[int, Tensor] = None,
        L: Union[int, Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        B, L_max = lengths.shape[0], lengths.max().item()
        device = lengths.device
        dtype = torch.float32

        if mask is None:
            if token_ids is not None:
                mask = token_ids != self.pad_id
            else:
                mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
        mask = mask.to(device=device)

        # Use embeddings if provided, otherwise use token embeddings
        if embeddings is not None:
            x = embeddings  # (B, L_max, d_model)
        else:
            x = self.token_emb(token_ids)
            if self.use_positional_encoding:
                pos_ids = torch.arange(L_max, device=device).unsqueeze(0).expand(B, L_max)
                x = x + self.pos_emb(pos_ids)
        
        x = self.dropout(x)

        # Pooled sequence representation
        seq_repr = self._masked_mean(x, mask)  # (B, d_model)

        # Scalar features (same as before)
        lengths = lengths.to(device=device, dtype=dtype)
        t = self._as_tensor(t, lengths)
        T = self._as_tensor(T, lengths)
        L = self._as_tensor(L, lengths)

        gap = lengths - L
        rem = T - t
        prog = t / (T.clamp_min(1.0))
        need = gap / (rem.clamp_min(1.0))
        len_ratio = lengths / (L.clamp_min(1.0))
        gap_ratio = gap / (lengths.clamp_min(1.0))
        rem_ratio = rem / (T.clamp_min(1.0))
        tgt_ratio = L / (T.clamp_min(1.0))

        feats = torch.stack([
            gap, rem, prog, need, len_ratio, gap_ratio, rem_ratio, tgt_ratio,
        ], dim=1)

        # Concatenate and predict
        fused = torch.cat([seq_repr, feats], dim=1)
        fused = self.feat_norm(fused)
        logit = self.mlp(fused).squeeze(-1)
        p_delete = torch.sigmoid(logit)
        return p_delete

