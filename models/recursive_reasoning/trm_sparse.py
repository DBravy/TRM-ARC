"""
TRM with CNN-Gated Sparse Attention

Key insight: The CNN "knows it when it sees it" - it can identify which pixels are wrong.
Instead of using this for freezing (post-hoc constraint), we use it for attention gating:
- Only positions the CNN thinks are WRONG emit queries (active reasoning)
- All positions serve as keys/values (full context available)
- Only active positions update through MLP

This implements cognitive attention: the visual system (CNN) directs where 
reasoning (TRM) happens, without controlling what it does.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import einops

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class TRMSparseInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMSparseCarry:
    inner_carry: TRMSparseInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRMSparseConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Architecture options
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # CNN-gated sparse attention config
    cnn_checkpoint_path: Optional[str] = None
    cnn_error_threshold: float = 0.5  # Below this = "needs work"
    cnn_warmup_steps: int = 0
    cnn_loss_weight: float = 0.1
    
    # Sparsity mode: "soft" (mask outputs) or "hard" (gather/scatter)
    sparsity_mode: str = "soft"
    
    # Minimum active ratio (prevent degenerate case of 0 active)
    min_active_ratio: float = 0.01


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class SparseAttention(nn.Module):
    """
    Attention with CNN-gated sparse queries.
    
    - Active positions (CNN says wrong): emit queries, gather context, update
    - Inactive positions (CNN says correct): serve as K/V context only, don't update
    
    Supports two modes:
    - "soft": All positions compute, but inactive outputs are masked (simpler, same gradients)
    - "hard": Only active positions compute (true sparsity, compute savings)
    """
    
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.output_size = head_dim * num_heads

        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        mode: str = "soft"
    ) -> torch.Tensor:
        """
        Args:
            cos_sin: Rotary embeddings (cos, sin)
            hidden_states: [B, L, D]
            active_mask: [B, L] bool tensor, True = active (needs work), False = inactive
            mode: "soft" or "hard"
        """
        if active_mask is None:
            # No sparsity - standard attention
            return self._full_attention(cos_sin, hidden_states)
        
        if mode == "soft":
            return self._soft_sparse_attention(cos_sin, hidden_states, active_mask)
        else:
            return self._hard_sparse_attention(cos_sin, hidden_states, active_mask)

    def _full_attention(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard full attention (no sparsity)."""
        B, L, _ = hidden_states.shape
        
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, L, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (q, k, v))
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = einops.rearrange(attn_out, 'B H S D -> B S H D')
        attn_out = attn_out.reshape(B, L, self.output_size)
        
        return self.o_proj(attn_out)

    def _soft_sparse_attention(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft sparsity: compute everything, mask inactive outputs.
        Same compute cost, but correct gradient flow.
        """
        B, L, _ = hidden_states.shape
        
        # Compute full QKV
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, L, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Zero out queries for inactive positions
        # active_mask: [B, L] -> [B, L, 1, 1] for broadcasting
        q = q * active_mask.unsqueeze(-1).unsqueeze(-1).to(q.dtype)

        q, k, v = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (q, k, v))
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = einops.rearrange(attn_out, 'B H S D -> B S H D')
        attn_out = attn_out.reshape(B, L, self.output_size)
        
        out = self.o_proj(attn_out)
        
        # Mask output for inactive positions (they shouldn't update)
        out = out * active_mask.unsqueeze(-1).to(out.dtype)
        
        return out

    def _hard_sparse_attention(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard sparsity: only compute for active positions.
        True compute savings, but more complex.
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # Compute K, V for ALL positions (they serve as context)
        qkv_full = self.qkv_proj(hidden_states)
        qkv_full = qkv_full.view(B, L, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        k_full = qkv_full[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        v_full = qkv_full[:, :, self.num_heads + self.num_key_value_heads:]
        
        if cos_sin is not None:
            cos, sin = cos_sin
            # Apply RoPE to keys (queries will be done per-batch below)
            _, k_full = apply_rotary_pos_emb(k_full, k_full, cos, sin)
        
        # Process each batch element separately (different active counts)
        outputs = torch.zeros(B, L, self.output_size, device=device, dtype=hidden_states.dtype)
        
        for b in range(B):
            active_idx = active_mask[b].nonzero(as_tuple=True)[0]  # [num_active]
            
            if len(active_idx) == 0:
                continue
            
            # Gather active hidden states
            h_active = hidden_states[b, active_idx]  # [num_active, D]
            
            # Compute Q only for active positions
            qkv_active = self.qkv_proj(h_active)  # [num_active, qkv_dim]
            qkv_active = qkv_active.view(len(active_idx), self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            q_active = qkv_active[:, :self.num_heads]  # [num_active, num_heads, head_dim]
            
            # Apply RoPE to queries at their actual positions
            if cos_sin is not None:
                cos, sin = cos_sin
                cos_active = cos[active_idx]  # [num_active, head_dim]
                sin_active = sin[active_idx]
                q_active = q_active.unsqueeze(0)  # [1, num_active, num_heads, head_dim]
                k_dummy = q_active.clone()
                q_active, _ = apply_rotary_pos_emb(q_active, k_dummy, cos_active, sin_active)
                q_active = q_active.squeeze(0)
            
            # Attention: active queries attend to full context
            # q_active: [num_active, num_heads, head_dim]
            # k_full[b]: [L, num_kv_heads, head_dim]
            # v_full[b]: [L, num_kv_heads, head_dim]
            
            q = einops.rearrange(q_active, 'S H D -> H S D').unsqueeze(0)  # [1, H, num_active, D]
            k = einops.rearrange(k_full[b], 'S H D -> H S D').unsqueeze(0)  # [1, H, L, D]
            v = einops.rearrange(v_full[b], 'S H D -> H S D').unsqueeze(0)
            
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [1, H, num_active, D]
            attn_out = einops.rearrange(attn_out.squeeze(0), 'H S D -> S (H D)')  # [num_active, output_size]
            
            out = self.o_proj(attn_out)  # [num_active, D]
            
            # Scatter back
            outputs[b, active_idx] = out
        
        return outputs


class SparseBlock(nn.Module):
    """Transformer block with CNN-gated sparse attention."""
    
    def __init__(self, config: TRMSparseConfig):
        super().__init__()
        self.config = config
        
        if config.mlp_t:
            self.puzzle_emb_len = config.puzzle_emb_len if config.puzzle_emb_len > 0 else -(config.puzzle_emb_ndim // -config.hidden_size)
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = SparseAttention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
            )
        
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        sparsity_mode: str = "soft"
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape
        
        if self.config.mlp_t:
            # MLP-T mode (transpose trick) - no sparsity here
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Sparse self-attention
            attn_out = self.self_attn(
                cos_sin=cos_sin,
                hidden_states=hidden_states,
                active_mask=active_mask,
                mode=sparsity_mode
            )
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        
        # MLP - also sparse if we have active_mask
        if active_mask is not None and sparsity_mode == "soft":
            mlp_out = self.mlp(hidden_states)
            mlp_out = mlp_out * active_mask.unsqueeze(-1).to(mlp_out.dtype)
            hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
        elif active_mask is not None and sparsity_mode == "hard":
            # Hard sparse MLP: only compute for active positions
            outputs = hidden_states.clone()
            for b in range(B):
                active_idx = active_mask[b].nonzero(as_tuple=True)[0]
                if len(active_idx) > 0:
                    h_active = hidden_states[b, active_idx]
                    mlp_out = self.mlp(h_active)
                    outputs[b, active_idx] = rms_norm(
                        h_active + mlp_out, 
                        variance_epsilon=self.norm_eps
                    )
            hidden_states = outputs
        else:
            mlp_out = self.mlp(hidden_states)
            hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
        
        return hidden_states


class SparseReasoningModule(nn.Module):
    """Reasoning module with sparse blocks."""
    
    def __init__(self, layers: List[SparseBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        sparsity_mode: str = "soft",
        **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                active_mask=active_mask,
                sparsity_mode=sparsity_mode,
                **kwargs
            )
        return hidden_states


class TRMSparse_Inner(nn.Module):
    """Inner TRM model with CNN-gated sparse attention."""
    
    def __init__(self, config: TRMSparseConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # I/O embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Puzzle embeddings
        self.puzzle_emb_len = config.puzzle_emb_len if config.puzzle_emb_len > 0 else -(config.puzzle_emb_ndim // -config.hidden_size)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype
            )

        # Position encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len, config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype
            )

        # Reasoning layers (sparse)
        self.L_level = SparseReasoningModule(
            layers=[SparseBlock(config) for _ in range(config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # Q head init (for halting)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

        # CNN for error detection (guides attention)
        self.correctness_cnn = None
        if config.cnn_checkpoint_path:
            from train_pixel_error_cnn import PixelErrorCNN
            if config.cnn_checkpoint_path == "init":
                self.correctness_cnn = PixelErrorCNN(hidden_dim=64)
            else:
                self.correctness_cnn = PixelErrorCNN.from_checkpoint(config.cnn_checkpoint_path)

        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device=None):
        return TRMSparseInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMSparseInnerCarry):
        H_init = self.H_init.to(carry.z_H.device)
        L_init = self.L_init.to(carry.z_L.device)
        return TRMSparseInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def _get_active_mask(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Get mask of positions that NEED WORK (active for computation).
        
        Returns:
            Tensor of shape [B, L] where True = active (CNN thinks wrong), False = inactive (CNN thinks correct).
            Returns None if CNN is not enabled or we're in warmup.
        """
        if self.correctness_cnn is None:
            return None
        
        if self.step_counter.item() < self.config.cnn_warmup_steps:
            return None

        with torch.no_grad():
            # Get current predictions
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
            current_output = logits.argmax(dim=-1)
            
            # Convert to color space
            current_output_colors = (current_output - 2).clamp(0, 9)
            current_output_2d = current_output_colors.view(-1, 30, 30).long()
            
            input_colors = (batch["inputs"] - 2).clamp(0, 9)
            input_2d = input_colors.view(-1, 30, 30).long()
            
            # CNN predicts correctness probability
            confidence = self.correctness_cnn.predict_proba(input_2d, current_output_2d)
            
            # ACTIVE = where CNN thinks we're WRONG (low confidence)
            active_2d = confidence < self.config.cnn_error_threshold
            active_seq = active_2d.view(-1, self.config.seq_len)
            
            # Ensure minimum active ratio (prevent degenerate case)
            min_active = int(self.config.seq_len * self.config.min_active_ratio)
            for b in range(active_seq.shape[0]):
                if active_seq[b].sum() < min_active:
                    # Force lowest confidence positions to be active
                    conf_flat = confidence[b].view(-1)
                    _, lowest_idx = conf_flat.topk(min_active, largest=False)
                    active_seq[b, lowest_idx] = True
            
            # Puzzle embedding positions are always active (they participate in reasoning)
            active_full = F.pad(active_seq, (self.puzzle_emb_len, 0), value=True)
            
            return active_full

    def _compute_cnn_loss(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute CNN loss for joint training."""
        if self.correctness_cnn is None:
            return None

        with torch.no_grad():
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
            current_output = logits.argmax(dim=-1)
            current_output_colors = (current_output - 2).clamp(0, 9)
            current_output_2d = current_output_colors.view(-1, 30, 30).long()

            input_colors = (batch["inputs"] - 2).clamp(0, 9)
            input_2d = input_colors.view(-1, 30, 30).long()

            labels = batch["labels"]
            valid_mask = (labels != IGNORE_LABEL_ID)
            label_colors = (labels.clamp(min=0) - 2).clamp(0, 9)
            label_2d = label_colors.view(-1, 30, 30)
            valid_2d = valid_mask.view(-1, 30, 30)

            actual_correct = (current_output_2d == label_2d).float()

        predicted_logits = self.correctness_cnn(input_2d, current_output_2d)

        if valid_2d.any():
            loss = F.binary_cross_entropy_with_logits(
                predicted_logits[valid_2d],
                actual_correct[valid_2d]
            )
            return loss
        return None

    def forward(
        self,
        carry: TRMSparseInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMSparseInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Dict]:
        
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_H, z_L = carry.z_H, carry.z_L

        use_sparsity = (
            self.correctness_cnn is not None and 
            self.step_counter.item() >= self.config.cnn_warmup_steps
        )
        sparsity_mode = self.config.sparsity_mode

        # Track sparsity stats
        total_active = 0
        total_positions = 0

        # H_cycles - 1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                active_mask = self._get_active_mask(z_H, batch) if use_sparsity else None
                
                if active_mask is not None:
                    total_active += active_mask.sum().item()
                    total_positions += active_mask.numel()

                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(
                        z_L, z_H + input_embeddings,
                        active_mask=active_mask,
                        sparsity_mode=sparsity_mode,
                        **seq_info
                    )
                z_H = self.L_level(
                    z_H, z_L,
                    active_mask=active_mask,
                    sparsity_mode=sparsity_mode,
                    **seq_info
                )

        # Final step with grad
        active_mask = self._get_active_mask(z_H, batch) if use_sparsity else None
        
        if active_mask is not None:
            total_active += active_mask.sum().item()
            total_positions += active_mask.numel()

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(
                z_L, z_H + input_embeddings,
                active_mask=active_mask,
                sparsity_mode=sparsity_mode,
                **seq_info
            )
        z_H = self.L_level(
            z_H, z_L,
            active_mask=active_mask,
            sparsity_mode=sparsity_mode,
            **seq_info
        )

        # CNN loss for joint training
        cnn_loss = self._compute_cnn_loss(z_H, batch) if self.training else None

        if self.training:
            self.step_counter.add_(1)

        # Stats
        stats = {}
        if total_positions > 0:
            stats["sparsity/active_ratio"] = total_active / total_positions
            stats["sparsity/inactive_ratio"] = 1 - (total_active / total_positions)

        # Outputs
        new_carry = TRMSparseInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), cnn_loss, stats


class TRMSparse(nn.Module):
    """Main TRM with CNN-gated sparse attention."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMSparseConfig(**config_dict)
        self.inner = TRMSparse_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRMSparseCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: TRMSparseCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMSparseCarry, Dict[str, torch.Tensor]]:

        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), cnn_loss, stats = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "cnn_loss": cnn_loss,
            **stats,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt, next_q_continue), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt, torch.maximum(next_q_halt, next_q_continue))
                    )

        return TRMSparseCarry(new_inner_carry, new_steps, halted, new_current_data), outputs