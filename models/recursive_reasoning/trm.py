from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # CNN-guided freezing and joint training
    cnn_checkpoint_path: Optional[str] = None
    cnn_freeze_threshold: float = 0.5
    cnn_loss_weight: float = 0.1  # Weight for CNN loss in joint training
    cnn_freeze_warmup_steps: int = 0  # Steps before using CNN for freezing

    # Dynamic iteration mode (replaces fixed H_cycles with CNN-guided stopping)
    dynamic_iterations: bool = False  # Enable dynamic iteration mode
    dynamic_error_threshold: float = 0.1  # Stop when CNN error rate < this (0.1 = 10% errors)
    dynamic_max_steps: int = 8  # Maximum iterations in dynamic mode (keep low for memory)
    dynamic_min_steps: int = 1  # Minimum iterations before checking threshold

    # Force error pixel changes (in-loop perturbation)
    force_error_changes: bool = False  # Force model to change pixels CNN marks as errors
    force_error_scale: float = 1.0  # Scale of perturbation applied to stuck error pixels

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        # Load CNN for correctness detection (jointly trained)
        self.correctness_cnn = None
        if self.config.cnn_checkpoint_path:
            from train_pixel_error_cnn import PixelErrorCNN
            if self.config.cnn_checkpoint_path == "init":
                # Initialize fresh CNN without loading weights
                self.correctness_cnn = PixelErrorCNN(hidden_dim=64)
            else:
                # Load from checkpoint as initialization
                self.correctness_cnn = PixelErrorCNN.from_checkpoint(self.config.cnn_checkpoint_path)
            # CNN is now trainable - no freezing

        # Step counter for CNN warmup
        self.register_buffer('cnn_step_counter', torch.tensor(0, dtype=torch.long))

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device=None):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        # Ensure init tensors are on the same device as carry
        H_init = self.H_init.to(carry.z_H.device)
        L_init = self.L_init.to(carry.z_L.device)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def _get_frozen_mask(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Get mask of positions that should be frozen.
        
        Returns:
            Tensor of shape [B, L, 1] where True = frozen (high confidence correct).
            Returns None if CNN is not enabled.
        """
        if self.correctness_cnn is None:
            return None
        
        with torch.no_grad():
            # Get current predictions from z_H
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]
            current_output = logits.argmax(dim=-1)  # [B, seq_len], values 0-11 (vocab space)
            
            # Convert from vocab space to color space
            # Vocab: PAD=0, EOS=1, colors 2-11 -> Color space: 0-9
            # Subtract 2 and clamp: PAD/EOS become 0, colors 2-11 become 0-9
            current_output_colors = (current_output - 2).clamp(0, 9)
            current_output_2d = current_output_colors.view(-1, 30, 30).long()  # [B, 30, 30]
            
            # Get input grid and convert from vocab space to color space
            input_colors = (batch["inputs"] - 2).clamp(0, 9)
            input_2d = input_colors.view(-1, 30, 30).long()  # [B, 30, 30]
            
            # Run CNN to get correctness confidence
            confidence = self.correctness_cnn.predict_proba(input_2d, current_output_2d)  # [B, 30, 30]
            
            # Create mask - True where frozen (high confidence = probably correct)
            frozen_2d = confidence > self.config.cnn_freeze_threshold  # [B, 30, 30]
            frozen_seq = frozen_2d.view(-1, self.config.seq_len)  # [B, seq_len]
            
            # Prepend False for puzzle_emb positions (never frozen)
            frozen_full = F.pad(frozen_seq, (self.puzzle_emb_len, 0), value=False)  # [B, seq_len + puzzle_emb_len]
            
            return frozen_full.unsqueeze(-1)  # [B, L, 1]

    def _compute_cnn_loss(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute CNN loss for joint training.

        The CNN learns to predict which pixels in the current TRM output are incorrect
        by comparing against ground truth labels.

        Returns:
            CNN loss tensor, or None if CNN is not enabled.
        """
        if self.correctness_cnn is None:
            return None

        # Get current TRM predictions (detached - CNN learns independently)
        with torch.no_grad():
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]
            current_output = logits.argmax(dim=-1)  # [B, seq_len]

            # Convert to color space (0-9)
            current_output_colors = (current_output - 2).clamp(0, 9)
            current_output_2d = current_output_colors.view(-1, 30, 30).long()

            # Input grid in color space
            input_colors = (batch["inputs"] - 2).clamp(0, 9)
            input_2d = input_colors.view(-1, 30, 30).long()

            # Ground truth labels
            labels = batch["labels"]
            # Handle IGNORE_LABEL_ID (-100) - these positions should be ignored
            valid_mask = (labels != IGNORE_LABEL_ID)
            label_colors = (labels.clamp(min=0) - 2).clamp(0, 9)
            label_2d = label_colors.view(-1, 30, 30)
            valid_2d = valid_mask.view(-1, 30, 30)

            # Actual correctness: 1.0 = correct, 0.0 = wrong
            actual_correct = (current_output_2d == label_2d).float()

        # CNN predicts correctness (this has gradients)
        predicted_logits = self.correctness_cnn(input_2d, current_output_2d)

        # BCE loss only on valid positions
        if valid_2d.any():
            loss = F.binary_cross_entropy_with_logits(
                predicted_logits[valid_2d],
                actual_correct[valid_2d]
            )
            return loss
        return None

    def _compute_cnn_error_rate(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> float:
        """Compute the average CNN-predicted error rate across the batch.

        Used for dynamic iteration stopping - returns the fraction of pixels
        that the CNN predicts are incorrect (error rate).

        Returns:
            Float between 0 and 1 representing the error rate.
            Returns 1.0 if CNN is not available.
        """
        if self.correctness_cnn is None:
            return 1.0

        with torch.no_grad():
            # Get current predictions from z_H
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]
            current_output = logits.argmax(dim=-1)  # [B, seq_len]

            # Convert from vocab space to color space
            current_output_colors = (current_output - 2).clamp(0, 9)
            current_output_2d = current_output_colors.view(-1, 30, 30).long()  # [B, 30, 30]

            # Get input grid and convert from vocab space to color space
            input_colors = (batch["inputs"] - 2).clamp(0, 9)
            input_2d = input_colors.view(-1, 30, 30).long()  # [B, 30, 30]

            # Get labels to create valid mask (only count output positions)
            labels = batch["labels"]
            valid_mask = (labels != IGNORE_LABEL_ID).view(-1, 30, 30)  # [B, 30, 30]

            # Run CNN to get correctness confidence (higher = more correct)
            confidence = self.correctness_cnn.predict_proba(input_2d, current_output_2d)  # [B, 30, 30]

            # Error rate = fraction of pixels with low confidence (CNN thinks they're wrong)
            # Only count valid output positions
            if valid_mask.any():
                error_pixels = (confidence < self.config.cnn_freeze_threshold) & valid_mask
                error_rate = error_pixels.float().sum() / valid_mask.float().sum()
                return error_rate.item()

            return 1.0

    def _get_error_mask(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Get mask of positions that CNN predicts are errors.
        
        Returns:
            Tensor of shape [B, seq_len] where True = error pixel that should change.
            Returns None if CNN is not enabled.
        """
        if self.correctness_cnn is None:
            return None
        
        with torch.no_grad():
            # Get current predictions from z_H
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]
            current_output = logits.argmax(dim=-1)  # [B, seq_len]
            
            # Convert from vocab space to color space
            current_output_colors = (current_output - 2).clamp(0, 9)
            current_output_2d = current_output_colors.view(-1, 30, 30).long()
            
            # Get input grid and convert from vocab space to color space
            input_colors = (batch["inputs"] - 2).clamp(0, 9)
            input_2d = input_colors.view(-1, 30, 30).long()
            
            # Run CNN to get correctness confidence
            confidence = self.correctness_cnn.predict_proba(input_2d, current_output_2d)  # [B, 30, 30]
            
            # Error mask - True where CNN thinks it's wrong (low confidence)
            error_2d = confidence < self.config.cnn_freeze_threshold  # [B, 30, 30]
            error_seq = error_2d.view(-1, self.config.seq_len)  # [B, seq_len]
            
            # Also mask out non-output positions (where labels are IGNORE_LABEL_ID)
            labels = batch["labels"]
            valid_mask = (labels != IGNORE_LABEL_ID)  # [B, seq_len]
            
            return error_seq & valid_mask  # [B, seq_len]

    def _force_error_changes_in_logits(
        self, 
        logits: torch.Tensor,
        error_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Modify logits to force different predictions for error pixels.
        
        For pixels that the CNN marks as errors, we mask out the current
        top prediction so the model must output something different.
        
        Args:
            logits: Output logits [B, seq_len, vocab]
            error_mask: Boolean mask [B, seq_len] where True = error pixel
            batch: Input batch (for reference)
            
        Returns:
            Modified logits with forced changes for error pixels
        """
        if not error_mask.any():
            return logits
        
        # Get current predictions
        current_preds = logits.argmax(dim=-1)  # [B, seq_len]
        
        # Create modified logits
        modified_logits = logits.clone()
        
        # For each position marked as error, set its current prediction logit to -inf
        # This forces argmax to pick a different token
        B, L, V = logits.shape
        batch_idx = torch.arange(B, device=logits.device)[:, None].expand(B, L)
        seq_idx = torch.arange(L, device=logits.device)[None, :].expand(B, L)
        
        # Scatter -inf to force different predictions for error pixels
        modified_logits[batch_idx[error_mask], seq_idx[error_mask], current_preds[error_mask]] = float('-inf')
        
        return modified_logits

    def _get_current_predictions(self, z_H: torch.Tensor) -> torch.Tensor:
        """Get current token predictions from hidden states.
        
        Args:
            z_H: Hidden states [B, L, D] (includes puzzle_emb positions)
            
        Returns:
            Predictions [B, seq_len] in vocab space
        """
        with torch.no_grad():
            logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [B, seq_len, vocab]
            return logits.argmax(dim=-1)  # [B, seq_len]

    def _apply_error_forcing_to_hidden(
        self,
        z_H: torch.Tensor,
        prev_preds: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply error forcing by perturbing hidden states for stuck error pixels.
        
        For pixels that the CNN marks as errors AND that haven't changed from
        previous predictions, we perturb z_H to push it away from the current
        prediction. This is done by subtracting a scaled version of the lm_head
        weight for the current prediction (like a gradient step).
        
        Args:
            z_H: Hidden states [B, L, D]
            prev_preds: Previous predictions [B, seq_len], or None for first iteration
            batch: Input batch
            
        Returns:
            Tuple of (modified z_H, current predictions for next iteration)
        """
        if self.correctness_cnn is None or not self.config.force_error_changes:
            current_preds = self._get_current_predictions(z_H)
            return z_H, current_preds
        
        perturbation_scale = self.config.force_error_scale
        
        # Get current predictions
        current_preds = self._get_current_predictions(z_H)  # [B, seq_len]
        
        # Get error mask from CNN
        error_mask = self._get_error_mask(z_H, batch)  # [B, seq_len]
        if error_mask is None or not error_mask.any():
            return z_H, current_preds
        
        # If we have previous predictions, only force pixels that are stuck
        if prev_preds is not None:
            stuck_mask = error_mask & (current_preds == prev_preds)
        else:
            # First iteration: force all error pixels
            stuck_mask = error_mask
        
        if not stuck_mask.any():
            return z_H, current_preds
        
        # Perturb z_H to push away from current predictions
        # Strategy: subtract the lm_head weight vector for the current prediction
        # This decreases the logit for that class
        
        # Get lm_head weights [vocab, hidden_size]
        lm_weight = self.lm_head.weight  # [vocab, hidden_size]
        
        # For stuck positions, get the weight vector of the current prediction
        # and subtract it from z_H (scaled)
        B, L_full, D = z_H.shape
        L = self.config.seq_len  # actual sequence length without puzzle_emb
        
        # Create perturbation tensor
        perturbation = torch.zeros_like(z_H)
        
        # Get the weight vectors for current predictions at stuck positions
        # stuck_mask: [B, seq_len], current_preds: [B, seq_len]
        # We need to index into lm_weight with current_preds where stuck
        
        # Expand stuck_mask to full sequence (add puzzle_emb prefix as False)
        stuck_mask_full = F.pad(stuck_mask, (self.puzzle_emb_len, 0), value=False)  # [B, L_full]
        
        # Also expand current_preds (pad with 0, won't be used)
        current_preds_full = F.pad(current_preds, (self.puzzle_emb_len, 0), value=0)  # [B, L_full]
        
        # Get weight vectors for all positions (we'll mask later)
        # lm_weight[current_preds_full] -> [B, L_full, D]
        pred_weights = lm_weight[current_preds_full]  # [B, L_full, D]
        
        # Apply perturbation only at stuck positions
        perturbation = torch.where(
            stuck_mask_full.unsqueeze(-1),
            -perturbation_scale * pred_weights,
            torch.zeros_like(pred_weights)
        )
        
        # Apply perturbation to z_H
        z_H_modified = z_H + perturbation.to(z_H.dtype)
        
        return z_H_modified, current_preds

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        # Check if we're past warmup period for CNN freezing
        use_cnn_freezing = (self.correctness_cnn is not None and
                           self.cnn_step_counter.item() >= self.config.cnn_freeze_warmup_steps)

        # Check if dynamic iteration mode is enabled
        use_dynamic = (self.config.dynamic_iterations and self.correctness_cnn is not None)

        if use_dynamic:
            # Dynamic iteration mode: iterate until CNN error rate < threshold or max steps
            max_steps = self.config.dynamic_max_steps
            min_steps = self.config.dynamic_min_steps
            error_threshold = self.config.dynamic_error_threshold

            # All iterations except the last run without grad
            # We track how many steps we've done and break early if error is low enough
            steps_done = 0
            prev_preds = None  # Track predictions for error forcing

            with torch.no_grad():
                while steps_done < max_steps - 1:  # -1 because final step is with grad
                    # Get frozen mask at start of this step (only after warmup)
                    frozen_mask = self._get_frozen_mask(z_H, batch) if use_cnn_freezing else None
                    if frozen_mask is not None:
                        z_H_cached, z_L_cached = z_H.clone(), z_L.clone()

                    # L_cycles iterations
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

                    # Restore frozen positions
                    if frozen_mask is not None:
                        z_H = torch.where(frozen_mask, z_H_cached, z_H)
                        z_L = torch.where(frozen_mask, z_L_cached, z_L)

                    # Apply error forcing: perturb z_H to change stuck error pixels
                    if self.config.force_error_changes:
                        z_H, prev_preds = self._apply_error_forcing_to_hidden(z_H, prev_preds, batch)

                    steps_done += 1

                    # Check if we can stop early (after min_steps)
                    if steps_done >= min_steps:
                        error_rate = self._compute_cnn_error_rate(z_H, batch)
                        if error_rate < error_threshold:
                            break

            # Final step with grad
            frozen_mask = self._get_frozen_mask(z_H, batch) if use_cnn_freezing else None
            if frozen_mask is not None:
                z_H_cached, z_L_cached = z_H.clone(), z_L.clone()

            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)

            # Restore frozen positions (with grad)
            if frozen_mask is not None:
                z_H = torch.where(frozen_mask, z_H_cached, z_H)
                z_L = torch.where(frozen_mask, z_L_cached, z_L)

            # Apply error forcing on final step too (no grad context for the forcing itself)
            if self.config.force_error_changes:
                with torch.no_grad():
                    z_H, _ = self._apply_error_forcing_to_hidden(z_H, prev_preds, batch)

        else:
            # Original fixed H_cycles mode
            # H_cycles-1 without grad
            prev_preds = None  # Track predictions for error forcing
            
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles-1):
                    # Get frozen mask at start of this H step (only after warmup)
                    frozen_mask = self._get_frozen_mask(z_H, batch) if use_cnn_freezing else None
                    if frozen_mask is not None:
                        z_H_cached, z_L_cached = z_H.clone(), z_L.clone()

                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

                    # Restore frozen positions
                    if frozen_mask is not None:
                        z_H = torch.where(frozen_mask, z_H_cached, z_H)
                        z_L = torch.where(frozen_mask, z_L_cached, z_L)

                    # Apply error forcing: perturb z_H to change stuck error pixels
                    if self.config.force_error_changes:
                        z_H, prev_preds = self._apply_error_forcing_to_hidden(z_H, prev_preds, batch)

            # 1 with grad
            # Get frozen mask for final step (only after warmup)
            frozen_mask = self._get_frozen_mask(z_H, batch) if use_cnn_freezing else None
            if frozen_mask is not None:
                z_H_cached, z_L_cached = z_H.clone(), z_L.clone()

            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)

            # Restore frozen positions (with grad - uses torch.where which is differentiable)
            if frozen_mask is not None:
                z_H = torch.where(frozen_mask, z_H_cached, z_H)
                z_L = torch.where(frozen_mask, z_L_cached, z_L)

            # Apply error forcing on final step too (no grad context for the forcing itself)
            if self.config.force_error_changes:
                with torch.no_grad():
                    z_H, _ = self._apply_error_forcing_to_hidden(z_H, prev_preds, batch)

        # Compute CNN loss for joint training (always, if CNN exists)
        cnn_loss = self._compute_cnn_loss(z_H, batch) if self.training else None

        # Increment step counter during training
        if self.training:
            self.cnn_step_counter.add_(1)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), cnn_loss


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), cnn_loss = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "cnn_loss": cnn_loss,  # May be None if CNN not enabled
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs