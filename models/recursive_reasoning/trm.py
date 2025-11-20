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


class SlotAttentionEncoder(nn.Module):
    """Slot Attention encoder for object-centric representations."""

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        hidden_size: int,
        num_iterations: int = 3,
        mlp_hidden_size: Optional[int] = None,
        epsilon: float = 1e-8,
        forward_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.forward_dtype = forward_dtype

        mlp_hidden_size = mlp_hidden_size or slot_dim

        # Learnable slot initializations (using Gaussian params)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Slot attention components
        self.norm_inputs = nn.LayerNorm(slot_dim, eps=epsilon)
        self.norm_slots = nn.LayerNorm(slot_dim, eps=epsilon)
        self.norm_mlp = nn.LayerNorm(slot_dim, eps=epsilon)

        # Attention: slots attend to input features
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, slot_dim)
        )

        # Project to hidden size if different
        if slot_dim != hidden_size:
            self.proj_to_hidden = nn.Linear(slot_dim, hidden_size)
        else:
            self.proj_to_hidden = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, num_inputs, slot_dim] - encoded grid features
        Returns:
            slots: [batch, num_slots, hidden_size]
        """
        batch_size, num_inputs, input_dim = inputs.shape

        # Initialize slots from learned distribution
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Normalize inputs
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # [batch, num_inputs, slot_dim]
        v = self.to_v(inputs)  # [batch, num_inputs, slot_dim]

        # Iterative slot attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention: slots attend to inputs
            q = self.to_q(slots)  # [batch, num_slots, slot_dim]

            # Compute attention weights
            scale = self.slot_dim ** -0.5
            attn_logits = torch.einsum('bnd,bmd->bnm', q, k) * scale  # [batch, num_slots, num_inputs]
            attn = F.softmax(attn_logits, dim=-1)  # Softmax over slots

            # Normalize attention weights across slots (competition)
            attn = attn / (attn.sum(dim=1, keepdim=True) + self.epsilon)  # [batch, num_slots, num_inputs]

            # Weighted mean of values
            updates = torch.einsum('bnm,bmd->bnd', attn, v)  # [batch, num_slots, slot_dim]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(batch_size, self.num_slots, self.slot_dim)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Project to hidden size
        slots = self.proj_to_hidden(slots)
        return slots.to(self.forward_dtype)


class CNNEncoder(nn.Module):
    """Simple CNN encoder to convert grids to feature maps."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        slot_dim: int,
        forward_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.forward_dtype = forward_dtype

        # Simple CNN: grid -> features
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, slot_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [batch, channels, height, width] or [batch, height, width]
        Returns:
            features: [batch, num_positions, slot_dim]
        """
        if grid.ndim == 3:
            grid = grid.unsqueeze(1)  # Add channel dimension

        # Encode
        features = self.encoder(grid.to(self.forward_dtype))  # [batch, slot_dim, height, width]

        # Flatten spatial dimensions
        batch_size, slot_dim, h, w = features.shape
        features = features.reshape(batch_size, slot_dim, h * w)
        features = features.permute(0, 2, 1)  # [batch, h*w, slot_dim]

        return features


class SpatialBroadcastDecoder(nn.Module):
    """Spatial broadcast decoder for slot attention."""

    def __init__(
        self,
        slot_dim: int,
        hidden_size: int,
        output_channels: int,
        output_height: int,
        output_width: int,
        decoder_hidden_dim: int = 64,
        forward_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.output_channels = output_channels
        self.output_height = output_height
        self.output_width = output_width
        self.forward_dtype = forward_dtype

        # Project from hidden_size to slot_dim if needed
        if hidden_size != slot_dim:
            self.proj_from_hidden = nn.Linear(hidden_size, slot_dim)
        else:
            self.proj_from_hidden = nn.Identity()

        # Spatial broadcast decoder
        # Input: [batch, num_slots, slot_dim + 2] (slot features + x,y coordinates)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(slot_dim + 2, decoder_hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_hidden_dim, decoder_hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_hidden_dim, decoder_hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_hidden_dim, output_channels + 1, kernel_size=5, stride=1, padding=2),  # +1 for alpha mask
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [batch, num_slots, hidden_size]
        Returns:
            reconstructed_grid: [batch, output_channels, output_height, output_width]
        """
        batch_size, num_slots, hidden_size = slots.shape

        # Project to slot_dim
        slots = self.proj_from_hidden(slots)  # [batch, num_slots, slot_dim]

        # Create spatial coordinates grid
        y_coords = torch.linspace(-1, 1, self.output_height, device=slots.device)
        x_coords = torch.linspace(-1, 1, self.output_width, device=slots.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=0)  # [2, height, width]
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, 2, height, width]

        # Decode each slot
        slot_reconstructions = []
        slot_masks = []

        for i in range(num_slots):
            slot = slots[:, i:i+1, :]  # [batch, 1, slot_dim]

            # Broadcast slot to spatial dimensions
            slot_spatial = slot.view(batch_size, self.slot_dim, 1, 1).expand(
                -1, -1, self.output_height, self.output_width
            )  # [batch, slot_dim, height, width]

            # Concatenate with coordinates
            decoder_input = torch.cat([slot_spatial, coords], dim=1)  # [batch, slot_dim+2, height, width]

            # Decode
            output = self.decoder(decoder_input.to(self.forward_dtype))  # [batch, output_channels+1, height, width]

            recon = output[:, :-1]  # [batch, output_channels, height, width]
            mask = output[:, -1:]   # [batch, 1, height, width]

            slot_reconstructions.append(recon)
            slot_masks.append(mask)

        # Stack and combine using masks
        recons = torch.stack(slot_reconstructions, dim=1)  # [batch, num_slots, channels, height, width]
        masks = torch.stack(slot_masks, dim=1)  # [batch, num_slots, 1, height, width]

        # Softmax over slots for masks
        masks = F.softmax(masks, dim=1)

        # Weighted sum
        reconstructed = (recons * masks).sum(dim=1)  # [batch, channels, height, width]

        return reconstructed


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

    # Slot Attention config
    use_slot_attention: bool = False  # Enable slot-based encoding
    num_slots: int = 10  # Number of object slots
    slot_dim: int = 64  # Dimension of each slot
    slot_iterations: int = 3  # Number of slot attention iterations
    slot_mlp_hidden: int = 128  # Hidden size for slot MLP
    grid_height: int = 30  # Height of input/output grids
    grid_width: int = 30  # Width of input/output grids
    grid_channels: int = 1  # Number of channels (1 for grayscale, can be num colors)
    cnn_hidden_dim: int = 64  # Hidden dim for CNN encoder
    decoder_hidden_dim: int = 64  # Hidden dim for decoder

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            # Use num_slots for slot attention, seq_len for token-based
            effective_seq_len = self.config.num_slots if self.config.use_slot_attention else self.config.seq_len
            self.mlp_t = SwiGLU(
                hidden_size=effective_seq_len + self.puzzle_emb_len, # L
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

        # Choose between token-based and slot-based encoding
        if self.config.use_slot_attention:
            # Slot Attention path
            self.cnn_encoder = CNNEncoder(
                input_channels=self.config.grid_channels,
                hidden_dim=self.config.cnn_hidden_dim,
                slot_dim=self.config.slot_dim,
                forward_dtype=self.forward_dtype
            )
            self.slot_encoder = SlotAttentionEncoder(
                num_slots=self.config.num_slots,
                slot_dim=self.config.slot_dim,
                hidden_size=self.config.hidden_size,
                num_iterations=self.config.slot_iterations,
                mlp_hidden_size=self.config.slot_mlp_hidden,
                forward_dtype=self.forward_dtype
            )
            self.decoder = SpatialBroadcastDecoder(
                slot_dim=self.config.slot_dim,
                hidden_size=self.config.hidden_size,
                output_channels=self.config.grid_channels,
                output_height=self.config.grid_height,
                output_width=self.config.grid_width,
                decoder_hidden_dim=self.config.decoder_hidden_dim,
                forward_dtype=self.forward_dtype
            )
            # seq_len becomes num_slots for slot attention
            effective_seq_len = self.config.num_slots
        else:
            # Token-based path (original)
            self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
            effective_seq_len = self.config.seq_len

        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Position encodings
        total_seq_len = effective_seq_len + self.puzzle_emb_len
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=total_seq_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(total_seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
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

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        if self.config.use_slot_attention:
            # Slot Attention path: input is a grid [batch, height, width] or [batch, channels, height, width]
            # CNN encode grid to features
            features = self.cnn_encoder(input)  # [batch, h*w, slot_dim]

            # Slot attention to extract object slots
            embedding = self.slot_encoder(features)  # [batch, num_slots, hidden_size]
        else:
            # Token-based path (original)
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

    def empty_carry(self, batch_size: int):
        effective_seq_len = self.config.num_slots if self.config.use_slot_attention else self.config.seq_len
        total_seq_len = effective_seq_len + self.puzzle_emb_len

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, total_seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, total_seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad

        if self.config.use_slot_attention:
            # Slot Attention path: decode slots to grid
            slots = z_H[:, self.puzzle_emb_len:]  # [batch, num_slots, hidden_size]
            output = self.decoder(slots)  # [batch, channels, height, width]
        else:
            # Token-based path (original)
            output = self.lm_head(z_H)[:, self.puzzle_emb_len:]  # [batch, seq_len, vocab_size]

        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        if hasattr(self.inner, 'puzzle_emb'):
            return self.inner.puzzle_emb
        return None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
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
