#!/usr/bin/env python3
"""
Slot Correspondence Visualization

Visualizes which objects in an ARC puzzle's input grid correspond to which
objects in the output grid using slot-based extraction and feature similarity.

Usage:
    python slot_correspondence_viz.py --puzzle-id 009d5c81
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as scipy_label, binary_fill_holes
import torch

from crm import VentralCNN, AffinitySlotAttention


# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: brown/maroon
]

# Distinct colors for slot correspondence visualization
SLOT_COLORS = [
    '#E6194B',  # red
    '#3CB44B',  # green
    '#FFE119',  # yellow
    '#4363D8',  # blue
    '#F58231',  # orange
    '#911EB4',  # purple
    '#46F0F0',  # cyan
    '#F032E6',  # magenta
    '#BCF60C',  # lime
    '#FABEBE',  # pink
    '#008080',  # teal
    '#E6BEFF',  # lavender
    '#9A6324',  # brown
    '#FFFAC8',  # beige
    '#800000',  # maroon
    '#AAFFC3',  # mint
    '#808000',  # olive
    '#FFD8B1',  # apricot
    '#000075',  # navy
    '#808080',  # grey
]


# =============================================================================
# Shape Feature Extraction (with optional Centroid-Based Location Encoding)
# =============================================================================

@dataclass
class ShapeFeatures:
    """Container for extracted shape features.

    Includes both position-invariant intrinsic properties and
    optional location encoding via centroid coordinates.
    """
    # Structural features
    area: int                          # Number of pixels
    bbox_width: int                    # Bounding box width
    bbox_height: int                   # Bounding box height
    perimeter: float                   # Boundary length
    density: float                     # area / bbox_area (1.0 = solid rectangle)
    aspect_ratio: float                # width / height (normalized to <= 1)
    compactness: float                 # 4π × area / perimeter² (1.0 = circle)
    euler_number: int                  # Topology: 1 - num_holes

    # Normalized moments (rotation/scale invariant)
    hu_moments: np.ndarray             # 7 Hu moments

    # Color information
    color_counts: np.ndarray           # (10,) count of each color in shape
    dominant_color: int                # Most common color
    num_colors: int                    # Number of distinct colors

    # Fourier descriptors (contour-based, position/rotation invariant)
    fourier_descriptors: np.ndarray    # Complex Fourier coefficients of contour

    # Location features (centroid-based encoding)
    centroid_y: float = 0.0            # Normalized y coordinate (0-1)
    centroid_x: float = 0.0            # Normalized x coordinate (0-1)
    
    def to_vector(self,
                  include_structural: bool = True,
                  include_moments: bool = True,
                  include_color: bool = True,
                  include_fourier: bool = True,
                  include_location: bool = True,
                  n_fourier: int = 16) -> np.ndarray:
        """Convert features to a fixed-size vector for similarity computation.

        Args:
            include_structural: Include basic structural features
            include_moments: Include Hu moments
            include_color: Include color distribution
            include_fourier: Include Fourier descriptors
            include_location: Include centroid-based location encoding
            n_fourier: Number of Fourier coefficients to include

        Returns:
            1D numpy array of features
        """
        parts = []

        if include_structural:
            # Normalize structural features to reasonable ranges
            structural = np.array([
                np.log1p(self.area) / 5.0,           # Log-scale area
                self.bbox_width / 30.0,               # Normalized by typical max
                self.bbox_height / 30.0,
                np.log1p(self.perimeter) / 4.0,      # Log-scale perimeter
                self.density,                         # Already 0-1
                self.aspect_ratio,                    # Already 0-1
                self.compactness,                     # Already 0-1
                (self.euler_number + 5) / 10.0,      # Shift and scale
            ], dtype=np.float32)
            parts.append(structural)

        if include_moments:
            # Hu moments are already normalized, but take log for stability
            hu_log = -np.sign(self.hu_moments) * np.log10(np.abs(self.hu_moments) + 1e-10)
            hu_normalized = hu_log / 20.0  # Scale to reasonable range
            parts.append(hu_normalized.astype(np.float32))

        if include_color:
            # Normalize color distribution
            color_dist = self.color_counts / (self.color_counts.sum() + 1e-10)
            parts.append(color_dist.astype(np.float32))

        if include_fourier:
            # Take magnitude of Fourier descriptors (phase encodes rotation)
            fd = self.fourier_descriptors[:n_fourier]
            if len(fd) < n_fourier:
                fd = np.pad(fd, (0, n_fourier - len(fd)))
            fd_mag = np.abs(fd)
            # Normalize by first coefficient (scale invariance)
            if fd_mag[0] > 1e-10:
                fd_mag = fd_mag / fd_mag[0]
            parts.append(fd_mag.astype(np.float32))

        if include_location:
            # Centroid-based location encoding (already normalized 0-1)
            location = np.array([
                self.centroid_y,
                self.centroid_x,
            ], dtype=np.float32)
            parts.append(location)

        return np.concatenate(parts)


class ShapeFeatureExtractor:
    """Extract features from shape masks.

    This class computes structural, topological, frequency-domain,
    and location features for shapes.
    """

    def __init__(self, n_fourier_coefficients: int = 32):
        """
        Args:
            n_fourier_coefficients: Number of Fourier descriptors to compute
        """
        self.n_fourier = n_fourier_coefficients

    def extract(self, mask: np.ndarray, color_grid: np.ndarray) -> ShapeFeatures:
        """Extract all features from a shape mask.

        Args:
            mask: (H, W) binary mask defining the shape
            color_grid: (H, W) integer array with color values 0-9

        Returns:
            ShapeFeatures dataclass with all extracted features
        """
        mask = mask.astype(bool)
        H, W = mask.shape

        # Handle empty mask
        if not mask.any():
            return self._empty_features()

        # Basic structural features
        area = int(mask.sum())

        # Bounding box (position-free: just dimensions)
        rows, cols = np.where(mask)
        bbox_height = rows.max() - rows.min() + 1
        bbox_width = cols.max() - cols.min() + 1
        bbox_area = bbox_height * bbox_width

        # Centroid-based location encoding (normalized to 0-1)
        centroid_y = float(rows.mean()) / max(H - 1, 1)
        centroid_x = float(cols.mean()) / max(W - 1, 1)
        
        # Density
        density = area / bbox_area if bbox_area > 0 else 0.0
        
        # Aspect ratio (always <= 1 for invariance to 90° rotation)
        aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
        
        # Perimeter (count of boundary pixels)
        perimeter = self._compute_perimeter(mask)
        
        # Compactness (isoperimetric quotient)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            compactness = min(compactness, 1.0)  # Clip numerical errors
        else:
            compactness = 1.0
        
        # Euler number (topology)
        euler_number = self._compute_euler_number(mask)
        
        # Hu moments (rotation/scale invariant)
        hu_moments = self._compute_hu_moments(mask)
        
        # Color features
        color_counts = np.zeros(10, dtype=np.int32)
        colors_in_shape = color_grid[mask]
        for c in range(10):
            color_counts[c] = (colors_in_shape == c).sum()
        dominant_color = int(np.argmax(color_counts))
        num_colors = int((color_counts > 0).sum())
        
        # Fourier descriptors of contour
        fourier_descriptors = self._compute_fourier_descriptors(mask)
        
        return ShapeFeatures(
            area=area,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            perimeter=perimeter,
            density=density,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            euler_number=euler_number,
            hu_moments=hu_moments,
            color_counts=color_counts,
            dominant_color=dominant_color,
            num_colors=num_colors,
            fourier_descriptors=fourier_descriptors,
            centroid_y=centroid_y,
            centroid_x=centroid_x,
        )
    
    def _empty_features(self) -> ShapeFeatures:
        """Return features for an empty mask."""
        return ShapeFeatures(
            area=0,
            bbox_width=0,
            bbox_height=0,
            perimeter=0.0,
            density=0.0,
            aspect_ratio=0.0,
            compactness=0.0,
            euler_number=0,
            hu_moments=np.zeros(7),
            color_counts=np.zeros(10, dtype=np.int32),
            dominant_color=0,
            num_colors=0,
            fourier_descriptors=np.zeros(self.n_fourier, dtype=np.complex128),
            centroid_y=0.0,
            centroid_x=0.0,
        )
    
    def _compute_perimeter(self, mask: np.ndarray) -> float:
        """Compute perimeter as count of boundary edges.
        
        Uses 4-connectivity: counts edges between foreground and background.
        """
        # Pad to handle edges
        padded = np.pad(mask, 1, mode='constant', constant_values=False)
        
        # Count transitions in each direction
        h_transitions = np.abs(padded[:, 1:].astype(int) - padded[:, :-1].astype(int)).sum()
        v_transitions = np.abs(padded[1:, :].astype(int) - padded[:-1, :].astype(int)).sum()
        
        return float(h_transitions + v_transitions)
    
    def _compute_euler_number(self, mask: np.ndarray) -> int:
        """Compute Euler number: #objects - #holes.
        
        For a single connected component, this is 1 - #holes.
        """
        # Count objects
        labeled, n_objects = scipy_label(mask)
        
        # Count holes: fill the shape and subtract
        filled = binary_fill_holes(mask)
        holes = filled & ~mask
        labeled_holes, n_holes = scipy_label(holes)
        
        return n_objects - n_holes
    
    def _compute_hu_moments(self, mask: np.ndarray) -> np.ndarray:
        """Compute 7 Hu moments (rotation, scale, translation invariant).
        
        These are derived from normalized central moments.
        """
        # Get coordinates relative to centroid (translation invariant)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return np.zeros(7)
        
        # Centroid
        m00 = len(rows)
        cy = rows.mean()
        cx = cols.mean()
        
        # Centered coordinates
        y = rows - cy
        x = cols - cx
        
        # Central moments (up to 3rd order)
        def mu(p, q):
            return np.sum((x ** p) * (y ** q))
        
        mu00 = m00
        mu20 = mu(2, 0)
        mu02 = mu(0, 2)
        mu11 = mu(1, 1)
        mu30 = mu(3, 0)
        mu03 = mu(0, 3)
        mu21 = mu(2, 1)
        mu12 = mu(1, 2)
        
        # Normalize by scale (using mu00)
        if mu00 == 0:
            return np.zeros(7)
        
        # Scale normalization factor
        def eta(mu_pq, p, q):
            gamma = (p + q) / 2 + 1
            return mu_pq / (mu00 ** gamma)
        
        n20 = eta(mu20, 2, 0)
        n02 = eta(mu02, 0, 2)
        n11 = eta(mu11, 1, 1)
        n30 = eta(mu30, 3, 0)
        n03 = eta(mu03, 0, 3)
        n21 = eta(mu21, 2, 1)
        n12 = eta(mu12, 1, 2)
        
        # 7 Hu moments (rotation invariant)
        hu = np.zeros(7)
        hu[0] = n20 + n02
        hu[1] = (n20 - n02)**2 + 4*n11**2
        hu[2] = (n30 - 3*n12)**2 + (3*n21 - n03)**2
        hu[3] = (n30 + n12)**2 + (n21 + n03)**2
        hu[4] = ((n30 - 3*n12) * (n30 + n12) * 
                 ((n30 + n12)**2 - 3*(n21 + n03)**2) +
                 (3*n21 - n03) * (n21 + n03) * 
                 (3*(n30 + n12)**2 - (n21 + n03)**2))
        hu[5] = ((n20 - n02) * ((n30 + n12)**2 - (n21 + n03)**2) +
                 4*n11 * (n30 + n12) * (n21 + n03))
        hu[6] = ((3*n21 - n03) * (n30 + n12) * 
                 ((n30 + n12)**2 - 3*(n21 + n03)**2) -
                 (n30 - 3*n12) * (n21 + n03) * 
                 (3*(n30 + n12)**2 - (n21 + n03)**2))
        
        return hu
    
    def _compute_fourier_descriptors(self, mask: np.ndarray) -> np.ndarray:
        """Compute Fourier descriptors of the shape contour.
        
        These capture the shape's boundary in a position and scale
        invariant way (rotation invariance via magnitude).
        
        The approach:
        1. Extract ordered contour points
        2. Represent contour as complex numbers: z = x + iy
        3. Compute FFT of the contour
        4. Normalize for scale/translation invariance
        """
        # Extract contour
        contour = self._extract_ordered_contour(mask)
        
        if len(contour) < 4:
            return np.zeros(self.n_fourier, dtype=np.complex128)
        
        # Convert to complex representation
        z = np.array([complex(x, y) for y, x in contour])
        
        # Center (translation invariance)
        z = z - z.mean()
        
        # Compute FFT
        Z = np.fft.fft(z)
        
        # Scale invariance: normalize by |Z[1]| (first harmonic)
        if np.abs(Z[1]) > 1e-10:
            Z = Z / np.abs(Z[1])
        
        # Take first n coefficients (low frequency = overall shape)
        # Skip Z[0] as it's the centroid (translation)
        n = min(self.n_fourier, len(Z) - 1)
        descriptors = np.zeros(self.n_fourier, dtype=np.complex128)
        descriptors[:n] = Z[1:n+1]
        
        return descriptors
    
    def _extract_ordered_contour(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Extract contour points in order (clockwise/counterclockwise).
        
        Uses a boundary tracing algorithm for discrete grids.
        """
        # Find boundary pixels using morphological operations
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        
        # If shape is 1 pixel thick, boundary is the shape itself
        if not boundary.any():
            boundary = mask.copy()
        
        # Get all boundary points
        rows, cols = np.where(boundary)
        if len(rows) == 0:
            return []
        
        # Start from topmost, leftmost point
        start_idx = np.lexsort((cols, rows))[0]
        start = (rows[start_idx], cols[start_idx])
        
        # Build set of boundary points for fast lookup
        boundary_set = set(zip(rows, cols))
        
        # Trace contour using 8-connectivity
        contour = [start]
        visited = {start}
        current = start
        
        # Direction vectors for 8-connectivity (clockwise from right)
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), 
                      (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        
        max_iterations = len(boundary_set) * 2
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            found_next = False
            
            # Try each direction
            for dy, dx in directions:
                next_point = (current[0] + dy, current[1] + dx)
                
                if next_point in boundary_set and next_point not in visited:
                    contour.append(next_point)
                    visited.add(next_point)
                    current = next_point
                    found_next = True
                    break
            
            if not found_next:
                # Check if we can close the loop
                for dy, dx in directions:
                    next_point = (current[0] + dy, current[1] + dx)
                    if next_point == start and len(contour) > 2:
                        return contour
                break
        
        return contour
    
    def extract_batch(self, masks: np.ndarray, color_grid: np.ndarray) -> List[ShapeFeatures]:
        """Extract features for multiple masks.
        
        Args:
            masks: (N, H, W) array of binary masks
            color_grid: (H, W) integer array with color values
            
        Returns:
            List of ShapeFeatures, one per mask
        """
        return [self.extract(mask, color_grid) for mask in masks]


def compute_shape_similarity(features1: ShapeFeatures,
                             features2: ShapeFeatures,
                             weights: Optional[Dict[str, float]] = None) -> float:
    """Compute similarity between two shapes based on their features.

    Args:
        features1, features2: Shape features to compare
        weights: Optional dict with keys 'structural', 'moments', 'color',
                 'fourier', 'location' specifying relative importance
                 (default: equal weights, location=0.0)

    Returns:
        Similarity score in [0, 1]
    """
    if weights is None:
        weights = {
            'structural': 1.0,
            'moments': 1.0,
            'color': 1.0,
            'fourier': 1.0,
            'location': 1.0,  # Off by default for backwards compatibility
        }

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    similarities = {}

    # Structural similarity (Euclidean distance -> similarity)
    if weights.get('structural', 0) > 0:
        v1 = features1.to_vector(include_structural=True, include_moments=False,
                                  include_color=False, include_fourier=False,
                                  include_location=False)
        v2 = features2.to_vector(include_structural=True, include_moments=False,
                                  include_color=False, include_fourier=False,
                                  include_location=False)
        dist = np.linalg.norm(v1 - v2)
        similarities['structural'] = np.exp(-dist)

    # Moment similarity
    if weights.get('moments', 0) > 0:
        v1 = features1.to_vector(include_structural=False, include_moments=True,
                                  include_color=False, include_fourier=False,
                                  include_location=False)
        v2 = features2.to_vector(include_structural=False, include_moments=True,
                                  include_color=False, include_fourier=False,
                                  include_location=False)
        dist = np.linalg.norm(v1 - v2)
        similarities['moments'] = np.exp(-dist)

    # Color similarity (histogram intersection)
    if weights.get('color', 0) > 0:
        c1 = features1.color_counts / (features1.color_counts.sum() + 1e-10)
        c2 = features2.color_counts / (features2.color_counts.sum() + 1e-10)
        similarities['color'] = np.minimum(c1, c2).sum()

    # Fourier similarity
    if weights.get('fourier', 0) > 0:
        fd1 = np.abs(features1.fourier_descriptors[:16])
        fd2 = np.abs(features2.fourier_descriptors[:16])
        # Normalize
        if fd1[0] > 1e-10:
            fd1 = fd1 / fd1[0]
        if fd2[0] > 1e-10:
            fd2 = fd2 / fd2[0]
        dist = np.linalg.norm(fd1 - fd2)
        similarities['fourier'] = np.exp(-dist)

    # Location similarity (Euclidean distance between centroids)
    if weights.get('location', 0) > 0:
        loc1 = np.array([features1.centroid_y, features1.centroid_x])
        loc2 = np.array([features2.centroid_y, features2.centroid_x])
        dist = np.linalg.norm(loc1 - loc2)
        # Scale factor of 2.0 so that max distance (sqrt(2)) gives ~0.24 similarity
        similarities['location'] = np.exp(-2.0 * dist)

    # Weighted average
    score = sum(weights[k] * similarities.get(k, 0) for k in weights) / total_weight
    return float(score)


def print_shape_features(features: ShapeFeatures, slot_idx: int, prefix: str = "") -> None:
    """Print shape features in a readable format.

    Args:
        features: ShapeFeatures to print
        slot_idx: Slot index for labeling
        prefix: Optional prefix for indentation
    """
    print(f"{prefix}Slot {slot_idx}:")
    print(f"{prefix}  Structural: area={features.area}, bbox={features.bbox_width}×{features.bbox_height}, "
          f"perim={features.perimeter:.1f}")
    print(f"{prefix}  Properties: density={features.density:.3f}, aspect={features.aspect_ratio:.3f}, "
          f"compact={features.compactness:.3f}, euler={features.euler_number}")
    print(f"{prefix}  Location: centroid=({features.centroid_y:.3f}, {features.centroid_x:.3f})")
    print(f"{prefix}  Color: dominant={features.dominant_color}, num_colors={features.num_colors}")
    
    # Show top Fourier descriptor magnitudes
    fd_mag = np.abs(features.fourier_descriptors[:8])
    if fd_mag[0] > 0:
        fd_mag = fd_mag / fd_mag[0]
    fd_str = ", ".join(f"{m:.2f}" for m in fd_mag)
    print(f"{prefix}  Fourier[0:8]: [{fd_str}]")


def extract_and_print_slot_features(masks: torch.Tensor, 
                                     color_grid: np.ndarray, 
                                     valid_indices: List[int],
                                     label: str = "") -> List[ShapeFeatures]:
    """Extract and print shape features for all valid slots.
    
    Args:
        masks: (max_slots, H, W) tensor of masks
        color_grid: (H, W) integer array
        valid_indices: Which slot indices to process
        label: Label for this set of slots (e.g., "Input" or "Output")
        
    Returns:
        List of ShapeFeatures for each valid slot
    """
    extractor = ShapeFeatureExtractor()
    features_list = []
    
    if label:
        print(f"\n{label} Shape Features:")
        print("-" * 60)
    
    for idx in valid_indices:
        mask = masks[idx].cpu().numpy() if isinstance(masks[idx], torch.Tensor) else masks[idx]
        features = extractor.extract(mask, color_grid)
        features_list.append(features)
        if label:
            print_shape_features(features, idx, "  ")
    
    return features_list


def load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined") -> Dict:
    """Load a single puzzle from the ARC dataset."""
    subsets = [
        ("training", "training"),
        ("evaluation", "evaluation"),
        ("training2", "training2"),
        ("evaluation2", "evaluation2"),
    ]

    for subset_name, subset_key in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset_key}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset_key}_solutions.json"

        if not os.path.exists(challenges_path):
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        if puzzle_id not in puzzles:
            continue

        puzzle = puzzles[puzzle_id]

        # Load solutions if available
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            if puzzle_id in solutions:
                for i, sol in enumerate(solutions[puzzle_id]):
                    if i < len(puzzle["test"]):
                        puzzle["test"][i]["output"] = sol

        return puzzle

    raise ValueError(f"Puzzle '{puzzle_id}' not found in dataset")


def grid_to_onehot(grid: np.ndarray) -> torch.Tensor:
    """Convert a color grid to one-hot encoding.

    Args:
        grid: (H, W) integer array with values 0-9

    Returns:
        (1, 10, H, W) one-hot encoded tensor
    """
    H, W = grid.shape
    onehot = np.zeros((10, H, W), dtype=np.float32)
    for c in range(10):
        onehot[c] = (grid == c).astype(np.float32)
    return torch.from_numpy(onehot).unsqueeze(0)  # (1, 10, H, W)


def process_grid(grid: List[List[int]],
                 encoder: VentralCNN,
                 slot_attention: AffinitySlotAttention,
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a single grid through VentralCNN and AffinitySlotAttention.

    Args:
        grid: List of lists with integer color values 0-9
        encoder: VentralCNN instance
        slot_attention: AffinitySlotAttention instance
        device: torch device

    Returns:
        slots: (max_slots, slot_dim) slot embeddings
        masks: (max_slots, H, W) hard pixel-to-slot assignments
    """
    # Convert to numpy array
    grid_np = np.array(grid, dtype=np.int64)
    H, W = grid_np.shape

    # One-hot encode
    onehot = grid_to_onehot(grid_np).to(device)  # (1, 10, H, W)

    # Get per-pixel features from encoder
    features = encoder(onehot)  # (1, H, W, feature_dim)

    # Get color grid as tensor
    color_grid = torch.from_numpy(grid_np).unsqueeze(0).to(device)  # (1, H, W)

    # Extract slots
    slots, masks = slot_attention(features, color_grid)  # (1, max_slots, slot_dim), (1, max_slots, H, W)

    return slots.squeeze(0), masks.squeeze(0)  # Remove batch dimension


def compute_slot_similarity(input_masks: torch.Tensor,
                            output_masks: torch.Tensor,
                            input_grid: np.ndarray,
                            output_grid: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    """Compute similarity between input and output slots using shape features.

    Only compares non-empty slots (where mask has > 0 pixels).

    Args:
        input_masks: (max_slots, H_in, W_in)
        output_masks: (max_slots, H_out, W_out)
        input_grid: (H_in, W_in) color grid for shape features
        output_grid: (H_out, W_out) color grid for shape features

    Returns:
        similarity: (num_valid_input, num_valid_output) similarity matrix
        valid_input_indices: indices of non-empty input slots
        valid_output_indices: indices of non-empty output slots
    """
    # Find non-empty slots
    input_counts = input_masks.sum(dim=(1, 2))  # (max_slots,)
    output_counts = output_masks.sum(dim=(1, 2))  # (max_slots,)

    valid_input_idx = (input_counts > 0).nonzero(as_tuple=True)[0].tolist()
    valid_output_idx = (output_counts > 0).nonzero(as_tuple=True)[0].tolist()

    if len(valid_input_idx) == 0 or len(valid_output_idx) == 0:
        return np.zeros((len(valid_input_idx), len(valid_output_idx))), valid_input_idx, valid_output_idx

    extractor = ShapeFeatureExtractor(n_fourier_coefficients=32)

    # Extract features for valid input slots
    input_features = []
    for idx in valid_input_idx:
        mask = input_masks[idx].cpu().numpy()
        features = extractor.extract(mask, input_grid)
        input_features.append(features)

    # Extract features for valid output slots
    output_features = []
    for idx in valid_output_idx:
        mask = output_masks[idx].cpu().numpy()
        features = extractor.extract(mask, output_grid)
        output_features.append(features)

    # Compute shape similarity matrix
    shape_similarity = np.zeros((len(valid_input_idx), len(valid_output_idx)))
    for i, in_feat in enumerate(input_features):
        for j, out_feat in enumerate(output_features):
            shape_similarity[i, j] = compute_shape_similarity(in_feat, out_feat)

    return shape_similarity, valid_input_idx, valid_output_idx


def find_correspondences(similarity_matrix: np.ndarray,
                         valid_input_idx: List[int],
                         valid_output_idx: List[int],
                         threshold: float = 0.3,
                         margin: float = 0.1) -> List[Tuple[int, int, float]]:
    """Find matches from similarity matrix, allowing many-to-many when scores are close.

    For each output slot, includes the best matching input slot plus any others
    whose score is within `margin` of the best. Similarly for each input slot.
    This allows many-to-many correspondences only when multiple slots are
    similarly good matches.

    Args:
        similarity_matrix: (num_input, num_output) similarity scores
        valid_input_idx: indices of valid input slots
        valid_output_idx: indices of valid output slots
        threshold: minimum similarity to consider a match
        margin: how close to the best score a match must be to be included
                (e.g., 0.1 means include if score >= best_score - 0.1)

    Returns:
        List of (input_slot_idx, output_slot_idx, similarity_score)
        sorted by similarity score (highest first)
    """
    if similarity_matrix.size == 0:
        return []

    # Compute best scores for each row (input) and column (output)
    best_per_output = similarity_matrix.max(axis=0)  # Best input score for each output
    best_per_input = similarity_matrix.max(axis=1)   # Best output score for each input

    correspondences_set = set()

    # Include a correspondence only if it passes the margin check from BOTH directions:
    # 1. Score is within margin of the best input for this output
    # 2. Score is within margin of the best output for this input
    for in_i, in_idx in enumerate(valid_input_idx):
        for out_i, out_idx in enumerate(valid_output_idx):
            score = similarity_matrix[in_i, out_i]
            if score < threshold:
                continue

            # Check margin from output's perspective (best input for this output)
            if score < best_per_output[out_i] - margin:
                continue

            # Check margin from input's perspective (best output for this input)
            if score < best_per_input[in_i] - margin:
                continue

            correspondences_set.add((in_idx, out_idx, float(score)))

    # Convert to list and sort by similarity score (highest first)
    correspondences = list(correspondences_set)
    correspondences.sort(key=lambda x: x[2], reverse=True)

    return correspondences


def get_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Get the centroid of a mask.

    Args:
        mask: (H, W) binary mask

    Returns:
        (y, x) centroid coordinates
    """
    if mask.sum() == 0:
        return (0, 0)

    y_coords, x_coords = np.where(mask > 0)
    return (y_coords.mean(), x_coords.mean())


def get_mask_outline(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Get the outline pixels of a mask.

    Args:
        mask: (H, W) binary mask

    Returns:
        List of (y, x) coordinates forming the outline
    """
    from scipy import ndimage

    # Dilate and subtract to get outline
    dilated = ndimage.binary_dilation(mask)
    outline = dilated & ~mask

    # Also include edge pixels of the mask itself
    eroded = ndimage.binary_erosion(mask)
    edge = mask & ~eroded

    combined = outline | edge
    y_coords, x_coords = np.where(combined)

    return list(zip(y_coords, x_coords))


def draw_grid(ax, grid: np.ndarray, title: str = ""):
    """Draw an ARC grid on a matplotlib axis.

    Args:
        ax: matplotlib axis
        grid: (H, W) integer array with values 0-9
        title: title for the subplot
    """
    H, W = grid.shape

    # Create RGB image
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(10):
        color = np.array([int(ARC_COLORS[c][i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        mask = (grid == c)
        rgb_image[mask] = color

    ax.imshow(rgb_image, interpolation='nearest')

    # Draw grid lines
    for i in range(H + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.5)
    for j in range(W + 1):
        ax.axvline(x=j - 0.5, color='gray', linewidth=0.5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # Invert y-axis
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')


def draw_slot_outlines(ax, masks: np.ndarray, slot_indices: List[int],
                       slot_color_map: Dict[int, List[str]], linewidth: float = 2):
    """Draw outlines around slot masks.

    Supports many-to-many correspondences by drawing multiple colored outlines
    with slight offsets when a slot has multiple colors.

    Args:
        ax: matplotlib axis
        masks: (max_slots, H, W) mask array
        slot_indices: which slot indices to draw
        slot_color_map: mapping from slot index to list of colors
        linewidth: line width for outlines
    """
    for slot_idx in slot_indices:
        if slot_idx not in slot_color_map:
            continue

        mask = masks[slot_idx].cpu().numpy()
        if mask.sum() == 0:
            continue

        colors = slot_color_map[slot_idx]
        n_colors = len(colors)

        # Find contours using simple edge detection
        H, W = mask.shape
        for color_i, color in enumerate(colors):
            # Offset each color slightly outward for visibility when multiple colors
            offset = 0.05 * color_i if n_colors > 1 else 0
            lw = linewidth if n_colors == 1 else max(1.5, linewidth - 0.3 * color_i)

            for y in range(H):
                for x in range(W):
                    if mask[y, x] > 0:
                        # Check neighbors - if any neighbor is outside or zero, draw edge
                        # Top edge
                        if y == 0 or mask[y-1, x] == 0:
                            ax.plot([x - 0.5 - offset, x + 0.5 + offset],
                                   [y - 0.5 - offset, y - 0.5 - offset],
                                   color=color, linewidth=lw)
                        # Bottom edge
                        if y == H - 1 or mask[y+1, x] == 0:
                            ax.plot([x - 0.5 - offset, x + 0.5 + offset],
                                   [y + 0.5 + offset, y + 0.5 + offset],
                                   color=color, linewidth=lw)
                        # Left edge
                        if x == 0 or mask[y, x-1] == 0:
                            ax.plot([x - 0.5 - offset, x - 0.5 - offset],
                                   [y - 0.5 - offset, y + 0.5 + offset],
                                   color=color, linewidth=lw)
                        # Right edge
                        if x == W - 1 or mask[y, x+1] == 0:
                            ax.plot([x + 0.5 + offset, x + 0.5 + offset],
                                   [y - 0.5 - offset, y + 0.5 + offset],
                                   color=color, linewidth=lw)


def draw_slot_thumbnail(ax, grid: np.ndarray, mask: np.ndarray, slot_idx: int,
                        border_colors: List[str] = None):
    """Draw a single slot as a thumbnail showing the masked region.

    Supports many-to-many correspondences by drawing multiple colored borders.

    Args:
        ax: matplotlib axis
        grid: (H, W) the full color grid
        mask: (H, W) binary mask for this slot
        slot_idx: slot index for labeling
        border_colors: list of colors for the border (if part of correspondences)
    """
    H, W = grid.shape

    # Create RGB image with mask applied
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(10):
        color = np.array([int(ARC_COLORS[c][i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        cell_mask = (grid == c)
        rgb_image[cell_mask] = color

    # Dim non-masked pixels
    mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
    for y in range(H):
        for x in range(W):
            if mask_np[y, x] == 0:
                rgb_image[y, x] = rgb_image[y, x] * 0.2 + 0.1  # Dim background

    ax.imshow(rgb_image, interpolation='nearest')

    # Draw border(s) if specified
    if border_colors and len(border_colors) > 0:
        # Use first color for main border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors[0])
            spine.set_linewidth(3)
        # If multiple colors, add indicator dots/marks
        if len(border_colors) > 1:
            # Draw small color swatches in corner to show all correspondences
            for i, bc in enumerate(border_colors[1:], 1):
                ax.plot([0.05 + i * 0.12], [0.95], 's', color=bc, markersize=4,
                       transform=ax.transAxes, clip_on=False)
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"S{slot_idx}", fontsize=8)


def draw_slot_grid(fig, grid_spec, grid: np.ndarray, masks: torch.Tensor,
                   valid_indices: List[int], slot_color_map: Dict[int, List[str]],
                   title: str = ""):
    """Draw a grid of slot thumbnails.

    Args:
        fig: matplotlib figure
        grid_spec: SubplotSpec for this area
        grid: (H, W) the color grid
        masks: (max_slots, H, W) all slot masks
        valid_indices: which slot indices are non-empty
        slot_color_map: mapping from slot index to list of correspondence colors
        title: title for the grid
    """
    n_slots = len(valid_indices)
    if n_slots == 0:
        ax = fig.add_subplot(grid_spec)
        ax.text(0.5, 0.5, "No slots", ha='center', va='center')
        ax.axis('off')
        return

    # Determine grid layout (aim for roughly square)
    n_cols = min(4, n_slots)
    n_rows = (n_slots + n_cols - 1) // n_cols

    inner_gs = grid_spec.subgridspec(n_rows + 1, n_cols, hspace=0.3, wspace=0.1,
                                      height_ratios=[0.15] + [1] * n_rows)

    # Title row
    ax_title = fig.add_subplot(inner_gs[0, :])
    ax_title.text(0.5, 0.5, title, ha='center', va='center', fontsize=10, fontweight='bold')
    ax_title.axis('off')

    # Draw each slot
    for i, slot_idx in enumerate(valid_indices):
        row = i // n_cols + 1  # +1 for title row
        col = i % n_cols
        ax = fig.add_subplot(inner_gs[row, col])

        mask = masks[slot_idx]
        border_colors = slot_color_map.get(slot_idx, None)
        draw_slot_thumbnail(ax, grid, mask, slot_idx, border_colors)


class ExampleNavigator:
    """Interactive navigator for viewing examples one at a time."""

    def __init__(self, puzzle_id: str, examples_data: List[Dict]):
        self.puzzle_id = puzzle_id
        self.examples_data = examples_data
        self.current_idx = 0
        self.fig = None

    def draw_example(self, ex_idx: int):
        """Draw a single example."""
        if self.fig is not None:
            self.fig.clear()
        else:
            self.fig = plt.figure(figsize=(16, 10))

        data = self.examples_data[ex_idx]
        input_grid = data['input_grid']
        output_grid = data['output_grid']
        input_masks = data['input_masks']
        output_masks = data['output_masks']
        valid_input_idx = data['valid_input_idx']
        valid_output_idx = data['valid_output_idx']
        correspondences = data['correspondences']

        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        # Assign colors to correspondence pairs based on dominant slot color
        # Each slot can have multiple colors if it participates in multiple correspondences
        slot_color_map_input: Dict[int, List[str]] = {}
        slot_color_map_output: Dict[int, List[str]] = {}
        for i, (in_idx, out_idx, score) in enumerate(correspondences):
            # Get the dominant color from the input slot's mask
            in_mask = input_masks[in_idx].cpu().numpy()
            colors_in_slot = input_grid[in_mask > 0]
            if len(colors_in_slot) > 0:
                dominant_color = int(np.bincount(colors_in_slot).argmax())
            else:
                dominant_color = 0
            color = ARC_COLORS[dominant_color]

            if in_idx not in slot_color_map_input:
                slot_color_map_input[in_idx] = []
            slot_color_map_input[in_idx].append(color)
            if out_idx not in slot_color_map_output:
                slot_color_map_output[out_idx] = []
            slot_color_map_output[out_idx].append(color)

        # Layout: 2 rows, 3 columns
        main_gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 0.3, 1],
                                        height_ratios=[1, 0.8],
                                        hspace=0.3, wspace=0.15)

        # Row 1: Input grid
        ax_in = self.fig.add_subplot(main_gs[0, 0])
        draw_grid(ax_in, input_grid, f"Input ({H_in}x{W_in})")
        draw_slot_outlines(ax_in, input_masks, list(slot_color_map_input.keys()),
                           slot_color_map_input, linewidth=3)

        # Row 1: Arrow space
        ax_arrows = self.fig.add_subplot(main_gs[0, 1])
        ax_arrows.set_xlim(0, 1)
        ax_arrows.set_ylim(0, 1)
        ax_arrows.axis('off')

        # Draw arrows for correspondences
        for i, (in_idx, out_idx, score) in enumerate(correspondences):
            # Use the dominant color from the input slot
            in_mask = input_masks[in_idx].cpu().numpy()
            colors_in_slot = input_grid[in_mask > 0]
            if len(colors_in_slot) > 0:
                dominant_color = int(np.bincount(colors_in_slot).argmax())
            else:
                dominant_color = 0
            color = ARC_COLORS[dominant_color]

            # Stagger arrows vertically to avoid overlap
            y_pos = 0.9 - (i * 0.12) % 0.8

            arrow = FancyArrowPatch(
                (0.1, y_pos),
                (0.9, y_pos),
                connectionstyle="arc3,rad=0.0",
                arrowstyle="->,head_width=0.1,head_length=0.08",
                color=color,
                linewidth=2,
                alpha=0.8
            )
            ax_arrows.add_patch(arrow)
            ax_arrows.text(0.5, y_pos + 0.03, f"S{in_idx}→S{out_idx}: {score:.2f}",
                          ha='center', va='bottom', fontsize=7, color=color)

        # Row 1: Output grid
        ax_out = self.fig.add_subplot(main_gs[0, 2])
        draw_grid(ax_out, output_grid, f"Output ({H_out}x{W_out})")
        draw_slot_outlines(ax_out, output_masks, list(slot_color_map_output.keys()),
                           slot_color_map_output, linewidth=3)

        # Row 2: Input slot thumbnails
        draw_slot_grid(self.fig, main_gs[1, 0], input_grid, input_masks,
                       valid_input_idx, slot_color_map_input,
                       f"Input Slots ({len(valid_input_idx)})")

        # Row 2: Empty middle
        ax_mid = self.fig.add_subplot(main_gs[1, 1])
        ax_mid.axis('off')

        # Row 2: Output slot thumbnails
        draw_slot_grid(self.fig, main_gs[1, 2], output_grid, output_masks,
                       valid_output_idx, slot_color_map_output,
                       f"Output Slots ({len(valid_output_idx)})")

        n_examples = len(self.examples_data)
        self.fig.suptitle(
            f"Puzzle: {self.puzzle_id} - Example {ex_idx + 1}/{n_examples}\n"
            f"[← / → to navigate, q to quit]",
            fontsize=12, fontweight='bold'
        )
        plt.subplots_adjust(top=0.90)
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right' or event.key == 'n':
            self.current_idx = (self.current_idx + 1) % len(self.examples_data)
            self.draw_example(self.current_idx)
        elif event.key == 'left' or event.key == 'p':
            self.current_idx = (self.current_idx - 1) % len(self.examples_data)
            self.draw_example(self.current_idx)
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """Display the interactive navigator."""
        self.draw_example(0)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


def visualize_all_examples(puzzle_id: str, examples_data: List[Dict]):
    """Visualize examples with interactive navigation.

    Args:
        puzzle_id: puzzle identifier
        examples_data: list of dicts containing:
            - input_grid, output_grid
            - input_masks, output_masks
            - valid_input_idx, valid_output_idx
            - correspondences
    """
    navigator = ExampleNavigator(puzzle_id, examples_data)
    navigator.show()


def test_shape_features():
    """Test and visualize shape feature extraction on sample shapes."""
    import matplotlib.pyplot as plt
    
    # Create some test shapes
    shapes = {}
    
    # Solid rectangle
    rect = np.zeros((10, 10), dtype=bool)
    rect[2:8, 3:7] = True
    shapes['rectangle'] = rect
    
    # Square
    square = np.zeros((10, 10), dtype=bool)
    square[2:7, 2:7] = True
    shapes['square'] = square
    
    # L-shape
    L = np.zeros((10, 10), dtype=bool)
    L[2:8, 2:4] = True
    L[6:8, 2:7] = True
    shapes['L-shape'] = L
    
    # T-shape
    T = np.zeros((10, 10), dtype=bool)
    T[2:4, 2:8] = True
    T[2:7, 4:6] = True
    shapes['T-shape'] = T
    
    # Plus/cross
    plus = np.zeros((10, 10), dtype=bool)
    plus[4:6, 2:8] = True
    plus[2:8, 4:6] = True
    shapes['plus'] = plus
    
    # Hollow rectangle (frame)
    frame = np.zeros((10, 10), dtype=bool)
    frame[2:8, 2:8] = True
    frame[3:7, 3:7] = False
    shapes['frame'] = frame
    
    # Diagonal line
    diag = np.zeros((10, 10), dtype=bool)
    for i in range(8):
        diag[1+i, 1+i] = True
    shapes['diagonal'] = diag
    
    # Create dummy color grid (all one color)
    color_grid = np.ones((10, 10), dtype=np.int64)
    
    # Extract features
    extractor = ShapeFeatureExtractor()
    features = {name: extractor.extract(mask, color_grid) for name, mask in shapes.items()}
    
    # Print feature comparison
    print("\n" + "="*80)
    print("SHAPE FEATURE EXTRACTION TEST")
    print("="*80)
    
    print("\nStructural Features:")
    print(f"{'Shape':<12} {'Area':>6} {'W×H':>6} {'Perim':>6} {'Dens':>6} {'Aspect':>6} {'Compact':>7} {'Euler':>6}")
    print("-"*70)
    for name, feat in features.items():
        print(f"{name:<12} {feat.area:>6} {feat.bbox_width}×{feat.bbox_height:>3} "
              f"{feat.perimeter:>6.1f} {feat.density:>6.2f} {feat.aspect_ratio:>6.2f} "
              f"{feat.compactness:>7.3f} {feat.euler_number:>6}")
    
    # Compute pairwise similarities
    print("\n\nPairwise Shape Similarities:")
    names = list(shapes.keys())
    print(f"{'':>12}", end='')
    for name in names:
        print(f"{name[:8]:>9}", end='')
    print()
    
    for i, name1 in enumerate(names):
        print(f"{name1:<12}", end='')
        for j, name2 in enumerate(names):
            sim = compute_shape_similarity(features[name1], features[name2])
            print(f"{sim:>9.3f}", end='')
        print()
    
    # Visualize shapes and their Fourier descriptors
    fig, axes = plt.subplots(2, len(shapes), figsize=(14, 6))
    
    for i, (name, mask) in enumerate(shapes.items()):
        # Original shape
        axes[0, i].imshow(mask, cmap='Blues', interpolation='nearest')
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].axis('off')
        
        # Fourier magnitude spectrum
        fd = features[name].fourier_descriptors
        fd_mag = np.abs(fd[:16])
        if fd_mag[0] > 0:
            fd_mag = fd_mag / fd_mag[0]
        axes[1, i].bar(range(len(fd_mag)), fd_mag, color='steelblue')
        axes[1, i].set_xlabel('Freq', fontsize=8)
        axes[1, i].set_ylabel('Mag' if i == 0 else '', fontsize=8)
        axes[1, i].set_ylim(0, 1.5)
        axes[1, i].tick_params(labelsize=7)
    
    axes[0, 0].set_ylabel('Shape', fontsize=10)
    axes[1, 0].set_ylabel('Fourier Desc.', fontsize=10)
    
    fig.suptitle('Shape Features: Masks and Fourier Descriptors', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return shapes, features


def main():
    parser = argparse.ArgumentParser(description="Visualize slot correspondences in ARC puzzles")
    parser.add_argument('--puzzle-id', required=False, help="ARC puzzle ID (e.g., 009d5c81)")
    parser.add_argument('--data-root', default="kaggle/combined", help="Path to ARC dataset")
    parser.add_argument('--feature-dim', type=int, default=64, help="VentralCNN output dimension")
    parser.add_argument('--slot-dim', type=int, default=64, help="Slot embedding dimension")
    parser.add_argument('--max-slots', type=int, default=40, help="Maximum number of slots")
    parser.add_argument('--threshold', type=float, default=0.3, help="Similarity threshold for correspondences")
    parser.add_argument('--margin', type=float, default=0.1,
                        help="Margin for many-to-many matching (include matches within this distance of best)")
    parser.add_argument('--test-shapes', action='store_true',
                        help="Run shape feature extraction test instead of puzzle visualization")
    parser.add_argument('--verbose-shapes', action='store_true',
                        help="Print detailed shape features for each slot")
    args = parser.parse_args()
    
    # Run shape test mode if requested
    if args.test_shapes:
        test_shape_features()
        return
    
    # Require puzzle-id for normal mode
    if not args.puzzle_id:
        parser.error("--puzzle-id is required unless using --test-shapes")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load puzzle
    print(f"Loading puzzle: {args.puzzle_id}")
    puzzle = load_puzzle(args.puzzle_id, args.data_root)
    print(f"Found {len(puzzle['train'])} training examples")

    # Initialize models with random weights
    encoder = VentralCNN(out_dim=args.feature_dim).to(device)
    slot_attention = AffinitySlotAttention(
        input_dim=args.feature_dim,
        slot_dim=args.slot_dim,
        max_slots=args.max_slots
    ).to(device)

    encoder.eval()
    slot_attention.eval()

    # Collect data for all examples
    examples_data = []

    # Process each training example
    for i, example in enumerate(puzzle['train']):
        print(f"\nProcessing example {i + 1}/{len(puzzle['train'])}...")

        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")

        # Extract slot masks from both grids
        with torch.no_grad():
            _, input_masks = process_grid(example['input'], encoder, slot_attention, device)
            _, output_masks = process_grid(example['output'], encoder, slot_attention, device)

        # Compute similarity using shape features
        similarity, valid_in, valid_out = compute_slot_similarity(
            input_masks, output_masks, input_grid, output_grid
        )
        print(f"  Valid input slots: {len(valid_in)}, Valid output slots: {len(valid_out)}")
        
        # Print detailed shape features if requested
        if args.verbose_shapes:
            extract_and_print_slot_features(input_masks, input_grid, valid_in, "Input")
            extract_and_print_slot_features(output_masks, output_grid, valid_out, "Output")

        # Find correspondences
        correspondences = find_correspondences(similarity, valid_in, valid_out, args.threshold, args.margin)
        print(f"  Found {len(correspondences)} correspondences")

        for in_idx, out_idx, score in correspondences:
            print(f"    Input slot {in_idx} -> Output slot {out_idx} (similarity: {score:.3f})")

        # Collect data for visualization
        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_masks': input_masks,
            'output_masks': output_masks,
            'valid_input_idx': valid_in,
            'valid_output_idx': valid_out,
            'correspondences': correspondences,
        })

    # Visualize all examples in one figure
    print("\nGenerating visualization...")
    visualize_all_examples(args.puzzle_id, examples_data)


if __name__ == "__main__":
    main()