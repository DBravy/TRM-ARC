"""
Correspondence Module for ARC Puzzle Solver

This module provides functions for finding correspondences between input and output
objects in ARC puzzles. It uses shape features including structural properties,
Hu moments, color distribution, Fourier descriptors, and location encoding.

Usage:
    from correspondence_module import find_object_correspondences, compute_iou

    # Command-line visualization:
    python correspondence_module.py --puzzle-id 1990f7a8
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as scipy_label, binary_fill_holes


# =============================================================================
# Shape Feature Extraction
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
            'location': 1.0,
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


def compute_shape_similarity_matrix(
    input_masks: np.ndarray,
    output_masks: np.ndarray,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Compute similarity matrix between input and output masks using shape features.

    Only compares non-empty masks (where mask has > 0 pixels).

    Args:
        input_masks: (N, H_in, W_in) array of binary masks
        output_masks: (M, H_out, W_out) array of binary masks
        input_grid: (H_in, W_in) color grid for shape features
        output_grid: (H_out, W_out) color grid for shape features
        weights: Optional dict with feature weights (structural, moments, color, fourier, location)

    Returns:
        similarity: (num_valid_input, num_valid_output) similarity matrix
        valid_input_indices: indices of non-empty input masks
        valid_output_indices: indices of non-empty output masks
    """
    # Find non-empty masks
    input_counts = input_masks.sum(axis=(1, 2))
    output_counts = output_masks.sum(axis=(1, 2))

    valid_input_idx = np.where(input_counts > 0)[0].tolist()
    valid_output_idx = np.where(output_counts > 0)[0].tolist()

    if len(valid_input_idx) == 0 or len(valid_output_idx) == 0:
        return np.zeros((len(valid_input_idx), len(valid_output_idx))), valid_input_idx, valid_output_idx

    extractor = ShapeFeatureExtractor(n_fourier_coefficients=32)

    # Extract features for valid input masks
    input_features = []
    for idx in valid_input_idx:
        features = extractor.extract(input_masks[idx], input_grid)
        input_features.append(features)

    # Extract features for valid output masks
    output_features = []
    for idx in valid_output_idx:
        features = extractor.extract(output_masks[idx], output_grid)
        output_features.append(features)

    # Compute shape similarity matrix
    similarity = np.zeros((len(valid_input_idx), len(valid_output_idx)))
    for i, in_feat in enumerate(input_features):
        for j, out_feat in enumerate(output_features):
            similarity[i, j] = compute_shape_similarity(in_feat, out_feat, weights)

    return similarity, valid_input_idx, valid_output_idx


def find_correspondences_with_shape_features(
    input_masks: np.ndarray,
    output_masks: np.ndarray,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.3,
    margin: float = 0.1,
    weights: Optional[Dict[str, float]] = None
) -> List[Tuple[int, int, float]]:
    """Find correspondences between input and output masks using shape features.

    This is the main entry point for shape-feature-based correspondence finding.
    It extracts shape features from masks and uses the similarity to find matches.

    Args:
        input_masks: (N, H_in, W_in) array of binary masks
        output_masks: (M, H_out, W_out) array of binary masks
        input_grid: (H_in, W_in) color grid for shape features
        output_grid: (H_out, W_out) color grid for shape features
        threshold: Minimum similarity to consider a match
        margin: How close to the best score a match must be to be included
        weights: Optional dict with feature weights

    Returns:
        List of (input_idx, output_idx, similarity_score) sorted by score (highest first)
    """
    similarity, valid_input_idx, valid_output_idx = compute_shape_similarity_matrix(
        input_masks, output_masks, input_grid, output_grid, weights
    )

    return find_correspondences_from_similarity_matrix(
        similarity, valid_input_idx, valid_output_idx, threshold, margin
    )


def find_correspondences_greedy_shape(
    input_masks: np.ndarray,
    output_masks: np.ndarray,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.3,
    weights: Optional[Dict[str, float]] = None
) -> List[Tuple[int, int, float]]:
    """Find correspondences using shape features with greedy one-to-one matching.

    This uses the shape feature system (structural, moments, color, Fourier, location)
    but enforces one-to-one matching by greedily selecting the best matches first.

    Args:
        input_masks: (N, H_in, W_in) array of binary masks
        output_masks: (M, H_out, W_out) array of binary masks
        input_grid: (H_in, W_in) color grid for shape features
        output_grid: (H_out, W_out) color grid for shape features
        threshold: Minimum similarity to consider a match
        weights: Optional dict with feature weights

    Returns:
        List of (input_idx, output_idx, similarity_score) sorted by score (highest first)
    """
    similarity, valid_input_idx, valid_output_idx = compute_shape_similarity_matrix(
        input_masks, output_masks, input_grid, output_grid, weights
    )

    if similarity.size == 0:
        return []

    # Greedy one-to-one matching
    matches = []
    used_input = set()
    used_output = set()
    num_in = len(valid_input_idx)
    num_out = len(valid_output_idx)

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_in):
            if i in used_input:
                continue
            for j in range(num_out):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < threshold:
            break

        # Map back to original indices
        in_idx = valid_input_idx[best_i]
        out_idx = valid_output_idx[best_j]
        matches.append((in_idx, out_idx, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    # Sort by score (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def find_object_correspondences_shape(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_objects: List,
    output_objects: List,
    threshold: float = 0.3,
    weights: Optional[Dict[str, float]] = None
) -> List[Tuple[int, int, float]]:
    """Find correspondences between Object instances using shape features.

    This is a convenience function that converts Object instances (with row, col,
    height, width, pixels attributes) to masks and uses shape feature matching.

    Args:
        input_grid: The input grid array
        output_grid: The output grid array
        input_objects: List of input objects with pixels attribute
        output_objects: List of output objects with pixels attribute
        threshold: Minimum similarity threshold for a match
        weights: Optional dict with feature weights

    Returns:
        List of (input_idx, output_idx, score) tuples
    """
    if not input_objects or not output_objects:
        return []

    H_in, W_in = input_grid.shape
    H_out, W_out = output_grid.shape

    # Convert objects to masks
    input_masks = np.zeros((len(input_objects), H_in, W_in), dtype=bool)
    for i, obj in enumerate(input_objects):
        for r, c in obj.pixels:
            input_masks[i, r, c] = True

    output_masks = np.zeros((len(output_objects), H_out, W_out), dtype=bool)
    for i, obj in enumerate(output_objects):
        for r, c in obj.pixels:
            output_masks[i, r, c] = True

    return find_correspondences_greedy_shape(
        input_masks, output_masks, input_grid, output_grid, threshold, weights
    )


# =============================================================================
# IoU (Intersection over Union)
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute intersection over union between two binary masks.

    Handles different-sized masks by padding the smaller one to match the larger.

    Args:
        mask1: First binary mask (H1, W1)
        mask2: Second binary mask (H2, W2)

    Returns:
        IoU score in [0, 1]
    """
    # Handle different-sized masks (e.g., input/output grids have different dimensions)
    if mask1.shape != mask2.shape:
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        max_h, max_w = max(h1, h2), max(w1, w2)

        # Create padded versions
        padded1 = np.zeros((max_h, max_w), dtype=bool)
        padded2 = np.zeros((max_h, max_w), dtype=bool)
        padded1[:h1, :w1] = mask1
        padded2[:h2, :w2] = mask2

        mask1, mask2 = padded1, padded2

    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_iou_from_pixels(pixels1: set, pixels2: set) -> float:
    """
    Compute Intersection over Union between two pixel sets.

    This is an alternative to mask-based IoU when objects are represented
    as sets of (row, col) tuples.

    Args:
        pixels1: Set of (row, col) tuples for first object
        pixels2: Set of (row, col) tuples for second object

    Returns:
        IoU score in [0, 1]
    """
    if not pixels1 or not pixels2:
        return 0.0
    intersection = len(pixels1 & pixels2)
    union = len(pixels1 | pixels2)
    if union == 0:
        return 0.0
    return intersection / union


# =============================================================================
# Pattern Matching
# =============================================================================

def _extract_pattern_from_mask(grid: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the pixel pattern within a mask's bounding box.

    Args:
        grid: Original grid with color values
        mask: Binary mask indicating object pixels

    Returns:
        Pattern array (colors within bounding box) or None if mask is empty
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    # Extract the bounding box region from the original grid
    return grid[r_min:r_max+1, c_min:c_max+1].copy()


def extract_pattern_from_bbox(grid: np.ndarray, row: int, col: int,
                               height: int, width: int) -> np.ndarray:
    """
    Extract the pixel pattern within a bounding box.

    This is useful when working with Object instances that have row, col,
    height, and width properties.

    Args:
        grid: Original grid with color values
        row: Top-left row of bounding box
        col: Top-left column of bounding box
        height: Height of bounding box
        width: Width of bounding box

    Returns:
        Pattern array (colors within bounding box)
    """
    return grid[row:row + height, col:col + width].copy()


def pattern_similarity(pattern1: Optional[np.ndarray], pattern2: Optional[np.ndarray]) -> float:
    """
    Compute similarity between two patterns (0 to 1, higher = more similar).

    Args:
        pattern1: First pattern array
        pattern2: Second pattern array

    Returns:
        Similarity score in [0, 1]
    """
    if pattern1 is None or pattern2 is None:
        return 0.0
    # Must be same size for exact match
    if pattern1.shape != pattern2.shape:
        return 0.0
    # Count matching pixels
    matches = np.sum(pattern1 == pattern2)
    total = pattern1.size
    return matches / total if total > 0 else 0.0


# Keep private alias for backwards compatibility within this module
_pattern_similarity = pattern_similarity


# =============================================================================
# Object Correspondence Finding
# =============================================================================

def find_object_correspondences(
    input_labels: np.ndarray,
    input_colors: List[int],
    output_labels: np.ndarray,
    output_colors: List[int],
    iou_threshold: float = 0.1,
    input_grid: Optional[np.ndarray] = None,
    output_grid: Optional[np.ndarray] = None
) -> List[Tuple[int, int, float]]:
    """
    Find correspondences between input and output objects.

    Uses a combination of:
    1. Same color (required)
    2. IoU overlap (for objects that don't move much)
    3. Area similarity
    4. Pattern matching (when grids provided) - compares actual pixel patterns

    When objects move (IoU=0), pattern matching becomes the primary discriminator.

    Args:
        input_labels: Label mask for input (each object has unique label 1, 2, ...)
        input_colors: Dominant color for each input object (indexed by label - 1)
        output_labels: Label mask for output
        output_colors: Dominant color for each output object
        iou_threshold: Minimum score threshold for a match
        input_grid: Original input grid (optional, enables pattern matching)
        output_grid: Original output grid (optional, enables pattern matching)

    Returns:
        List of (input_idx, output_idx, score) tuples, where indices are 0-based.
    """
    num_input = len(input_colors)
    num_output = len(output_colors)

    if num_input == 0 or num_output == 0:
        return []

    # Pre-extract patterns if grids provided
    use_pattern_matching = input_grid is not None and output_grid is not None
    input_patterns = []
    output_patterns = []

    if use_pattern_matching:
        for i in range(num_input):
            mask = (input_labels == i + 1)
            input_patterns.append(_extract_pattern_from_mask(input_grid, mask))
        for j in range(num_output):
            mask = (output_labels == j + 1)
            output_patterns.append(_extract_pattern_from_mask(output_grid, mask))

    # Compute similarity matrix
    similarity = np.zeros((num_input, num_output), dtype=np.float32)

    for i in range(num_input):
        input_mask = (input_labels == i + 1)
        input_color = input_colors[i]

        for j in range(num_output):
            output_mask = (output_labels == j + 1)
            output_color = output_colors[j]

            # Must be same color
            if input_color != output_color:
                continue

            # IoU
            iou = compute_iou(input_mask, output_mask)

            # Area similarity
            input_area = input_mask.sum()
            output_area = output_mask.sum()
            if input_area > 0 and output_area > 0:
                area_ratio = min(input_area, output_area) / max(input_area, output_area)
            else:
                area_ratio = 0.0

            # Pattern similarity (if grids provided)
            if use_pattern_matching:
                pattern_sim = _pattern_similarity(input_patterns[i], output_patterns[j])
                # When IoU is high, objects overlap - use IoU + area
                # When IoU is low (objects moved), pattern matching is critical
                if iou > 0.3:
                    # Objects overlap significantly, use traditional approach
                    similarity[i, j] = 0.4 * iou + 0.3 * area_ratio + 0.3 * pattern_sim
                else:
                    # Objects moved - pattern matching is primary discriminator
                    similarity[i, j] = 0.1 * iou + 0.2 * area_ratio + 0.7 * pattern_sim
            else:
                # No grids provided - fall back to original behavior
                similarity[i, j] = 0.5 * iou + 0.5 * area_ratio

    # Greedy matching
    matches = []
    used_input = set()
    used_output = set()

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_input):
            if i in used_input:
                continue
            for j in range(num_output):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < iou_threshold:
            break

        matches.append((best_i, best_j, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    return matches


def find_object_correspondences_from_objects(
    input_objects: List,
    output_objects: List,
    iou_threshold: float = 0.1
) -> List[Tuple[int, int, float]]:
    """
    Find correspondences between input and output Object instances.

    This is a convenience wrapper for use with ordering_module.Object instances
    or similar dataclasses that have 'color' and 'pixels' attributes.

    Args:
        input_objects: List of input Object instances with color and pixels attributes
        output_objects: List of output Object instances
        iou_threshold: Minimum score threshold for a valid match

    Returns:
        List of (input_idx, output_idx, score) tuples
    """
    if not input_objects or not output_objects:
        return []

    num_input = len(input_objects)
    num_output = len(output_objects)

    # Compute similarity matrix
    similarity = np.zeros((num_input, num_output), dtype=np.float32)

    for i, in_obj in enumerate(input_objects):
        for j, out_obj in enumerate(output_objects):
            # Must be same color
            if in_obj.color != out_obj.color:
                continue

            # IoU using pixel sets
            iou = compute_iou_from_pixels(in_obj.pixels, out_obj.pixels)

            # Area similarity
            in_area = in_obj.area if hasattr(in_obj, 'area') else len(in_obj.pixels)
            out_area = out_obj.area if hasattr(out_obj, 'area') else len(out_obj.pixels)
            if in_area > 0 and out_area > 0:
                area_ratio = min(in_area, out_area) / max(in_area, out_area)
            else:
                area_ratio = 0.0

            # Combined score
            similarity[i, j] = 0.5 * iou + 0.5 * area_ratio

    # Greedy matching
    matches = []
    used_input = set()
    used_output = set()

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_input):
            if i in used_input:
                continue
            for j in range(num_output):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < iou_threshold:
            break

        matches.append((best_i, best_j, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    return matches


def find_correspondences_from_similarity_matrix(
    similarity_matrix: np.ndarray,
    valid_input_idx: List[int],
    valid_output_idx: List[int],
    threshold: float = 0.3,
    margin: float = 0.1
) -> List[Tuple[int, int, float]]:
    """
    Find matches from a pre-computed similarity matrix.

    Allows many-to-many correspondences when scores are close to the best match.
    For each output slot, includes the best matching input slot plus any others
    whose score is within `margin` of the best. Similarly for each input slot.

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


def find_correspondences_by_pattern(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_objects: List,
    output_objects: List,
    threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """
    Find correspondences by comparing actual pixel patterns.

    This is more robust than color+IoU+area when objects move but maintain
    their internal pattern. Uses pure pattern matching without color or IoU checks.

    Args:
        input_grid: The input grid array
        output_grid: The output grid array
        input_objects: List of input objects with row, col, height, width attributes
        output_objects: List of output objects with row, col, height, width attributes
        threshold: Minimum pattern similarity threshold for a match

    Returns:
        List of (input_idx, output_idx, score) tuples
    """
    if not input_objects or not output_objects:
        return []

    # Extract patterns for all objects
    input_patterns = [
        extract_pattern_from_bbox(input_grid, obj.row, obj.col, obj.height, obj.width)
        for obj in input_objects
    ]
    output_patterns = [
        extract_pattern_from_bbox(output_grid, obj.row, obj.col, obj.height, obj.width)
        for obj in output_objects
    ]

    # Build similarity matrix
    num_in = len(input_objects)
    num_out = len(output_objects)
    similarity = np.zeros((num_in, num_out))

    for i, in_pat in enumerate(input_patterns):
        for j, out_pat in enumerate(output_patterns):
            similarity[i, j] = pattern_similarity(in_pat, out_pat)

    # Greedy matching (best match first)
    matches = []
    used_input = set()
    used_output = set()

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_in):
            if i in used_input:
                continue
            for j in range(num_out):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < threshold:
            break

        matches.append((best_i, best_j, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    return matches


# =============================================================================
# Visualization (for --puzzle-id mode)
# =============================================================================

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


def _load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined"):
    """Load a single puzzle from the ARC dataset."""
    import json
    import os

    subsets = ["training", "evaluation", "training2", "evaluation2"]

    for subset in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"

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


def _draw_grid(ax, grid: np.ndarray, title: str = ""):
    """Draw an ARC grid on a matplotlib axis."""
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
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')


def _draw_object_outlines(ax, objects: List, color_map: dict, linewidth: float = 2):
    """Draw outlines around objects."""
    for obj in objects:
        if obj.id not in color_map:
            continue

        color = color_map[obj.id]
        # Draw bounding box
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (obj.col - 0.5, obj.row - 0.5),
            obj.width, obj.height,
            linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add object ID label
        center_row = obj.row + obj.height / 2
        center_col = obj.col + obj.width / 2
        ax.annotate(f'{obj.id}', (center_col, center_row),
                   color='white', fontsize=10, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8))


class CorrespondenceNavigator:
    """Interactive navigator for viewing correspondence examples."""

    def __init__(self, puzzle_id: str, examples_data: List[dict]):
        self.puzzle_id = puzzle_id
        self.examples_data = examples_data
        self.current_idx = 0
        self.fig = None

    def draw_example(self, ex_idx: int):
        """Draw a single example."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        if self.fig is not None:
            self.fig.clear()
        else:
            self.fig = plt.figure(figsize=(14, 8))

        data = self.examples_data[ex_idx]
        input_grid = data['input_grid']
        output_grid = data['output_grid']
        input_objects = data['input_objects']
        output_objects = data['output_objects']
        correspondences = data['correspondences']

        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        # Assign colors to correspondence pairs based on input object color
        input_color_map = {}
        output_color_map = {}
        for in_idx, out_idx, score in correspondences:
            in_obj = input_objects[in_idx]
            color = ARC_COLORS[in_obj.color]
            input_color_map[in_idx] = color
            output_color_map[out_idx] = color

        # Layout: input grid, arrows, output grid
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 0.3, 1], wspace=0.15)

        # Input grid
        ax_in = self.fig.add_subplot(gs[0, 0])
        _draw_grid(ax_in, input_grid, f"Input ({H_in}x{W_in}) - {len(input_objects)} objects")
        _draw_object_outlines(ax_in, input_objects, input_color_map, linewidth=3)

        # Arrow space
        ax_arrows = self.fig.add_subplot(gs[0, 1])
        ax_arrows.set_xlim(0, 1)
        ax_arrows.set_ylim(0, 1)
        ax_arrows.axis('off')

        # Draw arrows for correspondences
        for i, (in_idx, out_idx, score) in enumerate(correspondences):
            in_obj = input_objects[in_idx]
            color = ARC_COLORS[in_obj.color]

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
            ax_arrows.text(0.5, y_pos + 0.03, f"In{in_idx}→Out{out_idx}: {score:.2f}",
                          ha='center', va='bottom', fontsize=8, color=color)

        # Output grid
        ax_out = self.fig.add_subplot(gs[0, 2])
        _draw_grid(ax_out, output_grid, f"Output ({H_out}x{W_out}) - {len(output_objects)} objects")
        _draw_object_outlines(ax_out, output_objects, output_color_map, linewidth=3)

        n_examples = len(self.examples_data)
        self.fig.suptitle(
            f"Correspondence: {self.puzzle_id} - Example {ex_idx + 1}/{n_examples}\n"
            f"Found {len(correspondences)} correspondences  [← / → to navigate, q to quit]",
            fontsize=12, fontweight='bold'
        )
        plt.subplots_adjust(top=0.88)
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle keyboard navigation."""
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
        self.draw_example(0)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


def visualize_puzzle_correspondences(puzzle_id: str, data_root: str = "kaggle/combined",
                                      threshold: float = 0.3):
    """
    Visualize object correspondences for a specific puzzle.

    Uses shape features (structural, moments, color, Fourier, location) with
    greedy one-to-one matching.

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "1990f7a8")
        data_root: Path to puzzle data
        threshold: Minimum similarity score for correspondence matching
    """
    from object_module import extract_objects_from_grid

    print(f"Loading puzzle: {puzzle_id}")
    puzzle = _load_puzzle(puzzle_id, data_root)
    print(f"Found {len(puzzle['train'])} training examples")

    examples_data = []

    for i, example in enumerate(puzzle['train']):
        if 'output' not in example:
            continue

        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        print(f"\nProcessing example {i + 1}...")
        print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid)

        print(f"  Input objects: {len(input_objects)}, Output objects: {len(output_objects)}")

        # Find correspondences using shape features (greedy one-to-one)
        correspondences = find_object_correspondences_shape(
            input_grid, output_grid, input_objects, output_objects,
            threshold=threshold
        )

        print(f"  Found {len(correspondences)} correspondences:")
        for in_idx, out_idx, score in correspondences:
            in_obj = input_objects[in_idx]
            out_obj = output_objects[out_idx]
            print(f"    In[{in_idx}] (color={in_obj.color}, pos=({in_obj.row},{in_obj.col})) -> "
                  f"Out[{out_idx}] (color={out_obj.color}, pos=({out_obj.row},{out_obj.col})) "
                  f"score={score:.3f}")

        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_objects': input_objects,
            'output_objects': output_objects,
            'correspondences': correspondences,
        })

    # Show interactive visualization
    print("\nGenerating visualization...")
    navigator = CorrespondenceNavigator(puzzle_id, examples_data)
    navigator.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Correspondence Module for ARC Puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python correspondence_module.py --puzzle-id 1990f7a8
    python correspondence_module.py --puzzle-id 009d5c81 --threshold 0.3
        """
    )

    parser.add_argument("--puzzle-id", type=str, required=True,
                        help="ARC puzzle ID to visualize (e.g., 1990f7a8)")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to puzzle data")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Minimum shape similarity threshold for correspondences (default: 0.3)")

    args = parser.parse_args()

    visualize_puzzle_correspondences(
        args.puzzle_id,
        data_root=args.data_root,
        threshold=args.threshold
    )
