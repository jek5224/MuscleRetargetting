import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
IMAGE_SIZE = 1  # Sampling within a 1x1 unit square
BLUE_NOISE_MULTIPLIER = 10


def distance(x1, y1, x2, y2, space_size):
    """
    Compute toroidal distance between two points.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx > space_size / 2:
        dx = space_size - dx
    if dy > space_size / 2:
        dy = space_size - dy
    return dx**2 + dy**2


def generate_blue_noise(samples):
    """
    Generate blue noise samples using a dart-throwing algorithm.
    """
    sample_positions = []

    for i in range(samples):
        best_dist = 0
        best_x, best_y = None, None
        num_candidates = len(sample_positions) * BLUE_NOISE_MULTIPLIER + 1

        for i in range(num_candidates):
            if i == 0:
                x, y = 0.5, 0.5
            else:
                x, y = random.uniform(0, IMAGE_SIZE), random.uniform(0, IMAGE_SIZE)
            min_dist = float("inf")

            for sx, sy in sample_positions:
                min_dist = min(min_dist, distance(x, y, sx, sy, IMAGE_SIZE))

            if min_dist > best_dist:
                best_dist = min_dist
                best_x, best_y = x, y

        sample_positions.append((best_x, best_y))

    return np.array(sample_positions)


# Example Usage
N = 2  # Number of samples
samples = generate_blue_noise(N)

# Plot the result
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], c='blue', s=10)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.grid(True, linestyle='--', alpha=0.3)
plt.title(f"Blue Noise Sampling (N={N})")
plt.show()
