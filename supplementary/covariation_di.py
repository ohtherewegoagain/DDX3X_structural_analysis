#!/usr/bin/env python3

# covariation_di.py: Compute and visualize a GREMLIN-like direct information (DI) matrix
# Usage: python covariation_di.py msa_file.csv output_di.csv [output_plot.png]
# Requires EVcouplings (pip install evcouplings), numpy, pandas, matplotlib
# Visualizes a heatmap with thresholded contacts, mimicking GREMLIN style

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evcouplings.couplings import Couplings
from matplotlib.colors import LinearSegmentedColormap

# Check command-line arguments
if len(sys.argv) not in [3, 4]:
    print("Usage: python covariation_di.py msa_file.csv output_di.csv [output_plot.png]")
    sys.exit(1)

msa_file = sys.argv[1]  # Input MSA file (CSV with one-hot encoded residues)
output_di_file = sys.argv[2]  # Output DI matrix CSV
output_plot = sys.argv[3] if len(sys.argv) == 4 else "di_matrix.png"  # Default plot name

# Initialize EVcouplings and compute DI matrix
coupler = Couplings(msa_file, sequence_type="protein")
di_matrix = coupler.compute_di()

# Convert to numpy array and ensure symmetry
di_array = di_matrix.to_numpy()
n_residues = di_array.shape[0]

# Threshold contacts (e.g., top L/5 pairs, where L is sequence length)
threshold = int(n_residues / 5)
flat_indices = np.unravel_index(np.argsort(di_array, axis=None)[-threshold:], di_array.shape)
contact_mask = np.zeros_like(di_array, dtype=bool)
contact_mask[flat_indices] = True

# Mask lower triangle (since DI matrix is symmetric)
mask = np.triu(np.ones_like(di_array), k=1).astype(bool)

# Create a custom colormap (red-blue gradient, GREMLIN-like)
colors = ["blue", "white", "red"]
n_bins = 256
cmap = LinearSegmentedColormap.from_list("gremlin_cmap", colors, N=n_bins)

# Plotting
plt.figure(figsize=(10, 8))
plt.imshow(di_array, cmap=cmap, vmin=0, vmax=di_array.max(), interpolation='nearest')
plt.imshow(~mask, cmap='gray', alpha=0.3)  # Mask lower triangle with transparency

# Add thresholded contacts as dots
contact_coords = np.where(contact_mask & mask)
plt.scatter(contact_coords[1], contact_coords[0], c='black', s=10, alpha=0.5)

# Customize plot
plt.colorbar(label="Direct Information (DI) Score")
plt.xlabel("Residue Position")
plt.ylabel("Residue Position")
plt.title("GREMLIN-like DI Contact Map for DDX3X")

# Set ticks (assuming residues 1 to n_residues)
plt.xticks(np.arange(0, n_residues, max(1, n_residues // 10)), np.arange(1, n_residues + 1, max(1, n_residues // 10)))
plt.yticks(np.arange(0, n_residues, max(1, n_residues // 10)), np.arange(1, n_residues + 1, max(1, n_residues // 10)))

# Adjust layout and save
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

# Save DI matrix to CSV
di_matrix.to_csv(output_di_file)
print(f"GREMLIN-like DI matrix saved to {output_di_file}")
print(f"Visualization saved to {output_plot}")
