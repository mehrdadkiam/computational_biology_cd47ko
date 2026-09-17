"""
MELD: pairwise Wasserstein distance between genotype-condition categories.

Pipeline:
  1. Load the AnnData object and define sample labels.
  2. PCA.
  3. MELD sample-associated density estimates (cached to CSV).
  4. Average the per-sample densities within each genotype-condition category.
  5. Normalized Wasserstein distance for all 15 category pairs + bar plot.
"""

import math
import os
from collections import defaultdict
from itertools import combinations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.cm import ScalarMappable
from scipy.stats import wasserstein_distance

import meld

np.random.seed(42)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FILE_DIR = "/home/kiamari/Single_cell/Single_cell_codes/"
FILE_NAME = (
    "adata_combined_YekBasteh_ALLGenes_Neuron_And_Glia_harmonized_"
    "DiffThrshld_0.1_batch_NO_mt_NO_SexInBatchEffect.h5ad"
)
DENSITY_FILE = "sample_densities_pca_YekBasteh.csv"
BIN_GRANULARITY = 0.002
OUT_FIG = "Wasserstein_YekBasteh.png"


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #
adata = sc.read_h5ad(os.path.join(FILE_DIR, FILE_NAME))
adata.obs["sample_labels"] = adata.obs["sample_id"]

print(adata.obs[["sample_labels", "group", "condition"]].head())
print("Samples:", adata.obs["sample_labels"].unique().tolist())


# --------------------------------------------------------------------------- #
# 2. PCA
# --------------------------------------------------------------------------- #
sc.tl.pca(adata, svd_solver="arpack")
print("X_pca shape:", adata.obsm["X_pca"].shape)


# --------------------------------------------------------------------------- #
# 3. MELD sample-associated densities (cached)
# --------------------------------------------------------------------------- #
if os.path.exists(DENSITY_FILE):
    print(f"Loading cached MELD densities from {DENSITY_FILE}")
    sample_densities = pd.read_csv(DENSITY_FILE, index_col=0)
else:
    print("Running MELD ...")
    meld_op = meld.MELD()
    sample_densities = meld_op.fit_transform(
        adata.obsm["X_pca"], sample_labels=adata.obs["sample_labels"]
    )
    sample_densities.to_csv(DENSITY_FILE)

sample_densities.columns = sample_densities.columns.astype(str)
print("sample_densities shape:", sample_densities.shape)


# --------------------------------------------------------------------------- #
# 4. Average density per genotype-condition category
#    e.g. WT_Veh, WT_KA3, WT_KA25, KO_Veh, KO_KA3, KO_KA25
# --------------------------------------------------------------------------- #
mapper_category_to_sample = defaultdict(list)
sample_dict = dict(zip(adata.obs["sample_id"], adata.obs["group_condition"]))
for sample, category in sample_dict.items():
    mapper_category_to_sample[category].append(str(sample))

print("Category -> samples:", dict(mapper_category_to_sample))

category_densities = {}
for category, sample_list in mapper_category_to_sample.items():
    avg_density = sample_densities[sample_list].mean(axis=1)  # mean over replicates
    avg_density = avg_density / avg_density.sum()             # renormalize to a PDF
    category_densities[category] = avg_density


# --------------------------------------------------------------------------- #
# 5. Pairwise normalized Wasserstein distance (15 pairs)
# --------------------------------------------------------------------------- #
def wasserstein_normalized(p, q):
    """1-D Wasserstein distance between two density vectors, scaled to [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    x = np.arange(len(p))
    w = wasserstein_distance(x, x, u_weights=p, v_weights=q)  # in bin units
    return w / (len(p) - 1) if len(p) > 1 else 0.0


res = []
for cat_1, cat_2 in combinations(category_densities.keys(), 2):
    d = wasserstein_normalized(category_densities[cat_1], category_densities[cat_2])
    d = math.floor(round(d, 3) / BIN_GRANULARITY) * BIN_GRANULARITY
    res.append((d, f"{cat_1},{cat_2}"))
    print(f"Wasserstein ({cat_1} vs {cat_2}): {d:.4f}")

res.sort(key=lambda x: x[0])
distances, pairs = zip(*res)


# --------------------------------------------------------------------------- #
# 6. Bar plot
# --------------------------------------------------------------------------- #
norm = mcolors.Normalize(vmin=min(distances), vmax=max(distances))
cmap = plt.colormaps["inferno"]
colors = [cmap(norm(d)) for d in distances]

fig, ax = plt.subplots(figsize=(27, 12))
ax.bar(pairs, distances, color=colors)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, location="left")
cbar.ax.tick_params(labelsize=14)

ax.set_ylabel("Wasserstein Distance", fontsize=16)
ax.set_xlabel("Category Pairs", fontsize=16)
ax.set_title(
    "Wasserstein Distance Between Pairwise Genotype-Condition Distributions",
    fontsize=16,
)
ax.tick_params(axis="x", rotation=65, labelsize=13)

plt.savefig(OUT_FIG, bbox_inches="tight")
plt.show()

print(res)
