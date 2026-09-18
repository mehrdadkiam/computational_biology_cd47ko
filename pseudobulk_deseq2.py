"""
Mouse-level pseudobulk differential expression with DESeq2 (PyDESeq2).

Written to answer the reviewer comment:
    "The analysis must be repeated using pseudobulk counts for each mouse and
     cell type, with an appropriate method such as DESeq2, edgeR or limma-voom."

PyDESeq2 is the Python re-implementation of the DESeq2 model (negative-binomial
GLM, median-of-ratios size factors, empirical-Bayes dispersion shrinkage, Wald
test, Cook's outlier filtering, independent filtering, BH FDR), so the analysis
stays in Python while using exactly the method the reviewer asked for.

INPUT  (same objects the notebook already builds)
    pseudobulk_adata                  pre-normalisation copy of adata_combined,
                                      raw counts in .X and in .layers["counts"],
                                      .obs carries sample_id_unique / batch /
                                      group / condition / sex
    adata_filtered_significant_genes  annotated object, supplies .obs["celltype"]

    If either object is not already in memory the script reads the .h5ad files
    written by the notebook (paths in the CONFIG block).

WHAT IT DOES
    1. attaches the cell-type labels to the raw-count object
    2. sums raw counts over nuclei within each mouse x cell type
       -> one integer count profile per animal per cell type
    3. filters lowly expressed genes (edgeR filterByExpr logic)
    4. runs DESeq2 KO vs WT separately in every cell type x treatment stratum
    5. optionally runs a combined model ~ condition + group per cell type
    6. writes per-stratum result tables, a merged table, a DEG summary, and the
       pseudobulk count matrices themselves (so the identical matrices can be
       re-run in edgeR or limma-voom in R if the reviewer prefers)

REQUIREMENTS
    pip install pydeseq2        (tested with pydeseq2 0.5.x; 0.4.x also handled)
"""

import os
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIG  (keys identical to the notebook)
# =============================================================================

SAMPLE_KEY = "sample_id_unique"      # one mouse
CELLTYPE_KEY = "celltype"
BATCH_KEY = "batch"
GROUP_KEY = "group"                  # WT / KO
CONDITION_KEY = "condition"          # Veh / KA3 / KA25
SEX_KEY = "sex"

KEEP_BATCH = "batch_2022"
KEEP_CONDITIONS = ["Veh", "KA3", "KA25"]
CASE_GROUP = "KO"
CONTROL_GROUP = "WT"

CELLTYPE_ORDER = ["Astro", "CA1", "CA3", "DG", "Inhibitory", "Microglia", "Oligo"]

# fall-backs, only used if the objects are not already in the session
FRACTION = 0.1
METHOD = "batch"
PSEUDOBULK_H5AD = (
    f"adata_pseudobulk_{FRACTION}_{METHOD}"
    "_NO_mt_NO_SexInBatchEffect_REVISION_Pseudobulk.h5ad"
)
ANNOTATED_H5AD = "adata_filtered_significant_genes_NO_mt_YekBasteh.h5ad"

# aggregation / filtering
MIN_CELLS = 10          # nuclei required for a mouse x cell type profile
MIN_MICE_PER_GROUP = 2  # DESeq2 needs replication in both arms
MIN_COUNT = 10          # filterByExpr: counts in a sample to call a gene present
MIN_TOTAL_COUNT = 15    # filterByExpr: counts summed over all samples

# testing
ALPHA = 0.05            # FDR
LFC_TH = 0.5            # |log2FC| used only for the DEG tally
COVARIATE_SEX = True    # add sex to the design when it is estimable
RUN_COMBINED_MODEL = True   # per cell type, ~ condition + group over all mice
DROP_CD47 = True        # Cd47 is the knocked-out gene, excluded as in the paper
N_CPUS = 4

SEX_GENES = ["Xist", "Tsix", "Uty", "Eif2s3y", "Kdm5d", "Ddx3y"]

OUTDIR = "pseudobulk_DESeq2"
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# 1. RESOLVE INPUT OBJECTS
# =============================================================================

def _resolve_inputs():
    """Take pseudobulk_adata / adata_filtered_significant_genes from the running
    session if they exist, otherwise read the .h5ad files the notebook wrote."""
    import scanpy as sc

    g = globals()

    if "pseudobulk_adata" in g:
        pb = g["pseudobulk_adata"]
        print("Using pseudobulk_adata from the session.")
    else:
        print(f"Reading {PSEUDOBULK_H5AD}")
        pb = sc.read_h5ad(PSEUDOBULK_H5AD)

    if "adata_filtered_significant_genes" in g:
        ann = g["adata_filtered_significant_genes"]
        print("Using adata_filtered_significant_genes from the session.")
    else:
        print(f"Reading {ANNOTATED_H5AD}")
        ann = sc.read_h5ad(ANNOTATED_H5AD)

    return pb, ann


def prepare_cells(pb, ann):
    """Raw counts + cell-type labels, restricted to the mice under analysis."""
    pb = pb.copy()

    if "counts" in pb.layers:
        pb.X = pb.layers["counts"].copy()
        print("Raw counts taken from .layers['counts'].")
    else:
        print("WARNING: no 'counts' layer, using .X and assuming it is raw counts.")

    X = pb.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    if not np.allclose(X.data, np.rint(X.data)):
        raise ValueError(
            "The count matrix is not integer-valued. DESeq2 needs raw counts, "
            "so pseudobulk_adata must be the pre-normalisation object."
        )
    pb.X = X

    # cell-type labels
    ct = ann.obs[CELLTYPE_KEY].dropna().astype(str)
    common = pb.obs_names.intersection(ct.index)
    if len(common) == 0:
        raise ValueError("No shared barcodes between the two objects.")
    pb = pb[common].copy()
    pb.obs[CELLTYPE_KEY] = ct.loc[pb.obs_names].values

    # batch / condition / genotype
    keep = (
        (pb.obs[BATCH_KEY].astype(str) == KEEP_BATCH)
        & (pb.obs[CONDITION_KEY].astype(str).isin(KEEP_CONDITIONS))
        & (pb.obs[GROUP_KEY].astype(str).isin([CASE_GROUP, CONTROL_GROUP]))
    )
    pb = pb[keep.values].copy()

    # sex-linked genes, and the targeted gene itself
    drop = list(SEX_GENES) + (["Cd47"] if DROP_CD47 else [])
    pb = pb[:, ~pb.var_names.astype(str).isin(drop)].copy()

    print(f"Nuclei x genes entering aggregation: {pb.shape}")
    return pb


# =============================================================================
# 2. AGGREGATE TO ONE PROFILE PER MOUSE x CELL TYPE
# =============================================================================

def make_pseudobulk(pb):
    """Sum raw counts over the nuclei of each mouse within each cell type.

    Returns {celltype: (counts DataFrame [mice x genes], metadata DataFrame)}.
    """
    obs = pb.obs
    genes = pb.var_names.astype(str)
    out = {}

    celltypes = [c for c in CELLTYPE_ORDER
                 if c in set(obs[CELLTYPE_KEY].astype(str))]
    celltypes += sorted(set(obs[CELLTYPE_KEY].astype(str)) - set(celltypes))

    for celltype in celltypes:
        ct_mask = (obs[CELLTYPE_KEY].astype(str) == celltype).values
        rows, meta = [], []

        for sample in obs.loc[ct_mask, SAMPLE_KEY].astype(str).unique():
            mask = ct_mask & (obs[SAMPLE_KEY].astype(str) == sample).values
            n_cells = int(mask.sum())
            if n_cells < MIN_CELLS:
                continue
            rows.append(np.asarray(pb.X[mask].sum(axis=0)).ravel())
            first = obs.loc[mask].iloc[0]
            meta.append({
                SAMPLE_KEY: sample,
                GROUP_KEY: str(first[GROUP_KEY]),
                CONDITION_KEY: str(first[CONDITION_KEY]),
                SEX_KEY: str(first[SEX_KEY]) if SEX_KEY in obs.columns else "NA",
                "n_cells": n_cells,
            })

        if not rows:
            print(f"{celltype}: no mouse reaches {MIN_CELLS} nuclei, skipped.")
            continue

        counts = pd.DataFrame(np.vstack(rows), columns=genes,
                              index=[m[SAMPLE_KEY] for m in meta])
        counts = counts.round().astype(int)
        metadata = pd.DataFrame(meta).set_index(SAMPLE_KEY)
        metadata.index.name = SAMPLE_KEY
        out[celltype] = (counts, metadata)

    return out


def filter_genes(counts, min_group_size):
    """edgeR filterByExpr logic: a gene is kept when it reaches MIN_COUNT in at
    least as many mice as the smaller group, and MIN_TOTAL_COUNT overall."""
    n_expressed = (counts >= MIN_COUNT).sum(axis=0)
    keep = (n_expressed >= max(min_group_size, 1)) & \
           (counts.sum(axis=0) >= MIN_TOTAL_COUNT)
    return counts.loc[:, keep]


# =============================================================================
# 3. DESeq2
# =============================================================================

def _make_dds(counts, metadata, design):
    """DeseqDataSet across pydeseq2 0.4.x / 0.5.x signatures."""
    try:                                     # 0.5.x formula interface
        return DeseqDataSet(counts=counts, metadata=metadata, design=design,
                            refit_cooks=True, n_cpus=N_CPUS, quiet=True)
    except TypeError:                        # 0.4.x
        factors = [t.strip() for t in design.lstrip("~").split("+") if t.strip()]
        return DeseqDataSet(counts=counts, metadata=metadata,
                            design_factors=factors,
                            ref_level=[GROUP_KEY, CONTROL_GROUP],
                            refit_cooks=True, n_cpus=N_CPUS, quiet=True)


def run_deseq2(counts, metadata, design, contrast, label):
    """Fit the DESeq2 model and return the Wald-test table for `contrast`."""
    dds = _make_dds(counts, metadata, design)
    dds.deseq2()

    stat = DeseqStats(dds, contrast=contrast, alpha=ALPHA,
                      cooks_filter=True, independent_filter=True,
                      n_cpus=N_CPUS, quiet=True)
    stat.summary()

    res = stat.results_df.copy()
    res.index.name = "gene"
    res = res.reset_index().sort_values("pvalue", na_position="last")
    res.insert(0, "stratum", label)
    res["design"] = design
    return res.reset_index(drop=True)


def choose_design(metadata):
    """~ sex + group when sex is estimable in this stratum, otherwise ~ group."""
    if not COVARIATE_SEX or SEX_KEY not in metadata.columns:
        return f"~{GROUP_KEY}"
    tab = pd.crosstab(metadata[SEX_KEY], metadata[GROUP_KEY])
    # need both sexes present, and sex not perfectly confounded with genotype
    if tab.shape[0] < 2 or (tab > 0).all(axis=1).sum() < 1 or (tab.sum(axis=1) < 2).any():
        return f"~{GROUP_KEY}"
    if ((tab > 0).sum(axis=0) < 2).all():          # each genotype one sex only
        return f"~{GROUP_KEY}"
    return f"~{SEX_KEY}+{GROUP_KEY}"


# =============================================================================
# 4. DRIVER
# =============================================================================

def main():
    pb_raw, ann = _resolve_inputs()
    pb = prepare_cells(pb_raw, ann)
    pseudobulk = make_pseudobulk(pb)

    # ---- design overview -----------------------------------------------------
    overview = pd.concat(
        [md.assign(celltype=ct) for ct, (_, md) in pseudobulk.items()]
    ).reset_index()
    print("\nPseudobulk profiles (one row = one mouse x cell type):")
    print(overview.pivot_table(index=[CONDITION_KEY, GROUP_KEY, SAMPLE_KEY],
                               columns="celltype", values="n_cells",
                               fill_value=0, observed=True).to_string())
    overview.to_csv(f"{OUTDIR}/pseudobulk_profiles_overview.csv", index=False)

    all_res, summary = [], []

    # ---- KO vs WT inside each treatment -------------------------------------
    for celltype, (counts, metadata) in pseudobulk.items():

        # counts matrix exported once per cell type, for edgeR / limma in R
        counts.T.to_csv(f"{OUTDIR}/counts_{celltype}.csv")
        metadata.to_csv(f"{OUTDIR}/coldata_{celltype}.csv")

        for condition in KEEP_CONDITIONS:
            sub = metadata[metadata[CONDITION_KEY] == condition]
            n_case = int((sub[GROUP_KEY] == CASE_GROUP).sum())
            n_ctrl = int((sub[GROUP_KEY] == CONTROL_GROUP).sum())
            label = f"{celltype}|{condition}|{CASE_GROUP}_vs_{CONTROL_GROUP}"

            if min(n_case, n_ctrl) < MIN_MICE_PER_GROUP:
                print(f"\n{label}: {CASE_GROUP} n={n_case}, "
                      f"{CONTROL_GROUP} n={n_ctrl} -> not testable "
                      f"(need >= {MIN_MICE_PER_GROUP} mice per genotype).")
                summary.append({"stratum": label, "n_KO": n_case, "n_WT": n_ctrl,
                                "n_genes_tested": 0, "n_sig": 0,
                                "n_up": 0, "n_down": 0, "status": "skipped"})
                continue

            c = filter_genes(counts.loc[sub.index], min(n_case, n_ctrl))
            design = choose_design(sub)
            print(f"\n{label}: {CASE_GROUP} n={n_case}, {CONTROL_GROUP} n={n_ctrl}, "
                  f"{c.shape[1]} genes, design {design}")

            try:
                res = run_deseq2(c, sub.loc[c.index], design,
                                 [GROUP_KEY, CASE_GROUP, CONTROL_GROUP], label)
            except Exception as exc:                      # never kill the loop
                print(f"  DESeq2 failed: {exc}")
                summary.append({"stratum": label, "n_KO": n_case, "n_WT": n_ctrl,
                                "n_genes_tested": int(c.shape[1]), "n_sig": 0,
                                "n_up": 0, "n_down": 0, "status": f"error: {exc}"})
                continue

            res["celltype"], res["condition"] = celltype, condition
            res.to_csv(f"{OUTDIR}/DESeq2_{celltype}_{condition}_"
                       f"{CASE_GROUP}vs{CONTROL_GROUP}.csv", index=False)
            all_res.append(res)

            sig = res[(res["padj"] < ALPHA) & (res["log2FoldChange"].abs() > LFC_TH)]
            n_up = int((sig["log2FoldChange"] > 0).sum())
            n_down = int((sig["log2FoldChange"] < 0).sum())
            print(f"  padj < {ALPHA} and |log2FC| > {LFC_TH}: "
                  f"{len(sig)} genes ({n_up} up in {CASE_GROUP}, {n_down} down)")
            summary.append({"stratum": label, "n_KO": n_case, "n_WT": n_ctrl,
                            "n_genes_tested": int(c.shape[1]), "n_sig": len(sig),
                            "n_up": n_up, "n_down": n_down, "status": "ok"})

    # ---- genotype effect adjusted for treatment ------------------------------
    if RUN_COMBINED_MODEL:
        print("\n" + "=" * 70)
        print(f"Combined model per cell type: ~{CONDITION_KEY}+{GROUP_KEY}")
        print("=" * 70)

        for celltype, (counts, metadata) in pseudobulk.items():
            n_case = int((metadata[GROUP_KEY] == CASE_GROUP).sum())
            n_ctrl = int((metadata[GROUP_KEY] == CONTROL_GROUP).sum())
            label = f"{celltype}|all_conditions|{CASE_GROUP}_vs_{CONTROL_GROUP}"

            if min(n_case, n_ctrl) < MIN_MICE_PER_GROUP or \
                    metadata[CONDITION_KEY].nunique() < 2:
                continue

            c = filter_genes(counts, min(n_case, n_ctrl))
            design = f"~{CONDITION_KEY}+{GROUP_KEY}"
            print(f"\n{label}: {CASE_GROUP} n={n_case}, {CONTROL_GROUP} n={n_ctrl}, "
                  f"{c.shape[1]} genes")

            try:
                res = run_deseq2(c, metadata.loc[c.index], design,
                                 [GROUP_KEY, CASE_GROUP, CONTROL_GROUP], label)
            except Exception as exc:
                print(f"  DESeq2 failed: {exc}")
                continue

            res["celltype"], res["condition"] = celltype, "all"
            res.to_csv(f"{OUTDIR}/DESeq2_{celltype}_allConditions_"
                       f"{CASE_GROUP}vs{CONTROL_GROUP}.csv", index=False)
            all_res.append(res)

            sig = res[(res["padj"] < ALPHA) & (res["log2FoldChange"].abs() > LFC_TH)]
            n_up = int((sig["log2FoldChange"] > 0).sum())
            print(f"  padj < {ALPHA} and |log2FC| > {LFC_TH}: {len(sig)} genes "
                  f"({n_up} up in {CASE_GROUP}, {len(sig) - n_up} down)")
            summary.append({"stratum": label, "n_KO": n_case, "n_WT": n_ctrl,
                            "n_genes_tested": int(c.shape[1]), "n_sig": len(sig),
                            "n_up": n_up, "n_down": len(sig) - n_up,
                            "status": "ok"})

    # ---- write everything ----------------------------------------------------
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{OUTDIR}/DESeq2_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    if all_res:
        merged = pd.concat(all_res, ignore_index=True)
        merged.to_csv(f"{OUTDIR}/DESeq2_all_results.csv", index=False)
        print(f"\nResults written to ./{OUTDIR}/")
    else:
        print("\nNo stratum could be tested.")

    return summary_df


if __name__ == "__main__":
    main()
