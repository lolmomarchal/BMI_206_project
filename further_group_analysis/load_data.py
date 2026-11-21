"""Load the supplement adjacency matrices once and cache the degree sequences."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "../data/12859_2019_2897_MOESM2_ESM.xlsx")
CACHE = os.path.join(HERE, "degrees.npz")
NAMES = ["E. coli", "B. subtilis", "G. metallireducens", "K. pneumoniae", "S. cerevisiae"]
KINDS = ["total", "in", "out"]

def load():
    if not os.path.exists(CACHE):
        import openpyxl
        wb = openpyxl.load_workbook(XLSX, read_only=True)
        out = {}
        for nm, ws in zip(NAMES, wb.worksheets):
            rows = list(ws.iter_rows(values_only=True))
            A = np.array([r[1:] for r in rows[1:]], dtype=float)
            ind, outd = A.sum(0).astype(int), A.sum(1).astype(int)
            out[nm + "|in"], out[nm + "|out"], out[nm + "|total"] = ind, outd, ind + outd
        np.savez(CACHE, **out)
    z = np.load(CACHE)
    return {nm: {k: z[f"{nm}|{k}"] for k in KINDS} for nm in NAMES}


MCACHE = os.path.join(HERE, "matrices.npz")
PAPER = {  # gammas printed in Kim et al. 2019, Fig 2: (total, in, out)
    "E. coli": (-1.29, -1.32, -1.50), "B. subtilis": (-1.03, -1.10, -1.12),
    "G. metallireducens": (-1.03, -1.14, -1.11), "K. pneumoniae": (-1.21, -1.31, -1.47),
    "S. cerevisiae": (-0.99, -1.08, -1.16)}


def load_mats():
    """adjacency matrices (uint8), cached. needed for induced-subgraph / node-removal tests."""
    if not os.path.exists(MCACHE):
        import openpyxl
        wb = openpyxl.load_workbook(XLSX, read_only=True)
        out = {}
        for nm, ws in zip(NAMES, wb.worksheets):
            rows = list(ws.iter_rows(values_only=True))
            out[nm] = np.array([r[1:] for r in rows[1:]], dtype=np.uint8)
        np.savez_compressed(MCACHE, **out)
    z = np.load(MCACHE)
    return {nm: z[nm] for nm in NAMES}


def degrees_from(A):
    ind, outd = A.sum(0).astype(int), A.sum(1).astype(int)
    return {"total": ind + outd, "in": ind, "out": outd}
