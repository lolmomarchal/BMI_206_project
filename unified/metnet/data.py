"""Paths, organism table, published values, and loaders for the supplement and the BiGG models."""
import gzip, json, re, subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # .../unified
DATA = ROOT.parent / "data"
OLD_BIGG = DATA / "old_bigg"                           # original (2019-era) BiGG files
XLSX = DATA / "12859_2019_2897_MOESM2_ESM.xlsx"        # Additional file 2: adjacency matrices

# short key -> (BiGG model id, display name, supplement sheet index)
ORGS = {
    "ecoli": ("iJO1366", "E. coli"),
    "bsub":  ("iYO844",  "B. subtilis"),
    "gmet":  ("iAF987",  "G. metallireducens"),
    "kpneu": ("iYL1228", "K. pneumoniae"),
    "scer":  ("iMM904",  "S. cerevisiae"),
}
KEYS = list(ORGS)
NAME = {k: v[1] for k, v in ORGS.items()}
MODEL = {k: v[0] for k, v in ORGS.items()}

# Table 1 of the paper (reaction graphs): nodes, arcs
PAPER_COUNTS = {"ecoli": (1251, 9099), "bsub": (748, 6489), "gmet": (900, 8049),
                "kpneu": (1137, 8084), "scer": (881, 10460)}
# gamma printed in Fig. 2: (total, in, out)
PAPER_GAMMA = {"ecoli": (-1.29, -1.32, -1.50), "bsub": (-1.03, -1.10, -1.12),
               "gmet": (-1.03, -1.14, -1.11), "kpneu": (-1.21, -1.31, -1.47),
               "scer": (-0.99, -1.08, -1.16)}
KIND_ORDER = ("total", "in", "out")


def paper_gamma(key, kind):
    return PAPER_GAMMA[key][KIND_ORDER.index(kind)]


# ------------------------------------------------------------------ supplement
def load_supplement(cache=ROOT / "results" / "supplement_matrices.npz"):
    """Adjacency matrices from Additional file 2 -> {key: (reaction_ids, A)} with A[i, j] = arc i -> j."""
    import openpyxl
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: (list(z[k + "_ids"]), z[k]) for k in KEYS}
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    out, save = {}, {}
    for k, ws in zip(KEYS, wb.worksheets):
        rows = list(ws.iter_rows(values_only=True))
        ids = [str(r[0]).strip() for r in rows[1:]]
        A = (np.array([r[1:] for r in rows[1:]], dtype=float) > 0).astype(np.uint8)
        out[k] = (ids, A)
        save[k], save[k + "_ids"] = A, np.array(ids, dtype=object)
    np.savez_compressed(cache, **save)
    return out


def degrees_from_matrix(A):
    """in = column sum, out = row sum, total = in + out."""
    ind, outd = A.sum(0).astype(int), A.sum(1).astype(int)
    return {"total": ind + outd, "in": ind, "out": outd}


# ------------------------------------------------------------------ BiGG models
@dataclass
class Rxn:
    id: str
    subs: list            # reactant species ids (compartment-suffixed)
    prods: list
    lb: float
    ub: float
    subsystem: str = ""

    @property
    def reversible(self):
        """Both directions possible. Uses flux bounds, not the SBML `reversible` flag (which is wrong
        for e.g. HMGCOAS in iYO844: flagged reversible but ub = 0)."""
        return self.lb < 0 < self.ub


@dataclass
class Model:
    id: str
    rxns: dict = field(default_factory=dict)
    species_comp: dict = field(default_factory=dict)   # species id -> compartment


_NS = {"s": "http://www.sbml.org/sbml/level3/version1/core",
       "f": "http://www.sbml.org/sbml/level3/version1/fbc/version2"}


def _download(model, ext):
    fn = OLD_BIGG / f"{model}.{ext}.gz"
    if not fn.exists():
        OLD_BIGG.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(fn), f"https://bigg.ucsd.edu/static/models/{model}.{ext}.gz"], check=True)
    return fn


def read_model(model_id, path=None, with_subsystems=True):
    """Parse an SBML (fbc v2) BiGG model. Subsystems (not in the SBML) come from the matching BiGG JSON."""
    path = Path(path) if path else _download(model_id, "xml")
    root = ET.parse(gzip.open(path)).getroot()
    mdl = root.find("s:model", _NS)
    bounds = {p.get("id"): float(p.get("value")) for p in mdl.find("s:listOfParameters", _NS)}
    subsys = {}
    if with_subsystems:
        subsys = {r["id"]: r.get("subsystem", "") for r in json.load(gzip.open(_download(model_id, "json")))["reactions"]}
    M = Model(model_id)
    for sp in mdl.find("s:listOfSpecies", _NS):
        M.species_comp[sp.get("id")[2:]] = sp.get("compartment")
    for r in mdl.find("s:listOfReactions", _NS):
        rid = r.get("id")[2:]  # strip R_
        subs = [x.get("species")[2:] for x in r.find("s:listOfReactants", _NS) or []]
        prods = [x.get("species")[2:] for x in r.find("s:listOfProducts", _NS) or []]
        lb = bounds[r.get("{%s}lowerFluxBound" % _NS["f"])]
        ub = bounds[r.get("{%s}upperFluxBound" % _NS["f"])]
        M.rxns[rid] = Rxn(rid, subs, prods, lb, ub, subsys.get(rid, ""))
    return M


def count_sbml(path):
    """(#reactions, #species) of any SBML file; used to compare old vs new BiGG releases."""
    root = ET.parse(gzip.open(path)).getroot()
    mdl = root.find("s:model", _NS)
    return len(mdl.find("s:listOfReactions", _NS)), len(mdl.find("s:listOfSpecies", _NS))
