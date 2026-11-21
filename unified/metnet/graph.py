"""Reaction-graph construction: ONE builder, with the filtering choices as explicit switches.

Arc a -> b exists if reaction a can make a metabolite that reaction b can consume. A reversible
reaction (lb < 0 < ub) both makes and consumes all of its metabolites. Each reaction is one node.
Pr(k) = (# reactions of degree k) / (# reactions in the graph); k = 0 is not plotted.

The five rule sets below are the same builder with different switches, so every sensitivity
analysis (tiers, Test 1, Test 2) differs from the baseline only in the switch being tested.
"""
import re
from dataclasses import dataclass, replace
import numpy as np

# Currency metabolites (base BiGG ids). The paper only names ATP/NAD/H2O. This list was recovered
# independently three times from the supplement (non-adjacent node pairs; per-metabolite arc audit;
# and it is the same set used for the raw-data reconstruction) - see notebook section 3.
CURRENCY = frozenset({"h", "h2o", "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph",
                      "pi", "ppi", "co2", "nh4", "o2"})
YEAST = "iMM904"


def base(met):
    return re.sub(r"_[a-z]+$", "", met)


def comp(met):
    return met.rsplit("_", 1)[1]


@dataclass(frozen=True)
class Rules:
    name: str
    currency: bool = False          # remove currency metabolites from link construction
    reactions: str = "all"          # "all" | "intracellular" | "reconstruction"
    drop_isolated: bool = True      # reactions with no arc are not nodes (as in the paper's matrices)


UNFILTERED = Rules("Unfiltered")
CURRENCY_ONLY = Rules("Currency filtered", currency=True)
COMPARTMENT_ONLY = Rules("Compartment filtered", reactions="intracellular")
FULL_SIMPLE = Rules("Fully filtered (simple)", currency=True, reactions="intracellular")
RECONSTRUCTION = Rules("Fully filtered (reconstruction)", currency=True, reactions="reconstruction")
TIERS = [UNFILTERED, CURRENCY_ONLY, COMPARTMENT_ONLY, FULL_SIMPLE, RECONSTRUCTION]


def is_currency(met, model_id):
    """Yeast: only cytosolic copies are currency; mitochondrial ATP/NADH/... are ordinary metabolites
    (their arcs are in the supplement). Bacteria: every compartment."""
    return base(met) in CURRENCY and (model_id != YEAST or comp(met) == "c")


def _is_boundary(r):
    return r.id.startswith(("EX_", "DM_", "SK_")) or not (r.subs and r.prods)


def _keep_intracellular(r, M):
    """Coarse rule: no boundary reactions, every species in the cytosol (+ mitochondria for yeast)."""
    allowed = {"c", "m"} if M.id == YEAST else {"c"}
    return not _is_boundary(r) and {M.species_comp[x] for x in r.subs + r.prods} <= allowed


def _keep_reconstruction(r, M):
    """Rules that reproduce the paper's node lists. 1-3 are well supported by the supplement's own
    reaction names; 4-5 are a best fit (the paper does not state them)."""
    sub = r.subsystem.lower()
    # transport / exchange from the BiGG subsystem (typos exist in iYL1228). The mixed
    # "inorganic ion transport and metabolism" subsystem is kept unless the reaction touches the periplasm.
    if "inorganic ion" not in sub and re.search(r"tran[a-z]*port|tranpsort|exchange", sub):
        return False
    if r.id.startswith(("EX_", "DM_", "SK_", "BIOMASS")):
        return False
    comps = {comp(x) for x in r.subs + r.prods if not is_currency(x, M.id)}
    if "e" in comps:
        return False
    if "inorganic ion" in sub and "p" in comps:
        return False
    if M.id == YEAST:                      # supplement only has cytosol, mitochondria, ER
        return not (comps - {"c", "m", "r"})
    if comps and "c" not in comps:         # periplasm-only reactions
        return False
    if M.id == "iJO1366" and re.search(r"pp(i)?$", r.id):   # BiGG's periplasm naming; none in the E. coli node list
        return False
    return True


def reaction_ids(M, rules):
    keep = {"all": lambda r, M: True, "intracellular": _keep_intracellular,
            "reconstruction": _keep_reconstruction}[rules.reactions]
    return [i for i, r in M.rxns.items() if keep(r, M)]


def build_graph(M, rules, nodes=None):
    """Return (reaction_ids, A) with A[i, j] = 1 for arc i -> j (boolean-int matrix, no self loops).

    nodes: fixed reaction set (used by Test 1 so only the currency switch changes); in that case
    isolated reactions are kept in the denominator.
    """
    ids = list(nodes) if nodes is not None else reaction_ids(M, rules)
    drop_cur = (lambda x: is_currency(x, M.id)) if rules.currency else (lambda x: False)
    mets = sorted({x for i in ids for x in M.rxns[i].subs + M.rxns[i].prods if not drop_cur(x)})
    midx = {x: j for j, x in enumerate(mets)}
    cons = np.zeros((len(ids), len(mets)), dtype=np.int32)
    prod = np.zeros_like(cons)
    for i, rid in enumerate(ids):
        r = M.rxns[rid]
        for x in r.subs:
            if x in midx:
                cons[i, midx[x]] = 1
                if r.reversible:
                    prod[i, midx[x]] = 1
        for x in r.prods:
            if x in midx:
                prod[i, midx[x]] = 1
                if r.reversible:
                    cons[i, midx[x]] = 1
    A = ((prod @ cons.T) > 0).astype(np.uint8)
    np.fill_diagonal(A, 0)
    if nodes is None and rules.drop_isolated:
        keep = (A.sum(0) + A.sum(1)) > 0
        ids = [i for i, k in zip(ids, keep) if k]
        A = A[keep][:, keep]
    return ids, A


def degrees(A):
    ind, outd = A.sum(0).astype(int), A.sum(1).astype(int)
    return {"total": ind + outd, "in": ind, "out": outd}


def degree_distribution(deg):
    """k > 0 and Pr(k) = count / (number of nodes)."""
    n = len(deg)
    k, c = np.unique(deg[deg > 0].astype(int), return_counts=True)
    return k, c / n
