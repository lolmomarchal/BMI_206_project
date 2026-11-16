# same fig 2 but this time rebuilding the reaction graphs ourselves from the original
# bigg models (bigg.ucsd.edu) instead of using the adjacency matrices from the supplement
import gzip
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

models = ["iJO1366", "iYO844", "iAF987", "iYL1228", "iMM904"]
names = ["E. coli", "B. subtilis", "G. metallireducens", "K. pneumoniae", "S. cerevisiae"]

# table 1 of the paper (reactions, arcs) so we can see how close we get
table1 = {"iJO1366": (1251, 9099), "iYO844": (748, 6489), "iAF987": (900, 8049),
          "iYL1228": (1137, 8084), "iMM904": (881, 10460)}

# gammas printed in the paper, (total, in, out)
paper_gamma = {"iJO1366": (-1.29, -1.32, -1.50), "iYO844": (-1.03, -1.10, -1.12),
               "iAF987": (-1.03, -1.14, -1.11), "iYL1228": (-1.21, -1.31, -1.47),
               "iMM904": (-0.99, -1.08, -1.16)}

# paper doesnt list its currency metabolites. this set makes the arcs match the supplement
# exactly (checked arc by arc, using the supplement's own reaction lists)
currency = {"h", "h2o", "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph",
            "pi", "ppi", "co2", "nh4", "o2"}

outdir = "../data/old_bigg"
os.makedirs(outdir, exist_ok=True)

ns = {"s": "http://www.sbml.org/sbml/level3/version1/core",
      "f": "http://www.sbml.org/sbml/level3/version1/fbc/version2"}


def base(met):
    # strip the compartment, atp_c -> atp
    return re.sub(r"_[a-z]+$", "", met)


def comp(met):
    return met.rsplit("_", 1)[1]


def download(m, ext):
    fn = "%s/%s.%s.gz" % (outdir, m, ext)
    if not os.path.exists(fn):
        print("downloading", fn)
        # urllib kept failing the ssl check on my python so just use curl
        subprocess.run(["curl", "-sL", "-o", fn, "https://bigg.ucsd.edu/static/models/%s.%s.gz" % (m, ext)], check=True)
    return fn


def read_model(m):
    root = ET.parse(gzip.open(download(m, "xml"))).getroot()
    model = root.find("s:model", ns)
    bounds = {p.get("id"): float(p.get("value")) for p in model.find("s:listOfParameters", ns)}

    # the sbml has no subsystems but the json from bigg does, and the paper must have used
    # them to throw out the transport reactions (see the notes at the bottom)
    subsys = {r["id"]: r.get("subsystem", "") for r in json.load(gzip.open(download(m, "json")))["reactions"]}

    rxns = {}
    for r in model.find("s:listOfReactions", ns):
        rid = r.get("id")[2:]  # get rid of R_
        subs = [x.get("species")[2:] for x in r.find("s:listOfReactants", ns) or []]
        prods = [x.get("species")[2:] for x in r.find("s:listOfProducts", ns) or []]
        lb = bounds[r.get("{%s}lowerFluxBound" % ns["f"])]
        ub = bounds[r.get("{%s}upperFluxBound" % ns["f"])]
        # only reversible if flux can really go both ways. the "reversible" flag in the file
        # is wrong for a few reactions (ex HMGCOAS in iYO844 has ub = 0), anything that
        # isnt two way is read forward as written, this is what the supplement does
        rxns[rid] = (subs, prods, lb < 0 and ub > 0, subsys.get(rid, ""))
    return rxns


def is_currency(met, m):
    # yeast: only the cytosolic atp, nadh etc get dropped, the mitochondrial ones stay
    return base(met) in currency and (m != "iMM904" or comp(met) == "c")


def drop_rxn(rid, subs, prods, subsystem, m):
    sub = subsystem.lower()

    # transport and exchange (the typos are in the iYL1228 annotation). the "inorganic ion
    # transport AND metabolism" subsystem has real metabolism in it so its handled below
    if "inorganic ion" not in sub and re.search(r"tran[a-z]*port|tranpsort|exchange", sub):
        return True
    if rid.startswith(("EX_", "DM_", "SK_", "BIOMASS")):
        return True

    comps = {comp(x) for x in subs + prods if not is_currency(x, m)}
    if "e" in comps:
        return True
    if "inorganic ion" in sub and "p" in comps:
        return True   # transporters hiding in there

    if m == "iMM904":
        # paper only has cytosol, mito and er reactions in yeast
        return bool(comps - {"c", "m", "r"})
    if comps and "c" not in comps:
        return True   # nothing in the cytosol, only periplasm stuff
    if m == "iJO1366" and re.search(r"pp(i)?$", rid):
        return True   # bigg names periplasmic reactions ...pp, none of them are in the E. coli node list
    return False


def make_graph(rxns, m):
    ids = [r for r, (s, p, rev, sub) in rxns.items() if not drop_rxn(r, s, p, sub, m)]

    # all the non currency metabolites
    mets = sorted({x for r in ids for x in rxns[r][0] + rxns[r][1] if not is_currency(x, m)})
    midx = {x: i for i, x in enumerate(mets)}

    cons = np.zeros((len(ids), len(mets)), dtype=bool)  # reaction consumes met
    prod = np.zeros((len(ids), len(mets)), dtype=bool)  # reaction makes met
    for i, r in enumerate(ids):
        subs, prods, rev, _ = rxns[r]
        for x in subs:
            if x in midx:
                cons[i, midx[x]] = True
                if rev:
                    prod[i, midx[x]] = True   # reversible so it can go both ways
        for x in prods:
            if x in midx:
                prod[i, midx[x]] = True
                if rev:
                    cons[i, midx[x]] = True

    # arc a -> b if a makes something b uses
    A = (prod.astype(int) @ cons.T.astype(int)) > 0
    np.fill_diagonal(A, False)

    # reactions with no arcs at all arent really in the graph
    connected = (A.sum(0) + A.sum(1)) > 0
    return A[connected][:, connected].astype(int)


def get_pk(deg):
    n = len(deg)
    deg = deg[deg > 0].astype(int)
    k, cnt = np.unique(deg, return_counts=True)
    return k, cnt / n


kinds = [("total", "black", "o"), ("in", "red", "s"), ("out", "blue", "^")]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, m in enumerate(models):
    A = make_graph(read_model(m), m)
    ax = axes[i]
    print(m, "nodes", len(A), "(paper %d)" % table1[m][0], "arcs", A.sum(), "(paper %d)" % table1[m][1])

    indeg = A.sum(axis=0)
    outdeg = A.sum(axis=1)
    degs = {"total": indeg + outdeg, "in": indeg, "out": outdeg}

    for j, (what, col, mk) in enumerate(kinds):
        k, p = get_pk(degs[what])
        ax.scatter(k, p, c=col, marker=mk, s=14, zorder=3)

        slope, b = np.polyfit(np.log10(k), np.log10(p), 1)
        x = np.array([k.min(), k.max()])
        ax.plot(x, 10 ** (b + slope * np.log10(x)), color=col, lw=1.3)

        g = np.trunc(slope * 100) / 100
        print("   gamma %-5s ours %.2f paper %.2f" % (what, g, paper_gamma[m][j]))
        ax.text(0.97, 0.93 - 0.09 * j, "$\\gamma$ (%s) = %.2f" % (what, g),
                transform=ax.transAxes, color=col, ha="right", va="top",
                fontsize=10, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** -0.2, 10 ** 2.5)
    ax.set_ylim(1e-4, 1)
    ax.text(0.05, 0.05, "abcde"[i], transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.text(0.17, 0.05, names[i], transform=ax.transAxes, fontsize=14,
            fontweight="bold", style="italic")

    if i in (0, 3):
        ax.set_ylabel("Pr($k$)", fontsize=13, fontweight="bold")
    if i in (2, 3, 4):
        ax.set_xlabel("degree ($k$)", fontsize=13, fontweight="bold")

# legend in the empty spot
axes[5].axis("off")
handles = [
    Line2D([], [], color="black", marker="o", ls="", label="Total degree"),
    Line2D([], [], color="red", marker="s", ls="", label="In-degree"),
    Line2D([], [], color="blue", marker="^", ls="", label="Out-degree"),
    Line2D([], [], color="black", label="Regression of total degrees"),
    Line2D([], [], color="red", label="Regression of in-degrees"),
    Line2D([], [], color="blue", label="Regression of out-degrees"),
]
axes[5].legend(handles=handles, loc="center", frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig("figure2_rebuilt.png", dpi=200)

# notes (how i got the filtering rules, the paper only describes them in words):
# - direction: reversible only if lb < 0 and ub > 0, else forward as written. with the
#   supplement's reaction lists this gives its arcs exactly for 4 of the 5 models
# - transport / exchange reactions are the ones whose bigg subsystem says so. thats why e.g.
#   3PEPTabcpp is a node for G. metallireducens ("Murein Recycling") but not other transporters
# - the rest (e/periplasm only/yeast organelles/pp suffix for E. coli) was tuned by comparing
#   to the supplement node lists, so its a best fit not something the paper states
# - the paper's iYL1228 list uses old reaction ids (ACACT2r..7r) that dont exist anymore
