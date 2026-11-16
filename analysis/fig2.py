# %% [markdown]
# # figure 2: degree distributions of the reaction-centric graphs

# for each species we need:
# 1. the directed reaction graph (reactions = nodes, arcs = shared metabolites)
# 2. in-, out- and total-degree per reaction
# 3. pr(k) = fraction of reactions with degree k
# 4. a log-log linear regression of pr(k) on k. the slope is gamma.
# %%
import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
DATA = HERE.parent / 'data'
XLSX = DATA / '12859_2019_2897_MOESM2_ESM.xlsx'

# bigg model id, xlsx sheet name, display name
SPECIES = {
    'ecoli': ('iJO1366', 'E.coli (iJ01366)', 'E. coli'),
    'bsub': ('iYO844', 'B. subtilis (iYO844)', 'B. subtilis'),
    'gmet': ('iAF987', 'G. metallireducens (iAF987)', 'G. metallireducens'),
    'kpneu': ('iYL1228', 'K. pneumoniae (iYL1228)', 'K. pneumoniae'),
    'scer': ('iMM904', 'S. cerevisiae (iMM904)', 'S. cerevisiae'),
}

# TODO: this needs more exploration of how to identify these currency metabolites

# # currency metabolites, with compartment suffix dropped.
# # the paper names atp, nad and h2o but does not give the full list.
# # this list was inferred: it is the set of metabolites whose arcs are almost
# # never present in the authors' e. coli matrix (see notes at the bottom).
# CURRENCY = {
#     "h", "h2o", "atp", "adp", "amp", "pi", "ppi",
#     "nad", "nadh", "nadp", "nadph", "co2", "nh4", "o2",
# }

# %% [markdown]
# ## functions

# %%
# use adjacency matrix from the supplement to build reaction graphs
def load_adjacency(xlsx: Path, sheet: str) -> pd.DataFrame:
    adj = pd.read_excel(xlsx, sheet_name=sheet, index_col=0)
    # clean up straying spaces
    adj.index = adj.index.astype(str).str.strip() 
    adj.columns = adj.columns.astype(str).str.strip()
    # check to make sure matrix has the same row and column labels
    assert list(adj.index) == list(adj.columns), 'row and column labels of matrix do not match'
    return adj.astype(int)

def adjacency_to_edges(adj: pd.DataFrame) -> pd.DataFrame:
    """turn an adjacency matrix into an edge list (source, target)"""
    src, dst = np.nonzero(adj.to_numpy())
    return pd.DataFrame({'source': adj.index[src], 'target': adj.columns[dst]})

##########
# calculate degrees and pr(k)
def node_degrees(edges: pd.DataFrame, nodes=None) -> pd.DataFrame:
    """in-, out- and total-degree per reaction.

    out = arcs leaving, in = arcs arriving, total = in + out.

    args:
        edges: source/target edge list.
        nodes: optional full node list, so isolated nodes get degree 0.
    """
    out_deg = edges['source'].value_counts()
    in_deg = edges['target'].value_counts()
    nodes = pd.Index(nodes if nodes is not None else out_deg.index.union(in_deg.index))
    deg = pd.DataFrame({'in': in_deg.reindex(nodes, fill_value=0),
                        'out': out_deg.reindex(nodes, fill_value=0)})
    deg['total'] = deg['in'] + deg['out']
    return deg


def degree_distribution(deg: pd.DataFrame) -> pd.DataFrame:
    """pr(k) for each degree type, long format.

    pr(k) = (# reactions with degree k) / (# reactions). only k >= 1,
    since k = 0 cannot go on a log axis.

    returns:
        columns: kind ("in"/"out"/"total"), k, pk.
    """
    n = len(deg)
    parts = []
    for kind in ['in', 'out', 'total']:
        counts = deg[kind].value_counts().sort_index()
        counts = counts[counts.index >= 1]
        parts.append(pd.DataFrame({'kind': kind, 'k': counts.index, 'pk': counts.to_numpy() / n}))
    return pd.concat(parts, ignore_index=True)


def fit_power_law(dist: pd.DataFrame) -> pd.DataFrame:
    """fit log10 pr(k) = gamma * log10 k + b by least squares, per degree type.

    every observed k counts once, unweighted. this includes the flat tail of
    k's seen in a single node, which is why the figure's lines look shallow.
    this is a plain regression, as in the paper. not a proper power-law mle.

    returns:
        columns: kind, gamma (slope), intercept, k_min, k_max.
    """
    rows = []
    for kind, g in dist.groupby('kind'):
        gamma, b = np.polyfit(np.log10(g['k']), np.log10(g['pk']), 1)
        rows.append({'kind': kind, 'gamma': gamma, 'intercept': b,
                     'k_min': g['k'].min(), 'k_max': g['k'].max()})
    return pd.DataFrame(rows)

##########
# plotting functions

# colors and shape matches for degree types
KIND_COLOR = {'total': 'black', 'in': 'red', 'out': 'blue'}
KIND_SHAPE = {'total': 'o', 'in': 's', 'out': '^'}
KIND_LABEL = {'total': 'total', 'in': 'in', 'out': 'out'}


def plot_degree_distribution(dist: pd.DataFrame, fit: pd.DataFrame):
    """pr(k) vs k on log-log axes, with regression lines.

    args:
        dist: output of degree_distribution.
        fit: output of fit_power_law.
        letter: panel letter, e.g. "a".
        name: species name for the panel label.
    """
    # regression lines drawn from k = 1 to each type's max k
    lines = pd.concat([
        pd.DataFrame({
            'kind': f.kind,
            'k': [1, f.k_max],
            'pk': 10 ** (f.intercept + f.gamma * np.log10([1, f.k_max])),
        })
        for f in fit.itertuples()
    ])
    # gamma labels stacked top-right, coloured like the data
    labels = fit.set_index('kind')['gamma']
    p = (
        pn.ggplot(dist, pn.aes('k', 'pk', color='kind', shape='kind'))
        + pn.geom_line(data=lines, mapping=pn.aes('k', 'pk', color='kind'), size=0.8, inherit_aes=False)
        + pn.geom_point(size=2.5)
        + pn.scale_x_log10(limits=(0.7, 400))
        + pn.scale_y_log10(limits=(1e-4, 1))
        + pn.scale_color_manual(KIND_COLOR, labels=KIND_LABEL, name='degree')
        + pn.scale_shape_manual(KIND_SHAPE, labels=KIND_LABEL, name='degree')
        + pn.labs(x='degree (k)', y='Pr(k)')
        + pn.theme_bw()
        + pn.theme(panel_grid=pn.element_blank(), panel_border=pn.element_rect(color='black'))
    )
    for i, kind in enumerate(['total', 'in', 'out']):
        p += pn.annotate('text', x=300, y=0.6 / 2.2 ** i, ha='right', color=KIND_COLOR[kind],
                      label=f'γ ({kind}) = {labels[kind]:.2f}')

    return p

# %%
for species, vals in SPECIES.items():
    model, sheet, name = vals
    
    # route 1: authors' matrix
    adj = load_adjacency(XLSX, sheet)
    edges_xlsx = adjacency_to_edges(adj)
    deg = node_degrees(edges_xlsx, nodes=adj.index)
    dist = degree_distribution(deg)
    fit = fit_power_law(dist)
    panel = plot_degree_distribution(dist, fit)
    print(name)
    panel.show()

# %% [markdown]
# ## trying to rebuild reaction graph from SBML files

# %%
def clean_id(raw: str, prefix: str) -> str:
    """strip bigg-r decoration from an sbml id

    `R_PGI` -> `PGI`, `M_atp_c` -> `atp_c`, `R___iJO1366__AACPS1` -> `AACPS1`.
    the `__model__` part marks model-specific ids in the newer bigg-r files.
    """
    raw = re.sub(rf'^{prefix}_', '', raw) # remove R_ or M_ prefix
    return re.sub(r'^__[A-Za-z0-9]+__', '', raw) # remove model name


def read_sbml(path: Path) -> tuple[dict, dict]:
    """read reactions and metabolite compartments from a bigg sbml(.gz).

    returns:
        reactions: {rxn_id: {"reversible": bool, "reactants": [...], "products": [...]}}
        compartment: {metabolite_id: compartment_id}
    """
    with gzip.open(path) as fh:
        root = ET.parse(fh).getroot()
    # `{*}` ignores the sbml xml namespace. note: findall supports it, iter does not.
    compartment = {
        clean_id(s.get('id'), 'M'): s.get('compartment')
        for s in root.findall('.//{*}species')
    }
    reactions = {}
    for r in root.findall('.//{*}reaction'):
        def species(tag):
            return [clean_id(s.get('species'), 'M')
                    for lst in r.findall(f'{{*}}{tag}') for s in lst]
        reactions[clean_id(r.get('id'), 'R')] = {
            'reversible': r.get('reversible') == 'true',
            'reactants': species('listOfReactants'),
            'products': species('listOfProducts'),
        }
    return reactions, compartment


def is_boundary(rxn_id: str) -> bool:
    """exchange, demand, sink and biomass reactions never link intracellular steps."""
    return bool(re.match(r'(EX_|DM_|SK_|sink_|BIOMASS)', rxn_id, flags=re.I))


def build_reaction_graph(reactions: dict, compartment: dict,
                         currency: set = CURRENCY) -> pd.DataFrame:
    """build the directed reaction-centric graph as an edge list.

    rules from the methods section:
    - arc b -> c when a product of b is consumed by c.
    - a reversible reaction acts as both producer and consumer of everything.
    - currency metabolites are ignored (they make meaningless short paths).
    - exchange and transport reactions are dropped.

    transport is taken to be any reaction that touches more than one
    compartment. this is our guess. the paper does not define it.

    returns:
        dataframe with columns source, target. no self loops, no duplicates.
    """
    producers, consumers = {}, {}
    for rid, rxn in reactions.items():
        if is_boundary(rid):
            continue
        if len({compartment[m] for m in rxn['reactants'] + rxn['products']}) > 1:
            continue  # transport
        # drop currency metabolites: "atp_c" -> base "atp"
        def keep(mets):
            return [m for m in mets if m.removesuffix(f'_{compartment[m]}') not in currency]
        reac, prod = keep(rxn['reactants']), keep(rxn['products'])
        if rxn['reversible']:
            made, used = reac + prod, reac + prod
        else:
            made, used = prod, reac
        for m in made:
            producers.setdefault(m, set()).add(rid)
        for m in used:
            consumers.setdefault(m, set()).add(rid)

    # one arc per (producer, consumer) pair that shares a metabolite
    arcs = {(a, b)
            for m in producers.keys() & consumers.keys()
            for a in producers[m] for b in consumers[m] if a != b}
    return pd.DataFrame(sorted(arcs), columns=['source', 'target'])

# %%
def match_ids(sbml_ids, xlsx_ids) -> dict:
    """map sbml reaction ids to xlsx ids.

    bigg-r adds `_1`, `_2` to duplicate ids, so fall back to stripping that.
    ids that match neither way are left out.
    """
    xlsx_ids = set(xlsx_ids)
    out = {}
    for rid in sbml_ids:
        alt = re.sub(r'_\d+$', '', rid)
        if rid in xlsx_ids:
            out[rid] = rid
        elif alt in xlsx_ids:
            out[rid] = alt
    return out

# %%
reactions, comp = read_sbml(DATA / f'{model}.biggr.sbml.gz')
edges_sbml = build_reaction_graph(reactions, comp)
print(len(reactions), 'reactions in sbml ->', len(edges_sbml), 'arcs')

# degree distribution from our own graph, for a direct comparison
deg_sbml = node_degrees(edges_sbml)
fit_sbml = fit_power_law(degree_distribution(deg_sbml))

# plot from our own sbml-derived graph
panel_sbml = plot_degree_distribution(degree_distribution(deg_sbml), fit_sbml)
panel_sbml