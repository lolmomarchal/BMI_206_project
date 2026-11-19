# %% [markdown]
# # figure 2: degree distributions of the reaction-centric graphs

# %%
import re
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import optimize
from scipy.special import zeta

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
DATA = HERE.parent / 'data'
OLD = DATA / 'old_bigg' # path to old BIGG SBML files
XLSX = DATA / '12859_2019_2897_MOESM2_ESM.xlsx' # path to the adjacency matrix supplement

# bigg model id, xlsx sheet name, display name
SPECIES = {
    'ecoli': ('iJO1366', 'E.coli (iJ01366)', 'E. coli'),
    'bsub': ('iYO844', 'B. subtilis (iYO844)', 'B. subtilis'),
    'gmet': ('iAF987', 'G. metallireducens (iAF987)', 'G. metallireducens'),
    'kpneu': ('iYL1228', 'K. pneumoniae (iYL1228)', 'K. pneumoniae'),
    'scer': ('iMM904', 'S. cerevisiae (iMM904)', 'S. cerevisiae'),
}

# %%
# data loading
##########

# supplement: authors' adjacency matrices, one sheet per species
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


adj_xlsx, edges_xlsx = {}, {}
for sp, (model, sheet, name) in SPECIES.items():
    adj_xlsx[sp] = load_adjacency(XLSX, sheet)
    edges_xlsx[sp] = adjacency_to_edges(adj_xlsx[sp])

# bigg: old sbml models read with cobra (reactions, metabolites, compartments, flux bounds)
models = {sp: cobra.io.read_sbml_model(str(OLD / f'{model}.xml.gz')) for sp, (model, _, _) in SPECIES.items()}
# table 1 of the paper: 2583, 1250, 1285, 2262, 1577 reactions
{sp: len(m.reactions) for sp, m in models.items()}

# %%
# calculations: degrees, pr(k), power-law fit
##########
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


# degree distribution and fit for the supplement graphs
fits_xlsx, dists_xlsx = {}, {}
for sp in SPECIES:
    deg = node_degrees(edges_xlsx[sp], nodes=adj_xlsx[sp].index)
    dists_xlsx[sp] = degree_distribution(deg)
    fits_xlsx[sp] = fit_power_law(dists_xlsx[sp])


# %%
# calculations: building a reaction graph from a bigg model
##########

def is_boundary(rxn):
    """exchange / demand / sink (cobra's rxn.boundary) and biomass."""
    return rxn.boundary or 'biomass' in rxn.id.lower()


def roles(rxn):
    """(made, used) metabolites. a reaction that can run backward makes and uses all of them.

    reversible = lower bound < 0. this matches the sbml `reversible` flag. cobra's own
    rxn.reversibility (lb < 0 < ub) is stricter: it calls lb < 0, ub = 0 reactions
    irreversible, and that loses supplement arcs.
    """
    both = rxn.reactants + rxn.products
    return (both, both) if rxn.lower_bound < 0 else (rxn.products, rxn.reactants)


def build_graph(model, only=None, currency=(), currency_in=None, node_rule=None):
    """directed reaction graph as an edge list. arc a -> b if a makes what b uses.

    args:
        only: keep only these reaction ids.
        currency: base metabolite ids to ignore (id without compartment suffix).
        currency_in: compartments where currency is ignored. None = all.
        node_rule: function (reaction) -> bool. false drops the reaction.
    """
    drop = lambda m: (m.id.removesuffix(f'_{m.compartment}') in currency
                      and (currency_in is None or m.compartment in currency_in))
    prod, cons = {}, {}
    for rxn in model.reactions:
        if is_boundary(rxn) or (only is not None and rxn.id not in only):
            continue
        if node_rule is not None and not node_rule(rxn):
            continue
        made, used = roles(rxn)
        for m in made:
            if not drop(m):
                prod.setdefault(m.id, set()).add(rxn.id)
        for m in used:
            if not drop(m):
                cons.setdefault(m.id, set()).add(rxn.id)
    arcs = {(a, b) for m in prod.keys() & cons.keys() for a in prod[m] for b in cons[m] if a != b}
    return pd.DataFrame(sorted(arcs), columns=['source', 'target'])


# %%
# plotting
##########

# colors and shape matches for degree types
KIND_COLOR = {'total': 'black', 'in': 'red', 'out': 'blue'}
KIND_SHAPE = {'total': 'o', 'in': 's', 'out': '^'}
KIND_LABEL = {'total': 'total', 'in': 'in', 'out': 'out'}

def plot_degree_distribution(dist: pd.DataFrame, fit: pd.DataFrame):
    """pr(k) vs k on log-log axes, with regression lines.

    args:
        dist: output of degree_distribution.
        fit: output of fit_power_law.
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
# figure 2 from the supplement matrices
for sp, (model, sheet, name) in SPECIES.items():
    print(name)
    plot_degree_distribution(dists_xlsx[sp], fits_xlsx[sp]).show()

# %%
# check: supplement vs the paper

# node and arc counts: table 1 of the paper (reaction graphs)
PAPER_COUNTS = {'ecoli': (1251, 9099), 'bsub': (748, 6489), 'gmet': (900, 8049),
                'kpneu': (1137, 8084), 'scer': (881, 10460)}
# gamma read off figure 2 of the paper
PAPER_GAMMA = {
    'ecoli': {'total': -1.29, 'in': -1.32, 'out': -1.50},
    'bsub': {'total': -1.03, 'in': -1.10, 'out': -1.12},
    'gmet': {'total': -1.03, 'in': -1.14, 'out': -1.11},
    'kpneu': {'total': -1.21, 'in': -1.31, 'out': -1.47},
    'scer': {'total': -0.99, 'in': -1.08, 'out': -1.16},
}
pd.DataFrame({
    sp: {'nodes': len(adj_xlsx[sp]), 'nodes (paper)': PAPER_COUNTS[sp][0],
         'arcs': len(edges_xlsx[sp]), 'arcs (paper)': PAPER_COUNTS[sp][1],
         **{f'gamma {k}': round(fits_xlsx[sp].set_index('kind').loc[k, 'gamma'], 2) for k in ['total', 'in', 'out']},
         **{f'gamma {k} (paper)': PAPER_GAMMA[sp][k] for k in ['total', 'in', 'out']}}
    for sp in SPECIES
}).T

# %%
# rebuilding the supplement graphs from bigg
##########

# attempt 0: no filtering. only exchange / demand / sink / biomass reactions removed.
naive = build_graph(models['ecoli'])
print(f"naive: {len(set(naive['source']) | set(naive['target']))} nodes, {len(naive)} arcs")
print(f"paper: {len(adj_xlsx['ecoli'])} nodes, {len(edges_xlsx['ecoli'])} arcs")
# way too many arcs. protons and water link nearly every reaction to every other.

# %%
# before comparing graphs: match sbml reaction ids to the supplement.
# bigg adds _1, _2 to duplicated ids, so also try with that suffix stripped.
id_maps = {}
rows = []
for sp, model in models.items():
    xi = set(adj_xlsx[sp].index)
    rids = [r.id for r in model.reactions]
    id_maps[sp] = {r: r if r in xi else re.sub(r'_\d+$', '', r) for r in rids # remove BIGG suffixes
                   if r in xi or re.sub(r'_\d+$', '', r) in xi}
    found = set(id_maps[sp].values())
    rows.append({'species': sp, 'supplement reactions': len(xi),
                 'found directly': len(xi & set(rids)), 'found with _N stripped': len(found),
                 'not found': len(xi - found)})
pd.DataFrame(rows)

# supplement reactions not found in the SBML mapping, for all species
for sp in SPECIES:
    print(sp, sorted(set(adj_xlsx[sp].index) - set(id_maps[sp].values())))


# %%
# compare a rebuilt graph with the supplement matrix
def score(edges, sp):
    """compare a rebuilt graph with the supplement matrix for species sp."""
    ours = {(id_maps[sp].get(a, a), id_maps[sp].get(b, b)) for a, b in edges.itertuples(index=False)}
    theirs = set(edges_xlsx[sp].itertuples(index=False, name=None))
    nodes = {x for e in ours for x in e}
    xlsx_nodes = set(adj_xlsx[sp].index)
    return {'species': sp, 'nodes': len(nodes), 'nodes_shared': len(nodes & xlsx_nodes),
            'nodes_missing': len(xlsx_nodes - nodes), 'arcs': len(ours),
            'arcs_shared': len(ours & theirs), 'only_ours': len(ours - theirs),
            'only_theirs': len(theirs - ours)}

# %%
# build arcs one metabolite at a time, using only reactions found in both datasets.
# if a metabolite makes many arcs and almost none are in the supplement, it was removed.
audit = {}
for sp, model in models.items():
    theirs = set(edges_xlsx[sp].itertuples(index=False, name=None))
    prod, cons = {}, {}
    for rxn in model.reactions:
        # remove boundary reactions and reactions not in the supplement
        if is_boundary(rxn) or rxn.id not in id_maps[sp]:
            continue
        made, used = roles(rxn)
        # record which metabolites are produced and consumed by this reaction
        for m in made:
            prod.setdefault(m, set()).add(id_maps[sp][rxn.id])
        for m in used:
            cons.setdefault(m, set()).add(id_maps[sp][rxn.id])
    rows = []
    for m in prod.keys() & cons.keys():
        # build arcs for this metabolite
        arcs = {(a, b) for a in prod[m] for b in cons[m] if a != b}
        if arcs:
            # metabolite = name
            # base = cleand up metabolite name
            # compartment = compartment of the metabolite
            # n_arcs = number of arcs for this metabolite
            # frac_in_supplement = fraction of arcs found in the supplement
            rows.append({'metabolite': m.id, 'base': m.id.removesuffix(f'_{m.compartment}'),
                         'compartment': m.compartment, 'n_arcs': len(arcs),
                         'frac_in_supplement': len(arcs & theirs) / len(arcs)})

    audit[sp] = pd.DataFrame(rows).sort_values('n_arcs', ascending=False, ignore_index=True)

# show summary table of audit
for sp, a in audit.items():
    print(sp)
    display(a.head(10))


# %%
# plot fraction of arcs in the supplement for each species
audit_long = pd.concat([a.assign(species=sp) for sp, a in audit.items()], ignore_index=True)

p_audit = (pn.ggplot(audit_long, pn.aes(x='frac_in_supplement'))
    + pn.geom_histogram(bins=20)
    # + pn.xlim(0, 1)
    + pn.facet_wrap('~species')
    + pn.labs(x='Fraction in supplement', y='Count')
    + pn.theme_classic() + pn.theme(figure_size=(10, 6)))
p_audit.show()

# each point is one metabolite: how many arcs it makes vs how many of them the supplement has.
# two clear groups: removed metabolites (many arcs, ~none in the supplement, bottom right)
# and kept ones (top). almost nothing in between.
# same cut as the rule in the next cell: at least 20 arcs and at most 5% in the supplement.
audit_long['removed'] = (audit_long['n_arcs'] >= 20) & (audit_long['frac_in_supplement'] <= 0.05)
p_removed = (pn.ggplot(audit_long, pn.aes('n_arcs', 'frac_in_supplement', color='removed'))
    + pn.geom_point(alpha=0.6)
    + pn.scale_x_log10()
    + pn.scale_color_manual({True: 'red', False: 'grey'}, labels={True: 'removed', False: 'kept'}, name='')
    + pn.facet_wrap('~species')
    + pn.labs(x='arcs made by metabolite (log)', y='fraction in supplement')
    + pn.theme_classic() + pn.theme(figure_size=(10, 6)))
p_removed.show()

# which metabolites were removed in which species, and in which compartment?
# blank = not removed there (or not enough arcs to tell).
removed = audit_long.query('removed')
order = removed.groupby('base')['n_arcs'].max().sort_values().index.tolist()
# one tile per species and metabolite. kpneu removed h in both c and p, so join them
tiles = (removed.groupby(['species', 'base'])
         .agg(compartments=('compartment', lambda c: '+'.join(sorted(c)))).reset_index())

p_tiles = (pn.ggplot(tiles, pn.aes('species', 'base', fill='compartments'))
    + pn.geom_tile(color='white')
    + pn.scale_y_discrete(limits=order)
    + pn.scale_fill_brewer(type='qual', palette='Set2')
    + pn.labs(x='', y='removed metabolite')
    + pn.theme_classic() + pn.theme(figure_size=(5, 5)))
p_tiles.show()


# %%
# turn the split into a rule: many arcs and (almost) none in the supplement.
hits = {sp: a.query('n_arcs >= 20 and frac_in_supplement <= 0.05') for sp, a in audit.items()}
for sp, h in hits.items():
    print(sp, {c: sorted(g['base']) for c, g in h.groupby('compartment')})

# %%
# is the list the same in every species?
by_species = {sp: set(h['base']) for sp, h in hits.items()}
CURRENCY = set.union(*by_species.values())
print('all species:', sorted(set.intersection(*by_species.values())))
print('some species:', {m: [sp for sp, s in by_species.items() if m in s]
                        for m in sorted(CURRENCY - set.intersection(*by_species.values()))})
# the paper only names atp, nad and h2o. the rest had to be found from the data.

# %%
# attempt 1: rebuild with this currency list, using only reactions found in both.
# this isolates the metabolite question from the question of which reactions are nodes.
rows = [score(build_graph(models[sp], only=id_maps[sp], currency=CURRENCY), sp) for sp in SPECIES]
pd.DataFrame(rows)
# s. cerevisiae is far off: half of the supplement arcs are missing from our graph.

# %%
# see what compartments the currency metabolites appear in for each species
for sp in SPECIES:
    a = audit[sp].query('base in @CURRENCY and n_arcs >= 5')
    print(sp)
    display(a.pivot(index='base', columns='compartment', values='frac_in_supplement').round(2))
# currency was removed in the cytosol (c) but kept in the organelles (m, r...).
# h_m, atp_m, nadh_m... all have arcs in the supplement.

# %%
# rebuild again, removing currency in cell-level compartments only (cytosol, periplasm,
# extracellular), not in yeast organelles. c alone is not enough: g. metallireducens
# has currency in the periplasm (p) that the authors also removed.
CUR_COMPS = {'c', 'p', 'e'}
rows = [score(build_graph(models[sp], only=id_maps[sp], currency=CURRENCY, currency_in=CUR_COMPS), sp)
        for sp in SPECIES]
pd.DataFrame(rows)
# yeast now matches: all 10460 supplement arcs are found. every species has all arcs
# except k. pneumoniae (325 supplement arcs missing) and e. coli (41).

# %%
# which reactions are nodes? supplement has no exchange reactions.
# for e. coli and yeast: compartments touched by each reaction, and is it in the supplement?
for sp in ['ecoli', 'scer']:
    m = pd.DataFrame([{'compartments': ''.join(sorted(r.compartments)), 'in_supplement': r.id in id_maps[sp]}
                      for r in models[sp].reactions if not is_boundary(r)])
    print(sp)
    print(m.groupby('compartments')['in_supplement'].agg(['sum', 'count'])
          .sort_values('count', ascending=False).head(10).T.to_string())
# nearly all supplement reactions are cytosol-only. reactions touching the periplasm (p) or
# extracellular space (e) are mostly absent. yeast also has organelles (m, x, n, r, v, g).

# %%
def is_transport(rxn, ignore=()):
    """true if one metabolite sits on both sides in different compartments (glc_e -> glc_c).

    args:
        ignore: base ids to skip, so h_c -> h_p does not make a reaction transport.
    """
    seen = {}
    for m in rxn.metabolites:
        b = m.id.removesuffix(f'_{m.compartment}')
        if b not in ignore:
            seen.setdefault(b, set()).add(m.compartment)
    return any(len(v) > 1 for v in seen.values())


# candidate rules for "keep this reaction as a node"
rules = {
    'cytosol only': lambda r: r.compartments == {'c'},
    'not transport': lambda r: not is_transport(r),
    'not transport (ignore currency)': lambda r: not is_transport(r, CURRENCY),
    'not transport (ignore currency), not e/p only':
        lambda r: not is_transport(r, CURRENCY) and not r.compartments <= {'e', 'p'},
}
rows = [{'rule': name, **score(build_graph(models[sp], currency=CURRENCY, currency_in=CUR_COMPS, node_rule=rule), sp)}
        for name, rule in rules.items() for sp in SPECIES]
pd.DataFrame(rows).drop(columns='nodes')
# the last rule gets all supplement nodes for b. subtilis and yeast (1 missing) and the
# arcs between them. it also keeps extra reactions the authors dropped (nodes - nodes_shared).

# %%
# final: final rebuild with the last rule. compare degree slopes with the supplement.
FINAL = rules['not transport (ignore currency), not e/p only']
rebuilt, rows = {}, []
for sp, (model, sheet, name) in SPECIES.items():
    edges = build_graph(models[sp], currency=CURRENCY, currency_in=CUR_COMPS, node_rule=FINAL)
    rebuilt[sp] = edges
    fit = fit_power_law(degree_distribution(node_degrees(edges)))
    rows.append({'species': sp, **score(edges, sp),
                 **{f'gamma {k}': round(g, 2) for k, g in zip(fit['kind'], fit['gamma'])},
                 **{f'gamma {k} (supp.)': round(g, 2) for k, g in
                    zip(fits_xlsx[sp]['kind'], fits_xlsx[sp]['gamma'])}})

pd.DataFrame(rows)
# filtering reactions
# 1. remove boundary reactions - exchange, demand, sink (based on rxn.boundary) and biomass
# 2. remove transport reactions, defined as having the same base ID metabolite on both sides) except those involving currency metabolites
# 3. remove reactions that are only in the extracellular (e) or periplasm (p) compartments

# filtering metabolites
# currencies = H, H20, ATP, ADP, AMP, PI, PPI, NADP, NADPH, CO2, NH4, O2
# ignored only in cytoplasm, periplasm, and extracellular compartments (c, p, e)
# keep currencies if they are in the mitochondria (m) compartment

# %%
# figure 2 from our rebuilt graphs
for sp, edges in rebuilt.items():
    dist = degree_distribution(node_degrees(edges))
    print(SPECIES[sp][2])
    plot_degree_distribution(dist, fit_power_law(dist)).show()


# %% [markdown]
# ## testing different regression models

# the paper never says how gamma was fit. it only says the degree distributions "followed a
# power-law" and shows "regression" lines on log axes. our guess: ordinary least squares on
# log10 Pr(k) vs log10 k, one point per observed k. check it against the published gammas.
ours = pd.DataFrame({sp: fits_xlsx[sp].set_index('kind')['gamma'] for sp in SPECIES}).T
paper = pd.DataFrame(PAPER_GAMMA).T
# difference in gamma, ours minus paper (the paper rounds to 2 decimals)
(ours - paper).round(3)
# e. coli matches exactly. the rest are within 0.01. the guess reproduces the paper.

# %%
def power_law_mle(x, kmin):
    """discrete power-law exponent by maximum likelihood (clauset et al. 2009).

    models Pr(k) = k^-alpha / zeta(alpha, kmin) for k >= kmin and fits alpha
    by minimizing the negative log likelihood. returns alpha (positive).
    """
    tail = x[x >= kmin]
    s = np.log(tail).sum()
    nll = lambda a: a * s + len(tail) * np.log(zeta(a, kmin))
    return optimize.minimize_scalar(nll, bounds=(1.01, 6), method='bounded').x


def best_kmin(x, min_tail=30):
    """kmin that makes the power-law fit closest to the data (smallest ks distance).

    this is how clauset et al. choose where the power law starts. fits with fewer
    than min_tail nodes in the tail are skipped.
    """
    best = None
    for kmin in np.unique(x):
        tail = np.sort(x[x >= kmin])
        if len(tail) < min_tail:
            break
        a = power_law_mle(x, kmin)
        k = np.unique(tail)
        empirical = np.searchsorted(tail, k, side='right') / len(tail)
        model = 1 - zeta(a, k + 1) / zeta(a, kmin)
        d = np.abs(empirical - model).max()
        if best is None or d < best[0]:
            best = (d, kmin, a)
    return best[1], best[2]


def fit_models(degrees):
    """fit one degree distribution with several models.

    args:
        degrees: degree of every node, for one degree type.
    returns:
        table: one row per model. `gamma` is the power-law exponent (negative, so
            comparable with the paper). exponential and linear fits have no gamma,
            so their slope goes in `coef`. `rmse_log10` = typical distance between data
            and fitted curve in log10 Pr(k), over the k each model was fit on.
        curves: dict model -> function k -> fitted Pr(k), for plotting.
    """
    x = np.asarray(degrees)
    n = len(x)
    x = x[x >= 1]
    k, c = np.unique(x, return_counts=True)
    pk = c / n
    ccdf = np.array([(x >= kk).sum() for kk in k]) / n    # Pr(K >= k)
    lk, lp = np.log10(k), np.log10(pk)
    rows, curves = [], {}
    # note: every curve lambda binds its fitted values as default arguments (a=a, b=b, ...).
    # without that, python looks the names up when the curve is called, after later fits
    # have overwritten them, and every curve would use the last fit.

    def add(name, family, gamma=np.nan, coef=np.nan, kmin=1, curve=None):
        # error against the data, only where the curve is defined and positive
        if curve is not None:
            keep = k >= kmin
            fit = curve(k[keep])
            ok = fit > 0
            rmse = np.sqrt(np.mean((lp[keep][ok] - np.log10(fit[ok])) ** 2))
            curves[name] = curve
        else:
            rmse = np.nan
        rows.append({'model': name, 'family': family, 'gamma': gamma, 'coef': coef,
                     'kmin': kmin, 'rmse_log10': rmse})

    # --- least squares ---
    # power law: straight line on log-log axes. slope = gamma. (what the paper seems to do)
    b, a = np.polyfit(lk, lp, 1)
    add('log-log OLS', 'power law', gamma=b, curve=lambda q, a=a, b=b: 10 ** (a + b * np.log10(q)))
    # same, but weight each k by how many nodes have it (rare k count for less)
    b, a = np.polyfit(lk, lp, 1, w=np.sqrt(c))
    add('log-log WLS (weight = count)', 'power law', gamma=b,
        curve=lambda q, a=a, b=b: 10 ** (a + b * np.log10(q)))
    # same, but drop k values seen in only one node (the flat floor at 1/N)
    m = c >= 2
    b, a = np.polyfit(lk[m], lp[m], 1)
    add('log-log OLS (k seen in >= 2 nodes)', 'power law', gamma=b,
        curve=lambda q, a=a, b=b: 10 ** (a + b * np.log10(q)))
    # cumulative distribution: Pr(K >= k) ~ k^(1 - alpha), so gamma = slope - 1
    b, a = np.polyfit(lk, np.log10(ccdf), 1)
    add('log-log OLS on cumulative', 'power law', gamma=b - 1)
    # log-linear: log Pr(k) against k. this is an exponential, not a power law
    b, a = np.polyfit(k, lp, 1)
    add('log-linear OLS (exponential)', 'exponential', coef=b, curve=lambda q, a=a, b=b: 10 ** (a + b * q))
    # linear: Pr(k) against k
    b, a = np.polyfit(k, pk, 1)
    add('linear OLS', 'linear', coef=b, curve=lambda q, a=a, b=b: a + b * q)

    # --- maximum likelihood ---
    alpha = power_law_mle(x, 1)
    add('MLE (kmin = 1)', 'power law', gamma=-alpha, curve=lambda q, alpha=alpha: q ** -alpha / zeta(alpha, 1))
    kmin, alpha = best_kmin(x)
    frac = (x >= kmin).sum() / n
    add('MLE (kmin by KS)', 'power law', gamma=-alpha, kmin=kmin,
        curve=lambda q, alpha=alpha, frac=frac, kmin=kmin: frac * q ** -alpha / zeta(alpha, kmin))
    return pd.DataFrame(rows), curves


# %%
# fit every species and degree type, using the supplement degrees
tables, all_curves = [], {}
for sp in SPECIES:
    deg = node_degrees(edges_xlsx[sp], nodes=adj_xlsx[sp].index)
    for kind in ['total', 'in', 'out']:
        t, curves = fit_models(deg[kind])
        tables.append(t.assign(species=sp, kind=kind, paper=PAPER_GAMMA[sp][kind]))
        all_curves[sp, kind] = curves
fit_table = pd.concat(tables, ignore_index=True)

# gamma by model. last column is the published value.
fit_table.pivot_table(index=['species', 'kind'], columns='model', values='gamma').join(
    fit_table.drop_duplicates(['species', 'kind']).set_index(['species', 'kind'])['paper']).round(2)

# %%
# the exponential and linear fits have no gamma (they are not power laws). their slope and the
# fit error are here. rmse_log10: how far the curve is from the data in log10 Pr(k), over the
# k each model was fit on. lower = closer. compare across models with care: the tail-only mle
# (kmin by ks) is scored on fewer points.
(fit_table.query("species == 'ecoli' and kind == 'total'")
 [['model', 'family', 'gamma', 'coef', 'kmin', 'rmse_log10']].round(3))

# %%
# gamma by model, all species. cross = published value.
paper_pts = fit_table.drop_duplicates(['species', 'kind'])
p_gamma = (pn.ggplot(fit_table.dropna(subset=['gamma']), pn.aes('model', 'gamma', color='kind'))
    + pn.geom_point(size=2.5)
    #  + pn.geom_point(pn.aes(y='paper'), data=paper_pts.assign(model='log-log OLS'),
    #                  shape='x', size=4, stroke=1)
    # draw a vertical line for the published value
    + pn.geom_hline(pn.aes(yintercept='paper', color='kind'), data=paper_pts, linetype='dashed')
    + pn.scale_color_manual(KIND_COLOR, name='degree')
    + pn.scale_x_discrete(limits=fit_table.dropna(subset=['gamma'])['model'].unique()[::-1])
    + pn.coord_flip()
    + pn.facet_wrap('~species', ncol=3)
    + pn.labs(x='', y='gamma')
    + pn.theme_classic() + pn.theme(figure_size=(12, 6)))

p_gamma.show()

# plot deltas of gamma from published values
delta_df = fit_table.dropna(subset=['gamma']).copy()
delta_df['delta'] = delta_df['gamma'] - delta_df['paper']

p_delta = (pn.ggplot(delta_df, pn.aes('model', 'delta', color='kind'))
    + pn.geom_hline(yintercept=0, linetype='dashed')
    + pn.geom_point(size=2.5)
    + pn.scale_color_manual(KIND_COLOR, name='degree')
    + pn.scale_x_discrete(limits=delta_df['model'].unique()[::-1])
    + pn.coord_flip()
    + pn.facet_wrap('~species', ncol=3)
    + pn.labs(x='', y='gamma - published')
    + pn.theme_classic() + pn.theme(figure_size=(12, 6)))

p_delta.show()


# %%
# what each model looks like on the data, every species, in the same style as the recreation.
# one panel per degree type. points: same colours/shapes as before. lines: one colour per model.
# log-linear and linear curves are not power laws, so on log-log axes they bend away from the points.
kinds = ['total', 'in', 'out']
for sp, (model, sheet, name) in SPECIES.items():
    dist = dists_xlsx[sp].assign(kind=lambda d: pd.Categorical(d['kind'], kinds))
    parts = []
    for kind in kinds:
        grid = np.arange(1, dist.query('kind == @kind')['k'].max() + 1)
        for m, f in all_curves[sp, kind].items():
            # the tail-only mle is only meaningful from its own kmin up
            kmin = fit_table.query('species == @sp and kind == @kind and model == @m')['kmin'].iloc[0]
            g = grid[grid >= kmin]
            parts.append(pd.DataFrame({'kind': kind, 'model': m, 'k': g, 'pk': f(g)}))
    curve_df = pd.concat(parts).query('pk > 0').assign(kind=lambda d: pd.Categorical(d['kind'], kinds))

    p = (pn.ggplot(dist, pn.aes('k', 'pk'))
         # fill (not color) for the points, so color is free for the model lines
        + pn.geom_line(pn.aes(color='model'), data=curve_df, size=0.8)
        + pn.geom_point(pn.aes(fill='kind', shape='kind'), color='none', size=2.5)
        + pn.scale_x_log10(limits=(0.7, 400))
        + pn.scale_y_log10(limits=(1e-4, 1))
        + pn.scale_fill_manual(KIND_COLOR, labels=KIND_LABEL, name='degree')
        + pn.scale_shape_manual(KIND_SHAPE, labels=KIND_LABEL, name='degree')
        + pn.scale_color_brewer(type='qual', palette='Dark2', name='model')
        + pn.facet_wrap('~kind', ncol=3)
        + pn.labs(x='degree (k)', y='Pr(k)', title=name)
        + pn.theme_classic()
        + pn.theme(panel_grid=pn.element_blank(), panel_border=pn.element_rect(color='black'),
                    figure_size=(14, 4.5)))
    p.show()



# %%
