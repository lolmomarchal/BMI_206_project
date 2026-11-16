# Recreating Figure 2 (Kim et al. 2019, BMC Bioinformatics 20:328)

Figure 2 is the degree distribution (in, out, total) of the reaction-centric graphs for the
five organisms, on log-log axes with a regression line and a gamma for each.

There are two scripts:

| script | what it does | output |
|---|---|---|
| `figure2.py` | plots straight from the adjacency matrices in the paper's supplement (Additional file 2) | `figure2.png` |
| `figure2_rebuild.py` | rebuilds the five reaction graphs from the original BiGG models and plots those | `figure2_rebuilt.png` |

Run from this folder, e.g. `uv run --with openpyxl --with numpy --with matplotlib python figure2.py`
(the rebuild script only needs numpy + matplotlib, it downloads the models with `curl` into
`../data/old_bigg/` if they are not there).

## figure2.py (supplement matrices)

Degrees come from the matrices: in-degree = column sum, out-degree = row sum, total = both.
Pr(k) is the number of nodes with degree k divided by the number of nodes (k = 0 left out).
gamma is the slope of a least squares line through log10 Pr(k) vs log10 k.

This matches the paper for 14 of the 15 gammas if the slope is cut off at 2 decimals instead of
rounded (e.g. -1.0377 is printed as -1.03). The one that doesnt is E. coli in-degree
(-1.316, the paper prints -1.32). I tried weighted fits, a flipped regression, total least
squares and dropping low count points, none of them did better than the plain fit.

The authors' github (sybirg/cascade, zenodo 2634927) only has the MATLAB code for the cascade
number, nothing for building the graphs or for this figure.

## figure2_rebuild.py (from the original models)

The paper only describes the conversion in words ("currency metabolites were removed", "transport
and exchange reactions were removed", arc B -> C if a product of B is consumed by C). It does
not list the currency metabolites or say how reversibility was handled, so the rules below were
worked out by comparing against the supplement matrices, which have the reaction names for every
node.

### Data

The models are the originals from the old BiGG portal (bigg.ucsd.edu). Their metabolite,
reaction and gene counts match Table 1 of the paper exactly. The files downloaded from the new
portal (`data/*.biggr.sbml.gz`) are slightly different (iYO844 and iMM904 have 2 extra
metabolites each) so they are not used. Two files are used per model:

- `<model>.xml.gz`, the SBML, for reactions, metabolites and flux bounds
- `<model>.json.gz`, only for the `subsystem` of each reaction (the SBML doesnt have it)

### Rules, and how each one was determined

1. **Direction.** A reaction is reversible only if lower bound < 0 and upper bound > 0.
   Anything else is read forward as written. The `reversible` flag in the SBML cant be used:
   e.g. HMGCOAS in iYO844 is flagged reversible but has ub = 0, and in the supplement it has no
   arcs from the reactions that make acetyl-CoA. Using the bounds fixed that.
2. **Currency metabolites.** h, h2o, atp, adp, amp, nad, nadh, nadp, nadph, pi, ppi, co2, nh4, o2.
   Found by taking every pair of supplement nodes with no arc and collecting the metabolites that
   would have linked them. For E. coli this list was 15 metabolites, 14 of them these plus one
   odd murein species that is just an artifact of the node set. In yeast only the cytosolic
   copies are currency, the mitochondrial atp, nadh etc are normal metabolites.
3. **Transport and exchange reactions.** Dropped when the BiGG subsystem says transport or
   exchange (there are typos in iYL1228, "Tranpsort"). Also dropped by id prefix: `EX_`, `DM_`,
   `SK_`, `BIOMASS`. The subsystem has to be what the paper used, because it explains cases
   structure cant: 3PEPTabcpp is a transporter but its subsystem is "Murein Recycling" and it is
   a node in the supplement for G. metallireducens, while the other transporters are not. The
   subsystem "Inorganic Ion Transport and Metabolism" is mixed (reductases in, transporters out)
   so it is kept unless the reaction touches the periplasm.
4. **Boundary compartments.** Reactions touching extracellular (e) species are dropped. For the
   bacteria, reactions with no cytosolic (non currency) species are dropped too, i.e. things that
   only happen in the periplasm. For yeast the supplement only has cytosol, mitochondria and
   ER reactions, so anything touching peroxisome, nucleus, golgi or vacuole is dropped.
5. **E. coli only:** reactions whose id ends in `pp` or `ppi` (BiGG's periplasm naming) are
   dropped. The supplement has none of them. Doing this for K. pneumoniae made it worse so it
   is not applied there.
6. **Isolated reactions.** Reactions with no arcs after all of the above are not nodes.

Rules 1 and 2 are solid: given the supplement's own node lists they reproduce its arcs exactly
for E. coli, B. subtilis, G. metallireducens and S. cerevisiae (K. pneumoniae could not be
checked this way, see below). Rule 3 is well supported. Rules 4 and 5 are tuned to match the
supplement's node lists, so they are a best fit and not something the paper states.

### Result

| organism | nodes ours / paper | arcs ours / paper |
|---|---|---|
| E. coli (iJO1366) | 1257 / 1251 | 9103 / 9099 |
| B. subtilis (iYO844) | 748 / 748 | 6489 / 6489 |
| G. metallireducens (iAF987) | 901 / 900 | 8050 / 8049 |
| K. pneumoniae (iYL1228) | 1146 / 1137 | 8182 / 8084 |
| S. cerevisiae (iMM904) | 890 / 881 | 10466 / 10460 |

11 of the 15 gammas match the paper to 2 decimals (the script prints ours next to the paper's);
the rest are off by 0.01 to 0.03.

### Known gaps

- A handful of reactions look the same but the paper keeps one and drops the other (e.g. in E.
  coli ADNCYC, AP4AH, AP4AS, PDE1, LIPATPT, LIPAMPL, NODOx, NODOy are not in the supplement).
  Probably the paper's older model files were annotated a bit differently.
- The paper's iYL1228 list uses old reaction ids (ACACT2r - ACACT7r) that dont exist in the
  current model, so that one couldnt be checked reaction by reaction. It is also the one furthest
  off.
- The paper's own gamma values look like they were cut off instead of rounded, and one of them
  (E. coli in-degree) doesnt fit either way.
