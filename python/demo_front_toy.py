"""Minimal COBRApy demonstration of the growth-reactivation front (mSystems R2 comment 3).

Self-contained: builds two toy models from scratch, no AGORA2, no MICOM, no data files.
Needs only cobra. Runs in about a second.

    python python/demo_front_toy.py

WHY TWO MODELS. The paper reports that forcing growth does not reduce SN-38 production. The obvious
worry is that the method simply cannot detect a trade-off. So the demo builds the same toy twice:

  Model A  hydrolysis is cofactor-free, as in AGORA2:   sn38g + h2o -> sn38 + glucuronate
           Growth needs the limiting nutrient; reactivation does not. Front should be FLAT.

  Model B  identical EXCEPT hydrolysis also costs ATP:  sn38g + h2o + atp -> sn38 + glucuronate
           Now reactivation and biomass compete for the same ATP. Front should SLOPE DOWN.

Model B is the positive control: the same procedure that returns a flat front for A returns a sloped
one for B, so flatness is a property of the chemistry and not of the method.
"""
import cobra
from cobra import Model, Reaction, Metabolite


def build(gus_costs_atp: bool) -> Model:
    m = Model("toy_with_atp_cost" if gus_costs_atp else "toy_cofactor_free")

    glc_e, glc_c = Metabolite("glc_e"), Metabolite("glc_c")
    atp = Metabolite("atp_c")
    bio = Metabolite("biomass_c")
    g_e, g_c = Metabolite("sn38g_e"), Metabolite("sn38g_c")
    s_c, s_e = Metabolite("sn38_c"), Metabolite("sn38_e")
    gcur = Metabolite("glcur_c")

    def rxn(rid, stoich, lb=0.0, ub=1000.0):
        r = Reaction(rid)
        r.bounds = (lb, ub)
        m.add_reactions([r])
        r.add_metabolites(stoich)
        return r

    # --- the limiting nutrient: glucose uptake is capped, and it is the ONLY source of ATP ---
    rxn("EX_glc", {glc_e: -1}, lb=-10, ub=1000)       # negative flux = uptake, capped at 10
    rxn("GLC_T", {glc_e: -1, glc_c: 1})
    rxn("ATPGEN", {glc_c: -1, atp: 2})                 # glucose -> energy

    # --- growth consumes the limiting nutrient and energy ---
    growth = rxn("BIOMASS", {glc_c: -1, atp: -2, bio: 1})
    rxn("EX_biomass", {bio: -1})

    # --- the drug substrate is supplied without limit, exactly as EX_sn38g_m = 1000 in the paper ---
    rxn("EX_sn38g", {g_e: -1}, lb=-1000, ub=1000)
    rxn("SN38G_T", {g_e: -1, g_c: 1})

    # --- the reaction under test ---
    gus_stoich = {g_c: -1, s_c: 1, gcur: 1}
    if gus_costs_atp:
        gus_stoich[atp] = -1                           # the ONLY difference between the two models
    rxn("GUS", gus_stoich)

    rxn("EX_sn38", {s_c: -1})
    rxn("SINK_glcur", {gcur: -1})                      # glucuronate leaves; not fed back, to keep it minimal

    m.objective = "EX_sn38"
    return m, growth


def front(m: Model, growth: Reaction, fractions):
    """g(f) = max EX_sn38 subject to growth >= f * gr_max."""
    m.objective = growth
    m.objective_direction = "max"
    gr_max = m.optimize().objective_value

    out = []
    for f in fractions:
        growth.lower_bound = f * gr_max
        m.objective = "EX_sn38"
        # MUST be set explicitly: the uptake step below installs a direction="min" objective,
        # and assigning a reaction afterwards restores only the EXPRESSION, not the direction.
        # Without this every floor after the first silently minimises SN-38 (CLAUDE.md gotcha 6).
        m.objective_direction = "max"
        sol = m.optimize()
        sn38 = sol.objective_value if sol.status == "optimal" else float("nan")

        # deterministic uptake: pin SN-38 at its optimum, then minimise total uptake
        ex = [r for r in m.reactions if r.id.startswith("EX_") and r.lower_bound < 0]
        sn = m.reactions.EX_sn38
        lo, hi = sn.bounds
        sn.bounds = (sn38 - 1e-9, sn38 + 1e-9)
        m.objective = m.problem.Objective(sum(r.reverse_variable for r in ex), direction="min")
        m.optimize()
        uptake = sum(-r.flux for r in ex if r.flux < 0)
        sn.bounds = (lo, hi)

        out.append((f, f * gr_max, sol.fluxes[growth.id], sn38, uptake))
    growth.lower_bound = 0.0
    return gr_max, out


FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def report(costs_atp, fractions=FRACTIONS):
    """Build one model, sweep the front, print the table. Returns (model, growth, rows)."""
    m, growth = build(costs_atp)
    gr_max, rows = front(m, growth, fractions)
    label = ("MODEL B - hydrolysis ALSO costs ATP (positive control)" if costs_atp
             else "MODEL A - cofactor-free hydrolysis, as in AGORA2")
    print("=" * 76)
    print(label)
    print(f"  GUS reaction: {m.reactions.GUS.reaction}")
    print(f"  max growth (gr_max) = {gr_max:.4f}")
    print()
    print(f'{"f":>7}{"growth DEMANDED":>18}{"growth ACHIEVED":>18}{"max SN-38":>12}{"total uptake":>15}')
    for f, dem, ach, sn38, upt in rows:
        print(f"{f:>7}{dem:>18.4f}{ach:>18.4f}{sn38:>12.4f}{upt:>15.4f}")
    lo, hi = rows[0][3], rows[-1][3]
    print()
    print(f"  SN-38 at f=0 {lo:.4f}  ->  at f=0.999 {hi:.4f}   "
          f"{'FLAT' if abs(lo - hi) < 1e-6 else f'FELL by {100*(1-hi/lo):.1f}%'}")
    print(f"  total uptake {rows[0][4]:.4f}  ->  {rows[-1][4]:.4f}   "
          f"({100*(rows[-1][4]/rows[0][4]-1):+.1f}%)")
    print()
    return m, growth, rows


if __name__ == "__main__":
    for costs in (False, True):
        report(costs)
    print("=" * 76)
    print("Both models were put through the identical procedure. A is flat, B is not.")
    print("The method detects a trade-off when one exists; it reports none for A because,")
    print("as in AGORA2, the hydrolysis consumes nothing that biomass synthesis needs.")
