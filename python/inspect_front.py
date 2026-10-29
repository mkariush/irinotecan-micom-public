"""Print the growth-reactivation front and uptake comparison for one community.

Read-only: needs pandas and data/processed/flux/growth_pareto.parquet. No MICOM, no cobra, no
model pickles, no solver. Everything here was computed once by python/32_growth_pareto.py.

    python python/inspect_front.py                      # a default sample
    python python/inspect_front.py CCIS00146684ST-4-0   # any sample in the 45
    python python/inspect_front.py --list               # show which samples are available
"""
import sys
import pandas as pd

SRC = "data/processed/flux/growth_pareto.parquet"


def main():
    p = pd.read_parquet(SRC)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "--list" in sys.argv:
        meta = p[["sample_id", "cohort", "study_condition"]].drop_duplicates("sample_id")
        cap = p[p.fraction == 0].set_index("sample_id")["sn38"]
        meta["capacity"] = meta.sample_id.map(cap).round(2)
        print(meta.sort_values(["cohort", "sample_id"]).to_string(index=False))
        return

    sample = args[0] if args else "CCIS00146684ST-4-0"
    d = p[p.sample_id == sample].sort_values("fraction")
    if d.empty:
        raise SystemExit(f"{sample!r} not in the 45-sample subset. Try --list.")

    gr_max = d.gr_max.iloc[0]
    print(f"Community: {sample}  ({d.cohort.iloc[0]}, {d.study_condition.iloc[0]})")
    print(f"Maximum possible growth rate (gr_max) = {gr_max:.4f} /h")
    print()
    print(f'{"f":>7}{"growth DEMANDED":>18}{"growth ACHIEVED":>18}{"binding?":>10}{"max SN-38":>12}')
    slack = False
    for _, r in d.iterrows():
        demanded = r.fraction * gr_max
        binds = abs(r.growth - demanded) < 1e-6 and demanded > 0
        # When the floor does NOT bind, growth is free above it: many solutions give the same
        # SN-38 and the solver returns an arbitrary one. That value is not reproducible between
        # runs, so it is flagged rather than printed as if it meant something.
        if binds:
            achieved = f"{r.growth:.4f}"
        else:
            achieved = f"{r.growth:.4f}*"
            slack = True
        print(f"{r.fraction:>7}{demanded:>18.4f}{achieved:>18}"
              f"{('yes' if binds else 'no'):>10}{r.sn38:>12.4f}")
    if slack:
        print()
        print("  * floor not binding: growth is free above it, so this value is one of many")
        print("    equally optimal solutions and will differ between runs. SN-38 is unaffected.")

    lo = d[d.fraction == 0].iloc[0]
    hi = d[d.fraction == d.fraction.max()].iloc[0]
    print()
    print(f'{"":>7}{"SN-38G uptake":>18}{"TOTAL uptake":>18}')
    for lbl, r in (("f=0", lo), (f"f={hi.fraction}", hi)):
        print(f"{lbl:>7}{r.sn38g_uptake:>18.4f}{r.total_uptake:>18.4f}")
    extra = 100 * (hi.total_uptake / lo.total_uptake - 1)
    print()
    print(f"  SN-38 unchanged across the whole front: "
          f"{abs(lo.sn38 - hi.sn38) < 1e-6}  (|diff| = {abs(lo.sn38 - hi.sn38):.2e})")
    print(f"  but growing flat out costs {extra:.1f}% more nutrient uptake")
    only = abs(lo.total_uptake - lo.sn38g_uptake) < 1e-6
    print(f"  at f=0 the community takes up SN-38G and nothing else: {only}")


if __name__ == "__main__":
    main()
