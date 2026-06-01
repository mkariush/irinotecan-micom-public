import os
import glob
import cobra

json_dir = r"databases\AGORA2_json"
target_rxns = {"SN38G_GLCAASE", "SN38G_GLCAASEe", "SN38G_GLCAASEepp"}
keywords = ["sn38", "sn_38"]

hits = []
for path in glob.glob(os.path.join(json_dir, "*.json")):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "sn38" not in content.lower():
                continue
        model = cobra.io.load_json_model(path)
        rxns = [r for r in model.reactions if r.id in target_rxns]
        ex_rxns = [r for r in model.reactions
                   if r.id.startswith("EX_") and any(k in r.id.lower() for k in keywords)]
        if rxns or ex_rxns:
            hits.append(os.path.basename(path))
            print(f"\n{os.path.basename(path)}:")
            for r in rxns:
                print(f"  Reaction  {r.id}: {r.reaction}")
            for r in ex_rxns:
                print(f"  Exchange  {r.id}: lb={r.lower_bound}, ub={r.upper_bound}")
            if len(hits) >= 5:
                break
    except Exception:
        continue

print(f"\nFound SN-38 reactions in {len(hits)} models")
