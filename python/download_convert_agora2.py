"""
Download AGORA2 .mat files from VMH and convert to JSON for MICOM.
Replaces the Nextflow+QIIME2 pipeline from micom-dev/databases.

Downloads only the species-level representative strains we need (~1368),
not the full 7302. Skips files already converted.
"""

import os
import ssl
import time
import urllib.request
import pandas as pd
import cobra
from concurrent.futures import ThreadPoolExecutor, as_completed

# Bypass SSL verification (equivalent to wget --no-check-certificate)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

AGORA2_DIR   = r"databases\AGORA2_SBML"
MAT_DIR      = r"databases\AGORA2_mat"
JSON_DIR     = r"databases\AGORA2_json"
MANIFEST_OUT = os.path.join(JSON_DIR, "manifest.csv")

VMH_URL = "https://www.vmh.life/files/reconstructions/AGORA2/version2.01/mat_files/individual_reconstructions"

THREADS = 4

os.makedirs(MAT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


def strain_to_id(strain_name: str) -> str:
    """Convert 'Abiotrophia defectiva ATCC 49176' -> 'Abiotrophia_defectiva_ATCC_49176'"""
    return strain_name.replace(" ", "_")


def download_mat(strain_id: str) -> tuple[str, bool, str]:
    mat_path = os.path.join(MAT_DIR, f"{strain_id}.mat")
    if os.path.exists(mat_path):
        return strain_id, True, "cached"
    url = f"{VMH_URL}/{strain_id}.mat"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as resp:
                with open(mat_path, "wb") as f:
                    f.write(resp.read())
            return strain_id, True, "downloaded"
        except Exception as e:
            if attempt == 2:
                return strain_id, False, f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)


def convert_mat_to_json(strain_id: str) -> tuple[str, bool, str]:
    mat_path  = os.path.join(MAT_DIR, f"{strain_id}.mat")
    json_path = os.path.join(JSON_DIR, f"{strain_id}.json")
    if os.path.exists(json_path):
        return strain_id, True, "cached"
    try:
        model = cobra.io.load_matlab_model(mat_path)
        cobra.io.save_json_model(model, json_path)
        return strain_id, True, "converted"
    except Exception as e:
        return strain_id, False, str(e)


if __name__ == "__main__":
    manifest = pd.read_csv(os.path.join(AGORA2_DIR, "manifest_all_strains.csv"))

    # Keep one representative strain per species (first alphabetically)
    spp_manifest = manifest.drop_duplicates(subset=["genus", "species"], keep="first").copy()
    spp_manifest["strain_id"] = spp_manifest["id"].apply(strain_to_id)
    print(f"Species to process: {len(spp_manifest)}")

    # --- Download .mat files ---
    print("\n=== Downloading .mat files ===")
    dl_failed = []
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(download_mat, row.strain_id): row.strain_id
                   for _, row in spp_manifest.iterrows()}
        done = 0
        for future in as_completed(futures):
            sid, ok, msg = future.result()
            done += 1
            if not ok:
                dl_failed.append((sid, msg))
            if done % 100 == 0 or not ok:
                print(f"  [{done}/{len(spp_manifest)}] {sid}: {msg if ok else 'FAIL: ' + msg}")

    print(f"\nDownload complete. Failed: {len(dl_failed)}")
    for sid, err in dl_failed[:10]:
        print(f"  {sid}: {err}")

    # --- Convert .mat → JSON ---
    print("\n=== Converting .mat → JSON ===")
    conv_failed = []
    downloaded = spp_manifest[~spp_manifest["strain_id"].isin([s for s, _ in dl_failed])]
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(convert_mat_to_json, row.strain_id): row.strain_id
                   for _, row in downloaded.iterrows()}
        done = 0
        for future in as_completed(futures):
            sid, ok, msg = future.result()
            done += 1
            if not ok:
                conv_failed.append((sid, msg))
            if done % 100 == 0 or not ok:
                status = "FAIL" if not ok else msg
                print(f"  [{done}/{len(downloaded)}] {sid}: {status}")

    print(f"\nConversion complete. Failed: {len(conv_failed)}")
    for sid, err in conv_failed[:10]:
        print(f"  {sid}: {err}")

    # --- Build clean manifest ---
    failed_ids = {s for s, _ in dl_failed} | {s for s, _ in conv_failed}
    clean = spp_manifest[~spp_manifest["strain_id"].isin(failed_ids)].copy()
    clean["file"]         = clean["strain_id"].apply(lambda s: f"{s}.json")
    clean["summary_rank"] = "species"
    clean[["id", "genus", "species", "strain", "file", "summary_rank"]].to_csv(
        MANIFEST_OUT, index=False
    )
    print(f"\nManifest written: {len(clean)} species -> {MANIFEST_OUT}")
    print(f"Skipped: {len(failed_ids)} species")
