import urllib.request
import ssl
import pandas as pd

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://raw.githubusercontent.com/ThieleLab/CodeBase/master/Scripts_AGORA2_Heinken_et_al_2022/input/knownDrugMetabolizers.xlsx"
out = "data/raw/knownDrugMetabolizers.xlsx"

urllib.request.urlretrieve(url, out)
print("Downloaded to", out)

xl = pd.ExcelFile(out)
print("Sheets:", xl.sheet_names)
df = xl.parse(xl.sheet_names[0])
print(f"Shape: {df.shape}")
print(df.columns.tolist())
print(df.head(5).to_string())
