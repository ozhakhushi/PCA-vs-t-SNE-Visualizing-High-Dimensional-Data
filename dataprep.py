import pandas as pd
from sklearn.preprocessing import StandardScaler

winedt = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\mlpractice\mlproject2\WineQT.csv")

s1 = StandardScaler()
winedtscaled = s1.fit_transform(winedt)
winedtscaled=pd.DataFrame(
    winedtscaled,
    columns=winedt.columns
)
winedtscaled.to_csv("winescaled.csv",index=False)
