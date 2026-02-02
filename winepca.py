from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

winedt1=pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\mlpractice\mlproject2\winescaled.csv")
m1=PCA(n_components=2)
winereduced=m1.fit_transform(winedt1)

plt.figure(figsize=(8,6))
sns.scatterplot(x=winereduced[:,0], y=winereduced[:,1])
plt.title("PCA of Wine Dataset")
plt.show()

print(m1.explained_variance_ratio_)

