import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("combined_kaggle_ectsum_100rows.csv")
df = df.sort_values(["Ticker","Date"])

# select numeric fields only
features = ["Open","High","Low","Close","Volume"]   # add others if numeric

X = StandardScaler().fit_transform(df[features])

kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Map clusters to interpretation based on avg close-open movement
cluster_means = df.groupby("cluster").apply(lambda x: (x["Close"] - x["Open"]).mean()).sort_values()

# lowest return cluster = SELL, middle = HOLD, highest = BUY
mapping = {
    cluster_means.index[0]: "SELL",
    cluster_means.index[1]: "HOLD",
    cluster_means.index[2]: "BUY"
}

df["target_decision"] = df["cluster"].map(mapping)
print(df["target_decision"].value_counts(normalize=True) * 100)

df.to_csv("combined_kaggle_ectsum_100rows_baseline_4.1.csv", index=False)
