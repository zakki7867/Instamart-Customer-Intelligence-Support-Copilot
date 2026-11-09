"""customer_segmentation.py -- RFM computation and K-Means segmentation for Part B.
Produces outputs/customer_segments.csv with columns:
customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ORDERS_PATH = os.path.join("data", "clean", "orders.csv")
OUTPUT_PATH = os.path.join("outputs", "customer_segments.csv")
os.makedirs("outputs", exist_ok=True)


def build_rfm(orders_df, snapshot_str="2025-10-01 00:00:00"):
    snapshot = pd.Timestamp(snapshot_str)
    completed = orders_df[orders_df["status"] == "completed"].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["customer_id", "recency", "frequency", "monetary"]]


def evaluate_clusters(X_scaled):
    print("K-Means Cluster Evaluation (K=2 to 8):")
    results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(sil, 3)})
        print(f"  K={k}: Inertia = {km.inertia_:.1f}, Silhouette = {sil:.3f}")
    return pd.DataFrame(results)


def segment_customers(rfm_df, k=4):
    X = rfm_df[["recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm_df["segment_id"] = km.fit_predict(X_scaled)

    # Compute cluster statistics to identify segments dynamically
    profile = rfm_df.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    # The cluster with the highest mean monetary value is marked as Premium
    premium_cluster_id = profile.loc[profile["mean_monetary"].idxmax(), "segment_id"]

    # Assign intuitive business names based on characteristics
    # Note: cluster numbers are arbitrary labels without ordering
    def assign_name(row):
        cid = row["segment_id"]
        if cid == premium_cluster_id:
            return "Premium Spenders", True
        elif row["mean_frequency"] > 12:
            return "Loyal Frequent", False
        elif row["mean_recency"] > 60:
            return "At-Risk / Inactive", False
        else:
            return "Occasional Regulars", False

    name_map = {}
    for _, row in profile.iterrows():
        name, is_prem = assign_name(row)
        name_map[row["segment_id"]] = (name, is_prem)

    rfm_df["segment_name"] = rfm_df["segment_id"].apply(lambda s: name_map[s][0])
    rfm_df["is_premium"] = rfm_df["segment_id"].apply(lambda s: name_map[s][1])

    # PCA Projection (2 components)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA Explained Variance Ratios: PC1 = {evr[0]:.3f}, PC2 = {evr[1]:.3f}, Total = {evr.sum():.3f}")

    return rfm_df, profile, X_scaled, evr


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION (PART B)")
    print("=" * 60)
    orders = pd.read_csv(ORDERS_PATH)
    rfm = build_rfm(orders)

    print(f"RFM Table Built for {len(rfm)} customers.")
    print(f"  Recency: median={rfm['recency'].median():.1f}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
    print(f"  Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"  Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")

    X = rfm[["recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _ = evaluate_clusters(X_scaled)

    # Choose K=4
    rfm_segmented, profile, _, _ = segment_customers(rfm, k=4)
    print("\nSegment Profiles (K=4):")
    print(profile.to_string(index=False))

    # Output CSV
    cols_to_save = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    rfm_segmented[cols_to_save].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved segmented customers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
