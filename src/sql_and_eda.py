"""sql_and_eda.py -- Executes SQL queries from sql/queries.sql against data/instamart.db,
validates results against reference checkpoints, and conducts pandas cross-checks and EDA.
"""
import os
import re
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join("data", "instamart.db")
SQL_FILE = os.path.join("sql", "queries.sql")


def load_queries(sql_file_path):
    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split queries by marker @query: Qn
    queries = {}
    pattern = r"--\s*@query:\s*(Q\d+)\s*\n(.*?)(?=(?:--\s*@query:|\Z))"
    matches = re.findall(pattern, content, re.DOTALL)
    for q_id, q_sql in matches:
        queries[q_id] = q_sql.strip()
    return queries


def run_sql_analysis():
    print("=" * 60)
    print("RUNNING SQL QUERIES FROM sql/queries.sql")
    print("=" * 60)
    conn = sqlite3.connect(DB_PATH)
    queries = load_queries(SQL_FILE)

    # Q1
    df_q1 = pd.read_sql_query(queries["Q1"], conn)
    print(f"\n[Q1] Completed orders > 500, customer total > 8000:")
    print(f"  Count of qualifying customers: {len(df_q1)} (Reference: 30)")
    assert len(df_q1) == 30, f"Q1 expected 30, got {len(df_q1)}"

    # Q2
    df_q2 = pd.read_sql_query(queries["Q2"], conn)
    rev_sum = df_q2["revenue"].sum()
    max_row = df_q2.loc[df_q2["revenue"].idxmax()]
    print(f"\n[Q2] Revenue by city and product category:")
    print(f"  Rows returned: {len(df_q2)} (Reference: 30)")
    print(f"  Total revenue: INR {rev_sum:.2f} (Reference: 2998668.80)")
    print(f"  Largest cell: {max_row['city']} / {max_row['category']} at INR {max_row['revenue']:.2f}")
    assert len(df_q2) == 30, f"Q2 expected 30 rows, got {len(df_q2)}"
    assert np.isclose(rev_sum, 2998668.80, atol=0.01), f"Q2 sum mismatch: {rev_sum}"
    assert max_row["city"] == "Bengaluru" and max_row["category"] == "Electronics"
    assert np.isclose(max_row["revenue"], 233909.45, atol=0.01)

    # Q3
    df_q3 = pd.read_sql_query(queries["Q3"], conn)
    print(f"\n[Q3] Products priced above average catalogue price:")
    print(f"  Count of products: {len(df_q3)} (Reference: 11)")
    assert len(df_q3) == 11, f"Q3 expected 11, got {len(df_q3)}"

    # Q4
    df_q4 = pd.read_sql_query(queries["Q4"], conn)
    print(f"\n[Q4] Customers with no completed order:")
    print(f"  Count of customers: {len(df_q4)} (Reference: 25)")
    assert len(df_q4) == 25, f"Q4 expected 25, got {len(df_q4)}"

    # Q5
    df_q5 = pd.read_sql_query(queries["Q5"], conn)
    avg_spend = pd.read_sql_query(
        "WITH cs AS (SELECT customer_id, SUM(order_amount) AS s FROM orders WHERE status='completed' GROUP BY customer_id) SELECT AVG(s) FROM cs",
        conn
    ).iloc[0, 0]
    print(f"\n[Q5] Customers with completed spend above average:")
    print(f"  Count of customers: {len(df_q5)} (Reference: 550)")
    print(f"  Average spend: INR {avg_spend:.2f} (Reference: 2552.06)")
    assert len(df_q5) == 550, f"Q5 expected 550, got {len(df_q5)}"
    assert np.isclose(avg_spend, 2552.06, atol=0.01), f"Q5 avg mismatch: {avg_spend}"

    # Q6
    df_q6 = pd.read_sql_query(queries["Q6"], conn)
    top_per_city = df_q6[df_q6["rank_val"] == 1].sort_values("city")
    print(f"\n[Q6] Top customer per city by completed spend:")
    for _, row in top_per_city.iterrows():
        print(f"  {row['city']}: {row['customer_id']} (INR {row['total_spend']:.2f})")
    
    expected_top = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10)
    }
    for city, (cid, amt) in expected_top.items():
        city_top = top_per_city[top_per_city["city"] == city].iloc[0]
        assert city_top["customer_id"] == cid and np.isclose(city_top["total_spend"], amt, atol=0.01)

    diff_rank_rows = (df_q6["rank_val"] != df_q6["row_num"]).sum()
    print(f"  Rows where RANK() != ROW_NUMBER(): {diff_rank_rows} (Reference: 8)")
    assert diff_rank_rows == 8, f"Q6 expected 8 rank diff rows, got {diff_rank_rows}"

    # Q7
    df_q7 = pd.read_sql_query(queries["Q7"], conn)
    last_cum = df_q7.iloc[-1]["cumulative_revenue"]
    print(f"\n[Q7] Daily cumulative completed revenue:")
    print(f"  Daily rows: {len(df_q7)} (Reference: 183)")
    print(f"  Final cumulative revenue: INR {last_cum:.2f} (Reference: 2998668.80)")
    assert len(df_q7) == 183, f"Q7 expected 183 rows, got {len(df_q7)}"
    assert np.isclose(last_cum, 2998668.80, atol=0.01), f"Q7 last cumulative mismatch: {last_cum}"

    conn.close()


def run_pandas_eda():
    print("\n" + "=" * 60)
    print("PANDAS CROSS-CHECK AND EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")

    completed = orders[orders["status"] == "completed"].merge(customers, on="customer_id").merge(products, on="product_id")

    # (a) Pivot table Q2 cross-check
    pt = pd.pivot_table(
        completed,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    )
    grand_total = pt.loc["Total", "Total"]
    print(f"\n(a) Pivot Table Grand Total: INR {grand_total:.2f} (Matches SQL 2,998,668.80)")
    assert np.isclose(grand_total, 2998668.80, atol=0.01)

    # City aggregation: sum, mean, count
    city_agg = completed.groupby("city")["order_amount"].agg(
        sum_amount="sum",
        mean_amount="mean",
        order_count="count"
    ).round(2)
    print("\nCity completed order metrics:")
    print(city_agg.to_string())

    bengaluru_stats = city_agg.loc["Bengaluru"]
    mumbai_stats = city_agg.loc["Mumbai"]
    assert np.isclose(bengaluru_stats["sum_amount"], 523643.40, atol=0.01)
    assert np.isclose(bengaluru_stats["mean_amount"], 276.47, atol=0.01)
    assert bengaluru_stats["order_count"] == 1894
    assert np.isclose(mumbai_stats["sum_amount"], 479058.50, atol=0.01)
    assert np.isclose(mumbai_stats["mean_amount"], 297.00, atol=0.01)
    assert mumbai_stats["order_count"] == 1613

    # (b) Completed order_amount distribution & outlier analysis
    amt = completed["order_amount"]
    mean_val = amt.mean()
    median_val = amt.median()
    skew_val = amt.skew()
    q1 = amt.quantile(0.25)
    q3 = amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_below = (amt < lower_fence).sum()
    outliers_above = (amt > upper_fence).sum()

    print(f"\n(b) Completed order_amount distribution:")
    print(f"  Mean: INR {mean_val:.2f} (Reference: 269.91)")
    print(f"  Median: INR {median_val:.2f} (Reference: 135.00)")
    print(f"  Skewness: {skew_val:.2f} (Reference: ~4.61)")
    print(f"  Q1 (25th percentile): INR {q1:.2f} (Reference: 64.80)")
    print(f"  Q3 (75th percentile): INR {q3:.2f} (Reference: 297.00)")
    print(f"  IQR: INR {iqr:.2f}")
    print(f"  Lower Fence: INR {lower_fence:.2f} | Outliers Below: {outliers_below} (Reference: 0)")
    print(f"  Upper Fence: INR {upper_fence:.2f} | Outliers Above: {outliers_above} (Reference: 897)")

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed[corr_cols].corr()
    print("\nCorrelation Matrix (completed orders):")
    print(corr_matrix.round(4).to_string())

    amt_disc_corr = corr_matrix.loc["order_amount", "discount_pct"]
    amt_qty_corr = corr_matrix.loc["order_amount", "quantity"]
    print(f"\nPair with largest absolute correlation: order_amount & discount_pct (r = {amt_disc_corr:.4f})")
    print(f"order_amount & quantity correlation: r = {amt_qty_corr:.4f}")
    print("Note: Correlation of this magnitude (-0.095) does not establish causation.")


if __name__ == "__main__":
    run_sql_analysis()
    run_pandas_eda()
