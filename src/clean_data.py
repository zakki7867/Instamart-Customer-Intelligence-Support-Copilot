"""clean_data.py -- Data cleaning and database creation for Part A.
Audits raw exports, applies documented cleaning rules, stores clean tables
in data/clean/ and data/instamart.db.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def run_audit(orders_raw, customers_raw, products_raw, tickets_raw):
    """Computes all required audit statistics."""
    audit = {}
    audit["orders_raw_rows"] = len(orders_raw)
    audit["orders_exact_duplicates"] = int(orders_raw.duplicated().sum())
    audit["orders_pending"] = int((orders_raw["status"] == "pending").sum())
    audit["orders_distinct_id"] = int(orders_raw["order_id"].nunique())

    non_num_amt = pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum()
    audit["orders_non_numeric_amount"] = int(non_num_amt)

    deliv_num = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
    audit["orders_negative_delivery"] = int((deliv_num < 0).sum())
    audit["orders_blank_delivery"] = int(orders_raw["delivery_minutes"].isna().sum())

    audit["customers_blank_age"] = int(customers_raw["age"].isna().sum())
    cust_age_num = pd.to_numeric(customers_raw["age"], errors="coerce")
    audit["customers_age_outside_10_100"] = int(((cust_age_num < 10) | (cust_age_num > 100)).sum())
    audit["customers_distinct_city_spellings"] = int(customers_raw["city"].nunique())

    non_blank_gender = customers_raw["gender"].dropna()
    audit["customers_distinct_non_blank_gender"] = int(non_blank_gender[non_blank_gender.astype(str).str.strip() != ""].nunique())
    audit["customers_blank_gender"] = int((customers_raw["gender"].isna() | (customers_raw["gender"].astype(str).str.strip() == "")).sum())

    return audit


def clean_customers(customers_raw):
    """Applies Customer cleaning rules (a)-(d)."""
    cust = customers_raw.copy()
    
    # (a) Standardise city: strip and title case
    cust["city"] = cust["city"].astype(str).str.strip().str.title()

    # (b) Standardise gender
    def map_gender(val):
        if pd.isna(val) or str(val).strip() == "":
            return "unknown"
        g = str(val).strip().lower()
        if g in ["male", "m"]:
            return "male"
        elif g in ["female", "f"]:
            return "female"
        return "unknown"

    cust["gender"] = cust["gender"].apply(map_gender)

    # (c) Treat ages outside 10-100 as missing, then impute with city median
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
    cust["age"] = cust.groupby("city")["age"].transform(lambda g: g.fillna(g.median()))

    # (d) Convert signup_date to datetime
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    return cust


def clean_orders(orders_raw, products_raw):
    """Applies Order cleaning rules 1-5 in exact specified order."""
    orders = orders_raw.copy()

    # Rule 1: Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # Rule 2: Remove exact duplicate rows
    orders = orders.drop_duplicates()

    # Rule 3: Keep only the record with the latest order_ts per order_id
    orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()

    # Rule 4: Convert order_amount to numeric, impute missing from product catalog
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders = orders.merge(products_raw[["product_id", "price_inr", "category"]], on="product_id", how="left")
    imputed_amount = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100.0)).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(imputed_amount)

    # Rule 5: Set negative delivery_minutes to missing; for completed orders, impute with category median; leave cancelled missing
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    completed_mask = orders["status"] == "completed"
    category_median_deliv = orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(category_median_deliv)

    # Drop temporary product columns joined for imputation
    orders = orders.drop(columns=["price_inr", "category"])

    return orders


def main():
    print("Loading raw tables...")
    orders_raw = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    customers_raw = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    products_raw = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    tickets_raw = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)

    print("Running raw data audit...")
    audit = run_audit(orders_raw, customers_raw, products_raw, tickets_raw)
    for k, v in audit.items():
        print(f"  {k}: {v}")

    print("Cleaning customers...")
    cust_clean = clean_customers(customers_raw)
    print("  Cities:", dict(cust_clean["city"].value_counts()))
    print("  Genders:", dict(cust_clean["gender"].value_counts()))
    print("  Missing age:", cust_clean["age"].isna().sum())
    print("  Age range:", cust_clean["age"].min(), "-", cust_clean["age"].max())

    print("Cleaning orders...")
    orders_clean = clean_orders(orders_raw, products_raw)
    print("  Clean orders rows:", len(orders_clean))
    print("  Status distribution:", dict(orders_clean["status"].value_counts()))
    print("  Completed revenue sum: INR", f"{orders_clean[orders_clean['status'] == 'completed']['order_amount'].sum():.2f}")

    # Write clean CSVs
    print("Writing clean CSVs to data/clean/...")
    cust_clean.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
    products_raw.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
    orders_clean.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

    # Store to SQLite database
    print(f"Storing tables to SQLite database {DB_PATH}...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    
    # Store order_ts as text YYYY-MM-DD HH:MM:SS
    orders_db = orders_clean.copy()
    orders_db["order_ts"] = orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cust_db = cust_clean.copy()
    cust_db["signup_date"] = cust_db["signup_date"].dt.strftime("%Y-%m-%d")

    cust_db.to_sql("customers", conn, index=False)
    products_raw.to_sql("products", conn, index=False)
    orders_db.to_sql("orders", conn, index=False)
    conn.close()

    print("Cleaning and storage complete!")


if __name__ == "__main__":
    main()
