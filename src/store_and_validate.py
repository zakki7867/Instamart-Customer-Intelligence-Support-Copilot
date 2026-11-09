"""store_and_validate.py -- Validates clean data and SQLite tables for Part A.
Asserts reference checkpoints from tasks 3 and 4.
Prints exactly 'ALL CHECKPOINTS PASSED' when all hold.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")


def validate_checkpoints():
    # 1. Load clean data
    assert os.path.exists(os.path.join(CLEAN_DIR, "customers.csv")), "customers.csv missing"
    assert os.path.exists(os.path.join(CLEAN_DIR, "products.csv")), "products.csv missing"
    assert os.path.exists(os.path.join(CLEAN_DIR, "orders.csv")), "orders.csv missing"
    assert os.path.exists(DB_PATH), "instamart.db missing"

    cust = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
    products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))
    orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))

    # Connect to DB and verify tables
    conn = sqlite3.connect(DB_PATH)
    cust_db = pd.read_sql_query("SELECT * FROM customers", conn)
    prod_db = pd.read_sql_query("SELECT * FROM products", conn)
    orders_db = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()

    # Checkpoint 1: Customers city distribution
    city_counts = cust["city"].value_counts().to_dict()
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184
    }
    assert city_counts == expected_cities, f"City counts mismatch: {city_counts}"

    # Checkpoint 2: Customers gender distribution
    gender_counts = cust["gender"].value_counts().to_dict()
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert gender_counts == expected_genders, f"Gender counts mismatch: {gender_counts}"

    # Checkpoint 3: Customers missing ages and age range
    assert cust["age"].isna().sum() == 0, "Missing ages found in customers"
    assert (cust["age"] >= 18).all() and (cust["age"] <= 60).all(), "Ages outside 18-60 found"

    # Checkpoint 4: Total clean order rows (12,309) and distinct order_ids
    assert len(orders) == 12309, f"Expected 12309 order rows, got {len(orders)}"
    assert orders["order_id"].nunique() == 12309, "Duplicate order_ids in clean orders"

    # Checkpoint 5: Order status breakdown (11,110 completed, 1,199 cancelled, 0 pending)
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed", 0) == 11110, f"Completed orders mismatch: {status_counts}"
    assert status_counts.get("cancelled", 0) == 1199, f"Cancelled orders mismatch: {status_counts}"
    assert "pending" not in status_counts, "Pending orders found in clean data"

    # Checkpoint 6: Missing order_amount (0 missing)
    assert orders["order_amount"].isna().sum() == 0, "Missing order_amount in orders"

    # Checkpoint 7: Missing delivery_minutes among completed (0) and cancelled (1,199)
    comp_orders = orders[orders["status"] == "completed"]
    canc_orders = orders[orders["status"] == "cancelled"]
    assert comp_orders["delivery_minutes"].isna().sum() == 0, "Missing delivery_minutes in completed orders"
    assert canc_orders["delivery_minutes"].isna().sum() == 1199, "Cancelled orders should have missing delivery_minutes"

    # Checkpoint 8: Total completed order_amount sum equals INR 2,998,668.80
    completed_revenue = comp_orders["order_amount"].sum()
    assert np.isclose(completed_revenue, 2998668.80, atol=0.01), f"Revenue sum mismatch: {completed_revenue}"

    # Checkpoint 9: Database table rows match clean CSV rows
    assert len(cust_db) == len(cust), "DB customers row count mismatch"
    assert len(prod_db) == len(products), "DB products row count mismatch"
    assert len(orders_db) == len(orders), "DB orders row count mismatch"

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate_checkpoints()
