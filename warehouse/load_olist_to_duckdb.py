from __future__ import annotations

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "olist_data"
DB_PATH = PROJECT_ROOT / "olist.duckdb"
VIEWS_SQL_PATH = PROJECT_ROOT / "warehouse" / "views.sql"

FILE_TABLE_MAP = {
    "olist_orders_dataset.csv": "orders",
    "olist_customers_dataset.csv": "customers",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "olist_geolocation_dataset.csv": "geolocation",
}


def normalize_table_name(csv_path: Path) -> str:
    if csv_path.name in FILE_TABLE_MAP:
        return FILE_TABLE_MAP[csv_path.name]

    stem = csv_path.stem.lower()
    if stem.endswith("_dataset"):
        stem = stem[: -len("_dataset")]
    stem = stem.replace("olist_", "")
    return stem


def load_csvs(conn: duckdb.DuckDBPyConnection) -> list[str]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {DATA_DIR}")

    loaded_tables: list[str] = []
    for csv_file in csv_files:
        table_name = normalize_table_name(csv_file)
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"""
            CREATE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{csv_file.as_posix()}', header=True)
            """
        )
        loaded_tables.append(table_name)
        print(f"Loaded: {csv_file.name} -> {table_name}")

    return loaded_tables


def cast_orders_timestamps(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("DROP TABLE IF EXISTS orders_raw")
    conn.execute("ALTER TABLE orders RENAME TO orders_raw")

    # 统一 orders 时间字段命名并转换到 TIMESTAMP，便于后续口径复用
    conn.execute(
        """
        CREATE TABLE orders AS
        SELECT
          order_id,
          customer_id,
          order_status,
          try_strptime(CAST(order_purchase_timestamp AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS order_purchase_ts,
          try_strptime(CAST(order_approved_at AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS order_approved_ts,
          try_strptime(CAST(order_delivered_carrier_date AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS order_delivered_carrier_ts,
          try_strptime(CAST(order_delivered_customer_date AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS order_delivered_customer_ts,
          try_strptime(CAST(order_estimated_delivery_date AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS order_estimated_delivery_ts
        FROM orders_raw
        """
    )

    conn.execute("DROP TABLE IF EXISTS orders_raw")


def create_views(conn: duckdb.DuckDBPyConnection) -> None:
    if not VIEWS_SQL_PATH.exists():
        raise FileNotFoundError(f"views.sql not found: {VIEWS_SQL_PATH}")

    sql = VIEWS_SQL_PATH.read_text(encoding="utf-8")
    conn.execute(sql)
    print("Created views: vw_eligible_orders, vw_fact_items, vw_delivered_orders")


def main() -> None:
    conn = duckdb.connect(str(DB_PATH))
    try:
        loaded_tables = load_csvs(conn)
        if "orders" not in loaded_tables:
            raise RuntimeError("orders table was not loaded. Please check olist_orders_dataset.csv")

        cast_orders_timestamps(conn)
        create_views(conn)
        print(f"DuckDB ready: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
