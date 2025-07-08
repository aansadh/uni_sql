import asyncio
import logging
from datetime import date
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import re # Import regex for parsing table names

# Import your core dependencies (assuming these paths are correct in your project)
from mcp_server.core.dependencies import create_engine
from mcp_server.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Set logging level for this script

# Define your schema DDL (copy-pasted from your prompt)
DATABASE_SCHEMA_DDL = """
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER   -- Current quantity in stock
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
    name VARCHAR(50), -- Name of the customer
    address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,   -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);
"""

async def drop_all_tables(session: AsyncSession):
    """Drops all tables defined in the schema, if they exist."""
    logger.info("Attempting to drop existing tables...")
    
    # Extract table names from the DDL. This is a robust way to get them.
    # We use a regex to find CREATE TABLE <table_name>
    table_names = re.findall(r"CREATE TABLE (\w+)", DATABASE_SCHEMA_DDL, re.IGNORECASE)
    
    # Drop tables in a specific order if there were foreign key dependencies.
    # For your current schema without explicit FKs, reverse order of creation is safe.
    # If sales had FKs to products, customers, salespeople, it should be dropped first.
    # product_suppliers has FK to products, so it should be dropped before products.
    # Let's define an explicit safe drop order based on potential dependencies:
    safe_drop_order = ["sales", "product_suppliers", "products", "customers", "salespeople"]
    
    # Filter table_names to include only those present in safe_drop_order,
    # and ensure they are dropped in the defined safe order.
    # This ensures we don't try to drop a table that wasn't in the DDL,
    # and respects the desired order.
    ordered_tables_to_drop = [table for table in safe_drop_order if table in table_names]
    
    for table_name in ordered_tables_to_drop:
        sql = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        try:
            await session.execute(text(sql))
            logger.info(f"Executed DDL: {sql}")
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            raise # Re-raise if dropping fails significantly

    await session.commit()
    logger.info("Finished dropping tables.")


async def create_tables_if_not_exists(session: AsyncSession):
    """Executes DDL to create tables if they don't already exist."""
    logger.info("Attempting to create tables if they don't exist...")
    ddl_statements = DATABASE_SCHEMA_DDL.strip().split(';')
    for statement in ddl_statements:
        sql = statement.strip()
        if sql:
            try:
                await session.execute(text(sql))
                logger.info(f"Executed DDL: {sql.splitlines()[0]}...")
            except Exception as e:
                # This warning is less critical now that we drop tables first,
                # but good to keep for robust error handling.
                logger.warning(f"DDL execution warning (may already exist): {e}. Statement: {sql.splitlines()[0]}...")
    await session.commit() # Commit DDL changes
    logger.info("Table creation attempt finished.")


def generate_dummy_data():
    """Generates lists of dummy data for each table."""

    products = [
        {"product_id": 101, "name": "Laptop", "price": Decimal("1200.00"), "quantity": 50},
        {"product_id": 102, "name": "Mouse", "price": Decimal("25.00"), "quantity": 200},
        {"product_id": 103, "name": "Keyboard", "price": Decimal("75.00"), "quantity": 100},
        {"product_id": 104, "name": "Monitor", "price": Decimal("300.00"), "quantity": 30},
        {"product_id": 105, "name": "Webcam", "price": Decimal("50.00"), "quantity": 150},
    ]

    customers = [
        {"customer_id": 1, "name": "Alice Smith", "address": "123 Main St, Anytown"},
        {"customer_id": 2, "name": "Bob Johnson", "address": "456 Oak Ave, Bigcity"},
        {"customer_id": 3, "name": "Charlie Brown", "address": "789 Pine Rd, Smallville"},
        {"customer_id": 4, "name": "Diana Prince", "address": "101 Hero Ln, Metropolis"},
    ]

    salespeople = [
        {"salesperson_id": 1001, "name": "John Doe", "region": "North"},
        {"salesperson_id": 1002, "name": "Jane Roe", "region": "South"},
        {"salesperson_id": 1003, "name": "Peter Pan", "region": "East"},
    ]

    sales = [
        {"sale_id": 1, "product_id": 101, "customer_id": 1, "salesperson_id": 1001, "sale_date": date(2023, 1, 15), "quantity": 1},
        {"sale_id": 2, "product_id": 102, "customer_id": 2, "salesperson_id": 1002, "sale_date": date(2023, 1, 16), "quantity": 2},
        {"sale_id": 3, "product_id": 103, "customer_id": 1, "salesperson_id": 1001, "sale_date": date(2023, 1, 17), "quantity": 1},
        {"sale_id": 4, "product_id": 104, "customer_id": 3, "salesperson_id": 1003, "sale_date": date(2023, 1, 18), "quantity": 1},
        {"sale_id": 5, "product_id": 101, "customer_id": 4, "salesperson_id": 1002, "sale_date": date(2023, 1, 19), "quantity": 1},
        {"sale_id": 6, "product_id": 105, "customer_id": 2, "salesperson_id": 1001, "sale_date": date(2023, 1, 20), "quantity": 3},
    ]

    product_suppliers = [
        {"supplier_id": 201, "product_id": 101, "supply_price": Decimal("1000.00")},
        {"supplier_id": 202, "product_id": 102, "supply_price": Decimal("20.00")},
        {"supplier_id": 203, "product_id": 103, "supply_price": Decimal("60.00")},
        {"supplier_id": 204, "product_id": 104, "supply_price": Decimal("250.00")}, # Changed from 201 to 204
        {"supplier_id": 205, "product_id": 105, "supply_price": Decimal("40.00")}, # Changed from 204 to 205 to keep it unique
    ]

    return {
        "products": products,
        "customers": customers,
        "salespeople": salespeople,
        "sales": sales,
        "product_suppliers": product_suppliers,
    }


async def populate_db_with_dummy_data(session: AsyncSession):
    """
    Populates the database with dummy data for all tables.
    """
    data = generate_dummy_data()

    table_insert_map = {
        "products": "INSERT INTO products (product_id, name, price, quantity) VALUES (:product_id, :name, :price, :quantity)",
        "customers": "INSERT INTO customers (customer_id, name, address) VALUES (:customer_id, :name, :address)",
        "salespeople": "INSERT INTO salespeople (salesperson_id, name, region) VALUES (:salesperson_id, :name, :region)",
        "sales": "INSERT INTO sales (sale_id, product_id, customer_id, salesperson_id, sale_date, quantity) VALUES (:sale_id, :product_id, :customer_id, :salesperson_id, :sale_date, :quantity)",
        "product_suppliers": "INSERT INTO product_suppliers (supplier_id, product_id, supply_price) VALUES (:supplier_id, :product_id, :supply_price)",
    }

    logger.info("Starting database population...")

    for table_name, insert_sql in table_insert_map.items():
        logger.info(f"Inserting data into {table_name}...")
        for row_data in data[table_name]:
            try:
                await session.execute(text(insert_sql), row_data)
            except Exception as e:
                logger.error(f"Failed to insert row into {table_name}: {row_data}. Error: {e}")
                # We raise here because a failure to insert usually means the data is not valid
                # for the schema, or a unique constraint was violated.
                raise 
        await session.commit() # Commit after each table to ensure atomicity for each table insert block
        logger.info(f"Finished inserting data into {table_name}.")

    logger.info("Database population complete!")


async def main():
    # It's better to get the database URL from settings.py
    # if you have `settings.DATABASE_URL` configured.
    # For demonstration, keeping it hardcoded as in your example.
    db_url = 'postgresql+asyncpg://postgres:password@localhost:5432/unisqldb'
    engine = create_engine(db_url)
    AsyncSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False
    )

    try:
        async with AsyncSessionLocal() as session:
            # Step 1: Drop all tables to ensure a clean slate
            await drop_all_tables(session)

            # Step 2: Create tables from scratch
            await create_tables_if_not_exists(session)

            # Step 3: Populate the tables with dummy data
            await populate_db_with_dummy_data(session)
            
            logger.info("Script finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during database operations: {e}", exc_info=True)
    finally:
        await engine.dispose() # Always dispose the engine to close connections cleanly

if __name__ == "__main__":
    asyncio.run(main())