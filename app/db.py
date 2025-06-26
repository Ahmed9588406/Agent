from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from supabase import create_client, Client
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")  # new

# Use the host from .env for the connection string
DB_URL = f"postgresql://postgres:{SUPABASE_KEY}@{SUPABASE_DB_HOST}:5432/postgres"

engine = create_engine(DB_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to fetch product details
def get_product_details(db, product_name):
    result = db.execute(
        text("SELECT name, in_stock, quantity, image_url FROM products WHERE name = :name"),
        {"name": product_name}
    )
    row = result.fetchone()
    if row:
        return {
            "name": row[0],
            "in_stock": row[1],
            "quantity": row[2],
            "image_url": row[3]
        }
    return None

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_product(product):
    """
    Insert a single product into Supabase, including description and category.
    """
    try:
        from app.llm_utils import supabase
        # Ensure all fields are included
        data = {
            "name": product.name,
            "in_stock": product.in_stock,
            "quantity": product.quantity,
            "image_url": product.image_url,
            "description": getattr(product, "description", ""),
            "category": getattr(product, "category", ""),
            "tags": getattr(product, "tags", []),
        }
        response = supabase.table("products").insert(data).execute()
        return response
    except Exception as e:
        import logging
        logging.error(f"Error inserting product: {e}")
        return None

def insert_products_bulk(products):
    """
    Insert multiple products into Supabase, including all info provided in ProductCreate.
    """
    try:
        from app.llm_utils import supabase
        data = []
        for product in products:
            # Store all fields provided by ProductCreate, including tags if present
            prod_dict = product.dict() if hasattr(product, "dict") else dict(product)
            data.append(prod_dict)
        response = supabase.table("products").insert(data).execute()
        return response
    except Exception as e:
        import logging
        logging.error(f"Error inserting products bulk: {e}")
        return None