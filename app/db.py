from supabase import create_client, Client
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

# --- REMOVE direct SQLAlchemy connection to Supabase Postgres ---

# Dependency to get a DB session (for local tables only)
def get_db():
    yield None  # No DB session needed for Supabase operations

# --- All product operations should use Supabase REST API ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_product_details(db, product_name):
    # Use Supabase client instead of direct SQL
    try:
        response = supabase.table("products").select("*").eq("name", product_name).execute()
        if response.data and len(response.data) > 0:
            row = response.data[0]
            return {
                "name": row.get("name"),
                "in_stock": row.get("in_stock"),
                "quantity": row.get("quantity"),
                "image_url": row.get("image_url"),
                "description": row.get("description", ""),
                "category": row.get("category", ""),
                "tags": row.get("tags", []),
            }
    except Exception as e:
        import logging
        logging.error(f"Error fetching product details: {e}")
    return None

def insert_product(product):
    """
    Insert a single product into Supabase, including description and category.
    """
    try:
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
        data = []
        for product in products:
            prod_dict = product.dict() if hasattr(product, "dict") else dict(product)
            data.append(prod_dict)
        response = supabase.table("products").insert(data).execute()
        return response
    except Exception as e:
        import logging
        logging.error(f"Error inserting products bulk: {e}")
        return None