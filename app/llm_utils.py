# llm_utils.py

import requests
import dotenv
import os
from supabase import create_client, Client
import logging
from sentence_transformers import SentenceTransformer
import unicodedata
import re
import time
from typing import List, Dict, Optional, Tuple
from rapidfuzz import process, fuzz

# Suppress TensorFlow info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load environment variables
dotenv.load_dotenv()

# API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = os.getenv("OPENROUTER_API")  # Make sure this is set in .env
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")  # <-- Add this line
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()

# Search thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.4
FUZZY_MATCH_THRESHOLD = 65
MAX_SEARCH_RESULTS = 10

def advanced_text_normalization(text: str) -> str:
    """
    Normalize text for consistent search across Arabic and English.
    Handles diacritics, numerals, punctuation, and whitespace.
    """
    if not text:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])

    # Replace Arabic numerals with English
    arabic_to_english = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for arabic, english in arabic_to_english.items():
        text = text.replace(arabic, english)

    # Normalize spaces and punctuation
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep Arabic characters
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def create_comprehensive_product_text(product: Dict) -> str:
    """
    Create a rich text representation of a product for embedding.
    Includes name (tripled weight), description, category, availability, and quantity cues.
    """
    components = []

    name = product.get('name', '')
    if name:
        normalized_name = advanced_text_normalization(name)
        components.extend([normalized_name] * 3)

    description = product.get('description', '')
    if description:
        components.append(advanced_text_normalization(description))

    category = product.get('category', '')
    if category:
        components.append(advanced_text_normalization(category))

    if product.get('in_stock'):
        components.extend(['available', 'in stock', 'متوفر', 'موجود'])
    else:
        components.extend(['out of stock', 'unavailable', 'غير متوفر', 'منتهي'])

    quantity = product.get('quantity', 0)
    if isinstance(quantity, (int, float)) and quantity > 0:
        components.append(f"quantity {quantity}")
        if quantity > 10:
            components.append('high stock')
        elif quantity <= 5:
            components.append('low stock')

    return ' '.join(components)


def extract_product_keywords(query: str) -> List[str]:
    """
    Extract potential product keywords from a query.
    Helps identify multiple products mentioned in a single query.
    """
    normalized_query = advanced_text_normalization(query)

    separators = ['و', 'and', 'مع', 'with', 'كمان', 'also', 'برضو', 'too']
    parts = [normalized_query]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend([p.strip() for p in part.split(sep) if p.strip()])
        parts = new_parts

    # Common product patterns
    product_patterns = [
        r'\b\w+\s*cable\b',
        r'\b\w+\s*charger\b',
        r'\b\w+\s*headphones?\b',
        r'\b\w+\s*watch\b',
        r'\b\w+\s*phone\b',
        r'\b\w+\s*laptop\b',
    ]
    extracted_keywords = []
    for pattern in product_patterns:
        matches = re.findall(pattern, normalized_query, re.IGNORECASE)
        extracted_keywords.extend(matches)

    all_keywords = parts + extracted_keywords
    return list(set([kw for kw in all_keywords if len(kw.strip()) >= 3]))


async def enhanced_product_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Enhanced hybrid search combining semantic similarity and fuzzy matching.
    Handles multiple products in a single query better.
    """
    try:
        from app.qdrant_utils import search_embeddings

        products = get_all_products_from_supabase()
        if not products:
            logging.warning("No products found in database")
            return []

        query_keywords = extract_product_keywords(query)
        all_results = []

        for keyword in query_keywords[:3]:  # Limit to prevent too many searches
            try:
                keyword_norm = advanced_text_normalization(keyword)
                query_vector = embedding_model.encode(keyword_norm)
                qdrant_results = search_embeddings(
                    collection_name="products",
                    query_vector=query_vector,
                    top_k=5
                )
                if qdrant_results:
                    semantic_products = [r.get("payload", {}) for r in qdrant_results]
                    all_results.extend(semantic_products)
            except Exception as e:
                logging.error(f"Qdrant search failed for keyword '{keyword}': {e}")
                continue

        if not all_results:
            logging.info("Falling back to fuzzy matching")
            product_names = [p.get('name', '') for p in products]
            normalized_query = advanced_text_normalization(query)
            matches = process.extract(
                normalized_query, 
                product_names, 
                scorer=fuzz.partial_ratio, 
                limit=top_k
            )
            matched_products = []
            for name, score, _ in matches:
                if score >= FUZZY_MATCH_THRESHOLD:
                    product = next((p for p in products if p.get('name') == name), None)
                    if product:
                        matched_products.append(product)
            all_results.extend(matched_products)

        seen = set()
        unique_results = []
        for product in all_results:
            identifier = product.get('id') or product.get('name', '')
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_results.append(product)

        return unique_results[:top_k]

    except Exception as e:
        logging.error(f"Enhanced product search failed: {e}")
        return []


async def sync_products_to_qdrant() -> bool:
    """
    Sync all Supabase products into Qdrant with embeddings.
    """
    try:
        from app.qdrant_utils import create_collection, insert_embeddings, delete_collection

        logging.info("Starting product sync to Qdrant...")
        products = get_all_products_from_supabase()
        if not products:
            logging.warning("No products found to sync")
            return False

        logging.info(f"Found {len(products)} products to sync")

        try:
            delete_collection("products")
        except Exception:
            pass

        create_collection("products", vector_size=EMBEDDING_DIMENSION)

        vectors = []
        ids = []
        payloads = []

        for i, product in enumerate(products):
            try:
                product_text = create_comprehensive_product_text(product)
                vector = embedding_model.encode(product_text)
                if len(vector) != EMBEDDING_DIMENSION:
                    logging.error(f"Vector dimension mismatch for product {product.get('name')}")
                    continue

                vectors.append(vector.tolist())
                ids.append(str(product.get('id')))
                payloads.append(product)

            except Exception as e:
                logging.error(f"Failed to process product {product.get('name', 'Unknown')}: {e}")
                continue

        if not vectors:
            logging.error("No valid vectors created")
            return False

        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_payloads = payloads[i:i+batch_size]

            try:
                insert_embeddings(
                    collection_name="products",
                    vectors=batch_vectors,
                    ids=batch_ids,
                    payloads=batch_payloads
                )
                logging.info(f"Inserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            except Exception as e:
                logging.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                return False

        logging.info(f"Successfully synced {len(vectors)} products to Qdrant")
        return True

    except Exception as e:
        logging.error(f"Product sync failed: {e}")
        return False


def get_all_products_from_supabase() -> List[Dict]:
    """Fetch all products from Supabase"""
    try:
        response = supabase.table("products").select("*").execute()
        if response.data:
            logging.info(f"Retrieved {len(response.data)} products from Supabase")
            return response.data
        else:
            logging.warning("No products found in Supabase")
            return []
    except Exception as e:
        logging.error(f"Error fetching products from Supabase: {e}")
        return []


async def add_single_product_with_embedding(product_data: Dict) -> bool:
    """
    Add a single product to both Supabase and Qdrant with proper embedding.
    """
    try:
        from app.qdrant_utils import create_collection, insert_embeddings

        # Insert into Supabase first
        response = supabase.table("products").insert(product_data).execute()
        if not response.data:
            logging.error("Failed to insert product into Supabase")
            return False

        inserted_product = response.data[0]

        # Ensure collection exists
        try:
            create_collection("products", vector_size=EMBEDDING_DIMENSION)
        except Exception:
            pass

        # Create embedding
        product_text = create_comprehensive_product_text(inserted_product)
        vector = embedding_model.encode(product_text)

        # Insert into Qdrant
        insert_embeddings(
            collection_name="products",
            vectors=[vector.tolist()],
            ids=[str(inserted_product.get('id'))],
            payloads=[inserted_product]
        )

        logging.info(f"Successfully added product: {inserted_product.get('name')}")
        return True

    except Exception as e:
        logging.error(f"Failed to add product with embedding: {e}")
        return False


def get_all_product_names_from_supabase():
    """Retrieve all product names from Supabase"""
    try:
        response = supabase.table("products").select("name").execute()
        if response.data:
            return [row["name"] for row in response.data if row.get("name")]
        return []
    except Exception as e:
        logging.error(f"Error fetching product names: {e}")
        return []


def call_deepseek(prompt: str):
    """Call DeepSeek via OpenRouter"""
    try:
        api_url = DEEPSEEK_API_URL  # Use the correct env variable
        if not api_url:
            logging.error("DEEPSEEK_API_URL is not set in environment variables.")
            return "عذراً، إعدادات الخدمة غير مكتملة. يرجى التواصل مع الدعم."
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "deepseek/deepseek-chat:free",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"DeepSeek API call failed: {e}")
        return "عذراً، حدث خطأ في الاتصال بالخدمة. يرجى المحاولة مرة أخرى."


def store_conversation(user_id: str, user_message: str, llm_response: str):
    """Store conversation in Supabase"""
    try:
        response = supabase.table("conversations").insert({
            "user_id": user_id,
            "user_message": user_message,
            "llm_response": llm_response
        }).execute()
        return response
    except Exception as e:
        logging.error(f"Error storing conversation: {e}")
        return None


def get_conversation_history(user_id: str, limit: int = 10):
    """Retrieve conversation history from Supabase"""
    try:
        response = supabase.table("conversations") \
            .select("user_message,llm_response,created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        if response.data:
            return list(reversed(response.data))
        return []
    except Exception as e:
        logging.error(f"Error fetching conversation history: {e}")
        return []


# Utility function for testing
async def test_product_search(query: str):
    """Test function to verify search functionality."""
    print(f"Testing search for: '{query}'")
    normalized = advanced_text_normalization(query)
    print(f"Normalized: '{normalized}'")
    keywords = extract_product_keywords(query)
    print(f"Extracted keywords: {keywords}")
    results = await enhanced_product_search(query)
    print(f"Found {len(results)} products:")
    for product in results:
        print(f"  - {product.get('name', 'Unknown')} (Stock: {product.get('quantity', 0)})")


if __name__ == "__main__":
    import asyncio
    async def main():
        print("Testing product search system...")
        test_queries = [
            "USB cable and headphones",
            "كابل شحن و سماعات",
            "charging cable headphones wireless",
            "laptop cooling pad smartwatch"
        ]
        for query in test_queries:
            print("\n" + "="*50)
            await test_product_search(query)
    asyncio.run(main())