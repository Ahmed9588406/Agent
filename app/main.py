# Complete FastAPI endpoints - Replace the relevant parts in your main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from app.db import get_db, insert_product, insert_products_bulk
from app.qdrant_utils import search_documents, insert_documents, create_collection, delete_document
from app.llm_utils import (
    call_deepseek,
    store_conversation,
    get_conversation_history,
    get_all_products_from_supabase,
    enhanced_product_search,
    add_single_product_with_embedding,
    advanced_text_normalization,
    create_comprehensive_product_text,
    EMBEDDING_DIMENSION,
    embedding_model,
    sync_products_to_qdrant,
)
from fastapi import File, UploadFile
import tempfile
import os
from typing import List, Optional
from PyPDF2 import PdfReader
import docx
import logging
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from supabase_auth import user_router
from fastapi.middleware.cors import CORSMiddleware
import re
# Initialize FastAPI app
app = FastAPI()
app.include_router(user_router)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Pydantic models
class ChatRequest(BaseModel):
    question: str
class ProductCreate(BaseModel):
    id: str = None  # allow explicit id if needed
    name: str
    in_stock: bool
    quantity: int
    image_url: str
    category: str = ""
    description: str = ""
class BulkProductCreate(BaseModel):
    products: List[ProductCreate]
class DocumentInsertRequest(BaseModel):
    document: str
    id: Optional[str] = None  # Use string id
class ProductUpdate(BaseModel):
    name: Optional[str] = None
    in_stock: Optional[bool] = None
    quantity: Optional[int] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
def get_user_id_from_request(request: Request):
    """Extract user ID from request session."""
    session_token = request.cookies.get("session_token")
    if not session_token:
        return None
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        user_resp = supabase.auth.get_user(session_token)
        if hasattr(user_resp, "user") and user_resp.user:
            user = user_resp.user
        elif isinstance(user_resp, dict) and "user" in user_resp:
            user = user_resp["user"]
        if user and hasattr(user, "id"):
            return user.id
        elif user and "id" in user:
            return user["id"]
    except Exception as e:
        logging.error(f"Error getting user id from session: {e}")
    return None
def get_current_user_id(request: Request) -> str:
    return get_user_id_from_request(request)
@app.get("/")
def read_root():
    return {"message": "Enhanced Customer Support System is running!"}
# --- NEW: Arabic normalization and synonym matching ---
def normalize_arabic(text):
    # Basic normalization: remove diacritics, unify forms, lowercase, remove punctuation
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)  # Remove Arabic diacritics
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text
def get_arabic_synonyms():
    # Expand this dictionary as needed for more robust matching
    return {
        "سماعات وايرلس": ["wireless headphones", "سماعة وايرلس", "سماعات لاسلكية", "سماعة لاسلكية"],
        "سماعة وايرلس": ["wireless headphones", "سماعات وايرلس", "سماعات لاسلكية", "سماعة لاسلكية"],
        "سماعات لاسلكية": ["wireless headphones", "سماعات وايرلس", "سماعة وايرلس", "سماعة لاسلكية"],
        "سماعة لاسلكية": ["wireless headphones", "سماعات وايرلس", "سماعة وايرلس", "سماعات لاسلكية"],
        # Add more as needed
    }
def query_matches_product(query, product):
    # Normalize query and product name/description
    norm_query = normalize_arabic(query)
    norm_name = normalize_arabic(product.get("name", ""))
    norm_desc = normalize_arabic(product.get("description", ""))
    # Direct match
    if norm_query in norm_name or norm_query in norm_desc:
        return True
    # Synonym match
    synonyms = get_arabic_synonyms()
    for key, syns in synonyms.items():
        if norm_query == normalize_arabic(key) or norm_query in [normalize_arabic(s) for s in syns]:
            for s in [key] + syns:
                if normalize_arabic(s) in norm_name or normalize_arabic(s) in norm_desc:
                    return True
    return False
# IMPROVED CHAT ENDPOINT
@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    user_id: str = Depends(get_current_user_id)
):
    """
    Enhanced chat endpoint with improved multi-product search and better Arabic support.
    """
    try:
        question = request.question
        logging.info(f"Processing question: {question}")
        # Get conversation history
        conversation_history = []
        if user_id:
            conversation_history = get_conversation_history(user_id, limit=5)
        history_context = ""
        if conversation_history:
            history_context = "\n".join([
                f"المستخدم: {turn['user_message']}\nالمساعد: {turn['llm_response']}" 
                for turn in conversation_history
            ])
        # Use enhanced product search
        matched_products = await enhanced_product_search(question, top_k=8)
        # --- Ensure accurate product info by fetching from main DB and matching normalized Arabic ---
        all_products = get_all_products_from_supabase()
        products_by_id = {str(p.get("id")): p for p in all_products if p.get("id") is not None}
        # Supplement Qdrant results with normalized Arabic/Egyptian matching
        extra_matches = []
        for product in all_products:
            if query_matches_product(question, product):
                # Avoid duplicates
                if not any(str(product.get("id")) == str(mp.get("id")) for mp in matched_products):
                    extra_matches.append(product)
        matched_products += extra_matches
        # Merge Qdrant results with up-to-date DB info
        accurate_products = []
        for p in matched_products:
            pid = str(p.get("id"))
            if pid in products_by_id:
                accurate_products.append(products_by_id[pid])
            else:
                accurate_products.append(p)  # fallback if not found
        if accurate_products:
            product_context = ""
            for i, product_data in enumerate(accurate_products, 1):
                in_stock_ar = "نعم ✅" if product_data.get('in_stock', False) else "لا ❌"
                quantity = product_data.get('quantity', 0)
                product_context += (
                    f"🔹 المنتج {i}: {product_data.get('name', '')}\n"
                    f"📦 متوفر في المخزون: {in_stock_ar}\n"
                    f"🔢 الكمية المتوفرة: {quantity}\n"
                    f"🌐 رابط الصورة: {product_data.get('image_url', '')}\n"
                )
                description = product_data.get('description', '')
                if description:
                    product_context += f"📄 الوصف: {description}\n"
                category = product_data.get('category', '')
                if category:
                    product_context += f"📂 التصنيف: {category}\n"
                product_context += "\n"
            # Create comprehensive prompt
            prompt = f"""
السياق المحادثة السابقة:
{history_context}
معلومات المنتجات المطابقة:
{product_context}
سؤال العميل: {question}
تعليمات الإجابة:
1. اجب بالعربية بطريقة ودودة ومفيدة
2. اذكر كل المنتجات المطلوبة مع تفاصيلها
3. وضح حالة التوفر والكمية لكل منتج
4. اذا كان المنتج غير متوفر، اقترح بدائل إن وجدت
5. اذا كان السؤال عن عدة منتجات، اجب عن كل منتج بوضوح
6. استخدم الرموز التعبيرية لجعل الإجابة أكثر وضوحاً
"""
            response = call_deepseek(prompt)
            # Store conversation
            if user_id:
                store_conversation(user_id, question, response)
            return {"response": response}
        # Fallback to FAQ search if no products found
        logging.info("No products found, searching FAQ")
        faq_results = search_documents(question, collection_name="support_docs")
        faq_context = "\n".join([result["text"] for result in faq_results]) if faq_results else "لا توجد معلومات متاحة في قاعدة المعرفة."
        prompt = f"""
السياق المحادثة السابقة:
{history_context}
معلومات من قاعدة المعرفة:
{faq_context}
سؤال العميل: {question}
اجب بالعربية. إذا لم تجد معلومات كافية، انصح العميل بالتواصل مع فريق الدعم.
"""
        response = call_deepseek(prompt)
        if user_id:
            store_conversation(user_id, question, response)
        return {"response": response}
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return {"response": "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى أو التواصل مع فريق الدعم."}
# IMPROVED EGYPTIAN CHAT ENDPOINT
@app.post("/egyptian-chat")
async def egyptian_chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    user_id: str = Depends(get_current_user_id)
):
    """
    Egyptian Arabic chat with enhanced product search.
    """
    try:
        question = request.question
        logging.info(f"Processing Egyptian chat question: {question}")
        # Get conversation history
        conversation_history = []
        if user_id:
            conversation_history = get_conversation_history(user_id, limit=5)
        history_context = ""
        if conversation_history:
            history_context = "\n".join([
                f"العميل: {turn['user_message']}\nالمساعد: {turn['llm_response']}" 
                for turn in conversation_history
            ])
        # Enhanced product search
        matched_products = await enhanced_product_search(question, top_k=8)
        if matched_products:
            logging.info(f"Found {len(matched_products)} products for Egyptian chat")
            product_context = ""
            for i, product_data in enumerate(matched_products, 1):
                in_stock_eg = "أيوه موجود ✅" if product_data.get('in_stock', False) else "لأ مش موجود ❌"
                quantity = product_data.get('quantity', 0)
                product_context += (
                    f"🔹 المنتج {i}: {product_data.get('name', '')}\n"
                    f"📦 متوفر: {in_stock_eg}\n"
                    f"🔢 الكمية: {quantity}\n"
                    f"🌐 لينك الصورة: {product_data.get('image_url', '')}\n"
                )
                description = product_data.get('description', '')
                if description:
                    product_context += f"📄 الوصف: {description}\n"
                category = product_data.get('category', '')
                if category:
                    product_context += f"📂 النوع: {category}\n"
                product_context += "\n"
            prompt = f"""
الكلام اللي فات:
{history_context}
معلومات المنتجات:
{product_context}
سؤال العميل: {question}
تعليمات:
1. جاوب باللهجة المصرية العامية
2. كون ودود وبسيط في الكلام
3. اذكر كل المنتجات اللي العميل سأل عنها
4. وضح إيه المتوفر وإيه مش متوفر
5. لو حاجة مش موجودة، اقترح بدايل
6. استخدم كلمات زي "أيوه"، "لأ"، "كده"، "يعني"
7. خلي الرد مفيد ومفهوم
"""
            response = call_deepseek(prompt)
            if user_id:
                store_conversation(user_id, question, response)
            return {"response": response}
        # FAQ fallback
        faq_results = search_documents(question, collection_name="support_docs")
        faq_context = "\n".join([result["text"] for result in faq_results]) if faq_results else "مفيش معلومات متاحة."
        prompt = f"""
الكلام اللي فات:
{history_context}
معلومات من قاعدة البيانات:
{faq_context}
سؤال العميل: {question}
جاوب باللهجة المصرية. لو مش لاقي معلومات كفاية، قول للعميل يتصل بالدعم.
"""
        response = call_deepseek(prompt)
        if user_id:
            store_conversation(user_id, question, response)
        return {"response": response}
    except Exception as e:
        logging.error(f"Egyptian chat error: {e}")
        return {"response": "عذراً يا فندم، حصل خطأ. ممكن تجرب تاني أو تتصل بالدعم؟"}
# IMPROVED PRODUCT ENDPOINTS
@app.post("/add-product/")
async def add_product(product: ProductCreate):
    """Add a single product with improved embedding generation."""
    try:
        logging.info(f"Adding product: {product.name}")
        # Ensure product id is string
        if not product.id:
            import uuid
            product.id = str(uuid.uuid4())
        # Use the improved function
        success = await add_single_product_with_embedding(product.dict())
        if success:
            return {"message": "Product added successfully with embedding", "id": product.id}
        else:
            raise HTTPException(status_code=500, detail="Failed to add product")
    except Exception as e:
        logging.error(f"Add product error: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding product: {str(e)}")
@app.post("/add-products-bulk/")
async def add_products_bulk(bulk_request: BulkProductCreate, db: Session = Depends(get_db)):
    """Add multiple products in bulk with embeddings."""
    try:
        logging.info(f"Adding {len(bulk_request.products)} products in bulk")
        # Convert to list of dicts and ensure string ids
        import uuid
        products_data = []
        for product in bulk_request.products:
            pdict = product.dict()
            if not pdict.get("id"):
                pdict["id"] = str(uuid.uuid4())
            products_data.append(pdict)
        # Fetch existing product IDs from Supabase
        existing_products = get_all_products_from_supabase()
        existing_ids = set(str(p.get("id")) for p in existing_products if p.get("id"))
        # Filter out products with duplicate IDs
        new_products = [p for p in products_data if str(p.get("id")) not in existing_ids]
        if not new_products:
            return {"message": "No new products to add (all IDs already exist)", "added": 0}
        # Add only new products to database
        success = insert_products_bulk(new_products)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add products to database")
        # --- Ensure Qdrant collection exists ---
        try:
            create_collection("products", EMBEDDING_DIMENSION)
        except Exception as e:
            logging.warning(f"Qdrant products collection might already exist: {e}")
        # Only embed and insert products that were actually added (not already existing)
        # Use the IDs from new_products, not from all_products
        inserted_ids = set(str(p.get("id")) for p in new_products)
        # Fetch the products just inserted from Supabase to ensure up-to-date info
        all_products = get_all_products_from_supabase()
        inserted_products = [p for p in all_products if str(p.get("id")) in inserted_ids]
        # --- Batch embedding and insert into Qdrant ---
        try:
            texts = []
            qdrant_ids = []
            payloads = []
            for product in inserted_products:
                text = create_comprehensive_product_text(product)
                texts.append(text)
                # Qdrant requires UUID or int as point ID, so generate a UUID for Qdrant
                import uuid
                qdrant_id = str(uuid.uuid4())
                qdrant_ids.append(qdrant_id)
                # Store the original product ID as a payload field for lookup
                payload = dict(product)
                payload["original_id"] = str(product.get("id"))
                payloads.append(payload)
            if texts:
                embeddings = embedding_model.encode(texts)
                # insert_documents should accept payloads if supported, otherwise add to your Qdrant insert logic
                insert_documents(
                    texts, 
                    qdrant_ids, 
                    collection_name="products", 
                    embeddings=embeddings, 
                    payloads=payloads
                )
        except Exception as e:
            logging.error(f"Failed to add embeddings for bulk products: {e}")
        return {"message": f"Successfully added {len(new_products)} new products with embeddings"}
    except Exception as e:
        logging.error(f"Bulk add products error: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding products: {str(e)}")
@app.get("/products/")
async def get_all_products():
    """Get all products from the database."""
    try:
        products = get_all_products_from_supabase()
        return {"products": products, "count": len(products)}
    except Exception as e:
        logging.error(f"Get products error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching products: {str(e)}")
@app.get("/products/{product_id}")
async def get_product(product_id: str, db: Session = Depends(get_db)):
    """Get a specific product by ID."""
    try:
        result = db.execute(text("SELECT * FROM products WHERE id = :id"), {"id": product_id})
        product = result.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return dict(product._mapping) if hasattr(product, '_mapping') else dict(product)
    except Exception as e:
        logging.error(f"Get product error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching product: {str(e)}")
@app.put("/products/{product_id}")
async def update_product(product_id: str, product_update: ProductUpdate, db: Session = Depends(get_db)):
    """Update a product and its embedding."""
    try:
        # Get current product
        result = db.execute(text("SELECT * FROM products WHERE id = :id"), {"id": product_id})
        current_product = result.fetchone()
        if not current_product:
            raise HTTPException(status_code=404, detail="Product not found")
        # Prepare update data
        update_data = {}
        update_fields = []
        if product_update.name is not None:
            update_data["name"] = product_update.name
            update_fields.append("name = :name")
        if product_update.in_stock is not None:
            update_data["in_stock"] = product_update.in_stock
            update_fields.append("in_stock = :in_stock")
        if product_update.quantity is not None:
            update_data["quantity"] = product_update.quantity
            update_fields.append("quantity = :quantity")
        if product_update.image_url is not None:
            update_data["image_url"] = product_update.image_url
            update_fields.append("image_url = :image_url")
        if product_update.category is not None:
            update_data["category"] = product_update.category
            update_fields.append("category = :category")
        if product_update.description is not None:
            update_data["description"] = product_update.description
            update_fields.append("description = :description")
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")
        # Update database
        update_data["id"] = product_id
        query = f"UPDATE products SET {', '.join(update_fields)} WHERE id = :id"
        db.execute(text(query), update_data)
        db.commit()
        # Get updated product
        result = db.execute(text("SELECT * FROM products WHERE id = :id"), {"id": product_id})
        updated_product = result.fetchone()
        # Update embedding in Qdrant
        try:
            product_dict = dict(updated_product._mapping) if hasattr(updated_product, '_mapping') else dict(updated_product)
            await add_single_product_with_embedding(product_dict)
        except Exception as e:
            logging.error(f"Failed to update embedding for product {product_id}: {e}")
        return {"message": "Product updated successfully", "product": product_dict}
    except Exception as e:
        logging.error(f"Update product error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating product: {str(e)}")
@app.delete("/products/{product_id}")
async def delete_product(product_id: str, db: Session = Depends(get_db)):
    """Delete a product from both database and Qdrant."""
    try:
        # Check if product exists
        result = db.execute(text("SELECT * FROM products WHERE id = :id"), {"id": product_id})
        product = result.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        # Delete from database
        db.execute(text("DELETE FROM products WHERE id = :id"), {"id": product_id})
        db.commit()
        # Try to delete from Qdrant (if collection exists)
        try:
            from app.qdrant_utils import delete_document
            delete_document(str(product_id), collection_name="products")
        except Exception as e:
            logging.warning(f"Could not delete from Qdrant: {e}")
        return {"message": "Product deleted successfully"}
    except Exception as e:
        logging.error(f"Delete product error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting product: {str(e)}")
@app.post("/sync-products/")
async def sync_products():
    """Sync all products to Qdrant with embeddings."""
    try:
        logging.info("Starting product sync to Qdrant")
        success = await sync_products_to_qdrant()
        if success:
            return {"message": "Products synced to Qdrant successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to sync products")
    except Exception as e:
        logging.error(f"Sync products error: {e}")
        raise HTTPException(status_code=500, detail=f"Error syncing products: {str(e)}")
@app.post("/search-products/")
async def search_products(request: ChatRequest):
    """Search products using enhanced search."""
    try:
        query = request.question
        products = await enhanced_product_search(query, top_k=10)
        return {
            "query": query,
            "products": products,
            "count": len(products)
        }
    except Exception as e:
        logging.error(f"Search products error: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching products: {str(e)}")
# DOCUMENT MANAGEMENT ENDPOINTS
@app.post("/insert-document/")
async def insert_document_endpoint(request: DocumentInsertRequest):
    """Insert a document into the support knowledge base."""
    try:
        import uuid
        doc_id = request.id if request.id else str(uuid.uuid4())
        # Create collection if it doesn't exist
        try:
            create_collection("support_docs", EMBEDDING_DIMENSION)
        except Exception as e:
            logging.warning(f"Collection might already exist: {e}")
        # Insert document
        success = insert_documents([request.document], [doc_id], collection_name="support_docs")
        if success:
            return {"message": "Document inserted successfully", "id": doc_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to insert document")
    except Exception as e:
        logging.error(f"Insert document error: {e}")
        raise HTTPException(status_code=500, detail=f"Error inserting document: {str(e)}")
@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document file (PDF, DOCX, TXT)."""
    try:
        logging.info(f"Uploading file: {file.filename}")
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            # Extract text based on file type
            if file.filename.lower().endswith('.pdf'):
                with open(temp_path, 'rb') as f:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            elif file.filename.lower().endswith('.docx'):
                doc = docx.Document(temp_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            elif file.filename.lower().endswith('.txt'):
                with open(temp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, or TXT files.")
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text content found in the file")
            # Insert into knowledge base
            import uuid
            doc_id = str(uuid.uuid4())
            # Create collection if needed
            try:
                create_collection("support_docs", EMBEDDING_DIMENSION)
            except Exception as e:
                logging.warning(f"Collection might already exist: {e}")
            # Insert document
            success = insert_documents([text], [doc_id], collection_name="support_docs")
            if success:
                return {
                    "message": "Document uploaded and processed successfully",
                    "filename": file.filename,
                    "id": doc_id,
                    "text_length": len(text)
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to process document")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:
        logging.error(f"Upload document error: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
@app.get("/search-documents/")
async def search_documents_endpoint(query: str, limit: int = 5):
    """Search documents in the knowledge base."""
    try:
        results = search_documents(query, collection_name="support_docs", top_k=limit)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logging.error(f"Search documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")
# CONVERSATION HISTORY ENDPOINTS
@app.get("/conversation-history/")
async def get_conversation_history_endpoint(
    request: Request,
    limit: int = 10,
    user_id: str = Depends(get_current_user_id)
):
    """Get conversation history for the current user."""
    try:
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        history = get_conversation_history(user_id, limit=limit)
        return {
            "user_id": user_id,
            "conversations": history,
            "count": len(history)
        }
    except Exception as e:
        logging.error(f"Get conversation history error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation history: {str(e)}")
# HEALTH CHECK AND STATUS ENDPOINTS
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        from app.db import engine
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        # Test Qdrant connection
        from app.qdrant_utils import client
        collections = client.get_collections()
        return {
            "status": "healthy",
            "database": "connected",
            "qdrant": "connected",
            "collections": len(collections.collections) if hasattr(collections, 'collections') else 0
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
@app.get("/status")
async def get_status():
    """Get system status and statistics."""
    try:
        # Get product count
        products = get_all_products_from_supabase()
        product_count = len(products)
        # Get collection info
        try:
            from app.qdrant_utils import client
            collections_info = client.get_collections()
            collections = []
            for collection in collections_info.collections:
                try:
                    info = client.get_collection(collection.name)
                    collections.append({
                        "name": collection.name,
                        "vectors_count": info.vectors_count,
                        "status": info.status
                    })
                except Exception as e:
                    collections.append({
                        "name": collection.name,
                        "error": str(e)
                    })
        except Exception as e:
            collections = [{"error": f"Could not fetch collections: {e}"}]
        return {
            "system_status": "running",
            "products_count": product_count,
            "collections": collections,
            "endpoints": {
                "chat": "/chat",
                "egyptian_chat": "/egyptian-chat",
                "add_product": "/add-product/",
                "bulk_products": "/add-products-bulk/",
                "search_products": "/search-products/",
                "upload_document": "/upload-document/"
            }
        }
    except Exception as e:
        logging.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)