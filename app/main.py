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
from uuid import uuid4
# Initialize FastAPI app
app = FastAPI()
app.include_router(user_router)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-production-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# This CORS config applies to all endpoints, including /user/info
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
# --- Conversation Embedding Collection Name ---
CONV_COLLECTION = "conversation_memory"

# --- Store Conversation with Embedding ---
def store_conversation_with_embedding(user_id, user_message, llm_response, db=None):
    """
    Store conversation turn in DB and Qdrant with embedding.
    """
    import datetime
    from app.qdrant_utils import insert_documents, create_collection

    conv_id = str(uuid4())
    created_at = datetime.datetime.utcnow()
    # Store in DB
    if db:
        db.execute(
            text(
                "INSERT INTO conversations (id, user_id, user_message, llm_message, created_at) VALUES (:id, :user_id, :user_message, :llm_message, :created_at)"
            ),
            {
                "id": conv_id,
                "user_id": user_id,
                "user_message": user_message,
                "llm_message": llm_response,
                "created_at": created_at,
            },
        )
        db.commit()
    # Store embedding in Qdrant
    try:
        # Create collection if not exists
        try:
            create_collection(CONV_COLLECTION, EMBEDDING_DIMENSION)
        except Exception:
            pass
        # Use user_message + llm_response as the text for embedding
        text_for_embedding = f"USER: {user_message}\nASSISTANT: {llm_response}"
        embedding = embedding_model.encode([text_for_embedding])
        payload = {
            "user_id": user_id,
            "user_message": user_message,
            "llm_message": llm_response,
            "created_at": str(created_at),
            "conversation_id": conv_id,
        }
        insert_documents(
            [text_for_embedding],
            [conv_id],
            collection_name=CONV_COLLECTION,
            embeddings=embedding,
            payloads=[payload],
        )
    except Exception as e:
        logging.error(f"Failed to store conversation embedding: {e}")
    return conv_id

# --- List Conversations for Sidebar ---
@app.get("/user/conversations/")
async def list_user_conversations(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = 20,
):
    """
    List all conversations for the current user (for sidebar/history).
    """
    try:
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        result = db.execute(
            text(
                "SELECT id, user_message, llm_message, created_at FROM conversations WHERE user_id = :user_id ORDER BY created_at DESC LIMIT :limit"
            ),
            {"user_id": user_id, "limit": limit},
        )
        conversations = [
            dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
            for row in result.fetchall()
        ]
        return {"user_id": user_id, "conversations": conversations, "count": len(conversations)}
    except Exception as e:
        logging.error(f"List user conversations error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

# --- Get Specific Conversation ---
@app.get("/user/conversations/{conversation_id}")
async def get_conversation_by_id(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Retrieve a specific conversation by id.
    """
    try:
        result = db.execute(
            text(
                "SELECT id, user_message, llm_message, created_at FROM conversations WHERE id = :id AND user_id = :user_id"
            ),
            {"id": conversation_id, "user_id": user_id},
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
        return conversation
    except Exception as e:
        logging.error(f"Get conversation by id error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

# --- Optionally: Semantic Search in Conversations ---
@app.get("/user/search-conversations/")
async def search_user_conversations(
    query: str,
    user_id: str = Depends(get_current_user_id),
    limit: int = 10,
):
    """
    Semantic search in user's conversations using Qdrant.
    """
    try:
        from app.qdrant_utils import search_documents
        results = search_documents(
            query,
            collection_name=CONV_COLLECTION,
            top_k=limit,
            filter={"user_id": user_id},
        )
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        logging.error(f"Search user conversations error: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching conversations: {str(e)}")
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
# --- IMPROVED CHAT ENDPOINT
@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
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
                store_conversation_with_embedding(user_id, question, response, db=db)
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
            store_conversation_with_embedding(user_id, question, response, db=db)
        return {"response": response}
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return {"response": "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى أو التواصل مع فريق الدعم."}
# IMPROVED EGYPTIAN CHAT ENDPOINT
@app.post("/egyptian-chat")
async def egyptian_chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
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
                store_conversation_with_embedding(user_id, question, response, db=db)
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
            store_conversation_with_embedding(user_id, question, response, db=db)
        return {"response": response}
    except Exception as e:
        logging.error(f"Egyptian chat error: {e}")
        return {"response": "عذراً يا فندم، حصل خطأ. ممكن تجرب تاني أو تتصل بالدعم؟"}