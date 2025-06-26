# qdrant_utils.py

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import os
import time
import logging
from typing import List, Optional, Union, Dict
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Qdrant client using environment variables
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_CLOUD_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_CLOUD_URL and QDRANT_API_KEY must be set in .env file")

client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)

# Embedding model (ensure this matches vector size used in create_collection)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()  # Should be 384


def create_collection(collection_name: str, vector_size: int = EMBEDDING_DIMENSION):
    """
    Create or recreate a collection in Qdrant Cloud with specified vector size.
    """
    try:
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created successfully")
    except Exception as e:
        logger.error(f"Error creating collection '{collection_name}': {e}")
        raise


def insert_documents(
    texts: List[str],
    ids: List[Union[int, str]],
    collection_name: str,
    payloads: Optional[List[Dict]] = None,
    embeddings: Optional[List[List[float]]] = None
):
    """
    Insert documents into Qdrant. If embeddings are not provided, they will be generated automatically.
    
    Args:
        texts: List of text documents to embed and store.
        ids: List of unique IDs for each point.
        collection_name: Target collection name.
        payloads: Optional list of metadata dictionaries.
        embeddings: Optional list of precomputed embeddings.
    """
    if embeddings is None:
        logger.info("Generating embeddings...")
        embeddings = embedding_model.encode(texts).tolist()
    else:
        logger.info("Using precomputed embeddings")

    points = []
    for i, (text, vector) in enumerate(zip(texts, embeddings)):
        payload = {"text": text}
        if payloads and i < len(payloads):
            payload.update(payloads[i])  # merge product info with text

        points.append(
            PointStruct(id=ids[i], vector=vector, payload=payload)
        )

    try:
        logger.info(f"Inserting {len(points)} points into collection '{collection_name}'")
        client.upsert(collection_name=collection_name, points=points)
        logger.info("Successfully inserted documents into Qdrant")
        return True
    except Exception as e:
        logger.error(f"Failed to insert documents into Qdrant: {e}")
        return False


def search_documents(query: str, collection_name: str, top_k: int = 5):
    """
    Search Qdrant for similar documents to the query string.
    Returns list of dicts with id, score, and payload.
    """
    try:
        query_vector = embedding_model.encode([query])[0].tolist()
        logger.debug(f"Searching for top {top_k} results in '{collection_name}'")
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        results = [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in hits
        ]
        logger.debug(f"Found {len(results)} matching documents")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def delete_document(doc_id: Union[int, str], collection_name: str):
    """
    Delete a single document from a Qdrant collection by ID.
    """
    try:
        logger.info(f"Deleting document {doc_id} from collection '{collection_name}'")
        client.delete(collection_name=collection_name, points_selector=[doc_id])
        return True
    except Exception as e:
        logger.warning(f"Failed to delete document {doc_id}: {e}")
        return False


def insert_embeddings(
    collection_name: str,
    vectors: List[List[float]],
    ids: List[Union[int, str]],
    payloads: Optional[List[Dict]] = None
):
    """
    Insert precomputed embeddings into Qdrant.
    Useful when embeddings are already available and need to be indexed.
    """
    points = []
    for i, vector in enumerate(vectors):
        payload = {}
        if payloads and i < len(payloads):
            payload = payloads[i]

        points.append(
            PointStruct(id=ids[i], vector=vector, payload=payload)
        )

    try:
        logger.info(f"Inserting {len(points)} precomputed embeddings into '{collection_name}'")
        client.upsert(collection_name=collection_name, points=points)
        return True
    except Exception as e:
        logger.error(f"Failed to insert embeddings: {e}")
        return False


def search_embeddings(collection_name: str, query_vector: List[float], top_k: int = 5):
    """
    Perform similarity search using a precomputed query vector.
    """
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in hits
        ]
    except Exception as e:
        logger.error(f"Embedding search failed: {e}")
        return []