import logging
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化句子嵌入模型
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded, dimension: {EMBEDDING_DIM}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        embedding_model = None
        EMBEDDING_DIM = 384  # fallback dimension
    app.state.embedding_model = embedding_model
    app.state.embedding_dim = EMBEDDING_DIM
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/embedding")
async def generate_embedding(text: str):
    """生成文本嵌入向量"""
    if app.state.embedding_model is None:
        return [0.0] * app.state.embedding_dim
    
    try:
        embedding = app.state.embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return [0.0] * app.state.embedding_dim