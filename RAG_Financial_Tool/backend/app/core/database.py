import chromadb
from chromadb.config import Settings as ChromaSettings
import redis
from typing import Optional

class DatabaseManager:
    def __init__(self):
        self.chroma_client = None
        self.redis_client = None
        self.collection = None

    def initialize_chroma(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSISTS_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_documents",
            metadata={"hnsw:space": "cosine"}
        )


    def initialize_redis(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)

    def get_collection(self):
        if not self.collection:
            self.initialize_chroma()
        return self.collection
    
    def get_redis(self):
        if not self.redis_client:
            self.initialize_redis()
        return self.redis_client
    
db_manager = DatabaseManager()