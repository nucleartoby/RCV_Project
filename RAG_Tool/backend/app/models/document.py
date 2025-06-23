from pydantic import BaseModel, Field
from typing import List, Optional, Dic, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):

    FILING_10K = "10-K"
    FILING_10Q = "10-Q"
    FILING_8K = "8-K"
    EARNINGS_CALL = "earnings_call"
    ANALYSIS_REPORT = "analysis_report"
    NEWS_ARTICLE = "news_article"

class DocumentMetadata(BaseModel):

    document_id: str
    document_type: DocumentType
    company_ticker: str
    company_name: str
    filing_date: Optional[datetime] = None
    period_end: Optional[datetime] = None
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentChunk(BaseModel):

    chunk_id: str
    document_id: str
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    chunk_index: str
    confidence_score: Optional[float] = None

class SearchQuery(BaseModel):
    query: str
    company_filters: Optional[List[str]] = None
    document_types: Optional[List[DocumentType]] = None
    date_range: Optional[Dict[str, datetime]] = None
    limit: int = 10

class SearchResult(BaseModel):
    chunks: List[DocumentChunk]
    total_results: int
    query_time: float