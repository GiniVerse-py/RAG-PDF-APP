from pydantic import BaseModel
import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]      
    sources: list[str]        
    page_numbers: list[int]   
    chunk_indices: list[int]  
    scores: list[float]      
    doc_ids: list[str]        


class RAGQueryResult(pydantic.BaseModel):
    answer: str               
    sources: list[str]        
    num_contexts: int        
    citations: list[dict]     


class QueryRequest(pydantic.BaseModel):
    question: str
    top_k: int = 5
    doc_ids: list[str] = None