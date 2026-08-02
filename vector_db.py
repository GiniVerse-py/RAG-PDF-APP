from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,      
    VectorParams,   
    PointStruct,    

    Filter,         
    FieldCondition, 
    MatchValue,     
    MatchAny,       
)


class QdrantStorage:

    def __init__(self, collection="pdf_documents", dim=3072):

        self.client = QdrantClient(path="./qdrant_storage")
        # Opens a LOCAL Qdrant instance stored in ./qdrant_storage/ folder
        # No separate server needed — Qdrant runs embedded inside this process
        # For production: QdrantClient(host="your-server", port=6333)

        self.collection = collection
        # "pdf_documents" — the name of our "table" in Qdrant

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,               # 3072 — must match EMBED_DIM in data_loader.py
                    distance=Distance.COSINE # use cosine similarity for text comparison
                )
            )



    def upsert(self, ids, vectors, payloads):
     
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
            # PointStruct = one stored item: unique ID + 3072-float vector + metadata dict
        ]
        self.client.upsert(self.collection, points=points)
      

    def search(self, query_vector, top_k: int = 5):
       
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points

        contexts = []
        sources = set()
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
        
    def search_with_filter(self, query_vector, top_k: int = 5,
                           doc_ids: list = None) -> list[dict]:

        filter_condition = None

        if doc_ids:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=doc_ids)
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,             
            query_filter=filter_condition,  
            with_payload=True,             
            limit=top_k                     
        ).points
    

        citations = []
        for r in results:
            payload = r.payload or {}
            citations.append({
                "text":        payload.get("text", ""),
                "source":      payload.get("source", ""),
                "page":        payload.get("page_number", 0),
                "chunk_index": payload.get("chunk_index", 0),
                "doc_id":      payload.get("doc_id", ""),
                "score":       round(float(r.score), 4),
            })

        return citations

    def get_chunks_by_index(self, chunk_indices: list[int], doc_ids: list = None) -> list[dict]:
        must = [FieldCondition(key="chunk_index", match=MatchAny(any=chunk_indices))]
        if doc_ids:
            must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))

        points, _ = self.client.scroll(
        collection_name=self.collection,
        scroll_filter=Filter(must=must),
        with_payload=True,
        with_vectors=False,
        limit=100
    )

        return [
        {
            "text":        (p.payload or {}).get("text", ""),
            "source":      (p.payload or {}).get("source", ""),
            "page":        (p.payload or {}).get("page_number", 0),
            "chunk_index": (p.payload or {}).get("chunk_index", 0),
            "doc_id":      (p.payload or {}).get("doc_id", ""),
            "score":       0.0,  # not a similarity match — forced inclusion
        }
        for p in points
    ]
   
    def delete_document(self, doc_id: str):
       
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                        
                    )
                ]
            )
        )
   
    def list_documents(self) -> list[dict]:
        
        docs = {}
        offset = None
        # Qdrant's scroll API is paginated — returns 100 items at a time
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,    # we need payload to read doc_id and source
                with_vectors=False,   
                limit=100,            # fetch 100 points per batch
                offset=offset         
            )
           

            for point in results:
                payload = point.payload or {}
                doc_id = payload.get("doc_id", "")
                source = payload.get("source", "")
                if doc_id and doc_id not in docs:
                   
                    docs[doc_id] = {"doc_id": doc_id, "filename": source}

            if next_offset is None:
                break
              
            offset = next_offset
            

        return list(docs.values())
        # dict.values() returns the dict values as a view
        # list() converts it to a plain Python list
   
