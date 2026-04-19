import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
from data_loader import load_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from groq import Groq

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src) -> RAGUpsertResult:
        if isinstance(chunks_and_src, dict):
            chunks = chunks_and_src.get("chunks", [])
            source_id = chunks_and_src.get("source_id", "")
        else:
            chunks = chunks_and_src.chunks
            source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    def _answer(question: str, contexts: list, sources: list) -> RAGQueryResult:
        gclient = Groq(api_key=os.getenv("GROQ_API_KEY"))
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
        )
        response = gclient.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
        )
        return RAGQueryResult(answer=response.choices[0].message.content, sources=sources, num_contexts=len(contexts))

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    if isinstance(found, dict):
        found = RAGSearchResult(**found)

    result = await ctx.step.run("generate-answer", lambda: _answer(question, found.contexts, found.sources), output_type=RAGQueryResult)

    if isinstance(result, dict):
        result = RAGQueryResult(**result)

    return result.model_dump()


app = FastAPI()

@app.post("/ingest")
async def ingest_pdf(pdf_path: str):
    await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": pdf_path}
        )
    )
    return {"status": "event sent", "pdf_path": pdf_path}

@app.post("/query")
async def query_pdf(question: str):
    await inngest_client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question}
        )
    )
    return {"status": "event sent", "question": question}

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])