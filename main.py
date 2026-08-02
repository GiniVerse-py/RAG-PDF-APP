import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
from data_loader import load_chunk_pdf, load_chunk_pdf_with_ocr, embed_texts
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
        chunks = load_chunk_pdf_with_ocr(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src) -> RAGUpsertResult:
        if isinstance(chunks_and_src, dict):
            chunks = chunks_and_src.get("chunks", [])
            source_id = chunks_and_src.get("source_id", "")
        else:
            chunks = chunks_and_src.chunks
            source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i], "chunk_index": i, "doc_id": doc_id,} for i in range(len(chunks))]
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

    def _search(ctx: inngest.Context) -> RAGSearchResult:
        question = ctx.event.data["question"]
        top_k = ctx.event.data.get("top_k", 5)
        doc_ids = ctx.event.data.get("doc_ids", None)

        # None -> search all PDFs
        query_vec = embed_texts([question])[0]

        store = QdrantStorage()
        citations = store.search_with_filter(query_vec, top_k, doc_ids)
        cover_chunks = store.get_chunks_by_index([0], doc_ids)
        seen = {(c["doc_id"], c["chunk_index"]) for c in citations}
        for c in cover_chunks:
            if (c["doc_id"], c["chunk_index"]) not in seen:
                citations.append(c)
                seen.add((c["doc_id"], c["chunk_index"]))

        contexts = [c["text"] for c in citations]
        sources = [c["source"] for c in citations]
        scores = [c["score"] for c in citations]
        chunk_indices = [c["chunk_index"] for c in citations]
        doc_ids_out = [c["doc_id"] for c in citations]

        return RAGSearchResult(
            contexts=contexts,
            sources=sources,
            page_numbers=[0] * len(contexts),  # Placeholder
            chunk_indices=chunk_indices,
            scores=scores,
            doc_ids=doc_ids_out
        )

    def _answer(question: str, search_result) -> RAGQueryResult:

        if isinstance(search_result, dict):
            contexts = search_result.get("contexts", [])
            sources = search_result.get("sources", [])
            scores = search_result.get("scores", [])
            chunk_indices = search_result.get("chunk_indices", [])
        else:
            contexts = search_result.contexts
            sources = search_result.sources
            scores = search_result.scores
            chunk_indices = search_result.chunk_indices

        gclient = Groq(api_key=os.getenv("GROQ_API_KEY"))

        context_block = "\n\n".join(
            f"[{i+1}] (Source: {sources[i] if i < len(sources) else 'unknown'}, "
            f"chunk {chunk_indices[i] if i < len(chunk_indices) else i}, "
            f"relevance: {scores[i] if i < len(scores) else 'N/A'})\n{c}"
            for i, c in enumerate(contexts)
        )

        prompt = (
            "Use the following numbered context sections to answer the question.\n"
            "Read every section carefully — the answer may be a short fact buried in "
            "a longer section.\n"
            "Answer directly and concisely. State only the facts asked for — no "
            "citation numbers, no extra commentary, no caveats.\n"
            "If the context does not contain the answer, say "
            "'I don't have enough information in the provided documents'.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer (concise, no citations):"
        )

        response = gclient.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        citation_list = []

        for i, ctx_text in enumerate(contexts):
            citation_list.append(
                {
                    "number": i + 1,
                    "source": sources[i] if i < len(sources) else "unknown",
                    "chunk_index": chunk_indices[i] if i < len(chunk_indices) else i,
                    "score": scores[i] if i < len(scores) else 0.0,
                    "preview": (
                        ctx_text[:100] + "..."
                        if len(ctx_text) > 100
                        else ctx_text
                    )
                }
            )

        unique_sources = list(dict.fromkeys(sources))
        
        return RAGQueryResult(
            answer=response.choices[0].message.content,
            sources=sources,
            num_contexts=len(contexts),
            citations=citation_list
        )

    search_result = await ctx.step.run(
        "embed-and-search",
        lambda: _search(ctx),
        output_type=RAGSearchResult
    )

    answer_result = await ctx.step.run(
        "generate-answer",
        lambda: _answer(ctx.event.data["question"], search_result),
        output_type=RAGQueryResult
    )

    if isinstance(answer_result, dict):
        answer_result = RAGQueryResult(**answer_result)

    return answer_result.model_dump()


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