import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests



load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

@st.cache_resource
def get_event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    return loop


def get_backend_url() -> str:
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def get_all_documents() -> list[dict]:
    try:
        resp = requests.get(f"{get_backend_url()}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.sidebar.error(f"Error fetching documents: {e}")
        return []

def delete_document(doc_id: str) -> bool:
    try:
        resp = requests.delete(f"{get_backend_url()}/documents/{doc_id}", timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False
 
st.session_state["chat_history"] = []
st.sidebar.header("📚 Indexed Documents")

all_docs = get_all_documents()


if not all_docs:
    st.sidebar.write("No PDFs indexed yet. Upload one above!")
   
else:
    for doc in all_docs:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(doc["filename"])
        if col2.button("🗑️", key=f"del_{doc['doc_id']}"):
            if delete_document(doc["doc_id"]):
                st.sidebar.success(f"Deleted {doc['filename']}")
                st.rerun()
        
selected_filenames = st.sidebar.multiselect(
    "🔍 Search only in:",
    options=[d["filename"] for d in all_docs],
    
    default=[]
   
)

selected_doc_ids = [
    d["doc_id"]
    for d in all_docs
    if d["filename"] in selected_filenames
    
]

def save_uploaded_pdf(file) -> Path:
  
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:

    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
               
                "source_id": pdf_path.name,
               
            },
        )
    )


def _inngest_api_base() -> str:
  
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
  
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0,
                        poll_interval_s: float = 0.5) -> dict:

    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )
        time.sleep(poll_interval_s)

async def send_rag_query_event(question: str, top_k: int,
                                doc_ids: list = None) -> str:
  
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k":    top_k,
                "doc_ids":  doc_ids or [],
                
            },
        )
    )
    return result[0]
   
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)


if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        get_event_loop().run_until_complete(send_rag_ingest_event(path))
        
        time.sleep(0.3)
  
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")
    st.rerun()
  

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")

    top_k = st.number_input(
        "How many chunks to retrieve",
        min_value=1, max_value=20, value=5, step=1
    )

    submitted = st.form_submit_button("Ask")
  

    if submitted and question.strip():
      

        with st.spinner("Generating answer..."):
            event_id = get_event_loop().run_until_complete(
                send_rag_query_event(
                    question.strip(),
                    int(top_k),
                    selected_doc_ids or None
                )
            )
            output = wait_for_run_output(event_id)

            answer    = output.get("answer", "")
            citations = output.get("citations", [])
        
        st.session_state["chat_history"].append({"role": "user",      "content": question.strip()})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer}) 
        st.subheader("Answer")
        st.write(answer or "(No answer)")

        
        if citations:
            st.markdown("---")
            st.caption("📎 Citations used to generate this answer:")

            for c in citations:
                st.markdown(
                    f"**[{c['number']}]** `{c['source']}` · "
                    f"chunk {c['chunk_index']} · "
                    f"relevance: {c['score']:.2%}"
                  
                )
                with st.expander(f"Preview chunk [{c['number']}]"):
                    st.write(c["preview"])
           
if st.session_state["chat_history"]:
    st.markdown("---")
    st.subheader("💬 Conversation History")

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.button("🗑️ Clear conversation history"):
        st.session_state["chat_history"] = []
        st.rerun()
        
