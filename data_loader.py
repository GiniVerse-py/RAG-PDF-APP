from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size = 1024, chunk_overlap = 256)

def load_chunk_pdf(path: str):
    docs = PDFReader().load_data(file = path)
    texts = [d.text for d in docs if getattr (d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        if isinstance(text, dict):
            text = text.get("text", str(text))

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=str(text)
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings

