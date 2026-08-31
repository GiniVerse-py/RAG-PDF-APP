from google import genai              #We use this library to create embeddings.
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter            #Breaks large text into smaller pieces called chunks.Instead of splitting randomly, it tries to keep complete sentences together.
from dotenv import load_dotenv
from ocr_utils import is_scanned_pdf, ocr_pdf
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))        #Creates a Google Gemini client.This client communicates with Google's servers to generate embeddings.


EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072          #means every embedding contains 3072 numbers.[0.25, -0.17, 0.49, ..., 3072 values]



splitter = SentenceSplitter(chunk_size = 1024, chunk_overlap = 256)             #Creates an object that splits text into chunks.Each chunk can contain up to 1024 tokens.

# Why is chunk overlap important?
# Suppose a sentence is split like this.
# Chunk 1
# The penalty is
# Chunk 2
# $5000.
# Neither chunk contains the complete idea.
# By adding overlap, both chunks contain enough information.
# This improves search accuracy.




#Reads the PDF from the given path.
def load_chunk_pdf(path: str):
    docs = PDFReader().load_data(file = path)
    texts = [d.text for d in docs if getattr (d, "text", None)]       #Extracts only the text from every page.
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


# Converts text into embeddings asynchronously.
async def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    processed_texts = []
    for text in texts:
        if isinstance(text, dict):
            text = text.get("text", str(text))
        processed_texts.append(str(text))

    batch_size = 100
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i + batch_size]
        result = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=batch
        )
        for emb in result.embeddings:
            embeddings.append(emb.values)
    return embeddings


def load_chunk_pdf_with_ocr(path: str) -> list[str]:
    """
    Smart PDF loader: tries normal text extraction first.
    If the PDF is scanned (no text layer), falls back to OCR.

    This function is a DROP-IN REPLACEMENT for load_chunk_pdf().
    The chunking logic at the end is identical — only the extraction step differs.

    FLOW:
      1. Check if PDF is scanned using is_scanned_pdf()
      2a. If NOT scanned: use the normal PDFReader (fast, accurate)
      2b. If scanned: run OCR to extract text (slower, uses Tesseract)
      3. Either way, split the text into chunks using SentenceSplitter
      4. Return the list of chunk strings
    """

    if is_scanned_pdf(path):
        # ── SCANNED PDF PATH ─────────────────────────────────────────────────
        print(f"[Loader] Detected scanned PDF: {path}. Running OCR...")

        # ocr_pdf() returns one big string of all text from all pages
        full_text = ocr_pdf(path)

        if not full_text.strip():
            # OCR also found nothing — possibly a completely blank or corrupted PDF
            print(f"[Loader] WARNING: OCR returned no text for {path}")
            return []   # return empty list — no chunks to embed

        # Split the big OCR text into chunks, just like we do for normal PDFs
        chunks = splitter.split_text(full_text)
        print(f"[Loader] OCR produced {len(chunks)} chunks from {path}")

    else:
        # ── NORMAL PDF PATH (text layer exists) ──────────────────────────────
        print(f"[Loader] Normal PDF detected: {path}. Using PDFReader...")

        docs = PDFReader().load_data(file=path)
        texts = [d.text for d in docs if getattr(d, "text", None)]

        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))

        print(f"[Loader] Extracted {len(chunks)} chunks from {path}")

    return chunks
