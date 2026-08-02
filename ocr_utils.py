from llama_index.readers.file import PDFReader  # same reader used in data_loader.py
from pdf2image import convert_from_path         # converts each PDF page → PIL image
import pytesseract                              # Python wrapper around Tesseract OCR
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def is_scanned_pdf(path: str) -> bool:
    """
    Returns True if the PDF has no extractable text (i.e., it is scanned).

    HOW IT WORKS:
      We try to extract text using PDFReader (the same reader used normally).
      If we get zero text from all pages → the PDF must be image-only → scanned.

    WHY NOT USE pdftotext (a command-line tool)?
      We want pure Python with no external command dependencies beyond Tesseract.
      PDFReader is already a dependency of this project, so we reuse it.
    """
    try:
        docs = PDFReader().load_data(file=path)

        # Collect all non-empty text from all pages
        all_text = " ".join(
            d.text for d in docs if getattr(d, "text", None)
        ).strip()

        # If the combined text from the whole PDF is empty → it's scanned
        return len(all_text) == 0

    except Exception:
        # If reading fails entirely, assume it might be scanned → return True
        # so we at least try OCR rather than silently failing
        return True


def ocr_pdf(path: str) -> str:
    from PIL import ImageFilter, ImageEnhance
    import pytesseract

    print(f"[OCR] Converting PDF pages to images: {os.path.basename(path)}")
    images = convert_from_path(path, dpi=400)  # higher DPI for better quality

    print(f"[OCR] Found {len(images)} pages. Running OCR on each...")

    page_texts = []
    for page_number, image in enumerate(images, start=1):

        # Step 1: Convert to grayscale (removes color noise)
        image = image.convert("L")

        # Step 2: Increase contrast (makes text sharper vs background)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Step 3: Sharpen the image
        image = image.filter(ImageFilter.SHARPEN)

        # Step 4: Run OCR with better config
        # --psm 3 = fully automatic page segmentation
        # --oem 3 = use LSTM neural network (most accurate)
        text = pytesseract.image_to_string(
            image,
            lang='eng',
            config='--psm 3 --oem 3'
        )

        if text.strip():
            page_texts.append(text)
            print(f"[OCR] Page {page_number}: extracted {len(text)} characters")
        else:
            print(f"[OCR] Page {page_number}: no text found")

    full_text = "\n\n".join(page_texts)
    print(f"[OCR] Done. Total characters: {len(full_text)}")
    return full_text