"""pdf_handler.py – Extract text from a PDF file (bytes)."""

import io


def extract_pdf_text(data: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages) if pages else "(PDF contains no extractable text)"
    except Exception as e:
        return f"(Could not extract PDF text: {e})"
