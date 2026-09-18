"""Turn inline PDF attachments into text before LiteLLM routes chat completions."""

import asyncio
import base64
import binascii
import io
import json

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from pypdf import PdfReader

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 100
MAX_TEXT_CHARS = 200_000  # All PDF attachments in one request; never silently truncate.


def is_pdf(part: dict) -> bool:
    if part.get("type") != "file" or not isinstance(part.get("file"), dict):
        return False
    attachment = part["file"]
    filename = attachment.get("filename", "")
    data = attachment.get("file_data", "")
    return (
        isinstance(filename, str) and filename.lower().endswith(".pdf")
    ) or (
        isinstance(data, str) and data.lower().startswith("data:application/pdf;")
    )


def extract_pdf(attachment: dict) -> str:
    encoded = attachment.get("file_data")
    if not isinstance(encoded, str) or not encoded:
        raise HTTPException(400, "PDF attachments require inline base64 file_data; file IDs and URLs are not supported.")
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or header.lower() != "data:application/pdf;base64":
            raise HTTPException(400, "PDF file_data must use data:application/pdf;base64,...")
    if len(encoded) > 4 * ((MAX_PDF_BYTES + 2) // 3):
        raise HTTPException(400, "PDF attachment exceeds the 25 MiB limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(400, "PDF file_data is not valid base64.") from exc
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(400, "PDF attachment exceeds the 25 MiB limit.")

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise HTTPException(400, "Encrypted PDFs are not supported; upload an unencrypted copy.")
        if len(reader.pages) > MAX_PAGES:
            raise HTTPException(400, "PDF attachment exceeds the 100-page limit.")
        pages = []
        chars = 0
        has_text = False
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text(
                extraction_mode="layout", layout_mode_space_vertically=False
            ).strip()
            has_text |= bool(text)
            if not text:
                text = "[No extractable text: this page is blank or needs OCR.]"
            text = f"[Page {number}]\n{text}"
            chars += len(text)
            if chars > MAX_TEXT_CHARS:
                raise HTTPException(400, "Extracted PDF text exceeds the 200,000-character request limit.")
            pages.append(text)
        if not has_text:
            raise HTTPException(400, "PDF has no extractable text. Scanned/image-only PDFs need OCR first.")
    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose PDF bytes, extracted content, or parser internals in errors.
        raise HTTPException(400, "Unable to extract PDF text. Upload a valid, text-based PDF.") from exc

    filename = json.dumps(attachment.get("filename") or "attachment.pdf", ensure_ascii=False)
    return f"[PDF attachment: {filename}]\n" + "\n\n".join(pages)


def convert_pdfs(data: dict) -> dict:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return data
    converted = []
    total_chars = 0
    changed = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        message_changed = False
        for part in content:
            if isinstance(part, dict) and is_pdf(part):
                text = extract_pdf(part["file"])
                total_chars += len(text)
                if total_chars > MAX_TEXT_CHARS:
                    raise HTTPException(400, "Extracted PDF text exceeds the 200,000-character request limit.")
                parts.append({"type": "text", "text": text})
                message_changed = True
            else:
                parts.append(part)
        converted.append({**message, "content": parts} if message_changed else message)
        changed |= message_changed
    return {**data, "messages": converted} if changed else data


class PdfTextHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type not in ("completion", "acompletion"):
            return data
        # Parsing runs off the gateway event loop; requests are never written to disk.
        return await asyncio.to_thread(convert_pdfs, data)


handler = PdfTextHook()
