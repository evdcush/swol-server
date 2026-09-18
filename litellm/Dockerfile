# pypdf is pure Python; only the installed package enters the runtime image.
FROM python:3.13-slim AS pdf-deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-compile --no-deps --target=/pdf-deps -r /tmp/requirements.txt

# Retain the upstream image channel used by the original Compose deployment.
FROM ghcr.io/berriai/litellm:latest
COPY --from=pdf-deps /pdf-deps /opt/pdf-python
ENV PYTHONPATH=/opt/pdf-python
COPY pdf_text_hook.py /app/pdf_text_hook.py
RUN python -c "from pypdf import PdfReader; from pdf_text_hook import handler"
