FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Download the dataset + model and build the Chroma index once, at image build time.
RUN python build_index.py

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
