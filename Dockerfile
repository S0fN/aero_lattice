FROM python:3.11-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the user FIRST
RUN useradd -m -u 1000 appuser
USER appuser
# Set Home and Path so pip installs are accessible
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR $HOME/app

# 3. Copy requirements and install as the user
# This avoids needing 'chown' later
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 4. Copy the rest of the app
COPY --chown=appuser:appuser . .

# 5. Fixed Entrypoint (No backslashes, all on one line or using Shell form)
EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]