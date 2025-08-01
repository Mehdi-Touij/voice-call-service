FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bot code
COPY bot.py .
COPY pipecat.toml .

# Copy .env.example as template
COPY .env.example .

# Set environment variables at runtime
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "-m", "pipecat", "run"]
