FROM python:3.9-slim

WORKDIR /app

# Copy app folder into container
COPY . /app

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]