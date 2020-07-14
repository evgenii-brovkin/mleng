# Tensorflow image
FROM tensorflow/tensorflow
LABEL version="0.0.4"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . .
CMD [ "python", "pipeline.py", "-d", "data", "-m", "models", "-o", "output" ]
