# docker run -it --rm --name tsfm-pipeline 
# -v ${pwd}/data:/app/data:ro 
# -v ${pwd}/models:/app/models:ro 
# -v ${pwd}/output:/app/output 
# tsfm


# Tensorflow image
FROM tensorflow/tensorflow
LABEL version="0.0.4"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . .
CMD [ "python", "pipeline.py", "-d", "data", "-m", "models", "-o", "output" ]


# python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"