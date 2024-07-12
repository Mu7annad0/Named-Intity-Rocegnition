FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

ENV MODEL_PATH=/code/models/ner_model

EXPOSE 8000

CMD ["python", "-m", "application.fast_app"]