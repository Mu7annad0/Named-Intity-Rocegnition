from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .utils import transform_ner_output, get_device
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from functools import lru_cache
import logging
from contextlib import asynccontextmanager
import uvicorn
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = os.getenv("MODEL_PATH", "./models/ner_model/")

max_input_size = 512  

class NERInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=max_input_size)
class NEROutput(BaseModel):
    entities: list

@lru_cache(maxsize=1)
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=get_device())
        return classifier
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to load NER model")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load NER model: {str(e)}")


app = FastAPI(title="NER API", description="API for Named Entity Recognition", version="1.0.0", lifespan=lifespan)

@app.post('/predict_ner/', response_model=NEROutput)
async def predict_ner(input_data: NERInput):
    try:
        classifier = load_model()
        output = classifier(input_data.text)
        transformed_output = transform_ner_output(output)
        return NEROutput(entities=transformed_output)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
