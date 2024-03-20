import os
import sys

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

sys.path.append(os.getcwd())
import config
from utils import app_helpers

app = FastAPI()

model_directory = config.MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForSequenceClassification.from_pretrained(model_directory)
text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

environment = config.ENVIRONMENT
security = HTTPBearer()

description_classifier = None

try:
    description_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
except Exception as e:
    print("Failed to load news classifier:", str(e))


@app.get("/")
async def ping(credentials: HTTPAuthorizationCredentials = Depends(security)):
    app_helpers.validate_authentication(credentials.credentials)
    app_helpers.validate_description_classifier(description_classifier)

    content = {"status": "ok", "environment": environment}

    return JSONResponse(content=content)


@app.post("/classify_news/")
async def classify_news(news_content: dict, credentials: HTTPAuthorizationCredentials = Depends(security)):
    app_helpers.validate_authentication(credentials.credentials)
    app_helpers.validate_description_classifier(description_classifier)

    news = news_content.get("news")
    if not news:
        raise HTTPException(status_code=400, detail="News content is required")

    label_mapping = {'fake': 0, 'real': 1}

    result = text_classifier(news)
    result = app_helpers.process_result(result, label_mapping)

    return JSONResponse(content=result)
