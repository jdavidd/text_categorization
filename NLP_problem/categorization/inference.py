# main.py
from typing import Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from model_class import model


class Settings(BaseSettings):
    path_to_model: str = '../categorization/files/model_trained'
    path_to_categories: str = '../categorization/files/categories.pkl'
    path_to_vectorizer: str = '../categorization/files/vectorizer.pkl'

class NewsArticle(BaseModel):
    text: str

settings = Settings()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/news_categorization/inference/")
def call_model(data: NewsArticle) -> Dict:
    modelC = model.ModelCategorization(settings.path_to_model, settings.path_to_categories, settings.path_to_vectorizer)
    result = modelC.get_top_5_categories(data)
    
    return {'categories': result}