from fastapi import FastAPI,File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.utils import collection_dropdown,llm_model,compute_embeddings
from src.logger import create_logger
from typing import List, Optional
from pathlib import Path
import os
from pydantic import BaseModel
import uvicorn

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = FastAPI()
# Allow requests from any origin (CORS)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,    
    allow_origins=origins,    
    allow_credentials=True,    
    allow_methods=["*"],    
    allow_headers=["*"],
    )

class FileInput(BaseModel):
    file: UploadFile


@app.get("/list_of_collections")
async def list_of_collections():
    list_of_collections = collection_dropdown()
    return list_of_collections


@app.post("/embed")
async def embed(files: List[UploadFile]):
    for uploaded_file in files:
        if not uploaded_file.filename.endswith(".txt"):
            return {"error": "Only .txt files are allowed"}
        
        collection_name = Path(uploaded_file.filename).stem
        content = await uploaded_file.read()  # Await the read() method
        documents = [content.decode()]  # Decode the content
        embeddings_result = compute_embeddings(documents, collection_name)
        # You can use embeddings_result as needed
    
    return {"message": "Files embedded successfully"}


@app.get("/process_query")
async def process_query(query_text: str, collection_name: str):
    # Call your function with the query parameter
    result = llm_model(query_text, collection_name)
    return {"result": result}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

