from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA, QAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Qdrant
from src.config import config
# from llm_guard import scan_prompt
# from llm_guard.input_scanners import PromptInjection, BanTopics
# from llm_guard.vault import Vault



def QdrantClient_setup():
    url = config['url']
    api_key = config['api_key']
    client = QdrantClient(url= url,api_key=api_key,)
    return client
  

def qdrant_instance_setup(collection_name):
    client = QdrantClient_setup()
    embeddings_model_name = config['embeddings_model_name']
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    qdrant_instance = Qdrant(client, collection_name, embeddings)
    return qdrant_instance

def llm_model_setup():
    openai_api_key = config['openai_api_key']
    openai_api_base = config ['openai_api_base']
    Open_ai_llm = OpenAI(openai_api_key = openai_api_key,openai_api_base=openai_api_base, model= "meta-llama/Llama-2-13b-chat-hf", logit_bias = None )
    return Open_ai_llm

def collection_dropdown():
    client = QdrantClient_setup()
    collections = client.get_collections()
    collections = [c.name for c in collections.collections]
    return collections


def file_uploader(uploaded_file,collection_name):
    url = config['url']
    api_key = config['api_key']
    embeddings_model_name = config['embeddings_model_name']
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        qdrant = Qdrant.from_documents(texts, embeddings, url=url, prefer_grpc=True, api_key=api_key, collection_name=collection_name)
        return qdrant
        

def llm_model(query_text, collection_name): 
    qdrant_instance = qdrant_instance_setup(collection_name)
    retriever = qdrant_instance.as_retriever(k = 10)
    Open_ai_llm = llm_model_setup()
    qa = RetrievalQA.from_chain_type(llm=Open_ai_llm, chain_type="stuff", retriever=retriever, return_source_documents = True)
    query = query_text
    res = qa(query)
    return res


# def llm_model(query_text, collection_name):
#     # Step 1: Define Vault and Input Scanners
#     vault = Vault()
#     input_scanners = [PromptInjection(threshold=0.6), BanTopics(topics=["violence"], threshold=0.6)]
    
#     # Step 2: Scan the Prompt
#     sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, query_text)
    
#     # Step 3: Check if the prompt is valid
#     if any(not result for result in results_valid.values()):
#         print(f"Alert: Detecting prompt injections! Your Prompt is not valid")
#         return "Alert: Detecting prompt injections! Your Prompt is not valid"
        
#     # Step 4: If the prompt is valid, continue processing
#     qdrant_instance = qdrant_instance_setup(collection_name)
#     retriever = qdrant_instance.as_retriever(k=10)
#     Open_ai_llm = llm_model_setup()
#     qa = RetrievalQA.from_chain_type(llm=Open_ai_llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
#     res = qa(query_text)
#     return res


def compute_embeddings(documents, collection_name):
    url = config['url']
    api_key = config['api_key']
    embeddings_model_name = config['embeddings_model_name']
    embeddings_instance = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    if documents is not None:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        qdrant = Qdrant.from_documents(texts, embeddings_instance, url=url, prefer_grpc=True, api_key=api_key, collection_name=collection_name)
        return qdrant


