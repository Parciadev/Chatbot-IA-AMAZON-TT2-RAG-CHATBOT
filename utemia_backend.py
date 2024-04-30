#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

#5c. Wrap within a function
def utemia_index():
    #2. Define the data source and load data with PDFLoader
    data_load=PyPDFLoader('https://vrac.utem.cl/wp-content/uploads/2021/05/reglamento-general-de-los-estudiantes-de-pregrado.pdf')
    #3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)
    #4. Create Embeddings -- Client connection
    data_embeddings=BedrockEmbeddings(
    credentials_profile_name= 'default',
    model_id='cohere.embed-multilingual-v3')
    #5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    #5b  Create index for HR Policy Document
    db_index=data_index.from_loaders([data_load])
    return db_index
#6a. Write a function to connect to Bedrock Foundation Model - Claude Foundation Model
def utemia_llm():
    llm=Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.9,
        "top_p": 0.9})
    return llm
#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def utemia_rag_response(index,question):
    utemia_rag_llm=utemia_llm()
    utemia_rag_query=index.query(question=question,llm=utemia_rag_llm)
    return utemia_rag_query
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html