import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

def utemia_index():
    docs_dir = 'docs'
    loaders = []

    for file_name in os.listdir(docs_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(docs_dir, file_name)
            print(f"Cargando archivo: {file_path}")
            loader = PyPDFLoader(file_path)
            loaders.append(loader)

    data_split = RecursiveCharacterTextSplitter( 
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "\n\n\n", "\n\n\n\n", "\n\n\n\n\n", "\n\n\n\n\n\n", "\n\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n"]
        )
    
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='cohere.embed-multilingual-v3')

    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)

    db_index = data_index.from_loaders(loaders)
    return db_index

def utemia_llm():
    llm=Bedrock(
    credentials_profile_name='default',
    model_id='anthropic.claude-v2',
    model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.5,
        "top_p": 0.9})
    return llm

def utemia_rag_response(index,question):
    utemia_rag_llm=utemia_llm()
    utemia_rag_query=index.query(question=question,llm=utemia_rag_llm)
    return utemia_rag_query