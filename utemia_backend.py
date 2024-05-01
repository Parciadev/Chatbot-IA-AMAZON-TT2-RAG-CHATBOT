import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

DATA_PATH = 'docs'

def utemia_index():
    loaders = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(DATA_PATH, file_name)
            loader = PyPDFLoader(file_path)
            print(f"Cargando archivo: {file_path}")
            loaders.append(loader)
    
    data_split = RecursiveCharacterTextSplitter( 
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    #chunks = data_split.split_documents(loaders)
    
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
    llm = Bedrock(
        credentials_profile_name='default',
        model_kwargs={
            "max_tokens_to_sample": 1000,
            "temperature": 0.9,
            "top_k": 250,
            "top_p": 0.9
        },
        model_id='anthropic.claude-v2'
    )
    
    return llm

def utemia_chatbot(bot_context=None):
    initial_prompt = """¡Hola! Soy Utem-ia, un asistente de IA diseñado para ayudar a los estudiantes y académicos de la universidad   
    con sus preguntas y tareas. Mi conocimiento proviene de un conjunto de documentos académicos cargados en mi base de datos. 
    Puedo responder preguntas sobre una amplia variedad de temas relacionados con la universidad.
    Por favor, siéntete libre de hacer cualquier pregunta o solicitar ayuda que necesites. Estoy aquí para ayudarte.

    {bot_context if bot_context else ''}"""
    return initial_prompt

def utemia_rag_response(index, question):
    utemia_rag_llm = utemia_llm()
    inicial_prompt = utemia_chatbot()
    utemia_rag_query = index.query(question=question, llm=utemia_rag_llm)
    return utemia_rag_query
