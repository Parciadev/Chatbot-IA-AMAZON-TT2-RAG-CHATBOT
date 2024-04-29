import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Función para cargar el índice del chatbot
def chatbot_index():
    # Define la fuente de datos y carga los datos con PyPDFLoader
    data_load = PyPDFLoader('https://vrac.utem.cl/wp-content/uploads/2021/05/reglamento-general-de-los-estudiantes-de-pregrado.pdf')
    # Divide el texto basado en caracteres, tokens, etc. - Dividir recursivamente por caracter - ["\n\n", "\n", " ", ""]
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=10)
    # Crea embeddings - Conexión de cliente
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='cohere.embed-multilingual-v3'
    )
    # Crea la base de datos de vectores, almacena embeddings e índice para búsqueda - VectorstoreIndexCreator
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )
    # Crea el índice para el documento de políticas del chatbot
    db_index = data_index.from_loaders([data_load])
    return db_index

# Función para conectar al modelo Bedrock del chatbot
def chatbot_llm():
    llm = Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
            "max_tokens_to_sample": 3000,
            "temperature": 0.1,
            "top_p": 0.9
        }
    )
    return llm

# Función que busca la pregunta del usuario, busca la mejor coincidencia en la base de datos de vectores y envía ambos al modelo de lenguaje
def chatbot_rag_response(index, question):
    rag_llm = chatbot_llm()
    chatbot_rag_query = index.query(question=question, llm=rag_llm)
    return chatbot_rag_query

# Función para iniciar el chatbot de demostración
def demo_chatbot():
    demo_llm = chatbot_llm()
    return demo_llm

# Función para crear la memoria del chatbot de demostración
def demo_memory():
    llm_data = demo_chatbot
    memory = ConversationBufferMemory(llm=llm_data, max_token_limit=256)
    return memory

# Función para iniciar la conversación con el chatbot
def demo_conversation(input_text, memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply

# Crear índice del chatbot
chatbot_index_instance = chatbot_index()

# Crear memoria para el chatbot de demostración
demo_memory_instance = demo_memory()
