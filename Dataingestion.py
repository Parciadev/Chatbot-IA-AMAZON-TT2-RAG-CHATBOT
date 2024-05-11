import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.embeddings import BedrockEmbeddings

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Reading Environment variables
host = os.environ.get('HOST', '54.164.65.54').strip()
database = os.environ.get('DATABASE', 'postgres').strip()
user = os.environ.get('USER', 'postgres').strip()
password = os.environ.get('PASSWORD', 'utemia').strip()
collection = os.environ.get('COLLECTION', 'utemia_collection').strip()

# Initialize OpenAI or Bedrock Embeddings
embeddings = BedrockEmbeddings(credentials_profile_name= 'default',model_id='cohere.embed-multilingual-v3')


# Build the connection string
conenction_string = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=host,
    port=5432,
    database=database,
    user=user,
    password=password,
)
logger.info(f"The Connection String is: {conenction_string}")
logger.info(f"Collection name is : {collection}")


class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def split_data(self):
        # Determine the file type based on the extension
        _, file_extension = os.path.splitext(self.file_path)

        if file_extension.lower() == '.pdf':
            logger.info(f"Loading: {self.file_path}")
            loader = PyPDFLoader(self.file_path)
        elif file_extension.lower() == '.csv':
            logger.info(f"Loading: {self.file_path}")
            loader = CSVLoader(self.file_path)
        elif file_extension.lower() == '.xlsx':
            logger.info(f"Loading: {self.file_path}")
            loader = UnstructuredExcelLoader(self.file_path, mode="elements")

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        return docs

    def push_data(self, docs):
        PGVector.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection,
            connection_string=conenction_string,
        )
        print(f"{self.file_path} is Pushed successfully into {collection}")

doc_processor = DocumentProcessor(file_path="test.pdf")
docs = doc_processor.split_data()
doc_processor.push_data(docs)