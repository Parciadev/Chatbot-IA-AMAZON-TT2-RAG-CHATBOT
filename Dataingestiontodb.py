import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Reading Environment variables
host = os.environ.get('HOST', '100.25.154.92').strip()
database = os.environ.get('DATABASE', 'postgres').strip()
user = os.environ.get('USER', 'postgres').strip()
password = os.environ.get('PASSWORD', 'utemia').strip()
collection = os.environ.get('COLLECTION', 'utemia_collection').strip()
openai_key = os.environ.get('OPENAI_API_KEY', "sk-proj-W5wleQFdUbMRHmbknp0oT3BlbkFJWd1nvKhL5RQNEIUTSxeg").strip()
os.environ['OPENAI_API_KEY'] = openai_key

# Initialize OpenAI
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

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

docs_folder_path = os.path.join(os.path.dirname(__file__), "docs")

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def split_data(self):
        # Determine the file type based on the extension
        _, file_extension = os.path.splitext(self.file_path)

        supported_extensions = ['.pdf', '.csv', '.xlsx']
        if file_extension.lower() not in supported_extensions:
            logger.info(f"Skipping file: {self.file_path} (unsupported file type)")
            return []

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
        print(f"{self.file_path} a sido correctamente añadido a la coleccion: {collection}")

# Add documents to the collection from the local folder "docs"
for root, dirs, files in os.walk(docs_folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        doc_processor = DocumentProcessor(file_path=file_path)
        docs = doc_processor.split_data()
        doc_processor.push_data(docs)