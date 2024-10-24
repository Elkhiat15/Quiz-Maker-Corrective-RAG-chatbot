from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from constants.constansts import (
    PERSIST_DIRECTORY,
    COLLECTION_NAME
)


class VectorDB:
    """
    A class representing a vector database with a retriever.
    """

    def __init__(self, documents, model="models/embedding-001"):
        """
        Initializes the VectorDB instance.

        Args:
        - documents (list): A list of documents to be indexed in the vector database.
        - collection_name (str, optional): The name of the collection in the vector database. Defaults to "rag-chroma".
        - model (str, optional): The model to use for embeddings. Defaults to "mistral-embed".
        """
        self.documents = documents
        self.model = model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.model)
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever()

    def _create_vectorstore(self):
        """
        Creates a Chroma vectorstore instance from the provided documents and embeddings.
        """
        return Chroma.from_documents(
            documents=self.documents,
            collection_name=COLLECTION_NAME,
            # persist_directory=PERSIST_DIRECTORY,
            embedding=self.embeddings
        )
    
    def get_retriever(self):
        return self.retriever

# Example usage:
#documents = [...]  # Your list of documents
#vector_db = VectorDB(documents)
