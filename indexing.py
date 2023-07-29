import nltk
from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import VertexAIEmbeddings


#function to load documents from GCS bucket
def load_docs(projectID, bucketName):
    loader = GCSDirectoryLoader(project_name=projectID, bucket=bucketName)
    documents = loader.load()
    return documents


#function to split documents into chunk size
def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs


documents = load_docs("My First Project", "js-test-serious-hall-371508")
docs = split_docs(documents)


#first use the embeddings AI model (HuggingFace) provided by Langchain

#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = VertexAIEmbeddings(model_name="textembedding-gecko")

import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="e7fb3d39-83eb-4961-a7ea-6afc42842f45",  # find at app.pinecone.io
    environment="us-west4-gcp-free"  # next to api key in console
)
index_name = "langchain-chatbot2"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

