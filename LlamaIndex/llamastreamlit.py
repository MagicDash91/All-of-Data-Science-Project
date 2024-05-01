import streamlit as st
from bs4 import BeautifulSoup
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import PromptTemplate
import chromadb

# Set up Streamlit page title and instructions
st.title("LlamaIndex + Google Gemini Web Article Question Answering")
st.write("Please input the URL of the webpage you'd like to analyze, and ask your question about it.")

# Input for the webpage URL
url = st.text_input("Enter URL:")

# Input for the question
question = st.text_input("Ask your question:")

# If both URL and question are provided, execute the code
if url and question:
    # Load webpage content
    web_documents = SimpleWebPageReader().load_data([url])
    html_content = web_documents[0].text

    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    p_tags = soup.findAll('p')
    text_content = ""
    for each in p_tags:
        text_content += each.text + "\n"

    # Convert to Document format
    documents = [Document(text=text_content)]

    # Initialize Gemini embedding model and LLAMA model
    gemini_api_key = "AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA"
    gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")
    llm = Gemini(api_key=gemini_api_key, model_name="models/gemini-pro")

    # Create a client and a new collection
    client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = client.get_or_create_collection("quickstart")

    # Create a vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create a storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Set Global settings
    Settings.llm = llm
    Settings.embed_model = gemini_embedding_model

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Define LLAMA prompt template
    template = (
        """ You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {query_str} \nContext: {context_str} \nAnswer:"""
    )
    llm_prompt = PromptTemplate(template)

    # Query data from the persisted index
    query_engine = index.as_query_engine(text_qa_template=llm_prompt)
    response = query_engine.query(question)

    # Display just the response text
    st.write("Answer:", response.response)

