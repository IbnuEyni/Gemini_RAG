import streamlit as st
import tempfile
from doc_loader import PDFLoader
from utils import split_text
from chroma import Chroma
from prompts import make_rag_prompt
from config import Config
from embedding_function import GeminiEmbeddingFunction

N_RESULTS = 3

# Function to create an index from a PDF
def create_index_pdf(file_path):
    pdf_loader = PDFLoader(file_path=file_path)
    text = pdf_loader.load().content

    # Split/chunk the text
    chunked_text = split_text(text)

    # Create index
    chroma_instance = Chroma(embedding_function=GeminiEmbeddingFunction())
    collection_name = chroma_instance.add(chunked_text)
    return collection_name

# Function to query the indexed text
def query_text(collection_name, query, n_results):
    chroma_instance = Chroma(collection_name=collection_name, embedding_function=GeminiEmbeddingFunction())
    response = chroma_instance.query_text(query=query, n_results=n_results)
    return response

# Function to query the Gemini model
def query_gemini(prompt, **kwargs):
    import google.generativeai as genai
    model = genai.GenerativeModel(Config.GEMINI_MODEL)
    answer = model.generate_content(prompt)
    return answer.text 

# Function to generate a response
def generate_response(query):
    response = query_text(Config.DEFAULT_COLLECTION_NAME, query, N_RESULTS)
    rag_prompt = make_rag_prompt(query=query, relevant_passage='\n'.join(response))
    answer = query_gemini(rag_prompt)
    return answer

# Streamlit app
st.title("PDF Text Indexing and Querying")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    # Create index
    st.write("Creating index...")
    collection_name = create_index_pdf(temp_file_path)
    st.write(f"Index created with collection name: {collection_name}")

    # Query input
    query = st.text_input("Enter your query")
    if query:
        st.write("Generating response...")
        answer = generate_response(query)
        st.write("Response:")
        st.write(answer)
