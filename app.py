# Importing necessary libraries
import streamlit as st
import os
from pypdf import PdfReader
import re
import tempfile
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Set up the GEMINI API key
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"] 

# Function to load a PDF and extract text
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split the text into chunks
def split_text(text, chunk_size=10000, chunk_overlap=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Custom embedding function for document retrieval
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

# Function to create or load ChromaDB and store vectors
def create_or_load_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    
    # Check if the collection already exists
    existing_collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in existing_collections]

    if name in collection_names:
        st.write(f"Loading existing collection: {name}")
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    else:
        st.write(f"Creating new collection: {name}")
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))
    
    return db, name

# Function to load an existing Chroma collection
def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db

# Function to retrieve the most relevant passage
def get_relevant_passage(query, db, n_results):
    results = db.query(query_texts=[query], n_results=n_results)
    passage = results['documents'][0] if results['documents'] else ""
    return passage

# Function to create a RAG prompt
def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'
    ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

def generate_response(query, relevant_passages):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    context = "\n\n".join(relevant_passages)
    prompt_template = f"""
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details. If the answer is not in the provided context, 
    just say, "The answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    answer = model.generate_content(prompt_template)
    return answer.text


# Streamlit UI and integration
def main():
    st.title("PDF Text Extraction and Q&A")

    # Uploading the PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        try:
            # Load and process the PDF
            st.write("Loading and extracting text from the PDF...")
            pdf_text = load_pdf(temp_file_path)
            
            # Split the text into chunks
            text_chunks = split_text(pdf_text) 
            
            # Create ChromaDB and store the vectors
            st.write("Creating ChromaDB and storing vectors...")
            db, collection_name = create_or_load_chroma_db(documents=text_chunks, path="chroma_db", name="pdf_collection")
            
            # User input for querying the content
            user_query = st.text_input("Enter your query:")
            
            if st.button("Get Answer"):
                st.write("Retrieving relevant passages...")
                relevant_passages = get_relevant_passage(user_query, db, n_results=3)
                
                if relevant_passages:
                    st.write("Generating answer...")
                    answer = generate_response(user_query, relevant_passages)
                    
                    # Display the generated answer
                    st.subheader("Generated Answer")
                    st.write(answer)
                else:
                    st.write("No relevant passages found. Please try a different query.")
        finally:
            # Cleanup the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

if __name__ == "__main__":
    main()