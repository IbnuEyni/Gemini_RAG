# Advanced Retrieval

import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from pypdf import PdfReader
import PyPDF2
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import umap
import matplotlib.pyplot as plt
import google.generativeai as genai
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

genai.configure(api_key=st.secrets['GEMINI_API_KEY'])

# Read PDF and extract texts
def _read_pdf(filename):
    reader = PyPDF2.PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    
    # Filter empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts    


# Split texts into chunks using both character and token-based strategies
def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


# Load Chroma collection with chunked texts and embedding function
def load_chroma(filename, collection_name, embedding_function):
    texts = _read_pdf(filename)
    chunks = _chunk_texts(texts)

    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

    ids = [str(i) for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection


# Word wrap utility function for better text display
def word_wrap(string, n_chars=72):
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)


# Project embeddings using UMAP dimensionality reduction
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


# Configure the embedding function for SentenceTransformer
embedding_function = SentenceTransformerEmbeddingFunction()


# Streamlit UI
st.title("PDF Document Query and Ranking")

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

# Load Chroma collection
if uploaded_file is not None:
    st.write("Loading PDF and creating Chroma collection...")
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Read the uploaded file
    pdf_texts = _read_pdf(uploaded_file)

    # Load the Chroma collection using the read PDF text
    chroma_collection = load_chroma(pdf_texts, collection_name=uploaded_file.name, embedding_function=embedding_function)
    st.write(f"Loaded {chroma_collection.count()} chunks.")


# # Load the Chroma collection
# chroma_collection = load_chroma(filename='data/microsoft_annual_report_2022.pdf', 
#                                 collection_name='micro', 
#                                 embedding_function=embedding_function)

# # Check number of documents in the collection
# print(f'Number of documents in collection: {chroma_collection.count()}')

# Create UMAP projection of dataset embeddings
# embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
# umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
# projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# Query example and visualization
    query = st.text_input('Enter your query:')
    if query:

        results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])
        
        if results['documents']:
            retrieved_documents = results['documents'][0]

            # Display retrieved documents
            for document in retrieved_documents:
                st.write(word_wrap(document))
            
            # Embedding projection for query
            embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
            umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
            
            projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
            query_embedding = embedding_function.embed([query])[0]
            retrieved_embeddings = results['embeddings'][0]
            
            projected_query_embedding = project_embeddings([query_embedding], umap_transform)
            projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)



        # Query Expansion using Generative AI
        def expand_query_with_answer(query):
            prompt = f"""You are a helpful expert financial research assistant. Provide an example answer to the given question, \
            that might be found in a document like an annual report of Microsoft. Keep it very simple and generic.
            Question: {query}"""
            model = genai.GenerativeModel('gemini-pro')
            answer = model.generate_content(prompt)
            return f"{query} {answer.text}"

        # Generate expanded query and perform a new retrieval
        joint_query = expand_query_with_answer(query)
        print(joint_query)
        
        results = chroma_collection.query(query_texts=joint_query, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = results['documents'][0]

        for doc in retrieved_documents:
            print(word_wrap(doc))
            print('')


        # Multi-query expansion to cover different aspects
        def augment_multiple_query(query):
            prompt = f"""Suggest up to five additional related questions to help them find the information they need for the provided question. 
                        Suggest a variety of short questions related to the original query.
                        Query: {query}"""
            model = genai.GenerativeModel('gemini-pro')
            answer = model.generate_content(prompt)    
            return answer.text.split("\n")


        augmented_queries = augment_multiple_query(query)
        queries = [query] + augmented_queries

        # Perform multi-query retrieval
        results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = results['documents']

        # Deduplicate and display results
        unique_documents = set()
        for documents in retrieved_documents:
            for document in documents:
                unique_documents.add(document)

        for i, documents in enumerate(retrieved_documents):
            print(f"Query: {queries[i]}")
            print('')
            print("Results:")
            for doc in documents:
                print(word_wrap(doc))
                print('')
            print('-'*100)


        # Re-ranking documents using CrossEncoder
        def rank_documents(cross_encoder: CrossEncoder, query: str, retrieved_documents: list):
            pairs = [[query, doc] for doc in retrieved_documents]
            scores = cross_encoder.predict(pairs)
            ranks = np.argsort(scores)[::-1]  # Sort in descending order
            ranked_docs = {rank: doc for rank, doc in zip(ranks, retrieved_documents)}
            return ranked_docs


        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        ranked_docs = rank_documents(cross_encoder, query, retrieved_documents[0])

        # Display ranked documents
        for rank, doc in ranked_docs.items():
            print(f"Rank {rank + 1}:")
            print(word_wrap(doc))
            print('')
