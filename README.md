# Gemini_RAG

## PDF Text Extraction and Q&A System

# Overview

This project is a PDF text extraction and question-answering system built using Streamlit, ChromaDB, and Google Generative AI. It extracts text from PDFs, processes it into chunks, stores it in a vector database, and enables users to query the content for relevant information. It also includes advanced retrieval techniques for improving the quality of search results.

# Setup and Installation
Prerequisites

Make sure you have Anaconda installed on your system. If not, download and install it from the official Anaconda website.
Create and Activate Anaconda Environment

    Create a new Anaconda environment:

    bash

conda create --name pdfqa python=3.8

Activate the environment:

bash

    conda activate pdfqa

Install Dependencies

    Install the required packages listed in requirements.txt:

    bash

    pip install -r requirements.txt

    Additional setup:

    Some functionalities require additional setup. Ensure that you have the correct API keys and configuration for Google Generative AI and other external services.

# Project Structure

    data/: Directory for storing any data files or datasets used in the project.
    src/: Source code files.
    .gitignore: Git ignore file to exclude certain files from version control.
    Advanced-retrieval.ipynb: Jupyter notebook for advanced retrieval techniques.
    RAG.ipynb: Jupyter notebook for Retrieval-Augmented Generation (RAG) techniques.
    README.md: Project documentation.
    app.py: Main application file for running the Streamlit app.
    requirements.txt: List of Python dependencies.
    util/: Directory for utility functions and scripts.

# How It Works
Basic Workflow

    Upload PDF: Users upload a PDF file via the Streamlit UI.
    Extract Text: The PDF text is extracted and split into manageable chunks.
    Create/Load ChromaDB: The extracted text is stored in a ChromaDB vector database for efficient retrieval.
    Query Processing: Users input queries to retrieve relevant passages from the database.
    Generate Response: The relevant passages are processed to generate a detailed answer using Google Generative AI.

Advanced Retrieval Techniques

    Text Chunking: Improved text chunking using RecursiveCharacterTextSplitter and SentenceTransformersTokenTextSplitter for better text management and retrieval.
    UMAP Projection: Utilizes UMAP for dimensionality reduction of embeddings to visualize and analyze the vector space.
    Query Expansion: Generates additional related queries using a generative model to enhance the search results.
    Cross-Encoder Re-Ranking: Uses a cross-encoder model to re-rank retrieved documents based on their relevance to the query.

# Improvements

    Enhanced Text Chunking: Implemented advanced text chunking methods to handle large documents more effectively.
    UMAP Visualization: Added UMAP visualization to analyze the distribution of embeddings and query results in the vector space.
    Query Expansion: Integrated query expansion to generate additional related queries for more comprehensive search results.
    Cross-Encoder Re-Ranking: Added cross-encoder re-ranking to refine the relevance of retrieved documents.

# Running the Application

    Start the Streamlit app:

    bash

streamlit run app.py

Interact with the app:

    Upload a PDF file.
    Enter your query and view the generated answer.