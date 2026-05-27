# ========================================
# STREAMLIT RAG SYSTEM CHATBOT
# ========================================

import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered"
)

st.title("🧠 RAG Chatbot")
st.write("Ask questions about AI concepts.")

# ---------------------------
# Knowledge Base
# ---------------------------

documents = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",

    "Deep learning is a branch of machine learning that uses neural networks with multiple layers.",

    "Natural Language Processing is a field of artificial intelligence focused on understanding and processing human language.",

    "Large Language Models are powerful AI systems trained on massive text datasets.",

    "Retrieval Augmented Generation combines information retrieval with text generation models."
]

# ---------------------------
# Load Model (Only Once)
# ---------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# Create Embeddings
# ---------------------------

@st.cache_data
def create_embeddings(docs):
    return model.encode(docs)

document_embeddings = create_embeddings(documents)

# ---------------------------
# RAG Function
# ---------------------------

def rag_system(query):

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    # Get top 2 most relevant documents
    top_indices = similarity_scores[0].argsort()[-2:][::-1]

    # Retrieve documents
    retrieved_docs = [documents[i] for i in top_indices]

    # Get best similarity score
    best_score = similarity_scores[0][top_indices[0]]

    return retrieved_docs, best_score

# ---------------------------
# Chat Interface
# ---------------------------

user_input = st.text_input("Ask a Question:")

if st.button("Get Answer"):

    if user_input.strip() != "":

        answers, score = rag_system(user_input)

        st.success("Answer:")

        for i, answer in enumerate(answers, start=1):
            st.write(f"📄 Result {i}:")
            st.write(answer)
            st.write("")

        st.info(f"Similarity Score: {score:.2f}")

    else:
        st.warning("Please enter a question.")