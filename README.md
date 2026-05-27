# 🧠 RAG Chatbot using Streamlit

A simple RAG (Retrieval-Augmented Generation) chatbot built using Python and Streamlit.

---

# 📌 Features

* Simple chatbot interface
* Semantic search using embeddings
* Uses Sentence Transformers
* Retrieves relevant answers using cosine similarity

---

# 🛠 Technologies Used

* Python
* Streamlit
* Sentence Transformers
* Scikit-learn
* NumPy

---

# 📂 Project Structure

```text id="dr2b7m"
RAG_Chatbot/
│
├── app.py
├── README.md
```

---

# ⚙️ Installation

Install required libraries:

```bash id="2k3vb9"
pip install streamlit
pip install sentence-transformers
pip install scikit-learn
pip install numpy
```

---

# ▶️ Run the Project

```bash id="n2vt3n"
streamlit run app.py
```

If not working:

```bash id="w5w5l4"
python -m streamlit run app.py
```

---

# 💻 app.py Code

```python id="y76k4f"
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page title
st.title("🧠 RAG Chatbot")

# Knowledge base
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks.",
    "NLP processes human language.",
    "Large Language Models are trained on huge datasets.",
    "RAG combines retrieval and generation."
]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
document_embeddings = model.encode(documents)

# RAG function
def rag_system(query):

    query_embedding = model.encode([query])

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    best_match = np.argmax(similarity_scores)

    return documents[best_match]

# User input
user_input = st.text_input("Ask a question")

# Button
if st.button("Get Answer"):

    if user_input != "":

        answer = rag_system(user_input)

        st.success(answer)

    else:
        st.warning("Please enter a question")
```

---

# 🧪 Sample Questions

```text id="4q1ytw"
What is machine learning?
```

```text id="nfn06j"
Explain NLP
```

```text id="kz4h6j"
What is RAG?
```

---

# 🔍 How It Works

```text id="cahgvi"
User Question
      ↓
Create Embedding
      ↓
Compare Similarity
      ↓
Retrieve Best Answer
```

---

# 🎯 Conclusion

This project demonstrates a basic RAG chatbot using semantic search and embeddings.
