# CLAT Mentor Chatbot 

An AI-powered chatbot to help law aspirants with their CLAT preparation. Built using `Streamlit` and `SentenceTransformers`, it allows users to ask natural language questions and receive instant, relevant answers from a curated knowledge base.

---

## Approach Summary

- The chatbot uses the `all-MiniLM-L6-v2` model from [SentenceTransformers](https://www.sbert.net/) to encode both user queries and knowledge base questions into dense vector embeddings.
- When a user enters a query:
  1. It is encoded into a vector embedding.
  2. Cosine similarity is calculated against pre-encoded embeddings of FAQ-style questions.
  3. The most relevant answer is retrieved and shown to the user.
- This transformer-based approach allows **semantic matching** instead of keyword-based matching, giving more accurate results even for paraphrased questions.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository or download as zip:** 

```bash
git clone https://github.com/your-username/clat-mentor-chatbot.git
cd clat-mentor-chatbot
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **run command on the terminal:**
```bash
streamlit run app.py
```
