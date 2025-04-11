import streamlit as st
st.set_page_config(page_title="CLAT Chatbot")

import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

data = {
    "question": [
        "What is the syllabus for CLAT 2025?",
        "How many questions are there in the English section?",
        "Give me last year’s cut-off for NLSIU Bangalore.",
        "What subjects are covered in CLAT?",
        "What is the duration of the CLAT exam?",
        "How many total questions are there in CLAT?",
        "Is there any negative marking in CLAT?",
        "What is the marking scheme for CLAT?",
        "When is the CLAT 2025 exam scheduled?",
        "What are the eligibility criteria for CLAT 2025?",
        "How can I apply for CLAT 2025?",
        "What is the application fee for CLAT 2025?",
        "Are there any changes in the CLAT 2025 exam pattern?",
        "Which NLUs accept CLAT scores?",
        "What is the age limit for appearing in CLAT?",
        "Can I appear for CLAT after Class 12?",
        "What is the mode of the CLAT exam?",
        "Is there any reservation policy in CLAT?",
        "How can I prepare for the Legal Reasoning section?",
        "What are some recommended books for CLAT preparation?",
    ],
    "answer": [
        "The syllabus includes English Language, Current Affairs, Legal Reasoning, Logical Reasoning, and Quantitative Techniques.",
        "There are 22-26 questions in the English Language section.",
        "In 2024, the general category cut-off for NLSIU Bangalore was around 85 marks.",
        "Subjects include English Language, Current Affairs, Legal Reasoning, Logical Reasoning, and Quantitative Techniques.",
        "The CLAT exam duration is 2 hours.",
        "There are 120 questions in total in CLAT.",
        "Yes, there is a negative marking of 0.25 marks for each wrong answer.",
        "Each correct answer awards 1 mark, and each incorrect answer deducts 0.25 marks.",
        "The CLAT 2025 exam is scheduled for December 1, 2024.",
        "Candidates must have completed 10+2 or equivalent with at least 45% marks (40% for SC/ST).",
        "You can apply for CLAT 2025 through the official website of the Consortium of NLUs.",
        "The application fee is INR 4,000 for General/OBC/PWD/NRI and INR 3,500 for SC/ST/BPL.",
        "Yes, the number of questions has been reduced to 120, and the focus is on comprehension-based questions.",
        "CLAT scores are accepted by 22 National Law Universities (NLUs) across India.",
        "There is no upper age limit for appearing in CLAT.",
        "Yes, candidates who have completed Class 12 or equivalent are eligible to appear for CLAT.",
        "The CLAT exam is conducted in offline mode (pen and paper-based).",
        "Yes, CLAT follows the reservation policies as per government norms and NLU policies.",
        "Focus on understanding legal principles, reading comprehension, and solving past papers.",
        "Some good books include 'Legal Aptitude for CLAT' by A.P. Bhardwaj and 'Objective Legal Aptitude' by R.S. Aggarwal.",
    ]
}

df = pd.DataFrame(data)
corpus_embeddings = model.encode(df["question"], convert_to_tensor=True)

def get_response(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_match_idx = scores.argmax().item()
    return df.iloc[best_match_idx]["answer"]

st.title("CLAT Chatbot")
st.markdown("Ask me anything about CLAT!")

user_query = st.text_input("🔎 Type your question:")

if user_query:
    answer = get_response(user_query)
    st.markdown(f"**Answer:** {answer}")
