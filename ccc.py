import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Knowledge Base ---
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
        "The syllabus includes English Language, Current Affairs (including General Knowledge), Legal Reasoning, Logical Reasoning, and Quantitative Techniques. :contentReference[oaicite:0]{index=0}",
        "There are 22-26 questions in the English Language section. :contentReference[oaicite:1]{index=1}",
        "In 2024, the general category cut-off for NLSIU Bangalore was around 85 marks. :contentReference[oaicite:2]{index=2}",
        "Subjects include English Language, Current Affairs, Legal Reasoning, Logical Reasoning, and Quantitative Techniques. :contentReference[oaicite:3]{index=3}",
        "The CLAT exam duration is 2 hours. :contentReference[oaicite:4]{index=4}",
        "There are 120 questions in total in CLAT. :contentReference[oaicite:5]{index=5}",
        "Yes, there is a negative marking of 0.25 marks for each wrong answer. :contentReference[oaicite:6]{index=6}",
        "Each correct answer awards 1 mark, and each incorrect answer deducts 0.25 marks. :contentReference[oaicite:7]{index=7}",
        "The CLAT 2025 exam is scheduled for December 1, 2024. :contentReference[oaicite:8]{index=8}",
        "Candidates must have completed 10+2 or equivalent with a minimum of 45% marks (40% for SC/ST). :contentReference[oaicite:9]{index=9}",
        "You can apply for CLAT 2025 through the official website of the Consortium of NLUs. :contentReference[oaicite:10]{index=10}",
        "The application fee is INR 4,000 for General/OBC/PWD/NRI candidates and INR 3,500 for SC/ST/BPL candidates. :contentReference[oaicite:11]{index=11}",
        "Yes, the number of questions has been reduced to 120, and the focus is on comprehension-based questions. :contentReference[oaicite:12]{index=12}",
        "CLAT scores are accepted by 22 National Law Universities (NLUs) across India. :contentReference[oaicite:13]{index=13}",
        "There is no upper age limit for appearing in CLAT. :contentReference[oaicite:14]{index=14}",
        "Yes, candidates who have completed Class 12 or equivalent are eligible to appear for CLAT. :contentReference[oaicite:15]{index=15}",
        "The CLAT exam is conducted in offline mode (pen and paper-based). :contentReference[oaicite:16]{index=16}",
        "Yes, CLAT follows the reservation policies as per government norms and individual NLU policies. :contentReference[oaicite:17]{index=17}",
        "Focus on understanding principles of law, practice reading comprehension, and solve previous year papers.",
        "Some recommended books include 'Legal Aptitude for the CLAT and other Law Entrance Examinations' by A.P. Bhardwaj and 'Objective Legal Aptitude' by R.S. Aggarwal.",
    ]
}

df_qa = pd.DataFrame(data)

# --- Search Logic ---
def get_response(query, df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["question"])
    
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    
    best_match_index = similarity.argmax()
    return df.iloc[best_match_index]["answer"]

# --- Streamlit UI ---
st.set_page_config(page_title="CLAT Mentor Chatbot", page_icon="⚖️")

st.title("⚖️ CLAT Mentor Chatbot")
st.write("Ask me anything about CLAT exams!")

user_query = st.text_input("Type your question below:")

if user_query:
    response = get_response(user_query, df_qa)
    st.markdown(f"**Answer:** {response}")
