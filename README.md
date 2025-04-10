# CLAT Mentor Chatbot

A simple AI-powered chatbot built using Streamlit and basic NLP techniques to help law aspirants with CLAT (Common Law Admission Test) related queries.

---


It uses TF-IDF and cosine similarity to identify the most relevant question from a predefined knowledge base and returns the associated answer.

---


## 🧠 How It Works

1. The user enters a question in natural language.
2. The app uses `TfidfVectorizer` to convert the query and knowledge base questions into vector format.
3. Cosine similarity is calculated between the user query and all questions in the knowledge base.
4. The closest match is selected and the associated answer is returned.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/clat-mentor-chatbot.git
cd clat-mentor-chatbot
