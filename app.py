# app.py
import os
import certifi

# Auto-fix SSL cert path so we don't need to set it manually before running
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from flask import Flask, render_template, request
from dotenv import load_dotenv

from src.prompt import prompt_template
from src.helper import download_hugging_face_embeddings

from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

app = Flask(__name__)
load_dotenv()

embeddings = download_hugging_face_embeddings()

API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=API_KEY)
index_name = "medical-chatbot-llama2"

vector_store = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings,
)

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = CTransformers(
    model="./model/llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama",
    config={
        "max_new_tokens": 256,  # was 100 — too short for nuanced safety answers
        "temperature": 0.4,  # was 0.5 — slightly more focused, less rambling
        "gpu_layers": 0,
        "context_length": 2048,  # was 1024 — needed for k=3 + longer prompt
    },
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # was 1
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False,
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_question = request.form["msg"].strip()
    q_lower = user_question.lower().strip()

    # Exact greetings only
    if q_lower in [
        "salam",
        "hello",
        "hi",
        "hey",
        "bonjour",
        "ahlan",
        "السلام عليكم",
        "hola",
    ]:
        if "salam" in q_lower:
            return "**Wa alaykoum salam!** 😊 I'm MediQ, your medical assistant. How can I help you today?"
        return (
            "**Hello!** 😊 I'm MediQ, your medical assistant. How can I help you today?"
        )

    # Exact thanks only
    if q_lower in [
        "thank you",
        "thanks",
        "shukran",
        "شكرا",
        "merci",
        "thank you so much",
        "thx",
        "ty",
    ]:
        return "My pleasure! 💙 Stay healthy and feel free to ask me anything anytime."

    # Exact goodbyes only
    if q_lower in ["bye", "goodbye", "see you", "مع السلامة", "bslama", "ciao"]:
        return "Take care and stay healthy! 💙 Goodbye!"

    # Medical question — send to LLM
    result = qa.invoke({"query": user_question})
    answer = str(result["result"]).strip()

    if not answer:
        return "I'm not sure, please consult a doctor. 🏥"

    return answer


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
