from src.helper import embedd_model
from src.prompt import *
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_pinecone import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embeddings = embedd_model()

index_name = "medibot"

docsearch = Pinecone.from_existing_index(embedding=embeddings,index_name="medibot")
ret = docsearch.as_retriever(serarch_type="similarity",search_kwargs={"k": 3})

llm = OpenAI(temperature=0.4,max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

ret_chain = create_stuff_documents_chain(llm,prompt=prompt)
rag_chain = create_retrieval_chain(ret,ret_chain)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input":msg})
    return str(response["answer"])

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080)
