import os
from typing import Optional
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_experimental.sql import SQLDatabaseChain

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from data_loader import load_data, preprocess_data

# ✅ Load environment variables
load_dotenv()
print("🔑 OpenAI Key:", os.getenv("OPENAI_API_KEY")[:8], "..." if os.getenv("OPENAI_API_KEY") else "None")

# ✅ Load and preprocess documents
docs = load_data("data")
chunks = preprocess_data(docs)

# ✅ Create vector store retriever
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# ✅ Initialize memory and LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)

# ✅ Setup SQL DB Chain
sql_db = SQLDatabase.from_uri("sqlite:///flower.db")
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=sql_db, verbose=True)

# ✅ Define LangGraph State
class QAState(BaseModel):
    question: str
    answer: Optional[str] = None

# ✅ Routing logic
def route_question(state: QAState) -> dict:
    q = state.question.lower()
    if any(kw in q for kw in ["employee", "department", "start date", "email", "name"]):
        return {"answer": db_chain.run(q)}
    return {"answer": retrieval_chain.run(q)}

def respond(state: QAState) -> dict:
    return {"answer": state.answer}

# ✅ Build graph
graph_builder = StateGraph(QAState)
graph_builder.add_node("route", RunnableLambda(route_question))
graph_builder.add_node("respond", RunnableLambda(respond))
graph_builder.set_entry_point("route")
graph_builder.add_edge("route", "respond")
graph_builder.add_edge("respond", END)
graph = graph_builder.compile()

# ✅ Optional: ASCII + Mermaid diagrams
print("\n📊 ASCII Diagram:")
print(graph.get_graph().draw_ascii())

print("\n📌 Mermaid Diagram (copy to https://mermaid.live):")
print(graph.get_graph().draw_mermaid())

# ✅ REPL Loop
print("\n💬 Ask me anything about flower shop or employees (type 'exit' to quit)")
while True:
    q = input("You: ")
    if q.strip().lower() in ["exit", "quit"]:
        break
    try:
        res = graph.invoke({"question": q})
        print("AI:", res["answer"])
    except Exception as e:
        print("⚠️ Error:", e)

    # Optional: show memory history
    print("\n🧠 Chat History:")
    for m in memory.chat_memory.messages:
        print(f"- {m.type.upper()}: {m.content}")
