import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase

from langchain.chains import RetrievalQA
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ✅ Load environment variables
load_dotenv()

# ✅ Import document loading and preprocessing functions
from data_loader import load_data
from data_preprocessing import preprocess_data

# ✅ Load and chunk documents
docs = load_data("data")
chunks = preprocess_data(docs)

# ✅ Build vector retriever
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# ✅ Initialize LLM and memory
llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)

# ✅ SQL prompt template: restrict to field names + no markdown
custom_sql_prompt = PromptTemplate(
    input_variables=["input", "dialect"],
    template="""
You are an expert SQLite developer. Given a natural language question, write a syntactically correct {dialect} SQL query.
Only use the table: employees
Only use the columns: id, name, department, start_date, email
Do NOT use markdown, code fences, or explanations. Only return raw SQL.

Question: {input}
SQL Query:
"""
)

# ✅ Initialize SQL database query chain (direct result output)
sql_db = SQLDatabase.from_uri("sqlite:///flower.db")
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=sql_db,
    prompt=custom_sql_prompt,
    return_direct=True,
    verbose=True
)

# ✅ Clean up LLM output to remove markdown syntax
def clean_sql_output(text: str) -> str:
    return text.strip().replace("```sql", "").replace("```", "").strip()

# ✅ Format database query result with adaptive output logic
def format_db_result(result):
    if isinstance(result, str):
        try:
            import ast
            result = ast.literal_eval(result)
        except:
            return result

    if isinstance(result, list) and all(isinstance(row, (tuple, list)) for row in result):
        formatted = []
        for row in result:
            if len(row) == 6:
                formatted.append(f"🆔 {row[0]}\n👤 **{row[1]}**\n🏢 {row[2]}\n📅 {row[3]}\n✉️ {row[4]}")
            elif len(row) == 5:
                formatted.append(f"👤 **{row[1]}**\n🏢 {row[2]}\n📅 {row[3]}\n✉️ {row[4]}")
            elif len(row) == 1:
                formatted.append(f"👤 {row[0]}")
            else:
                formatted.append(" | ".join(str(cell) for cell in row))
        return "\n\n".join(formatted)
    return str(result)

# ✅ Streamlit page setup
st.set_page_config(page_title="🌸 Flower QA Assistant", layout="wide")
st.title("🌸 Flower QA Assistant")

# ✅ Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Receive user input
question = st.chat_input("Ask me anything about flower shop or employees...")

if question:
    st.session_state.chat_history.append(("user", question))

    try:
        # ✅ Determine which pipeline to use based on keywords
        if any(word in question.lower() for word in ["employee", "department", "start date", "email", "name"]):
            raw_result = db_chain.run(question)
            answer = format_db_result(raw_result)
        else:
            answer = retrieval_chain.run(question)
    except Exception as e:
        answer = f"⚠️ An error occurred: {e}"

    st.session_state.chat_history.append(("ai", answer))

# ✅ Render chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# ✅ Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain, LangGraph & Streamlit")
