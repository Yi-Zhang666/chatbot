# 🌸 Chatbot - GPT-based Flower Knowledge Assistant

This project implements a flower-themed chatbot powered by a fine-tuned GPT-3.5-turbo model. It includes a retrieval-augmented question-answering system built with Streamlit.

## 🔧 Key Components

- `qa_system.py` – Main pipeline for semantic search and answer generation
- `data_loader.py` / `data_preprocessing.py` – Load and process documents (PDF, DOCX, TXT)
- `streamlit_app.py` – Streamlit UI for interactive demo
- `flower.db` – Local vector database for document embeddings
- `requirements.txt` – Python dependencies

## 📚 Included Documents

- `Flower_Employee_Handbook.pdf`
- `Flower_Operations_Guide.docx`
- `The_Complete_Guide_to_Flower_Language.txt`

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
