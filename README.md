# QA Bot

## Overview

The **QA Bot** is a web-based application designed to demonstrate the power of **Retrieval-Augmented Generation (RAG)** using LangChain. It allows users to upload PDF documents and query their contents using a custom-built question-answering system. The project showcases key components such as document loaders, text splitters, embeddings, vector databases, retrievers, and a user-friendly interface built with **Gradio**.

## Features

- Upload any PDF document and ask questions based on its content.  
- Utilizes LangChain's RAG pipeline to retrieve and generate accurate responses.  
- Integrates free, open-source models for embeddings and language generation.  
- Features a customizable Gradio interface for easy interaction.  

## How to Use

1. **Upload a PDF:** Use the "Upload PDF File" section to add your document.  
2. **Input Query:** Type your question in the "Input Query" box (e.g., "What is this document about?").  
3. **Get Response:** Click "Submit" to see the answer in the "Output" box.  

## Installation

### Local Setup

1. Clone the repository or download the files.  
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your Gemini API key in a .env file:

```bash
# Create a .env file in the project root
GOOGLE_API_KEY=your-api-key-here
```
4. Add .env to .gitignore.

5. Run the app:

```bash
python app.py
```

6. Access it at http://127.0.0.1:7860.

## Deployment on Hugging Face Spaces

1. **Create a new Space** on Hugging Face.  
2. **Upload** `app.py` and `requirements.txt`.  
3. **Add your Gemini API key as a secret**:  
   - Go to **Space Settings > Repository secrets > New secret**  
   - **Name:** `GOOGLE_API_KEY`  
   - **Value:** your API key  
4. **Commit and deploy.** Access the live app via the provided URL.

## Requirements

- Python 3.8+  
- Dependencies listed in `requirements.txt`

## Technical Highlights

- **LangChain:** Powers the RAG pipeline, including document loading and retrieval.  
- **Loaders:** Uses `PyPDFLoader` to extract text from PDFs.  
- **Splitters:** Implements `RecursiveCharacterTextSplitter` for chunking text.  
- **Embeddings:** Leverages `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`.  
- **Vector DB:** Utilizes `Chroma` for efficient storage and retrieval.  
- **Retriever:** Configures a retriever with `k=3` for context-rich responses.  
- **Gradio:** Provides an interactive UI with customizable styling.

## License

This project is open-source. Feel free to use and modify it!

## Acknowledgments

Built with the help of open-source tools from **LangChain**, **Hugging Face**, and **Google**. Made by [Safiullah Aamir].
