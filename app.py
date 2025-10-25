import os
from dotenv import load_dotenv
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings
load_dotenv()
# Suppress warnings
warnings.filterwarnings('ignore')

# Set up Gemini API key (use environment variable or replace with your key)
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyBQX3_Z3HEe_Yy69ejpWQHXOSpx1s63hGY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# LLM using Gemini
def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Updated to a more likely supported model
            max_output_tokens=512,
            temperature=0.5
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise

# Document loader
def document_loader(file):
    try:
        loader = PyPDFLoader(file)
        loaded_document = loader.load()
        return loaded_document
    except Exception as e:
        print(f"Error loading document: {e}")
        raise

# Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Embedding model using Hugging Face
def get_embeddings():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU for local and Hugging Face compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model

# Vector database
# def vector_database(chunks):
#     embedding_model = get_embeddings()
#     vectordb = Chroma.from_documents(chunks, embedding_model)
#     return vectordb
def vector_database(chunks):
    embedding_model = get_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=None  # Do NOT store on disk
    )
    return vectordb
# Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# QA Chain
# def retriever_qa(file, query):
#     try:
#         llm = get_llm()
#         retriever_obj = retriever(file)
#         qa = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever_obj,
#             return_source_documents=False
#         )
#         response = qa.invoke(query)
#         return response['result']
#     except Exception as e:
#         return f"Error processing query: {e}"

# QA Chain with enhanced retrieval and prompt
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    # Increase the number of documents retrieved
    retriever_obj.search_kwargs = {"k": 3}  # Retrieve 3 documents instead of the default
    prompt_template = """You are a helpful QA bot. Use the following pieces of context to answer the question. If the context is insufficient, summarize what you can from the document. Context: {context} Question: {question} Answer:"""
    PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    response = qa.invoke(query)
    return response['result']


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output", lines=9, max_lines=20),
    title="Your Personalized QA Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
    article="Made with Gradio and LangChain. (by Safiullah)"
)

# Launch the app (for local testing; comment out for Hugging Face deployment)
if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)
