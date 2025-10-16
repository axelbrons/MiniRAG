from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama  # Solution locale
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import torch
import warnings
warnings.filterwarnings('ignore')
import tqdm as notebook_tqdm
from langchain.prompts import PromptTemplate

# --- Modèle de langage (local, rapide) ---
llm = Ollama(model="mistral", temperature=0.1, num_predict=256)

# --- Vectorstore et embeddings initialisés vides ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = None

template = """
Tu es un assistant intelligent spécialisé dans l'analyse de CV pour les recruteurs.
Tu dois répondre **en français** de manière claire, concise et professionnelle.

Tu disposes des informations suivantes issues d'une base de CV :
{context}

Consignes :
- Utilise uniquement les informations présentes dans le contexte pour répondre.
- Si la réponse n'est pas clairement indiquée, dis simplement : "L'information n'est pas disponible dans les CV."
- Si plusieurs candidats semblent correspondre, mentionne leurs prénoms et explique brièvement pourquoi.
- Si un candidat se démarque particulièrement, indique-le clairement et justifie ton choix.

Question du recruteur : {question}

Réponse :
"""

prompt_fr = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# --- Fonction d'upload de PDF ---
def add_cvs(files):
    global vector_store
    documents = []
    for file in files:
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())
    # Découpage en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    # Création ou ajout au vectorstore
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embedding_model)
    else:
        vector_store.add_documents(chunks)
    return f"{len(chunks)} chunks ajoutés au vectorstore."

# --- Fonction de question ---
def ask_question(question):
    if vector_store is None:
        return "Aucun CV chargé."
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt_fr},return_source_documents=True)
    response = qa_chain(question)
    sources = response["source_documents"]
    res = response["result"]
    
    if sources:
        # Récupère la métadonnée 'source' du premier document
        source_file = sources[0].metadata.get("source", "Inconnu")
    else:
        source_file = "Inconnu (aucune source récupérée)"
    
    return f"{res}\n\nSource la plus pertinente : {source_file}"

# --- Interface Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("## Mini RAG pour CVs — Assistant Recruteur")
    with gr.Tab("Ajouter CV"):
        uploader = gr.Files(file_types=[".pdf"], label="Upload CVs")
        add_btn = gr.Button("Ajouter au vectorstore")
        output_add = gr.Textbox()
        add_btn.click(add_cvs, inputs=uploader, outputs=output_add)
    with gr.Tab("Poser une question"):
        question_input = gr.Textbox(label="Question du recruteur")
        ask_btn = gr.Button("Poser la question")
        output_answer = gr.Textbox(lines=12)
        ask_btn.click(ask_question, inputs=question_input, outputs=output_answer)

demo.launch()
