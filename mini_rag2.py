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
from langchain.schema import Document
import json

# Modèle llm
llm = Ollama(model="mistral", temperature=0.1, num_predict=256) #mistral ou phi3 (petit et rapide)

# Vectorstore et embeddings initialisés vides 
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") #384
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #768
vector_store = None

template = """
Tu es un assistant intelligent spécialisé dans l'analyse de CV pour les recruteurs.
Tu dois répondre **en français** de manière claire, concise et professionnelle.

Tu disposes des informations suivantes issues d'une base de CV :
{context}

Consignes :
- Utilise uniquement les informations présentes dans le contexte pour répondre.
- Si la réponse n'est pas clairement indiquée, dis simplement : "L'information n'est pas disponible dans les CV."
- Si plusieurs candidats semblent correspondre, mentionne leurs **NOM ET PRÉNOM COMPLET** et explique brièvement pourquoi.
- Si un candidat se démarque particulièrement, indique-le clairement en donnant son **NOM ET PRÉNOM COMPLET** et justifie ton choix.

Question du recruteur : {question}

Réponse :
"""

prompt_fr = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# --- PROMPT SPÉCIFIQUE D'EXTRACTION ---
# Ce prompt demande au LLM d'agir comme un extracteur de données
extraction_template = """
Tu es un extracteur de données intelligent. Ton but est d'analyser le texte brut d'un CV et de le transformer en un format JSON structuré.

Tu dois impérativement retourner UNIQUEMENT la structure JSON, sans aucun commentaire ni texte explicatif.

Structure JSON requise :
{{
    "nom_complet": "Nom et Prénom du candidat",
    "competences_techniques": ["Liste des compétences techniques clés"],
    "experience_principale": "Synthèse de l'expérience professionnelle la plus pertinente",
    "cv_source": "{source_file}"
}}

TEXTE DU CV :
{text}

JSON :
"""

extraction_prompt = PromptTemplate(
    input_variables=["text", "source_file"],
    template=extraction_template
)

def extract_json_from_cv(text_document, llm_model):
    """Utilise le LLM pour extraire le JSON structuré à partir d'un document texte."""
    source_file = text_document.metadata.get("source", "inconnu.pdf")
    
    # Remplir le prompt d'extraction
    prompt_value = extraction_prompt.format(
        text=text_document.page_content, 
        source_file=source_file
    )
    

    try:
        json_output = llm_model.invoke(prompt_value)
        
        if json_output.startswith("```json"):
            json_output = json_output.strip().strip("```json").strip("```").strip()
            
        data = json.loads(json_output)
        
        # Convertir le JSON structuré en une chaîne simple et propre pour l'embedding
        # Ceci est la chaîne qui sera vectorisée et stockée dans FAISS
        structured_text = (
            f"Candidat : {data.get('nom_complet', 'Inconnu')}. "
            f"Compétences : {', '.join(data.get('competences_techniques', []))}. "
            f"Expérience : {data.get('experience_principale', 'Non spécifiée')}."
        )
        
        new_metadata = text_document.metadata
        new_metadata['structured_data'] = data
        
        return Document(page_content=structured_text, metadata=new_metadata)
        
    except Exception as e:
        print(f"Erreur d'extraction JSON pour {source_file}: {e}")
        # En cas d'échec, on retourne le texte brut original
        return text_document


# Fonction upload PDF
def add_cvs(files):
    global vector_store
    
    documents = []
    structured_documents = []

    for file in files:
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())
    
    print("Début de l'extraction JSON par le LLM...")
    for doc in documents:
        structured_doc = extract_json_from_cv(doc, llm)
        structured_documents.append(structured_doc)
    print("Extraction JSON terminée.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(structured_documents) # On split les documents structurés
    
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embedding_model)
    else:
        vector_store.add_documents(chunks)
        
    return f"{len(chunks)} chunks structurés ajoutés au vectorstore."

# Fonction question
def ask_question(question):
    if vector_store is None:
        return "Aucun CV chargé"
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
    
    #return f"{res}\n\nSource la plus pertinente : {source_file}" #pour afficher la source
    return res

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Assistant Recruteur")
    with gr.Tab("Ajouter CV"):
        uploader = gr.Files(file_types=[".pdf"], label="Upload CVs")
        add_btn = gr.Button("Découpage et ajouter au vectorstore")
        output_add = gr.Textbox()
        add_btn.click(add_cvs, inputs=uploader, outputs=output_add)
    with gr.Tab("Poser une question"):
        question_input = gr.Textbox(label="Question du recruteur")
        ask_btn = gr.Button("Poser la question")
        output_answer = gr.Textbox(lines=12)
        ask_btn.click(ask_question, inputs=question_input, outputs=output_answer)

demo.launch()
