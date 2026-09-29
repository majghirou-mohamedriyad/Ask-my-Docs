import os
from pathlib import Path
import streamlit as st

from src.config import DOCS_DIR, GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE
from src.rag_engine import RAGEngine

# Configuration de la page Streamlit sans sidebar
st.set_page_config(
    page_title="AskMyDocs - RAG Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS personnalise épuré et moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155;
        margin-right: 8px;
    }
    .badge-accent {
        background-color: #EFF6FF;
        border-color: #BFDBFE;
        color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- GESTION DE SESSION -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ----------------- EN-TETE PRINCIPAL -----------------
st.markdown('<div class="main-title">AskMyDocs — Assistant RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Systeme d\'interrogation documentaire en langage naturel avec tracabilite et citation des sources.</div>',
    unsafe_allow_html=True
)

# Badges techniques
st.markdown(f"""
<div style="margin-bottom: 16px;">
    <span class="badge-pill badge-accent">LLM : Groq {LLM_MODEL}</span>
    <span class="badge-pill">Embeddings : HuggingFace Multilingual</span>
    <span class="badge-pill">Vector DB : ChromaDB (Local)</span>
    <span class="badge-pill">Chunk Size : {CHUNK_SIZE} tokens</span>
</div>
""", unsafe_allow_html=True)

# ----------------- GESTION DU CORPUS (VOLET RETRACTABLE) -----------------
with st.expander("Gestion du Corpus Documentaire & Indexation", expanded=False):
    col_upload, col_files = st.columns([1, 1])
    
    with col_upload:
        st.markdown("**Importer des documents :**")
        uploaded_files = st.file_uploader(
            "Selectionner des fichiers (.pdf, .txt, .md)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                dest_path = DOCS_DIR / uploaded_file.name
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"{len(uploaded_files)} document(s) ajoute(s).")
            
        if st.button("Re-indexer la base vectorielle", use_container_width=True):
            if not GROQ_API_KEY:
                st.error("Cle GROQ_API_KEY non definie dans le fichier .env")
            else:
                with st.spinner("Indexation vectorielle ChromaDB en cours..."):
                    try:
                        engine = RAGEngine(groq_api_key=GROQ_API_KEY)
                        engine.build_vector_store()
                        st.session_state.rag_engine = engine
                        st.session_state.indexed = True
                        st.success("Base vectorielle indexee avec succes.")
                    except Exception as e:
                        st.error(f"Erreur d'indexation : {e}")

    with col_files:
        existing_files = list(DOCS_DIR.glob("*.*"))
        st.markdown(f"**Documents presents dans l'index ({len(existing_files)}) :**")
        if existing_files:
            for f in existing_files:
                st.markdown(f"- `{f.name}`")
        else:
            st.info("Aucun document dans le dossier source.")

st.divider()

# Initialisation automatique du moteur RAG en arriere-plan
if GROQ_API_KEY and st.session_state.rag_engine is None:
    try:
        engine = RAGEngine(groq_api_key=GROQ_API_KEY)
        if not engine.load_existing_vector_store():
            engine.build_vector_store()
        st.session_state.rag_engine = engine
        st.session_state.indexed = True
    except Exception:
        pass

# ----------------- QUESTIONS SUGGEREES -----------------
st.markdown("**Exemples de questions rapides :**")
col1, col2, col3 = st.columns(3)
quick_prompt = None

with col1:
    if st.button("Budget formation collaborateur", use_container_width=True):
        quick_prompt = "Quel est le budget formation annuel par collaborateur ?"
with col2:
    if st.button("Politique de teletravail", use_container_width=True):
        quick_prompt = "Combien de jours de teletravail sont autorises par semaine ?"
with col3:
    if st.button("Regles de securite informatique", use_container_width=True):
        quick_prompt = "Quelles sont les regles de securite concernant les mots de passe et le 2FA ?"

# ----------------- HISTORIQUE DU CHAT -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources documentaires consultees"):
                for src in msg["sources"]:
                    page_str = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                    st.markdown(f"**Source {src['index']} : `{src['file']}`{page_str}**")
                    st.caption(f"_{src['content']}_")

# ----------------- ZONE DE SAISIE -----------------
user_input = st.chat_input("Posez votre question sur les documents...") or quick_prompt

if user_input:
    if not GROQ_API_KEY:
        st.error("Cle GROQ_API_KEY introuvable dans le fichier .env.")
    else:
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Reponse du RAG
        with st.chat_message("assistant"):
            with st.spinner("Recherche vectorielle et synthese en cours..."):
                try:
                    if st.session_state.rag_engine is None:
                        st.session_state.rag_engine = RAGEngine(groq_api_key=GROQ_API_KEY)
                    
                    response = st.session_state.rag_engine.query(user_input)
                    answer = response["answer"]
                    sources = response["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources documentaires consultees"):
                            for src in sources:
                                page_str = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                                st.markdown(f"**Source {src['index']} : `{src['file']}`{page_str}**")
                                st.caption(f"_{src['content']}_")

                    # Sauvegarde
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Erreur d'execution : {e}")
