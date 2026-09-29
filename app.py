import os
from pathlib import Path
import streamlit as st

from src.config import DOCS_DIR, GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE
from src.rag_engine import RAGEngine

# Configuration de la page Streamlit
st.set_page_config(
    page_title="AskMyDocs - RAG Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé épuré et moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #334155;
        margin-right: 6px;
    }
    .badge-accent {
        background-color: #EFF6FF;
        border-color: #BFDBFE;
        color: #1D4ED8;
    }
    .source-container {
        background-color: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
    }
    .section-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 8px;
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


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="section-label">Configuration Moteur</div>', unsafe_allow_html=True)
    
    # Gestion de la clé API Groq
    api_key_input = st.text_input(
        "Clé API Groq",
        type="password",
        value=GROQ_API_KEY,
        help="Clé d'authentification API Groq"
    )
    
    st.divider()
    
    st.markdown('<div class="section-label">Corpus Documentaire</div>', unsafe_allow_html=True)
    
    # Upload de nouveaux fichiers
    uploaded_files = st.file_uploader(
        "Importer des documents (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            dest_path = DOCS_DIR / uploaded_file.name
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} document(s) ajouté(s).")
    
    # Affichage des fichiers présents dans le corpus
    existing_files = list(DOCS_DIR.glob("*.*"))
    if existing_files:
        st.caption(f"**Index actuel ({len(existing_files)} document(s)) :**")
        for f in existing_files:
            st.markdown(f"- `{f.name}`")
    else:
        st.info("Aucun document dans le dossier source.")
        
    # Bouton d'indexation
    if st.button("Ré-indexer le corpus", use_container_width=True):
        if not api_key_input:
            st.error("Clé API Groq requise pour l'initialisation.")
        else:
            with st.spinner("Indexation vectorielle ChromaDB en cours..."):
                try:
                    engine = RAGEngine(groq_api_key=api_key_input)
                    engine.build_vector_store()
                    st.session_state.rag_engine = engine
                    st.session_state.indexed = True
                    st.success("Base vectorielle indexée.")
                except Exception as e:
                    st.error(f"Erreur d'indexation : {e}")

    st.divider()
    st.markdown('<div class="section-label">Spécifications Techniques</div>', unsafe_allow_html=True)
    st.markdown(f"- **Moteur LLM :** `Groq ({LLM_MODEL})`")
    st.markdown(f"- **Embeddings :** `HuggingFace (MiniLM Multilingue)`")
    st.markdown(f"- **Vector Store :** `ChromaDB (Local)`")
    st.markdown(f"- **Chunk Size :** `{CHUNK_SIZE} tokens`")


# ----------------- CORPS PRINCIPAL -----------------
st.markdown('<div class="main-title">AskMyDocs — Assistant RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interrogation documentaire en langage naturel avec traçabilité et citation des sources.</div>',
    unsafe_allow_html=True
)

# Badges informatifs
st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="badge-pill badge-accent">LLM: Llama 3.3 70B</span>
    <span class="badge-pill">Vector DB: Chroma</span>
    <span class="badge-pill">Embeddings: Multilingual</span>
</div>
""", unsafe_allow_html=True)

# Initialisation automatique du moteur RAG si la clé est fournie
if api_key_input and st.session_state.rag_engine is None:
    try:
        engine = RAGEngine(groq_api_key=api_key_input)
        if not engine.load_existing_vector_store():
            engine.build_vector_store()
        st.session_state.rag_engine = engine
        st.session_state.indexed = True
    except Exception:
        pass

# Suggestions de questions rapides
st.markdown("**Questions fréquentes :**")
col1, col2, col3 = st.columns(3)
quick_prompt = None

with col1:
    if st.button("Budget formation collaborateur"):
        quick_prompt = "Quel est le budget formation annuel par collaborateur ?"
with col2:
    if st.button("Politique de télétravail"):
        quick_prompt = "Combien de jours de télétravail sont autorisés par semaine ?"
with col3:
    if st.button("Règles de sécurité informatique"):
        quick_prompt = "Quelles sont les règles de sécurité concernant les mots de passe et le 2FA ?"

# Affichage de l'historique des conversations
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources documentaires"):
                for src in msg["sources"]:
                    page_str = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                    st.markdown(f"**Source {src['index']} : `{src['file']}`{page_str}**")
                    st.caption(f"_{src['content']}_")

# Zone de saisie utilisateur
user_input = st.chat_input("Saisissez votre question sur les documents...") or quick_prompt

if user_input:
    if not api_key_input:
        st.warning("Veuillez renseigner une clé API Groq valide dans le panneau de configuration.")
    else:
        # Ajout du message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Génération de la réponse RAG
        with st.chat_message("assistant"):
            with st.spinner("Recherche vectorielle et synthèse en cours..."):
                try:
                    if st.session_state.rag_engine is None:
                        st.session_state.rag_engine = RAGEngine(groq_api_key=api_key_input)
                    
                    response = st.session_state.rag_engine.query(user_input)
                    answer = response["answer"]
                    sources = response["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources documentaires"):
                            for src in sources:
                                page_str = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                                st.markdown(f"**Source {src['index']} : `{src['file']}`{page_str}**")
                                st.caption(f"_{src['content']}_")

                    # Sauvegarde dans la session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Erreur d'exécution : {e}")
