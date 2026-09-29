import os
import re
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import (
    DOCS_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    LLM_MODEL,
    EMBEDDING_MODEL,
)


class RAGEngine:
    """
    Moteur RAG propulsé par Groq (LLM ultra-rapide) et HuggingFace Embeddings (Local/Gratuit).
    """

    def __init__(self, groq_api_key: str = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Clé API Groq introuvable. Veuillez renseigner GROQ_API_KEY."
            )

        # 1. Modèle d'Embeddings Multilingue (local et gratuit)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        # 2. LLM Groq (Llama 3.3 70B ultra-rapide)
        self.llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=0.1,
            groq_api_key=self.api_key
        )

        self.vector_store = None
        self.retriever = None

    def load_documents(self, folder_path: Path = DOCS_DIR) -> List[Document]:
        """Charge tous les fichiers .pdf, .txt et .md d'un dossier."""
        documents = []
        folder = Path(folder_path)

        if not folder.exists():
            return documents

        for file_path in folder.glob("**/*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                try:
                    if ext == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                        documents.extend(loader.load())
                    elif ext in [".txt", ".md"]:
                        loader = TextLoader(str(file_path), encoding="utf-8")
                        documents.extend(loader.load())
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_path.name}: {e}")

        return documents

    def build_vector_store(self, documents: List[Document] = None) -> Chroma:
        """Découpe les documents et crée/met à jour l'index vectoriel Chroma."""
        if documents is None:
            documents = self.load_documents()

        if not documents:
            raise ValueError("Aucun document trouvé pour l'indexation.")

        # Chunking intelligent
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = text_splitter.split_documents(documents)

        # Indexation Chroma persistante
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )
        return self.vector_store

    def load_existing_vector_store(self) -> bool:
        """Charge un index vectoriel existant depuis le disque si des documents y sont presents."""
        try:
            db_file = VECTOR_DB_DIR / "chroma.sqlite3"
            if not db_file.exists():
                return False
            self.vector_store = Chroma(
                persist_directory=str(VECTOR_DB_DIR),
                embedding_function=self.embeddings
            )
            if self.vector_store._collection.count() == 0:
                return False
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K_RESULTS}
            )
            return True
        except Exception:
            return False

    def query(self, question: str) -> Dict[str, Any]:
        """
        Interroge le RAG : récupère le contexte pertinent et génère la réponse
        avec citation des sources.
        """
        if self.retriever is None:
            if not self.load_existing_vector_store():
                self.build_vector_store()

        # 1. Récupération des documents similaires
        relevant_docs = self.retriever.invoke(question)

        # 2. Construction du contexte
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # 3. Prompt RAG strict anti-hallucination
        system_prompt = (
            "Tu es un assistant IA expert et rigoureux chargé de répondre aux questions des utilisateurs "
            "en te basant STRICTEMENT et UNIQUEMENT sur le contexte documentaire fourni ci-dessous en français.\n\n"
            "Règles impératives :\n"
            "1. Si la réponse ne figure pas dans le contexte, dis clairement que l'information n'est pas présente dans les documents.\n"
            "2. Reste concis, précis et professionnel.\n"
            "3. Cite des éléments clés du texte si pertinent.\n\n"
            "Contexte documentaire :\n"
            "{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        chain = prompt_template | self.llm | StrOutputParser()

        # 4. Génération de la réponse et nettoyage des balises internes
        raw_answer = chain.invoke({
            "context": context_text,
            "question": question
        })
        answer = re.sub(r'【.*?】', '', raw_answer).strip()

        # 5. Extraction des métadonnées des sources
        sources = []
        for i, doc in enumerate(relevant_docs, start=1):
            source_file = Path(doc.metadata.get("source", "Inconnu")).name
            page = doc.metadata.get("page", None)
            sources.append({
                "index": i,
                "file": source_file,
                "page": page,
                "content": doc.page_content.strip()
            })

        return {
            "answer": answer,
            "sources": sources
        }
