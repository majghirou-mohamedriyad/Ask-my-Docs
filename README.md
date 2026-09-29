# AskMyDocs — Systeme RAG Haute Performance

Projet d'evaluation technique realise pour le poste d'Ingenieur IA chez CIRES TECHNOLOGIES (Filiale du Groupe TANGER MED).

---

## Sommaire

1. [Contexte et Objectifs du Projet](#1-contexte-et-objectifs-du-projet)
2. [Architecture Globale du Systeme](#2-architecture-globale-du-systeme)
3. [Pipeline RAG Etape par Etape](#3-pipeline-rag-etape-par-etape)
4. [Justification des Choix Techniques](#4-justification-des-choix-techniques)
5. [Controle des Hallucinations et Tracabilite](#5-controle-des-hallucinations-et-tracabilite)
6. [Structure du Repertoire](#6-structure-du-repertoire)
7. [Guide d'Installation et d'Execution](#7-guide-dinstallation-et-dexecution)
8. [Jeu de Donnees et Tests de Demonstration](#8-jeu-de-donnees-et-tests-de-demonstration)
9. [Perspectives d'Amelioration en Production](#9-perspectives-damelioration-en-production)

---

## 1. Contexte et Objectifs du Projet

Dans les organisations disposant d'importants volumes de donnees non structurees (politiques internes, rapports techniques, procedures operationnelles), la recherche d'information est souvent lente et sujette aux erreurs d'interpretation.

**AskMyDocs** est une application complete de **Retrieval-Augmented Generation (RAG)** concue pour :
- Centraliser et indexer automatiquement des corpus documentaires heterogenes (PDF, TXT, Markdown).
- Permettre aux utilisateurs d'interroger la base documentaire en langage naturel (francais / anglais).
- Fournir des reponses fiables et synthetiques en temps reel, avec citation stricte des sources consultees.
- Garantir un risque d'hallucination minimal grace a un cadrage systeme rigoureux.

---

## 2. Architecture Globale du Systeme

Le flux d'information combine une etape d'ingestion/indexation hors-ligne (offline) et une etape d'interrogation/generation en ligne (online) :

```mermaid
flowchart TD
    subgraph INGESTION ["Phase 1 : Ingestion & Indexation"]
        D[Documents PDF / TXT / MD] --> L[Document Loaders]
        L --> S[RecursiveCharacterTextSplitter\nChunk: 600 tokens | Overlap: 100]
        S --> E[HuggingFace Embeddings\nMiniLM Multilingue]
        E --> V[(ChromaDB - Base Vectorielle Persistante)]
    end

    subgraph RETRIEVAL_GENERATION ["Phase 2 : Recherche & Generation"]
        U[Question Utilisateur] --> Q_EMB[Vectorisation de la Requete]
        Q_EMB --> R[Retriever Top-K Similarity Search]
        V --> R
        R --> C[Contexte Documentaire Extrait + Metadonnees]
        
        U --> P[ChatPromptTemplate Anti-hallucination]
        C --> P
        P --> LLM[Groq LPU Engine\nLlama-3.3-70b-versatile]
        LLM --> ANS[Reponse Formulee + Citations]
        ANS --> UI[Interface Web Streamlit]
    end
```

---

## 3. Pipeline RAG Etape par Etape

### Etape 1 : Ingestion des Documents
- Les fichiers deposes dans `data/documents/` ou charges via l'interface web sont automatiquement detectes.
- Utilisation de `PyPDFLoader` pour les fichiers PDF et de `TextLoader` (encodage UTF-8) pour les fichiers texte et Markdown.

### Etape 2 : Decoupage Semantique (Chunking)
- Algorithme : `RecursiveCharacterTextSplitter`.
- **Taille de chunk (`chunk_size`)** : 600 tokens. Cette dimension est ideale pour conserver une unite logique d'information (un paragraphe complet ou un article de reglementation) sans diluer le signal semantique.
- **Recouvrement (`chunk_overlap`)** : 100 tokens. Assure la continuite contextuelle entre deux segments consecutifs et evite la perte d'informations situees aux frontieres des blocs.

### Etape 3 : Vectorisation (Embeddings)
- Modele : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Transformation des textes en vecteurs denses de 384 dimensions captures dans un espace semantique aligne pour plus de 50 langues (avec une excellente sensibilite sur le francais technique et administratif).

### Etape 4 : Indexation et Recherche Vectorielle
- Moteur : **ChromaDB**.
- Les vecteurs et leurs metadonnees associees (`source`, `page`, `chunk_id`) sont indexes localement sur disque (`data/chroma_db/`).
- Lors d'une requete, le moteur effectue une recherche de similarite par distance cosinus pour extraire les **Top-4 chunks** les plus pertinents.

### Etape 5 : Generation Augmentee et Formattage
- Le LLM (**Llama 3.3 70B**) recoit la requete de l'utilisateur enrichie du contexte strict extrait par le retriever.
- Le systeme formate ensuite la reponse avec des blocs de citation detaillant le fichier d'origine, le numero de page et le texte source.

---

## 4. Justification des Choix Techniques

| Composant | Technologie Choisie | Raison du Choix & Avantages | Alternatives Considerees |
| :--- | :--- | :--- | :--- |
| **Moteur d'Inference LLM** | **Groq (Llama 3.3 70B)** | Vitesse d'inference exceptionnelle (~500 tokens/seconde grace aux puces LPU), modele open-weights de niveau GPT-4 sans cout exorbitant. | OpenAI GPT-4 (plus lent, cout par token eleve), Ollama local (limite par le GPU local). |
| **Modele d'Embeddings** | **HuggingFace Multilingual MiniLM** | Execution 100% locale, gratuite, aucune dependance reseau, support natif du francais et faible empreinte memoire (~120 Mo). | OpenAI text-embedding-3-small (payant, dependance reseau). |
| **Base Vectorielle** | **ChromaDB** | Base vectorielle legere, sans serveur externe requis, integration native avec LangChain et persistance disque immediate. | FAISS (pas de persistance native des metadonnees sans surcouche), Pinecone (SaaS distant). |
| **Framework d'Orchestration** | **LangChain (LCEL)** | Conception modulaire avec *LangChain Expression Language*, standard de l'industrie, composabilite des chaines de traitement. | Code Python natif (moins extensible), LlamaIndex. |
| **Interface Utilisateur** | **Streamlit** | Deploiement rapide, interface reactive, support du streaming et affichage personnalise des sources. | Gradio, FastAPI + React. |

---

## 5. Controle des Hallucinations et Tracabilite

Pour repondre aux exigences strictes du monde de l'entreprise, le systeme applique une strategie de fiabilisation a plusieurs niveaux :

1. **Prompt System Cadrant** :
   Le systeme impose formellement au modele de repondre uniquement a partir du contexte documentaire fourni. En l'absence d'information dans les documents, le modele indique explicitement son incapacite a repondre au lieu d'extrapoler.
2. **Temperature Basse (`temperature=0.1`)** :
   Reduction maximale de l'aléa generationnel pour privilegier la precision factuelle et le determinisme.
3. **Inspecteur de Sources Retractable** :
   Chaque reponse generee est accompagnee d'un module retractable affichant le fichier source, la page ainsi que le paragraphe exact sur lequel s'est basee la deduction.

---

## 6. Structure du Repertoire

```
AskMyDocs/
|-- data/
|   |-- documents/               # Repertoire source des documents (PDF, TXT, MD)
|   |   `-- politique_entreprise.md # Document exemple precharge
|   `-- chroma_db/               # Base vectorielle persistante (generee)
|-- src/
|   |-- __init__.py
|   |-- config.py                # Configuration globale et parametres
|   `-- rag_engine.py            # Logique d'ingestion, chunking, retrieval et generation
|-- app.py                       # Interface web applicative Streamlit
|-- requirements.txt             # Dependances Python du projet
|-- .env.example                 # Modele de configuration d'environnement
|-- .gitignore                   # Exclusion des fichiers temporaires et des secrets
`-- README.md                    # Documentation technique exhaustive
```

---

## 7. Guide d'Installation et d'Execution

### Prerequis
- Python 3.10 ou superieur installe sur la machine.
- Une cle API Groq (disponible gratuitement sur [console.groq.com](https://console.groq.com)).

### 1. Recuperation du projet
```bash
git clone https://github.com/majghirou-mohamedriyad/Ask-my-Docs.git
cd Ask-my-Docs
```

### 2. Creation d'un environnement virtuel (Recommande)
```bash
python -m venv venv

# Activation sous Windows :
.\venv\Scripts\activate

# Activation sous Linux/macOS :
source venv/bin/activate
```

### 3. Installation des dependances
```bash
pip install -r requirements.txt
```

### 4. Configuration des variables d'environnement
Creez un fichier `.env` a la racine du projet :
```env
GROQ_API_KEY=votre_cle_api_groq_ici
```
*(Note : Il est egalement possible de saisir la cle API directement depuis le panneau lateral de l'application web).*

### 5. Lancement de l'application
```bash
python -m streamlit run app.py
```
L'interface est accessible a l'adresse locale : `http://localhost:8501`.

---

## 8. Jeu de Donnees et Tests de Demonstration

Le projet est livre avec un document de demonstration precharge : `data/documents/politique_entreprise.md`.

### Exemples de requetes evaluables lors du test :

| Question Posee | Resultat Attendu | Source Verifiee |
| :--- | :--- | :--- |
| **"Quel est le budget formation accorde par collaborateur ?"** | 1 500 euros par collaborateur et par an pour des certifications ou cours. | `politique_entreprise.md` (Section 3) |
| **"Combien de jours de teletravail sont autorises ?"** | 2 jours par semaine apres validation du manager. | `politique_entreprise.md` (Section 1) |
| **"Quelle est la regle pour les mots de passe et le 2FA ?"** | Authentification 2FA obligatoire, mot de passe de 14 caracteres minimum sur 1Password. | `politique_entreprise.md` (Section 4) |
| **"Quel est le chiffre d'affaires de l'entreprise en 2025 ?"** | L'assistant declare que l'information n'est pas presente dans les documents (Test anti-hallucination reussi). | Aucune source (Refus de generer du contenu fictif) |

---

## 9. Perspectives d'Amelioration en Production

Pour adapter cette architecture a des volumes massifs (plusieurs dizaines de milliers de documents) en environnement industriel :

1. **Recherche Hybride (Dense + Sparse)** :
   Combiner la recherche vectorielle dense (embeddings) avec un algorithme BM25 sparse pour ameliorer la precision sur les mots-cles techniques rares, acronymes et codes produits.
2. **Re-ranking avec Cross-Encoder** :
   Ajouter une etape de reranking (ex: `bge-reranker-large` ou `Cohere Rerank`) sur les Top-20 resultats afin de ne transmettre au LLM que les Top-3 les plus pertinents.
3. **Chunking Semantique Avance** :
   Mettre en place un decoupage base sur la structure hierarchique des documents (Markdown header-based ou parser de mise en page PDF avance type Docling / Unstructured).
4. **Evaluation Continue (RAGAS / TruLens)** :
   Integrer un pipeline d'evaluation automatique mesurant la fidelite (*faithfulness*), la pertinence du contexte (*context relevance*) et la pertinence de la reponse (*answer relevance*).
