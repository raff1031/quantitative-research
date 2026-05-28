# filename: tutor_app_chroma.py
import io
import os
import re
import shutil
import tempfile
import logging
from typing import List, Tuple

import streamlit as st

# ---------- Parser PDF ----------
USE_UNSTRUCTURED = False
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader  # opzionale
    USE_UNSTRUCTURED = True
except Exception:
    pass

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    from PyPDF2 import PdfReader
    HAS_PYMUPDF = False

try:
    import ocrmypdf  # OCR opzionale
    HAS_OCR = True
except Exception:
    HAS_OCR = False

# ---------- LangChain / Ollama / Chroma ----------
from langchain_ollama import ChatOllama, OllamaEmbeddings  # :contentReference[oaicite:2]{index=2}
from langchain_community.vectorstores import Chroma  # :contentReference[oaicite:3]{index=3}
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain  # :contentReference[oaicite:4]{index=4}
from langchain.chains import create_retrieval_chain  # :contentReference[oaicite:5]{index=5}

# ---------- Streamlit ----------
st.set_page_config(page_title="Tutor — gpt-oss:20b + RAG + OCR (Chroma)", layout="wide")
st.title("🎓 Tutor Virtuale — gpt-oss:20b + RAG + OCR (Chroma)")
st.caption("Carica PDF (anche scannerizzati), indicizza con Chroma e fai domande.")

# ---------- Persistenza ----------
CHROMA_DIR = os.path.join(".", "storage", "chroma")
COLLECTION_NAME = "docs"
os.makedirs(CHROMA_DIR, exist_ok=True)
CHROMA_SETTINGS = Settings(anonymized_telemetry=False, is_persistent=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ LLM (Ollama)")
    llm_model = st.selectbox("Modello", ["gpt-oss:20b"], index=0)
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.4, 0.1)
    ctx_tokens = st.slider("Contesto (num_ctx)", 2048, 131072, 8192, step=1024)
    unlimited = st.checkbox("Output illimitato (num_predict = -1)", value=False)
    max_tokens = -1 if unlimited else st.slider("Max tokens risposta (num_predict)", 128, 8192, 4096, 128)

    st.markdown("---")
    st.subheader("🔎 Retrieval")
    k_results = st.number_input("Passages da recuperare (top-k)", 1, 10, 4)
    fetch_k = st.number_input("fetch_k (pool MMR)", 10, 200, max(20, int(k_results)*5))
    show_sources = st.checkbox("Mostra i brani usati", value=True)

    st.markdown("---")
    st.subheader("📄 Documenti")
    pdf_docs = st.file_uploader("Carica PDF (multipli)", type="pdf", accept_multiple_files=True)
    prefer_unstructured = st.checkbox("Parser Unstructured (se disponibile)", value=USE_UNSTRUCTURED,
                                      help="Richiede il pacchetto 'unstructured'.")

    use_ocr = st.checkbox("Applica OCR se il PDF non ha testo", value=HAS_OCR,
                          help="OCRmyPDF aggiunge un layer testuale al PDF scannerizzato.")
    ocr_lang = st.text_input("Lingua OCR (Tesseract)", value="ita+eng")

    colA, colB = st.columns(2)
    with colA:
        process_btn = st.button("Processa / Aggiorna indice", type="primary")
    with colB:
        clear_btn = st.button("Svuota indice")

# ---------- Stato ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = OllamaEmbeddings(model="mxbai-embed-large")

# ---------- LLM ----------
def build_llm(model_name: str, temperature: float, num_ctx: int, num_predict: int):
    return ChatOllama(model=model_name, temperature=temperature, num_ctx=num_ctx, num_predict=num_predict)

# ---------- Estrazione testo ----------
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def clean_text(s: str) -> str:
    return CTRL_RE.sub(" ", s).strip()

def _ocr_to_temp_pdf(file_bytes: bytes, lang: str) -> str:
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_in.write(file_bytes); tmp_in.flush(); tmp_in.close()
    out_path = tmp_in.name.replace(".pdf", ".ocred.pdf")
    ocrmypdf.ocr(tmp_in.name, out_path, language=lang, force_ocr=True,
                 progress_bar=False, optimize=3, output_type="pdf")
    return out_path  # OCRmyPDF aggiunge un layer testo al PDF :contentReference[oaicite:6]{index=6}

def extract_with_pymupdf(b: bytes, source: str) -> List[Document]:
    docs = []
    doc = fitz.open(stream=b, filetype="pdf")
    for i, page in enumerate(doc, start=1):
        # sort=True riordina top-left → bottom-right :contentReference[oaicite:7]{index=7}
        txt = page.get_text("text", sort=True) or ""
        txt = clean_text(txt)
        if txt.strip():
            docs.append(Document(page_content=txt, metadata={"source": source, "page": i}))
    return docs

def extract_with_pypdf(b: bytes, source: str) -> List[Document]:
    out = []
    rdr = PdfReader(io.BytesIO(b))
    for i, p in enumerate(rdr.pages, start=1):
        txt = (p.extract_text() or "")
        txt = clean_text(txt)
        if txt.strip():
            out.append(Document(page_content=txt, metadata={"source": source, "page": i}))
    return out

def extract_with_unstructured(path: str, source: str) -> List[Document]:
    # Unstructured gestisce layout / elementi; ottimo per PDF “ricchi” :contentReference[oaicite:8]{index=8}
    loader = UnstructuredPDFLoader(path, mode="elements")
    pages = []
    for d in loader.load():
        content = clean_text(d.page_content or "")
        md = d.metadata or {}
        page = md.get("page_number") or md.get("page") or None
        if content.strip():
            pages.append(Document(page_content=content, metadata={"source": source, "page": page}))
    return pages

def read_pdfs(files: List, prefer_unstructured: bool, use_ocr: bool, lang: str) -> List[Document]:
    all_docs: List[Document] = []
    for f in files:
        b = f.read()
        name = getattr(f, "name", "PDF")
        # 1) Unstructured (se presente e richiesto)
        if prefer_unstructured and USE_UNSTRUCTURED:
            # se serve OCR prima:
            tmp_path = None
            try:
                if use_ocr and HAS_OCR:
                    tmp_path = _ocr_to_temp_pdf(b, lang)
                else:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(b); tmp.flush(); tmp.close()
                    tmp_path = tmp.name
                all_docs += extract_with_unstructured(tmp_path, name)
                continue
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass

        # 2) PyMuPDF → 3) PyPDF2, con OCR se vuoto
        docs = extract_with_pymupdf(b, name) if HAS_PYMUPDF else []
        if not docs and use_ocr and HAS_OCR:
            ocr_path = _ocr_to_temp_pdf(b, lang)
            with open(ocr_path, "rb") as pf:
                docs = extract_with_pymupdf(pf.read(), name) if HAS_PYMUPDF else []
            try: os.remove(ocr_path)
            except: pass
        if not docs and not HAS_PYMUPDF:
            docs = extract_with_pypdf(b, name)
        if not docs:
            st.warning(f"⚠️ Nessun testo estraibile da: {name}")
        else:
            all_docs += docs
    return all_docs

def chunk_docs_by_page(docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    out = []
    for d in docs:
        for part in splitter.split_text(d.page_content):
            out.append(Document(page_content=part, metadata=d.metadata))
    return out

# ---------- Pulisci indice ----------
if clear_btn:
    try:
        if os.path.isdir(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        st.session_state.vector_store = None
        st.success("Indice Chroma eliminato.")
    except Exception as e:
        st.error(f"Errore eliminando l'indice: {e}")

# ---------- Processa / indicizza ----------
if process_btn:
    if not pdf_docs:
        st.error("Carica almeno un PDF prima di processare.")
    else:
        with st.spinner("🧠 Parsing/OCR e indicizzazione (Chroma) in corso..."):
            pages = read_pdfs(pdf_docs, prefer_unstructured, use_ocr, ocr_lang)
            if pages:
                chunks = chunk_docs_by_page(pages, 1000, 200)
                emb = st.session_state.embeddings
                vs = Chroma.from_documents(
                    documents=chunks,
                    embedding=emb,
                    persist_directory=CHROMA_DIR,
                    collection_name=COLLECTION_NAME,
                    client_settings=CHROMA_SETTINGS,
                )  # persist_directory + collection_name → persistenza su disco :contentReference[oaicite:9]{index=9}
                vs.persist()
                st.session_state.vector_store = vs
                try:
                    count = vs._collection.count()
                    st.info(f"Chroma: documenti indicizzati = {count}")
                except Exception:
                    pass
                st.success(f"Documento/i processati! Chunks indicizzati: {len(chunks)}")
            else:
                st.error("Nessun testo indicizzabile trovato.")

# ---------- Carica indice se esiste ----------
if st.session_state.vector_store is None and os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    try:
        st.session_state.vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=st.session_state.embeddings,
            collection_name=COLLECTION_NAME,
            client_settings=CHROMA_SETTINGS,
        )
        try:
            count = st.session_state.vector_store._collection.count()
            st.info(f"Indice Chroma caricato da disco. Documenti = {count}")
        except Exception:
            st.info("Indice Chroma caricato da disco.")
    except Exception as e:
        st.warning(f"Impossibile caricare l'indice da disco. Dettagli: {e}")

# ---------- Chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Prompt RAG forte (system) ----------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Sei un tutor universitario. Rispondi SEMPRE in italiano usando esclusivamente il CONTEXT fornito "
     "(estratto dai PDF indicizzati). NON dire mai che non puoi aprire o visualizzare PDF: disponi già del testo. "
     "Se l'informazione non è nel CONTEXT, scrivi solo: 'Non l'ho trovato nei tuoi appunti.' "
     "Se la domanda è vaga, fai 1–2 domande di chiarimento basandoti sui temi presenti nel CONTEXT."),
    ("human",
     "CONTEXT:\n{context}\n\nDOMANDA: {input}\n\nRispondi in modo chiaro e strutturato.")
])

# ---------- Input utente ----------
prompt = st.chat_input("Fai la tua domanda sul contenuto indicizzato (o chiedimi altro).")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto pensando..."):
            if st.session_state.vector_store is None:
                llm = build_llm(llm_model, temperature, ctx_tokens, max_tokens)
                st.markdown("Indice vuoto. Premi **Processa / Aggiorna indice** dopo aver caricato i PDF.")
                response = ""
            else:
                llm = build_llm(llm_model, temperature, ctx_tokens, max_tokens)

                # 1) retriever MMR (fetch_k alto) :contentReference[oaicite:10]{index=10}
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": int(k_results), "fetch_k": int(fetch_k)}
                )

                # 2) prima “dry run” per capire se abbiamo CONTEXT
                candidates = retriever.get_relevant_documents(prompt)

                if not candidates:
                    # No fallback all'LLM nudo: chiedi chiarimento (stile ML Studio)
                    response = ("Non trovo riferimenti nei tuoi appunti per questa domanda. "
                                "Specificami l’argomento o la sezione (es. 'Chi-quadrato: densità e atteso', "
                                "'quantile esponenziale eq. (5.23)').")
                    st.markdown(response)
                    if show_sources:
                        st.info("Nessun passage recuperato.")
                else:
                    # 3) chain RAG “nuova” (docs -> {context} -> LLM) :contentReference[oaicite:11]{index=11}
                    qa_docs_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
                    rag_chain = create_retrieval_chain(retriever, qa_docs_chain)
                    result = rag_chain.invoke({"input": prompt})
                    response = result.get("answer") or result.get("result", "")
                    st.markdown(response)

                    if show_sources:
                        with st.expander("🔎 Passages utilizzati"):
                            ctx_docs = result.get("context", []) or candidates
                            for i, d in enumerate(ctx_docs, 1):
                                md = d.metadata or {}
                                where = f"(pag. {md.get('page')})" if md.get('page') else ""
                                src = f" — {md.get('source')}" if md.get('source') else ""
                                pc = d.page_content
                                st.markdown(f"**Passage {i} {where}{src}:**\n\n{pc[:1200]}{'…' if len(pc)>1200 else ''}")

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---------- Logging ----------
logging.getLogger("streamlit").setLevel(logging.ERROR)
