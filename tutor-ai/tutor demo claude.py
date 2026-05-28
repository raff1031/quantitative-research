# filename: tutor_app_universal.py
# Tutor AI Universale — Login/Onboarding + RAG (Chroma/Ollama) + Migrazione DB sicura

import io
import os
import re
import shutil
import tempfile
import logging
import sqlite3
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import streamlit as st
import pandas as pd
import plotly.express as px

# =============== STREAMLIT CONFIG ===============
st.set_page_config(page_title="🎓 AI Tutor Universal", layout="wide")

# ---------- Parser PDF ----------
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    from PyPDF2 import PdfReader
    HAS_PYMUPDF = False

# ---------- LangChain / Ollama / Chroma ----------
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ---------- Vision & Advanced Processing ----------
try:
    import PIL.Image as PILImage
    HAS_PIL = True
except:
    HAS_PIL = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except:
    HAS_NETWORKX = False

# ---------- Costanti ----------
BASE_DIR = "storage"
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")
DB_PATH = os.path.join(BASE_DIR, "users.db")
CURRENT_USER_FILE = os.path.join(BASE_DIR, "current_user.json")
KNOWLEDGE_GRAPH_PATH = os.path.join(BASE_DIR, "knowledge_graph.json")
CHROMA_SETTINGS = Settings(anonymized_telemetry=False, is_persistent=True)

# Modelli Ollama
MAIN_MODEL = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"  # Qwen3 Coder (migliore!)
VISION_MODEL = "llama3.2-vision:11b"  # Fallback per PDF scannerizzati
EMBEDDING_MODEL = "nomic-embed-text"  # Embeddings (raccomandato per RAG)

# Configurazione Performance (GPU offload massimo per RTX 3080 Ti)
GPU_LAYERS = 50  # Numero layer su GPU (aumenta per più offload, max ~55 per 20B su 12GB)
NUM_GPU = 1      # Numero GPU da usare
NUM_THREAD = 16  # Thread CPU (metà dei core del 9950X3D per lasciare spazio)
OLLAMA_NUM_PARALLEL = 1  # Richieste parallele (1 = massima velocità singola inferenza)

# Ottimizzazioni Avanzate
FAST_ANALYSIS_MODEL = "llama3.2:3b"  # Modello lightweight per analisi struttura (opzionale)
MAX_WORKERS = 3  # Thread paralleli per batch processing
BATCH_SIZE_MIN = 3  # Batch size minimo
BATCH_SIZE_MAX = 10  # Batch size massimo
ADAPTIVE_BATCHING = True  # Abilita batch size dinamico
STRUCTURE_CACHE_FILE = os.path.join(BASE_DIR, "structure_cache.json")

os.makedirs(BASE_DIR, exist_ok=True)

# ==================== Persistenza utente ====================
def save_current_user(user_id: str):
    with open(CURRENT_USER_FILE, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id}, f)

def load_current_user() -> Optional[str]:
    try:
        if os.path.exists(CURRENT_USER_FILE):
            with open(CURRENT_USER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("user_id")
    except Exception:
        pass
    return None

# ==================== DATABASE (migrazione sicura) ====================
EXPECTED_USER_PROFILE_COLS = [
    "user_id",
    "nickname",
    "subject_area",
    "course_name",
    "education_level",
    "goal",
    "exam_date",
    "days_to_exam",
    "target_grade",
    "study_urgency",
    "study_hours_week",
    "learning_style",
    "self_assessment_theory",
    "self_assessment_exercises",
    "exam_anxiety",
    "difficulty_preference",
    "created_at",
    "total_points",
    "difficulty_level",
    "custom_topics",
    "prior_knowledge",
]

CREATE_USER_PROFILE_SQL = '''
CREATE TABLE IF NOT EXISTS user_profile
 (user_id TEXT PRIMARY KEY,
  nickname TEXT,
  subject_area TEXT,
  course_name TEXT,
  education_level TEXT,
  goal TEXT,
  exam_date TEXT,
  days_to_exam INTEGER,
  target_grade INTEGER,
  study_urgency TEXT,
  study_hours_week INTEGER,
  learning_style TEXT,
  self_assessment_theory INTEGER,
  self_assessment_exercises INTEGER,
  exam_anxiety INTEGER,
  difficulty_preference TEXT,
  created_at TEXT,
  total_points INTEGER DEFAULT 0,
  difficulty_level REAL DEFAULT 3.0,
  custom_topics TEXT,
  prior_knowledge TEXT)
'''

class EnhancedDatabase:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_database()

    # ---- util ----
    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None

    def _get_columns(self, conn: sqlite3.Connection, table: str) -> List[str]:
        try:
            cur = conn.execute(f"PRAGMA table_info({table});")
            return [row[1] for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    # ---- rebuild sicuro user_profile ----
    def _rebuild_user_profile(self, conn: sqlite3.Connection):
        """
        Rebuild user_profile in modo sicuro:
          - FK OFF
          - se esiste user_profile → rinomina in user_profile_old
          - crea NUOVA user_profile (schema atteso)
          - copia solo colonne comuni da user_profile_old (se esiste)
          - drop user_profile_old (se esiste)
          - FK ON
        """
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.execute("BEGIN;")
        try:
            renamed = False
            if self._table_exists(conn, "user_profile"):
                conn.execute("ALTER TABLE user_profile RENAME TO user_profile_old;")
                renamed = True

            # crea tabella nuova con schema atteso (nome definitivo)
            conn.execute(CREATE_USER_PROFILE_SQL.replace("IF NOT EXISTS ", "").strip().rstrip(";"))

            if renamed and self._table_exists(conn, "user_profile_old"):
                old_cols = set(self._get_columns(conn, "user_profile_old"))
                common = [c for c in EXPECTED_USER_PROFILE_COLS if c in old_cols]
                if common:
                    cols = ", ".join(common)
                    conn.execute(
                        f"INSERT INTO user_profile ({cols}) SELECT {cols} FROM user_profile_old;"
                    )
                conn.execute("DROP TABLE user_profile_old;")

            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON;")

    def _migrate_user_profile_if_needed(self, conn: sqlite3.Connection):
        if not self._table_exists(conn, "user_profile"):
            conn.execute(CREATE_USER_PROFILE_SQL)
            return

        existing_cols = set(self._get_columns(conn, "user_profile"))
        expected_cols = set(EXPECTED_USER_PROFILE_COLS)
        if not expected_cols.issubset(existing_cols):
            # schema non compatibile: rebuild
            self._rebuild_user_profile(conn)

    # ---- bootstrap DB ----
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")

        # user_profile con migrazione
        self._migrate_user_profile_if_needed(conn)

        # altre tabelle
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions
                     (session_id TEXT PRIMARY KEY,
                      user_id TEXT,
                      start_time TEXT,
                      questions_count INTEGER,
                      points_earned INTEGER,
                      FOREIGN KEY(user_id) REFERENCES user_profile(user_id))''')

        conn.execute('''CREATE TABLE IF NOT EXISTS questions
                     (question_id TEXT PRIMARY KEY,
                      session_id TEXT,
                      user_id TEXT,
                      timestamp TEXT,
                      question_text TEXT,
                      help_seeking_type TEXT,
                      topic TEXT,
                      showed_attempt INTEGER,
                      points_earned INTEGER,
                      difficulty_level REAL,
                      FOREIGN KEY(session_id) REFERENCES sessions(session_id))''')

        conn.execute('''CREATE TABLE IF NOT EXISTS knowledge_state
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id TEXT,
                      topic TEXT,
                      mastery_level REAL,
                      interactions_count INTEGER,
                      last_interaction TEXT,
                      FOREIGN KEY(user_id) REFERENCES user_profile(user_id))''')

        conn.execute('''CREATE TABLE IF NOT EXISTS study_plan
                     (plan_id TEXT PRIMARY KEY,
                      user_id TEXT,
                      topic TEXT,
                      hours_allocated REAL,
                      priority INTEGER,
                      completed INTEGER DEFAULT 0,
                      created_at TEXT,
                      FOREIGN KEY(user_id) REFERENCES user_profile(user_id))''')

        conn.commit()
        conn.close()

    # ---- API app ----
    def create_user(self, profile_data: Dict) -> str:
        user_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:16]
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        c = conn.cursor()

        difficulty = self._calculate_initial_difficulty(profile_data)

        c.execute(
            '''INSERT INTO user_profile
               (user_id, nickname, subject_area, course_name, education_level, goal,
                exam_date, days_to_exam, target_grade, study_urgency, study_hours_week,
                learning_style, self_assessment_theory, self_assessment_exercises,
                exam_anxiety, difficulty_preference, created_at, total_points,
                difficulty_level, custom_topics, prior_knowledge)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (user_id,
             profile_data.get('nickname'),
             profile_data.get('subject_area'),
             profile_data.get('course_name'),
             profile_data.get('education_level'),
             profile_data.get('goal'),
             profile_data.get('exam_date'),
             profile_data.get('days_to_exam'),
             profile_data.get('target_grade'),
             profile_data.get('study_urgency'),
             profile_data.get('study_hours_week'),
             profile_data.get('learning_style'),
             profile_data.get('self_assessment_theory'),
             profile_data.get('self_assessment_exercises'),
             profile_data.get('exam_anxiety'),
             profile_data.get('difficulty_preference'),
             datetime.now().isoformat(),
             0,
             difficulty,
             profile_data.get('custom_topics'),
             profile_data.get('prior_knowledge'))
        )

        self._generate_study_plan(user_id, profile_data, conn)
        conn.commit()
        conn.close()
        return user_id

    def _calculate_initial_difficulty(self, profile: Dict) -> float:
        theory = profile.get('self_assessment_theory', 3)
        exercises = profile.get('self_assessment_exercises', 3)
        base_score = (theory + exercises) / 2
        anxiety = profile.get('exam_anxiety', 3)
        anxiety_penalty = (anxiety - 3) * 0.15
        difficulty = base_score - anxiety_penalty
        return max(1.0, min(5.0, difficulty))

    def _generate_study_plan(self, user_id: str, profile: Dict, conn: sqlite3.Connection):
        days_left = profile.get('days_to_exam', 30)
        hours_week = profile.get('study_hours_week', 10)
        total_hours = (days_left / 7) * hours_week
        urgency = profile.get('study_urgency', 'Normale')

        custom_topics_json = profile.get('custom_topics', '[]')
        custom_topics = json.loads(custom_topics_json) if custom_topics_json else []
        if not custom_topics:
            custom_topics = ["Concetti fondamentali", "Teoria di base", "Esercizi pratici", "Argomenti avanzati"]

        plan = []
        for i, topic in enumerate(custom_topics):
            priority = 1 if i < 3 else 2 if i < 6 else 3
            hours = 10 / (priority * 0.5)
            plan.append({'name': topic, 'hours': hours, 'priority': priority})

        if urgency in ['Emergenza', 'Molto intensa']:
            plan = [t for t in plan if t['priority'] == 1]

        plan_hours = sum(t['hours'] for t in plan)
        if plan_hours > total_hours and total_hours > 0:
            scale = total_hours / plan_hours
            for topic in plan:
                topic['hours'] *= scale

        c = conn.cursor()
        for topic in plan:
            pid = hashlib.md5((user_id + topic['name']).encode()).hexdigest()[:16]
            c.execute('''INSERT OR REPLACE INTO study_plan
                         (plan_id, user_id, topic, hours_allocated, priority, completed, created_at)
                         VALUES (?,?,?,?,?,?,?)''',
                      (pid, user_id, topic['name'], topic['hours'], topic['priority'], 0, datetime.now().isoformat()))

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM user_profile WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_study_plan(self, user_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""SELECT topic, hours_allocated, priority, completed
                     FROM study_plan WHERE user_id=? ORDER BY priority""", (user_id,))
        rows = c.fetchall()
        conn.close()
        return [{'topic': r[0], 'hours': r[1], 'priority': r[2], 'completed': r[3]} for r in rows]

    def start_session(self, user_id: str) -> str:
        sid = hashlib.md5((user_id + datetime.now().isoformat()).encode()).hexdigest()[:16]
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO sessions (session_id, user_id, start_time, questions_count, points_earned)
                     VALUES (?, ?, ?, 0, 0)''',
                  (sid, user_id, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return sid

    def log_question(self, session_id: str, user_id: str, question_data: Dict):
        qid = hashlib.md5((session_id + datetime.now().isoformat()).encode()).hexdigest()[:16]
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?)''',
                  (qid, session_id, user_id, datetime.now().isoformat(),
                   question_data.get('text', ''),
                   question_data.get('help_seeking_type', 'unknown'),
                   question_data.get('topic', 'general'),
                   question_data.get('showed_attempt', 0),
                   question_data.get('points', 0),
                   question_data.get('difficulty', 3.0)))
        c.execute("UPDATE sessions SET questions_count = questions_count + 1 WHERE session_id=?", (session_id,))
        c.execute("UPDATE user_profile SET total_points = total_points + ? WHERE user_id=?",
                  (question_data.get('points', 0), user_id))
        conn.commit()
        conn.close()

    def update_knowledge_state(self, user_id: str, topic: str, mastery_delta: float):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM knowledge_state WHERE user_id=? AND topic=?", (user_id, topic))
        existing = c.fetchone()
        if existing:
            new_mastery = min(1.0, existing[3] + mastery_delta)
            c.execute('''UPDATE knowledge_state
                         SET mastery_level=?, interactions_count=interactions_count+1,
                             last_interaction=?
                         WHERE user_id=? AND topic=?''',
                      (new_mastery, datetime.now().isoformat(), user_id, topic))
        else:
            c.execute('''INSERT INTO knowledge_state
                         (user_id, topic, mastery_level, interactions_count, last_interaction)
                         VALUES (?, ?, ?, 1, ?)''',
                      (user_id, topic, max(0, mastery_delta), datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_stats(self, user_id: str) -> Dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?", (user_id,))
        total_sessions = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM questions WHERE user_id=?", (user_id,))
        total_questions = c.fetchone()[0]
        c.execute("SELECT total_points FROM user_profile WHERE user_id=?", (user_id,))
        total_points = c.fetchone()[0]
        c.execute("""SELECT help_seeking_type, COUNT(*) FROM questions 
                     WHERE user_id=? GROUP BY help_seeking_type""", (user_id,))
        help_seeking = dict(c.fetchall())
        c.execute("SELECT topic, mastery_level FROM knowledge_state WHERE user_id=?", (user_id,))
        knowledge = dict(c.fetchall())
        conn.close()
        return {
            'total_sessions': total_sessions,
            'total_questions': total_questions,
            'total_points': total_points,
            'help_seeking': help_seeking,
            'knowledge': knowledge
        }

# ==================== Helper RAG / PDF ====================
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
def clean_text(s: str) -> str:
    return CTRL_RE.sub(" ", s).strip()

# ==================== CACHE MANAGER ====================
class StructureCache:
    """Cache intelligente per strutture PDF analizzate"""

    def __init__(self, cache_file=STRUCTURE_CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.lock = threading.Lock()

    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Cache load failed: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Cache save failed: {e}")

    def get_hash(self, text: str) -> str:
        """Genera hash per testo (per identificare pagine già analizzate)"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[Dict]:
        """Recupera struttura dalla cache"""
        text_hash = self.get_hash(text)
        return self.cache.get(text_hash)

    def set(self, text: str, structure: Dict):
        """Salva struttura in cache"""
        with self.lock:
            text_hash = self.get_hash(text)
            self.cache[text_hash] = structure
            self._save_cache()

    def clear(self):
        """Pulisci cache"""
        with self.lock:
            self.cache = {}
            self._save_cache()

# ========== PDF EXTRACTION AVANZATO CON OTTIMIZZAZIONI ===========
class EnhancedPDFProcessor:
    """Processore PDF intelligente con fallback Vision e ottimizzazioni avanzate"""

    def __init__(self, skip_structure_analysis=False, use_fast_model=False):
        # Scegli modello per analisi struttura
        analysis_model = FAST_ANALYSIS_MODEL if use_fast_model else MAIN_MODEL

        # GPU offload massimo
        self.main_llm = ChatOllama(
            model=analysis_model,
            temperature=0.1,
            num_ctx=16384,
            num_gpu=NUM_GPU,
            num_thread=NUM_THREAD
        )
        self.vision_llm = ChatOllama(
            model=VISION_MODEL,
            temperature=0.4,
            num_ctx=32000,  # Ridotto a limite sicuro (era 128K)
            num_gpu=NUM_GPU,
            num_thread=NUM_THREAD
        )
        self.skip_structure = skip_structure_analysis
        self.cache = StructureCache()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.use_fast_model = use_fast_model

    def is_scanned_page(self, page, threshold=50) -> bool:
        """Verifica se una pagina è scannerizzata (poco testo estraibile)"""
        if HAS_PYMUPDF:
            text = page.get_text("text", sort=True) or ""
            return len(text.strip()) < threshold
        return False

    def extract_page_with_vision(self, page, page_num: int, source: str) -> Document:
        """Usa Llama Vision per estrarre contenuto da pagine scannerizzate"""
        if not HAS_PYMUPDF or not HAS_PIL:
            return None

        try:
            # Converti pagina in immagine
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
            img_data = pix.tobytes("png")

            # Salva temporaneamente
            temp_path = os.path.join(tempfile.gettempdir(), f"page_{page_num}.png")
            with open(temp_path, "wb") as f:
                f.write(img_data)

            # Analizza con Vision
            prompt = f"""Analizza questa pagina di dispensa accademica (pagina {page_num}).

Estrai in formato Markdown:
1. TITOLI e SOTTOTITOLI (usa #, ##, ###)
2. TUTTO il testo leggibile
3. FORMULE matematiche in LaTeX (usa $...$ o $$...$$)
4. DEFINIZIONI e TEOREMI (evidenzia con **bold**)
5. DIAGRAMMI: descrivi cosa rappresentano

Sii accurato e preserva la struttura gerarchica!"""

            # Nota: Llama Vision in Ollama supporta immagini via file path
            response = self.vision_llm.invoke([
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"file://{temp_path}"}
            ])

            content = getattr(response, "content", str(response))

            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass

            return Document(
                page_content=content,
                metadata={
                    "source": source,
                    "page": page_num,
                    "extraction_method": "vision",
                    "type": "scanned"
                }
            )
        except Exception as e:
            logging.warning(f"Vision extraction failed for page {page_num}: {e}")
            return None

    def extract_structure_from_text(self, text: str) -> Dict:
        """Estrae struttura gerarchica (titoli, definizioni, teoremi) da testo"""
        prompt = f"""Analizza questo testo di dispensa e estrai la struttura in JSON:

```
{text[:4000]}
```

Restituisci SOLO JSON valido con questa struttura:
{{
  "main_title": "titolo principale se presente, altrimenti null",
  "sections": [
    {{
      "title": "nome sezione",
      "level": 1-3,
      "type": "section|definition|theorem|lemma|corollary|example|exercise",
      "content_summary": "breve riassunto (max 100 char)"
    }}
  ],
  "formulas": ["formula1_latex", "formula2_latex"],
  "key_concepts": ["concetto1", "concetto2"]
}}

IMPORTANTE: Restituisci SOLO il JSON, nient'altro!"""

        try:
            response = self.main_llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            # Estrai JSON dal response (potrebbe avere markdown wrapper)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {}
        except Exception as e:
            logging.warning(f"Structure extraction failed: {e}")
            return {}

    def calculate_optimal_batch_size(self, pages_data: List[Tuple[int, str]]) -> int:
        """Calcola batch size ottimale in base alla lunghezza media delle pagine"""
        if not ADAPTIVE_BATCHING or not pages_data:
            return BATCH_SIZE_MIN

        # Calcola lunghezza media
        avg_length = sum(len(text) for _, text in pages_data) / len(pages_data)

        # Batch size inversamente proporzionale alla lunghezza
        if avg_length < 500:
            return BATCH_SIZE_MAX  # Pagine corte → batch grandi
        elif avg_length < 1500:
            return 6  # Pagine medie → batch medi
        else:
            return BATCH_SIZE_MIN  # Pagine lunghe → batch piccoli

    def extract_structure_batch_parallel(self, pages_data: List[Tuple[int, str]],
                                        progress_callback=None) -> Dict[int, Dict]:
        """Analisi struttura BATCH PARALLELO con caching

        Args:
            pages_data: lista di tuple (page_num, text)
            progress_callback: funzione callback per progress (opzionale)

        Returns:
            dict {page_num: structure_dict}
        """
        if not pages_data:
            return {}

        results = {}
        to_process = []

        # Check cache prima
        for page_num, text in pages_data:
            cached = self.cache.get(text)
            if cached:
                results[page_num] = cached
            else:
                to_process.append((page_num, text))

        # Se tutto in cache, ritorna subito
        if not to_process:
            return results

        # Calcola batch size ottimale
        optimal_batch_size = self.calculate_optimal_batch_size(to_process)

        # Processa in batch con parallelizzazione
        futures = []
        total_batches = (len(to_process) + optimal_batch_size - 1) // optimal_batch_size

        for batch_idx in range(0, len(to_process), optimal_batch_size):
            batch = to_process[batch_idx:batch_idx + optimal_batch_size]
            future = self.executor.submit(self._process_single_batch, batch)
            futures.append((future, batch))

        # Raccoglie risultati man mano che arrivano
        completed = 0
        for future, batch in futures:
            try:
                batch_results = future.result(timeout=60)  # 60s timeout per batch
                results.update(batch_results)

                # Salva in cache
                for page_num, text in batch:
                    if page_num in batch_results:
                        self.cache.set(text, batch_results[page_num])

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_batches)

            except Exception as e:
                logging.error(f"Batch processing failed: {e}")

        return results

    def _sanitize_json_string(self, json_str: str) -> str:
        """Sanitizza JSON rimuovendo escape sequences invalide e caratteri problematici"""
        # Fix backslash escape comuni che causano problemi
        json_str = json_str.replace('\\n', ' ')  # Newline
        json_str = json_str.replace('\\r', ' ')  # Carriage return
        json_str = json_str.replace('\\t', ' ')  # Tab

        # Fix escape sequences matematiche comuni che causano errori
        # Queste sono tipiche in testo matematico ma invalide in JSON
        json_str = json_str.replace('\\alpha', 'alpha')
        json_str = json_str.replace('\\beta', 'beta')
        json_str = json_str.replace('\\gamma', 'gamma')
        json_str = json_str.replace('\\delta', 'delta')
        json_str = json_str.replace('\\epsilon', 'epsilon')
        json_str = json_str.replace('\\theta', 'theta')
        json_str = json_str.replace('\\lambda', 'lambda')
        json_str = json_str.replace('\\mu', 'mu')
        json_str = json_str.replace('\\sigma', 'sigma')
        json_str = json_str.replace('\\pi', 'pi')
        json_str = json_str.replace('\\sum', 'sum')
        json_str = json_str.replace('\\int', 'int')
        json_str = json_str.replace('\\partial', 'partial')
        json_str = json_str.replace('\\nabla', 'nabla')
        json_str = json_str.replace('\\infty', 'infinity')
        json_str = json_str.replace('\\forall', 'forall')
        json_str = json_str.replace('\\exists', 'exists')

        # Fix generic invalid backslash (ultima risorsa - molto aggressivo)
        # Rimuove backslash non seguiti da caratteri JSON-validi (", /, b, f, n, r, t, u)
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', '', json_str)

        return json_str

    def _process_single_batch(self, pages_data: List[Tuple[int, str]]) -> Dict[int, Dict]:
        """Processa un singolo batch (chiamato in thread separato)"""
        if not pages_data:
            return {}

        # Costruisci prompt batch ottimizzato
        pages_text = ""
        for page_num, text in pages_data:
            # Limita dinamicamente in base al numero di pagine nel batch
            max_chars = min(1500, 6000 // len(pages_data))  # Ridotto ulteriormente

            # Pulizia MOLTO aggressiva del testo per evitare problemi JSON
            clean = text[:max_chars]
            # Rimuovi TUTTI i backslash prima di includere nel prompt
            clean = clean.replace('\\', ' ')
            # Converti virgolette in apostrofi
            clean = clean.replace('"', "'")
            clean = clean.replace('"', "'")
            clean = clean.replace('"', "'")
            # Rimuovi newline e caratteri di controllo
            clean = clean.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Rimuovi caratteri unicode problematici
            clean = re.sub(r'[^\x20-\x7E\u00C0-\u024F]', ' ', clean)
            # Comprimi spazi multipli
            clean = re.sub(r'\s+', ' ', clean).strip()

            pages_text += f"\n\n=== PAGINA {page_num} ===\n{clean}\n"

        prompt = f"""Analizza queste {len(pages_data)} pagine di dispensa ed estrai la struttura per OGNUNA.

{pages_text}

CRITICO: Restituisci SOLO un JSON array VALIDO.
- NON usare backslash (\\) nel JSON
- NON usare simboli matematici LaTeX
- Usa solo lettere, numeri, spazi, apostrofi
- Esempio JSON VALIDO:
[
  {{
    "page": 1,
    "main_title": "Titolo principale",
    "sections": [{{"title": "Sezione 1", "level": 1, "type": "section", "content_summary": "Breve sommario"}}],
    "formulas": ["formula semplificata"],
    "key_concepts": ["concetto 1", "concetto 2"]
  }}
]

Restituisci SOLO il JSON array, nient'altro!"""

        try:
            response = self.main_llm.invoke(prompt)
            content = getattr(response, "content", str(response))

            # Rimozione robusta di wrapper markdown
            content = content.strip()
            content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
            content = content.strip()

            # Estrai JSON array con pattern più robusto
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)

                # SANITIZZA il JSON prima di parsarlo
                json_str = self._sanitize_json_string(json_str)

                try:
                    results = json.loads(json_str)

                    # Validazione: deve essere una lista
                    if not isinstance(results, list):
                        logging.warning(f"Batch response non è una lista: {type(results)}")
                        return self._fallback_single_page_processing(pages_data)

                    # Converti in dict {page_num: structure}
                    output = {}
                    for idx, r in enumerate(results):
                        if isinstance(r, dict):
                            page_num = r.get('page', pages_data[idx][0] if idx < len(pages_data) else idx + 1)
                            output[page_num] = r

                    return output

                except json.JSONDecodeError as je:
                    logging.warning(f"JSON decode failed after sanitization: {je}")
                    # Fallback immediato
                    return self._fallback_single_page_processing(pages_data)
            else:
                logging.warning(f"No JSON array found in response")
                return self._fallback_single_page_processing(pages_data)

        except Exception as e:
            logging.error(f"Single batch processing failed: {e}")
            return self._fallback_single_page_processing(pages_data)

    def _fallback_single_page_processing(self, pages_data: List[Tuple[int, str]]) -> Dict[int, Dict]:
        """Fallback: processa pagine una alla volta se batch fallisce"""
        logging.info(f"Fallback: processing {len(pages_data)} pages individually")
        results = {}

        for page_num, text in pages_data:
            try:
                # Usa il metodo single-page esistente
                structure = self.extract_structure_from_text(text)
                if structure:
                    results[page_num] = structure
            except Exception as e:
                logging.warning(f"Fallback processing failed for page {page_num}: {e}")
                # Struttura minima di fallback
                results[page_num] = {
                    "page": page_num,
                    "main_title": None,
                    "sections": [],
                    "formulas": [],
                    "key_concepts": []
                }

        return results

    def extract_documents(self, pdf_bytes: bytes, source: str) -> Tuple[List[Document], Dict]:
        """Estrazione intelligente con fallback Vision"""
        docs = []
        structure = {"pages": [], "has_scanned_pages": False, "knowledge_graph": {}}

        if HAS_PYMUPDF:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Batch processing per analisi struttura (più efficiente!)
            pages_for_batch_analysis = []

            for i, page in enumerate(doc, start=1):
                # Prova estrazione testuale standard
                txt = page.get_text("text", sort=True) or ""
                txt = clean_text(txt)

                is_scanned = self.is_scanned_page(page)

                if is_scanned:
                    # Usa Vision
                    structure["has_scanned_pages"] = True
                    vision_doc = self.extract_page_with_vision(page, i, source)
                    if vision_doc:
                        docs.append(vision_doc)
                        txt = vision_doc.page_content
                    else:
                        # Fallback: usa il poco testo disponibile
                        if txt.strip():
                            docs.append(Document(
                                page_content=txt,
                                metadata={"source": source, "page": i, "extraction_method": "text_fallback"}
                            ))
                else:
                    # Estrazione testuale normale
                    if txt.strip():
                        docs.append(Document(
                            page_content=txt,
                            metadata={"source": source, "page": i, "extraction_method": "text"}
                        ))

                # Accumula pagine per analisi batch (se abbastanza testo e non skip)
                if not self.skip_structure and len(txt) > 100:
                    pages_for_batch_analysis.append((i, txt))

            # Analisi struttura in BATCH PARALLELO (10-20x più veloce!)
            if pages_for_batch_analysis and not self.skip_structure:
                # Progress callback per tracking
                def progress_callback(completed, total):
                    pass  # Può essere usato per aggiornare UI

                batch_results = self.extract_structure_batch_parallel(
                    pages_for_batch_analysis,
                    progress_callback=progress_callback
                )

                # Aggiungi risultati batch a structure (ordinati per page_num)
                for page_num in sorted(batch_results.keys()):
                    structure["pages"].append({
                        "page_num": page_num,
                        "structure": batch_results[page_num]
                    })
        else:
            # Fallback PyPDF2 (senza Vision)
            rdr = PdfReader(io.BytesIO(pdf_bytes))
            pages_for_batch_analysis = []

            for i, p in enumerate(rdr.pages, start=1):
                txt = (p.extract_text() or "")
                txt = clean_text(txt)
                if txt.strip():
                    docs.append(Document(
                        page_content=txt,
                        metadata={"source": source, "page": i, "extraction_method": "pypdf2"}
                    ))
                    if not self.skip_structure and len(txt) > 100:
                        pages_for_batch_analysis.append((i, txt))

            # Analisi batch anche per PyPDF2 (parallelo)
            if pages_for_batch_analysis and not self.skip_structure:
                batch_results = self.extract_structure_batch_parallel(
                    pages_for_batch_analysis,
                    progress_callback=None
                )

                for page_num in sorted(batch_results.keys()):
                    structure["pages"].append({
                        "page_num": page_num,
                        "structure": batch_results[page_num]
                    })

        return docs, structure

def extract_with_pymupdf(b: bytes, source: str) -> List[Document]:
    """Wrapper legacy - mantenuto per compatibilità"""
    processor = EnhancedPDFProcessor()
    docs, _ = processor.extract_documents(b, source)
    return docs

def process_pdfs(files, fast_mode=True, use_fast_model=False) -> Tuple[Chroma, int, Dict]:
    """Processa PDF con estrazione intelligente e knowledge graph

    Args:
        files: lista file PDF
        fast_mode: se True, salta analisi struttura (MOLTO più veloce)
        use_fast_model: se True, usa modello lightweight per analisi (più veloce ma meno accurato)
    """
    processor = EnhancedPDFProcessor(
        skip_structure_analysis=fast_mode,
        use_fast_model=use_fast_model
    )
    all_docs = []
    all_structures = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    analysis_info = st.empty()

    for idx, f in enumerate(files):
        status_text.text(f"📄 Processando {f.name}... ({idx+1}/{len(files)})")
        if not fast_mode:
            model_name = FAST_ANALYSIS_MODEL if use_fast_model else MAIN_MODEL
            analysis_info.info(f"🔬 Analisi struttura con {model_name.split('/')[-1]}")

        b = f.read()
        docs, structure = processor.extract_documents(b, getattr(f, "name", "PDF"))
        all_docs.extend(docs)
        all_structures.append({
            "filename": f.name,
            "structure": structure
        })
        progress_bar.progress((idx + 1) / len(files))

    status_text.text("✂️ Chunking intelligente...")

    # Chunking context-aware (più intelligente)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Chunks più grandi per preservare contesto
        chunk_overlap=300,  # Overlap maggiore per continuità
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],  # Rispetta struttura markdown
    )

    chunks: List[Document] = []
    for d in all_docs:
        for part in splitter.split_text(d.page_content):
            # Arricchisci metadata con info strutturali
            metadata = d.metadata.copy()
            metadata["chunk_length"] = len(part)
            metadata["has_formula"] = bool(re.search(r'\$.*?\$|\\\[.*?\\\]', part))
            metadata["has_definition"] = bool(re.search(r'\*\*Definizione\*\*|\*\*Teorema\*\*', part, re.IGNORECASE))
            chunks.append(Document(page_content=part, metadata=metadata))

    status_text.text("🧠 Indicizzazione in Chroma...")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=st.session_state.embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="docs",
        client_settings=CHROMA_SETTINGS,
    )
    vs.persist()

    # Salva strutture estratte per future analisi
    # Conta pagine dal numero di documenti estratti (più accurato)
    total_pages_from_docs = len(all_docs)
    total_pages_from_structure = sum(len(s["structure"]["pages"]) for s in all_structures)

    structure_info = {
        "processed_at": datetime.now().isoformat(),
        "files": all_structures,
        "total_pages": total_pages_from_docs if fast_mode else total_pages_from_structure,
        "total_docs": total_pages_from_docs,
        "has_scanned": any(s["structure"]["has_scanned_pages"] for s in all_structures)
    }

    progress_bar.empty()
    status_text.empty()
    analysis_info.empty()

    return vs, len(chunks), structure_info

def classify_help_seeking(question: str) -> Tuple[str, int]:
    q = question.lower()
    executive = [r'dammi (la |il )?soluzione', r'risolvi', r'fai( tu)?', r'^come si fa', r'qual[eè] (la|il) risultato', r'rispondi tu']
    instrumental = [r'(non )?capisco', r'spiega', r'perch[eé]', r'ho provato.*sbaglio', r'questo passaggio', r'come funziona']
    optimal = ['ho provato', 'ho fatto', 'secondo me', 'credo che', 'penso']
    for p in executive:
        if re.search(p, q): return ('executive', -1 if st.session_state.get('executive_count', 0) >= 3 else 0)
    has_attempt = any(x in q for x in optimal)
    for p in instrumental:
        if re.search(p, q): return ('instrumental', 2 if has_attempt else 1)
    return ('neutral', 0)

def extract_topic(text: str, user_topics: List[str]) -> str:
    t = text.lower()
    for topic in user_topics:
        for w in topic.lower().split():
            if len(w) > 3 and w in t:
                return topic
    return "general"

# ==================== Init stato app ====================
db = EnhancedDatabase()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
if "pdf_structures" not in st.session_state:
    st.session_state.pdf_structures = None
if "executive_count" not in st.session_state:
    st.session_state.executive_count = 0
if "struggle_count" not in st.session_state:
    st.session_state.struggle_count = 0
if "last_question_time" not in st.session_state:
    st.session_state.last_question_time = datetime.now()

# prova a caricare user corrente salvato
if st.session_state.user_id is None:
    persisted = load_current_user()
    if persisted and db.get_user_profile(persisted):
        st.session_state.user_id = persisted
        st.session_state.session_id = db.start_session(persisted)

# ==================== Schermata Accesso / Onboarding ====================
def onboarding_form():
    st.header("➕ Crea nuovo profilo")
    with st.form("enhanced_onboarding"):
        col1, col2 = st.columns(2)
        with col1:
            nickname = st.text_input("Nome*", placeholder="Marco")
            education_level = st.selectbox("Livello*", ["Scuola Superiore","Università - Triennale","Università - Magistrale","Dottorato","Autodidatta","Altro"])
            study_hours_week = st.slider("Ore/settimana", 0, 40, 10, 5)
            learning_style = st.radio("Imparo meglio con:", ["Teoria dettagliata","Esempi pratici","Visualizzazioni/grafici","Video","Esercizi","Mix di tutto"])
        with col2:
            subject_area = st.selectbox("Area*", ["Matematica","Fisica","Chimica","Biologia","Informatica","Statistica","Ingegneria","Economia","Medicina","Diritto","Filosofia","Storia","Letteratura","Lingue","Arte","Altro"])
            course_name = st.text_input("Corso/Esame*", placeholder="Es: Analisi, Storia Moderna, Programmazione…")
            goal = st.selectbox("Obiettivo*", ["Preparazione esame","Recupero debito","Approfondimento personale","Tesi/Progetto","Certificazione","Altro"])
            exam_date = st.date_input("Scadenza", value=datetime.now()+timedelta(days=30))
        topics_input = st.text_area("Argomenti* (separa con virgole)", placeholder="Derivate, Integrali, Limiti…", height=90)
        prior_knowledge = st.text_area("Cosa sai già? (opzionale)", height=70)
        colA, colB = st.columns(2)
        with colA:
            self_assessment_theory = st.slider("📖 Teoria (1-5)", 1, 5, 3)
            exam_anxiety = st.slider("😰 Ansia (1-5)", 1, 5, 3)
        with colB:
            self_assessment_exercises = st.slider("✏️ Pratica (1-5)", 1, 5, 3)
            difficulty_preference = st.radio("Difficoltà:", ["Parti da facili","Subito difficili","Mix","Adatta tu"])

        submitted = st.form_submit_button("🚀 Crea profilo", type="primary", use_container_width=True)
        if submitted:
            if not nickname or not course_name or not topics_input:
                st.error("Compila: Nome, Corso/Esame e Argomenti.")
                return
            topics = [t.strip() for t in topics_input.split(",") if t.strip()]
            days_to_exam = (exam_date - datetime.now().date()).days
            profile_data = {
                'nickname': nickname,
                'subject_area': subject_area,
                'course_name': course_name,
                'education_level': education_level,
                'goal': goal,
                'exam_date': exam_date.isoformat(),
                'days_to_exam': days_to_exam,
                'target_grade': 27,
                'study_urgency': "Normale",
                'study_hours_week': study_hours_week,
                'learning_style': learning_style,
                'self_assessment_theory': self_assessment_theory,
                'self_assessment_exercises': self_assessment_exercises,
                'exam_anxiety': exam_anxiety,
                'difficulty_preference': difficulty_preference,
                'custom_topics': json.dumps(topics),
                'prior_knowledge': prior_knowledge
            }
            user_id = db.create_user(profile_data)
            st.session_state.user_id = user_id
            st.session_state.session_id = db.start_session(user_id)
            save_current_user(user_id)
            st.success(f"Profilo creato! Il tuo user_id è **{user_id}** (salvato).")
            st.balloons()
            st.rerun()

def login_form():
    st.header("🔐 Accedi con user_id esistente")
    uid = st.text_input("user_id (16 caratteri)", value="")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Accedi", type="primary", use_container_width=True):
            if not uid:
                st.error("Inserisci un user_id.")
            else:
                prof = db.get_user_profile(uid)
                if prof:
                    st.session_state.user_id = uid
                    st.session_state.session_id = db.start_session(uid)
                    save_current_user(uid)
                    st.success("Accesso eseguito!")
                    st.rerun()
                else:
                    st.error("user_id non trovato nel database.")
    with col2:
        if st.button("Usa utente salvato", use_container_width=True):
            persisted = load_current_user()
            if persisted and db.get_user_profile(persisted):
                st.session_state.user_id = persisted
                st.session_state.session_id = db.start_session(persisted)
                st.success("Utente salvato caricato!")
                st.rerun()
            else:
                st.warning("Nessun utente salvato trovato.")

if st.session_state.user_id is None:
    st.title("👋 Benvenuto nel Tutor AI")
    tab1, tab2 = st.tabs(["🔐 Accedi", "➕ Nuovo Profilo"])
    with tab1:
        login_form()
    with tab2:
        onboarding_form()
    st.stop()

# ==================== GENERATORI CONTENUTI ====================
class ContentGenerator:
    """Genera flashcards, quiz e riassunti dal materiale"""

    def __init__(self, llm_model=MAIN_MODEL):
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.2,
            num_ctx=8192,  # Ridotto per generazione veloce
            num_gpu=NUM_GPU,
            num_thread=NUM_THREAD
        )

    def generate_flashcards(self, docs: List[Document], num_cards: int = 10) -> List[Dict]:
        """Genera flashcards da documenti"""
        # Concatena testo da più documenti
        text_sample = "\n\n".join([d.page_content for d in docs[:5]])[:8000]

        prompt = f"""Dal seguente materiale didattico, genera {num_cards} flashcards per lo studio.

MATERIALE:
```
{text_sample}
```

Genera SOLO un JSON array con questa struttura:
[
  {{
    "front": "Domanda o concetto da ricordare",
    "back": "Risposta o spiegazione dettagliata",
    "difficulty": 1-5,
    "tags": ["tag1", "tag2"]
  }}
]

Criteri:
- Copri concetti chiave, definizioni, formule importanti
- Varia la difficoltà
- Domande chiare e risposte complete
- SOLO JSON valido, nient'altro!"""

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return []
        except Exception as e:
            logging.error(f"Flashcard generation failed: {e}")
            return []

    def generate_quiz(self, docs: List[Document], num_questions: int = 5, difficulty: int = 3) -> List[Dict]:
        """Genera quiz a risposta multipla"""
        text_sample = "\n\n".join([d.page_content for d in docs[:5]])[:8000]

        prompt = f"""Dal seguente materiale, genera {num_questions} domande quiz a risposta multipla (difficoltà {difficulty}/5).

MATERIALE:
```
{text_sample}
```

Genera SOLO un JSON array:
[
  {{
    "question": "Testo domanda",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct": 0,
    "explanation": "Spiegazione della risposta corretta",
    "difficulty": {difficulty},
    "topic": "argomento"
  }}
]

Criteri:
- Domande che testano comprensione profonda
- Distractors plausibili ma distinguibili
- Spiegazione chiara del perché la risposta è corretta
- SOLO JSON valido!"""

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return []
        except Exception as e:
            logging.error(f"Quiz generation failed: {e}")
            return []

    def extract_concepts_map(self, structure_info: Dict) -> Dict:
        """Estrae mappa concetti dai documenti processati"""
        concepts = {}
        for file_info in structure_info.get('files', []):
            for page_info in file_info['structure']['pages']:
                page_struct = page_info['structure']
                key_concepts = page_struct.get('key_concepts', [])
                for concept in key_concepts:
                    if concept not in concepts:
                        concepts[concept] = {
                            'pages': [],
                            'type': 'concept',
                            'related': []
                        }
                    concepts[concept]['pages'].append({
                        'file': file_info['filename'],
                        'page': page_info['page_num']
                    })
        return concepts

# ==================== MENU ====================
page = st.sidebar.radio("📍 Menu", ["💬 Chat", "📊 Dashboard", "🃏 Flashcards", "🎯 Quiz", "⚙️ Impostazioni"])

# ==================== DASHBOARD ====================
def show_dashboard():
    profile = db.get_user_profile(st.session_state.user_id)
    stats = db.get_stats(st.session_state.user_id)
    study_plan = db.get_study_plan(st.session_state.user_id)

    st.title(f"👋 Ciao, {profile['nickname']}!")
    st.caption(f"📚 {profile['subject_area']} — {profile['course_name']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Punti", stats['total_points'])
    c2.metric("📚 Sessioni", stats['total_sessions'])
    c3.metric("💬 Domande", stats['total_questions'])
    c4.metric("⏰ Giorni", profile['days_to_exam'])

    tab1, tab2 = st.tabs(["📅 Piano Studio", "🗺️ Knowledge Map"])
    with tab1:
        st.subheader("Piano")
        if study_plan:
            for i, item in enumerate(study_plan, 1):
                status = "✅" if item['completed'] else "⏳"
                prio = {1:"🔴",2:"🟡",3:"🟢"}.get(item['priority'],"⚪")
                colA, colB, colC = st.columns([3,1,1])
                colA.write(f"{status} {prio} **{item['topic']}**")
                colB.caption(f"{item['hours']:.1f} h")
                if colC.button("✓", key=f"done_{i}"):
                    st.info("Segna completato (demo).")
        else:
            st.info("Nessun piano ancora. Crea/aggiorna un profilo.")
    with tab2:
        st.subheader("Padronanza (demo)")
        know = stats['knowledge']
        if know:
            df = pd.DataFrame([{'Argomento': k, 'Padronanza': v*100} for k, v in know.items()])
            fig = px.bar(df, x='Argomento', y='Padronanza', range_y=[0,100], color='Padronanza', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ancora vuota: crescerà man mano che studi.")

if page == "📊 Dashboard":
    show_dashboard()
    st.stop()

# ==================== FLASHCARDS ====================
if page == "🃏 Flashcards":
    st.title("🃏 Flashcards Generate dal Materiale")

    if st.session_state.vector_store is None:
        st.warning("⚠️ Carica prima dei PDF dalla sezione Chat!")
        st.stop()

    if "flashcards" not in st.session_state:
        st.session_state.flashcards = []
    if "current_card" not in st.session_state:
        st.session_state.current_card = 0
    if "show_back" not in st.session_state:
        st.session_state.show_back = False

    col1, col2 = st.columns([3, 1])
    with col1:
        num_cards = st.slider("Quante flashcards?", 5, 30, 10)
    with col2:
        if st.button("🎲 Genera", type="primary", use_container_width=True):
            with st.spinner("🧠 Generando flashcards..."):
                generator = ContentGenerator()
                # Prendi sample random di documenti dal vector store
                all_docs = st.session_state.vector_store.get()
                if all_docs and 'documents' in all_docs:
                    sample_size = min(10, len(all_docs['documents']))
                    sample_docs = [
                        Document(page_content=all_docs['documents'][i],
                                metadata=all_docs['metadatas'][i] if 'metadatas' in all_docs else {})
                        for i in range(sample_size)
                    ]
                    cards = generator.generate_flashcards(sample_docs, num_cards)
                    st.session_state.flashcards = cards
                    st.session_state.current_card = 0
                    st.session_state.show_back = False
                    st.success(f"✅ Generate {len(cards)} flashcards!")
                    st.rerun()

    if st.session_state.flashcards:
        total = len(st.session_state.flashcards)
        current = st.session_state.current_card

        # Navigazione
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("◀ Precedente", disabled=current == 0):
                st.session_state.current_card -= 1
                st.session_state.show_back = False
                st.rerun()
        with col2:
            st.markdown(f"<h3 style='text-align: center'>Card {current+1}/{total}</h3>", unsafe_allow_html=True)
        with col3:
            if st.button("Successiva ▶", disabled=current >= total-1):
                st.session_state.current_card += 1
                st.session_state.show_back = False
                st.rerun()

        # Card display
        card = st.session_state.flashcards[current]
        difficulty_color = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"}
        st.markdown(f"{difficulty_color.get(card.get('difficulty', 3), '⚪')} Difficoltà: {card.get('difficulty', 3)}/5")

        if st.session_state.show_back:
            # Mostra risposta
            st.markdown("### ❓ Domanda:")
            st.info(card['front'])
            st.markdown("### ✅ Risposta:")
            st.success(card['back'])
            if card.get('tags'):
                st.caption(f"🏷️ Tags: {', '.join(card['tags'])}")
            if st.button("🔄 Nascondi risposta", use_container_width=True):
                st.session_state.show_back = False
                st.rerun()
        else:
            # Mostra solo domanda
            st.markdown("### ❓ Domanda:")
            st.markdown(f"<div style='background: #1e1e1e; padding: 30px; border-radius: 10px; font-size: 1.2em'>{card['front']}</div>", unsafe_allow_html=True)
            st.markdown("")
            if st.button("👁️ Mostra risposta", type="primary", use_container_width=True):
                st.session_state.show_back = True
                st.rerun()
    else:
        st.info("👆 Clicca su 'Genera' per creare le flashcards dal materiale caricato!")

    st.stop()

# ==================== QUIZ ====================
if page == "🎯 Quiz":
    st.title("🎯 Quiz Interattivo")

    if st.session_state.vector_store is None:
        st.warning("⚠️ Carica prima dei PDF dalla sezione Chat!")
        st.stop()

    profile = db.get_user_profile(st.session_state.user_id)
    difficulty_level = int(profile['difficulty_level'])

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_current" not in st.session_state:
        st.session_state.quiz_current = 0
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    col1, col2, col3 = st.columns(3)
    with col1:
        num_questions = st.slider("N° domande", 3, 15, 5)
    with col2:
        quiz_difficulty = st.slider("Difficoltà", 1, 5, difficulty_level)
    with col3:
        if st.button("🎲 Genera Quiz", type="primary", use_container_width=True):
            with st.spinner("🧪 Generando quiz..."):
                generator = ContentGenerator()
                all_docs = st.session_state.vector_store.get()
                if all_docs and 'documents' in all_docs:
                    sample_size = min(10, len(all_docs['documents']))
                    sample_docs = [
                        Document(page_content=all_docs['documents'][i],
                                metadata=all_docs['metadatas'][i] if 'metadatas' in all_docs else {})
                        for i in range(sample_size)
                    ]
                    questions = generator.generate_quiz(sample_docs, num_questions, quiz_difficulty)
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_current = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.success(f"✅ Generate {len(questions)} domande!")
                    st.rerun()

    if st.session_state.quiz_questions:
        questions = st.session_state.quiz_questions

        if not st.session_state.quiz_submitted:
            # Modalità risposta
            for idx, q in enumerate(questions):
                with st.expander(f"❓ Domanda {idx+1}/{len(questions)}", expanded=idx==0):
                    st.markdown(f"**{q['question']}**")
                    answer = st.radio(
                        "Scegli la risposta:",
                        options=range(len(q['options'])),
                        format_func=lambda x: q['options'][x],
                        key=f"q_{idx}",
                        index=None
                    )
                    if answer is not None:
                        st.session_state.quiz_answers[idx] = answer

            st.markdown("---")
            if st.button("📝 Consegna Quiz", type="primary", use_container_width=True, disabled=len(st.session_state.quiz_answers) < len(questions)):
                st.session_state.quiz_submitted = True
                st.rerun()

        else:
            # Mostra risultati
            correct = 0
            total_points = 0

            for idx, q in enumerate(questions):
                user_answer = st.session_state.quiz_answers.get(idx)
                is_correct = user_answer == q['correct']
                if is_correct:
                    correct += 1
                    points = q.get('difficulty', 3) * 2
                    total_points += points

                with st.expander(f"{'✅' if is_correct else '❌'} Domanda {idx+1}", expanded=not is_correct):
                    st.markdown(f"**{q['question']}**")
                    st.markdown("**Tua risposta:**")
                    if user_answer is not None:
                        st.info(q['options'][user_answer])
                    st.markdown("**Risposta corretta:**")
                    st.success(q['options'][q['correct']])
                    st.markdown("**Spiegazione:**")
                    st.write(q.get('explanation', 'N/A'))

            # Riepilogo
            score = (correct / len(questions)) * 100
            st.markdown("---")
            st.markdown(f"## 🎯 Risultato: {correct}/{len(questions)} ({score:.1f}%)")
            st.metric("Punti guadagnati", total_points)

            # Aggiorna DB
            question_data = {
                'text': f"Quiz con {len(questions)} domande",
                'help_seeking_type': 'quiz',
                'topic': 'quiz_practice',
                'showed_attempt': 1,
                'points': total_points,
                'difficulty': quiz_difficulty
            }
            db.log_question(st.session_state.session_id, st.session_state.user_id, question_data)

            if st.button("🔄 Nuovo Quiz", type="primary"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
    else:
        st.info("👆 Clicca su 'Genera Quiz' per iniziare!")

    st.stop()

# ==================== IMPOSTAZIONI ====================
if page == "⚙️ Impostazioni":
    st.title("⚙️ Impostazioni")
    prof = db.get_user_profile(st.session_state.user_id)
    with st.expander("👤 Profilo"):
        st.json(prof)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset utente corrente"):
            try:
                if os.path.exists(CURRENT_USER_FILE):
                    os.remove(CURRENT_USER_FILE)
            except Exception:
                pass
            st.session_state.user_id = None
            st.session_state.session_id = None
            st.success("Utente corrente azzerato.")
            st.rerun()
    with col2:
        if st.button("🗑️ Reset DB (cancella users.db)"):
            try:
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
                st.success("DB eliminato. Riavvia l'app.")
            except Exception as e:
                st.error(f"Errore: {e}")
    with col3:
        if st.button("🧹 Svuota indice Chroma"):
            try:
                if os.path.isdir(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR)
                st.session_state.vector_store = None
                st.session_state.pdf_structures = None
                st.success("Indice Chroma eliminato. Ricarica i PDF per creare nuovo indice.")
            except Exception as e:
                st.error(f"Errore: {e}")
    st.stop()

# ==================== CHAT (RAG) ====================
st.title("💬 Chat")
profile = db.get_user_profile(st.session_state.user_id)
subject_area = profile['subject_area']
difficulty_level = profile['difficulty_level']
user_topics = json.loads(profile.get('custom_topics', '[]'))

with st.sidebar:
    st.metric("🎚️ Livello", f"{difficulty_level:.1f}/5")
    st.caption(f"{['Principiante','Base','Intermedio','Avanzato','Esperto'][int(difficulty_level)-1]}")
    st.subheader("📄 Materiale")
    pdf_docs = st.file_uploader("Carica PDF", type="pdf", accept_multiple_files=True)

    # Modalità processing
    fast_mode = st.checkbox("⚡ Modalità Veloce", value=True, help="Salta analisi struttura (raccomandato per PDF grandi)")

    # Opzioni avanzate (solo se modalità completa)
    use_fast_model = False
    if not fast_mode:
        with st.expander("🔧 Opzioni Avanzate"):
            use_fast_model = st.checkbox(
                "Usa modello lightweight per analisi",
                value=False,
                help=f"Usa {FAST_ANALYSIS_MODEL} invece di {MAIN_MODEL.split('/')[-1]} per analisi struttura (3-5x più veloce, leggermente meno accurato)"
            )
            st.caption(f"📊 Batch size: {BATCH_SIZE_MIN}-{BATCH_SIZE_MAX} (dinamico)")
            st.caption(f"⚡ Parallelizzazione: {MAX_WORKERS} thread")
            st.caption(f"💾 Cache: attiva")

    if pdf_docs and st.button("Processa 🚀", type="primary"):
        mode_text = "⚡ VELOCE" if fast_mode else ("🔬 COMPLETA + FAST MODEL" if use_fast_model else "🔬 COMPLETA")
        with st.spinner(f"📄 Elaborazione {mode_text} in corso..."):
            vs, count, structures = process_pdfs(pdf_docs, fast_mode=fast_mode, use_fast_model=use_fast_model)
            st.session_state.vector_store = vs
            st.session_state.pdf_structures = structures
            st.success(f"✅ Indicizzati {count} chunks da {structures['total_pages']} pagine!")
            if structures['has_scanned']:
                st.info("🔍 Rilevate pagine scannerizzate → usato Llama Vision!")
            if not fast_mode:
                st.info(f"🧠 Analisi struttura completata con {'modello lightweight' if use_fast_model else 'modello principale'}")

    # Mostra info sui PDF caricati
    if st.session_state.pdf_structures:
        with st.expander("📚 Struttura documenti"):
            for file_info in st.session_state.pdf_structures['files']:
                st.markdown(f"**{file_info['filename']}**")
                st.caption(f"Pagine: {len(file_info['structure']['pages'])}")

    st.markdown("---")
    st.subheader("🔍 Strategia")
    search_mode = st.radio("Retrieval", ["🎯 Singola", "🚀 Multi-Query"], index=0)

# storia chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Chiedi qualcosa sugli appunti indicizzati…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            st.warning("⚠️ Prima carica e indicizza PDF dalla sidebar.")
        else:
            try:
                llm = ChatOllama(
                    model=MAIN_MODEL,
                    temperature=0.6,
                    num_ctx=32000,  # Massimo supportato da Qwen3-Coder-30B (era 128K → troppo!)
                    num_predict=8190,  # Ridotto per velocità
                    num_gpu=NUM_GPU,
                    num_thread=NUM_THREAD
                )

                # Retriever avanzato con filtri intelligenti
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 8,  # Più documenti
                        "fetch_k": 50,  # Pool più grande
                        "lambda_mult": 0.6  # Più diversità
                    }
                )
                with st.spinner("🔎 Cerco nel materiale…"):
                    docs = retriever.get_relevant_documents(prompt)

                if not docs:
                    st.warning("Nessun passaggio rilevante trovato. Prova a riformulare.")
                    reply = "🔍 Non ho trovato informazioni rilevanti."
                else:
                    # Costruisci context
                    unique_docs, seen = [], set()
                    for d in docs:
                        key = (d.metadata.get('source'), d.metadata.get('page'))
                        if key not in seen:
                            seen.add(key)
                            unique_docs.append(d)
                        if len(unique_docs) >= 10:
                            break

                    context = "\n\n---\n\n".join(
                        f"[Doc {i+1} - Pag.{d.metadata.get('page')} — {d.metadata.get('source')}]\n{d.page_content}"
                        for i, d in enumerate(unique_docs)
                    )

                    RAG_PROMPT = ChatPromptTemplate.from_messages([
                        ("system", f"""Sei un tutor esperto di {subject_area}. Livello studente: {int(difficulty_level)}/5.

REGOLE FERREE:
1. Usa ESCLUSIVAMENTE il CONTEXT fornito - non inventare nulla
2. Se l'informazione non è nel CONTEXT, rispondi: "Non trovo questa informazione nel materiale caricato"
3. CITA SEMPRE le fonti: [Doc X, pag. Y] per ogni affermazione
4. Per FORMULE matematiche: usa LaTeX con $ inline $ o $$ blocco $$
5. Per DEFINIZIONI/TEOREMI: usa **grassetto** e struttura chiara
6. Spiega con esempi quando possibile
7. Adatta la complessità al livello {int(difficulty_level)}/5

STILE:
- Chiaro e didattico
- Step-by-step per dimostrazioni
- Evidenzia collegamenti tra concetti"""),
                        ("human", "CONTEXT:\n{context}\n\nDOMANDA: {input}\n\nRispondi in modo completo e strutturato:")
                    ])
                    messages = RAG_PROMPT.format_messages(context=context, input=prompt)
                    res = llm.invoke(messages)
                    reply = getattr(res, "content", str(res))
                    st.markdown(reply)

                    with st.expander(f"📖 Passages usati ({len(unique_docs)})"):
                        for i, d in enumerate(unique_docs, 1):
                            st.markdown(f"**[Doc {i}]** Pag. {d.metadata.get('page')} — *{d.metadata.get('source')}*")
                            st.caption(d.page_content[:400] + "…")
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"❌ Errore: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Errore: {e}"})

logging.getLogger("streamlit").setLevel(logging.ERROR)
