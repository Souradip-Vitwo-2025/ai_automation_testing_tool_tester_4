#!/usr/bin/env python3
"""
GitHub Code + SQL Hybrid Analyzer (Rule-based + RAG + LLM)
- Auto-detects programming language (extension; LLM fallback if unknown)
- Prompts schema source options if SQL found (1-5 as requested)
- Hybrid validation: rule-based checks, RAG similarity lookup, LLM rewrite
- Minimal final output per file: only buggy line numbers and suggested fix
- Supports SQLite schema build, and optional external DB connection strings
- Improved: language-aware SQL extraction for Python, PHP, Java, C, JS, etc.

Enhancements:
- Prints a compact "Buggy Line Numbers" list in final output
- Adds fuzzy column correction for INSERT/UPDATE column names based on schema
"""

import os
import re
import ast
import json
import sqlite3
import difflib
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from github import Github
from sqlalchemy import create_engine, inspect
import sqlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional LLM imports (graceful)
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import AzureChatOpenAI
except Exception:
    PromptTemplate = None
    StrOutputParser = None
    AzureChatOpenAI = None

# -----------------------
# Configuration / Env
# -----------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

gh = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Initialize LLM client if available
llm = None
if AzureChatOpenAI and AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT:
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            azure_deployment=AZURE_DEPLOYMENT,
            api_version="2025-03-01-preview",
            temperature=0.0,
        )
    except Exception:
        llm = None

# Temporary artifacts
TEMP_DB = "temp_schema_db.sqlite3"
TEMP_SQL_FILE = "temp_schema.sql"
REPORTS_DIR = "analysis_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------
# Utilities
# -----------------------
def levenshtein_distance(a: str, b: str) -> int:
    a, b = a or "", b or ""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]

# Common extension -> language map
EXT_TO_LANG = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".php": "PHP",
    ".cpp": "C++",
    ".c": "C",
    ".cs": "C#",
    ".rb": "Ruby",
    ".go": "Go",
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
    ".r": "R",
    ".dart": "Dart",
    ".lua": "Lua",
}

def detect_language_from_extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return EXT_TO_LANG.get(ext, "Unknown")

# LLM fallback for language detection (template)
def llm_detect_language(code_snippet: str) -> str:
    """
    Uses LLM as fallback to return one-word language name.
    Requires llm client (AzureChatOpenAI via langchain_openai).
    """
    if llm is None or PromptTemplate is None:
        return "Unknown"
    try:
        prompt = PromptTemplate(
            input_variables=["code"],
            template=(
                "Identify the programming language of the following code. "
                "Return a single word (e.g., Python, JavaScript, PHP, Java, C++, Unknown).\n\n{code}"
            ),
        )
        chain = prompt | llm | StrOutputParser()
        res = chain.invoke({"code": code_snippet}).strip().splitlines()[0].strip()
        res = re.sub(r"[^A-Za-z0-9\+\#]", "", res)
        return res or "Unknown"
    except Exception:
        return "Unknown"

# -----------------------
# GitHub helpers
# -----------------------
def get_all_paths(repo, path: str = "") -> List[str]:
    paths: List[str] = []
    try:
        contents = repo.get_contents(path)
    except Exception:
        return []
    if isinstance(contents, list):
        for c in contents:
            if c.type == "dir":
                paths.extend(get_all_paths(repo, c.path))
            else:
                paths.append(c.path)
    elif contents:
        paths.append(contents.path)
    return paths


def download_file(repo, file_path: str, local_path: str) -> Optional[str]:
    try:
        obj = repo.get_contents(file_path)
        with open(local_path, "wb") as f:
            f.write(obj.decoded_content)
        return local_path
    except Exception:
        return None


def read_repo_file_text(repo, file_path: str, ref: str = "main") -> Optional[str]:
    try:
        obj = repo.get_contents(file_path, ref=ref)
        return obj.decoded_content.decode("utf-8", errors="ignore")
    except Exception:
        return None

# -----------------------
# Language-aware SQL extraction
# -----------------------
def extract_sql_queries(code: str, language: str = "Unknown") -> List[str]:
    """
    Language-aware SQL extraction. Returns list of SQL snippets found in code.
    Supports Python, PHP, Java, JavaScript, C/C++ (some DB APIs), and a generic fallback.
    """
    if not code:
        return []

    found: List[str] = []

    # Normalize language string
    lang = (language or "Unknown").lower()

    # Helper to add candidate if it looks like SQL
    def add_if_sql(s: str) -> None:
        if not s:
            return
        s2 = s.strip()
        if re.search(
            r"\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bCREATE\b|\bALTER\b|\bDROP\b",
            s2,
            flags=re.I,
        ):
            if s2 not in found:
                found.append(s2)

    try:
        if "python" in lang or "django" in lang or "flask" in lang:
            patterns = [
                r'(?:execute|executemany|execute_query|raw)\s*\(\s*(?:r?["\']{1,3})(.+?)(?:["\']{1,3})',
                r'("""|\'\'\')(.*?)(\1)',
                r'["\']\s*(SELECT .*? FROM .*?)["\']',
            ]
        elif "php" in lang:
            patterns = [
                r'mysqli_query\s*\(.*?,\s*(["\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'->query\s*\(\s*(["\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'->prepare\s*\(\s*(["\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'\$\w*sql\s*=\s*(["\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
            ]
        elif "java" in lang:
            patterns = [
                r'executeQuery\s*\(\s*(["\'])(SELECT .*?)\1',
                r'executeUpdate\s*\(\s*(["\'])(INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'prepareStatement\s*\(\s*(["\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'Statement\s+stmt\s*=\s*conn\.createStatement\(\)\s*;\s*stmt\.execute\s*\(\s*(["\'])(SELECT .*?)\1',
            ]
        elif any(x in lang for x in ("javascript", "node", "ts", "typescript")):
            patterns = [
                r'query\s*\(\s*(["`\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'execute\s*\(\s*(["`\'])(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)\1',
                r'`\s*(SELECT .*?FROM .*?)\s*`',
            ]
        elif any(x in lang for x in ("c", "c++", "cpp")):
            patterns = [
                r'sqlite3_exec\s*\(.*?,\s*"(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)"',
                r'PQexec\s*\(.*?,\s*"(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)"',
                r'mysql_query\s*\(\s*"(SELECT .*?|INSERT .*?|UPDATE .*?|DELETE .*?)"',
            ]
        else:
            patterns = [
                r'(?:query|execute|executemany|prepare|execute_query)\s*\(\s*(?:r?["`\']{1,3})(.+?)(?:["`\']{1,3})',
                r'(["`\'])(SELECT .*? FROM .*?)\1',
                r'(["`\'])(INSERT INTO .*?)\1',
                r'(["`\'])(UPDATE .*?)\1',
                r'(["`\'])(DELETE FROM .*?)\1',
                r'("""|\'\'\')(.*?)(\1)',
            ]

        for pat in patterns:
            for m in re.finditer(pat, code, flags=re.IGNORECASE | re.DOTALL):
                for g in m.groups():
                    if not g:
                        continue
                    s = str(g).strip()
                    add_if_sql(s)

        if not found:
            for ln in code.splitlines():
                if re.search(r"\b(SELECT|INSERT INTO|UPDATE|DELETE FROM|CREATE TABLE)\b", ln, flags=re.I):
                    mq = re.search(r'(["`\'])(.*?(SELECT|INSERT|UPDATE|DELETE).*?)\1', ln, flags=re.I)
                    if mq:
                        q = mq.group(2)
                    else:
                        q = ln.strip()
                    add_if_sql(q)

    except Exception:
        pass

    cleaned: List[str] = []
    for q in found:
        qq = q.strip()
        if qq not in cleaned:
            cleaned.append(qq)
    return cleaned

# -----------------------
# RAG (simple) for SQL similarity
# -----------------------
def rag_similar_sqls(query: str, repo_sqls: Dict[str, str], top_k: int = 3) -> List[str]:
    if not repo_sqls:
        return []
    scores = []
    for path, content in repo_sqls.items():
        sm = difflib.SequenceMatcher(None, query.lower(), content.lower())
        score = sm.quick_ratio()
        scores.append((score, path, content))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [content for (score, path, content) in scores[:top_k]]

# -----------------------
# LLM SQL rewrite (optional)
# -----------------------
def llm_rewrite_sql(query: str, context: List[str]) -> str:
    if llm is None or PromptTemplate is None:
        return query
    try:
        ctx = "\n\n".join(context[:3])
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are a SQL assistant. Given an original SQL query and context SQL/schema snippets, "
                "return a corrected SQL query only. If the query is fine, return it unchanged.\n\n"
                "Context:\n{context}\n\nOriginal Query:\n{query}\n"
            ),
        )
        chain = prompt | llm | StrOutputParser()
        out = chain.invoke({"query": query, "context": ctx})
        return out.strip()
    except Exception:
        return query

# -----------------------
# Schema building & DB helpers
# -----------------------
def create_db_from_sql_file(sql_path: str, out_db: str = TEMP_DB) -> Optional[str]:
    try:
        if os.path.exists(out_db):
            os.remove(out_db)
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_text = f.read()
        conn = sqlite3.connect(out_db)
        conn.executescript(sql_text)
        conn.close()
        return out_db
    except Exception:
        return None


def build_schema_from_sqlite(db_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(db_path):
        return {}
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        insp = inspect(engine)
        tables = insp.get_table_names()
        schema: Dict[str, List[str]] = {t: [c["name"] for c in insp.get_columns(t)] for t in tables}
        return schema
    except Exception:
        return {}


def build_schema_from_external_db(conn_str: str) -> Dict[str, List[str]]:
    """
    Accepts SQLAlchemy connection string for external DB (sqlite:///..., postgresql://..., mysql+pymysql://...).
    Be careful with credentials.
    """
    try:
        engine = create_engine(conn_str)
        insp = inspect(engine)
        tables = insp.get_table_names()
        return {t: [c["name"] for c in insp.get_columns(t)] for t in tables}
    except Exception:
        return {}

# -----------------------
# Prompt schema source menu (exact wording)
# -----------------------
def prompt_schema_source() -> Dict:
    print("\nSchema source options:")
    print("1) Use db.sqlite3 from this repository (default path: db.sqlite3)")
    print("2) Use an .sql file from this repository (choose path)")
    print("3) Use an .sql file from another GitHub repository")
    print("4) Use a local .sql file on disk")
    print("5) Auto-detect (default)")
    sel = input("Select schema source [1-5] (default 5): ").strip() or "5"
    try:
        sel_i = int(sel)
    except Exception:
        sel_i = 5

    if sel_i == 1:
        return {"type": "db_in_repo"}
    if sel_i == 2:
        path = input(
            "Enter path to .sql file in this repository (e.g., db/schema.sql): "
        ).strip()
        return {"type": "sql_in_repo", "path": path}
    if sel_i == 3:
        owner_repo = input("Enter owner/repo (e.g., org/repo): ").strip()
        path = input("Enter SQL file path inside that repo: ").strip()
        return {"type": "sql_other_repo", "owner_repo": owner_repo, "path": path}
    if sel_i == 4:
        local_path = input("Enter full local path to .sql file: ").strip()
        return {"type": "local_sql", "path": local_path}
    return {"type": "auto-detect"}

# -----------------------
# Core SQL validator (Hybrid)
# -----------------------
def _fuzzy_best_match(candidate: str, options: List[str]) -> Tuple[str, int]:
    """Return (best_option, distance). If options empty, returns (candidate, large_num)."""
    if not options:
        return candidate, 10 ** 6
    best = min(options, key=lambda o: levenshtein_distance(o.lower(), candidate.lower()))
    dist = levenshtein_distance(best.lower(), candidate.lower())
    return best, dist


def _strip_identifier_quotes(identifier: str) -> str:
    return identifier.strip().strip('`').strip('"')


def validate_and_fix_sql(
    query: str,
    schema: Dict[str, List[str]],
    repo_sql_files: Dict[str, str],
    db_path: str = TEMP_DB,
) -> Tuple[List[Tuple[int, str]], str]:
    """
    Return: (list of (line_number, short_issue)), corrected_query
    Note: line numbers are for the SQL snippet (1-based).
    """
    issues: List[Tuple[int, str]] = []
    corrected = query.strip()

    # Basic parse via sqlparse (best-effort; do not early-return on error to allow best-effort fixes)
    try:
        parsed = sqlparse.parse(corrected)
        if not parsed:
            issues.append((1, "Invalid SQL syntax"))
    except Exception:
        issues.append((1, "SQL parse error"))

    # Rule-based fuzzy table/column checks using provided schema
    all_tables: List[str] = list(schema.keys())

    # Try to find table tokens in FROM/INTO/UPDATE (first occurrence)
    m_from = re.search(r'\bFROM\s+([`\"]?)(\w+)\1', corrected, flags=re.I)
    m_into = re.search(r'\bINTO\s+([`\"]?)(\w+)\1', corrected, flags=re.I)
    m_update = re.search(r'\bUPDATE\s+([`\"]?)(\w+)\1', corrected, flags=re.I)
    target_table: Optional[str] = None
    for mm in (m_from, m_into, m_update):
        if mm:
            target_table = mm.group(2)
            break

    # Fuzzy-fix table name if not in schema
    if target_table:
        if target_table not in schema:
            if all_tables:
                best_table, dist = _fuzzy_best_match(target_table, all_tables)
                if dist <= 2:
                    corrected = re.sub(r"\\b" + re.escape(target_table) + r"\\b", best_table, corrected)
                    issues.append((1, f"Fixed table-like token '{target_table}' -> '{best_table}'"))
                    target_table = best_table
                else:
                    issues.append((1, f"Table '{target_table}' not found in schema"))
            else:
                issues.append((1, f"Table '{target_table}' not found (no schema)"))

    # Helper to get table-specific columns (original case) and a lowercase lookup
    def get_table_columns(table_name: Optional[str]) -> Tuple[List[str], Dict[str, str]]:
        if not table_name or table_name not in schema:
            return [], {}
        cols = schema.get(table_name, [])
        lower_map = {c.lower(): c for c in cols}
        return cols, lower_map

    table_cols, table_cols_lower_map = get_table_columns(target_table)

    # Fuzzy-fix column names in simple SELECT column list (single table)
    if corrected.strip().upper().startswith("SELECT"):
        m = re.match(r"SELECT\s+(.*?)\s+FROM\s+([^\s;]+)", corrected, flags=re.I | re.S)
        if m:
            fields_raw = m.group(1)
            fields = [f.strip() for f in fields_raw.split(",")]
            new_fields: List[str] = []
            for f in fields:
                if f == "*" or "." in f:
                    new_fields.append(f)
                    continue
                base = _strip_identifier_quotes(f)
                if base.lower() in table_cols_lower_map:
                    new_fields.append(f)
                else:
                    best_col, dist = _fuzzy_best_match(base, table_cols)
                    if dist <= 2:
                        new_fields.append(best_col)
                        issues.append((1, f"Fixed column '{base}' -> '{best_col}'"))
                    else:
                        issues.append((1, f"Column '{base}' not found in schema"))
                        new_fields.append(f)
            if ", ".join(fields) != ", ".join(new_fields):
                # Replace only the fields segment
                start, end = m.start(1), m.end(1)
                corrected = corrected[:start] + ", ".join(new_fields) + corrected[end:]

    # Fuzzy-fix column names in INSERT column list
    upcase = corrected.strip().upper()
    if upcase.startswith("INSERT"):
        m_ins = re.search(
            r"INSERT\s+INTO\s+([`\"]?)(\w+)\1\s*\((.*?)\)\s*VALUES\s*\(",
            corrected,
            flags=re.I | re.S,
        )
        if m_ins:
            table_in_insert = m_ins.group(2)
            # If earlier table correction happened, ensure we use corrected table
            if table_in_insert not in schema and target_table and target_table in schema:
                table_in_insert = target_table
            cols_text = m_ins.group(3)
            raw_cols = [c.strip() for c in cols_text.split(",")]
            fixed_cols: List[str] = []
            tcols, tcols_lower_map = get_table_columns(table_in_insert)
            for c in raw_cols:
                base = _strip_identifier_quotes(c)
                if base.lower() in tcols_lower_map:
                    fixed_cols.append(tcols_lower_map[base.lower()])
                else:
                    best_col, dist = _fuzzy_best_match(base, tcols)
                    if dist <= 3:  # allow a slightly larger threshold for INSERT typos
                        fixed_cols.append(best_col)
                        issues.append((1, f"Fixed column '{base}' -> '{best_col}'"))
                    else:
                        fixed_cols.append(c)
                        issues.append((1, f"Column '{base}' not found in schema"))
            if ", ".join(raw_cols) != ", ".join(fixed_cols):
                corrected = (
                    corrected[: m_ins.start(3)] + ", ".join(fixed_cols) + corrected[m_ins.end(3) :]
                )

    # Fuzzy-fix column names in UPDATE SET assignments
    if upcase.startswith("UPDATE"):
        m_upd = re.search(
            r"UPDATE\s+([`\"]?)(\w+)\1\s+SET\s+(.*?)\s+(WHERE|$)",
            corrected,
            flags=re.I | re.S,
        )
        if m_upd:
            table_in_update = m_upd.group(2)
            if table_in_update not in schema and target_table and target_table in schema:
                table_in_update = target_table
            assigns_text = m_upd.group(3)
            parts = [p.strip() for p in assigns_text.split(",")]
            tcols, tcols_lower_map = get_table_columns(table_in_update)
            new_parts: List[str] = []
            for p in parts:
                if "=" not in p:
                    new_parts.append(p)
                    continue
                left, right = p.split("=", 1)
                base = _strip_identifier_quotes(left.strip())
                if base.lower() in tcols_lower_map:
                    new_parts.append(f"{tcols_lower_map[base.lower()]} = {right.strip()}")
                else:
                    best_col, dist = _fuzzy_best_match(base, tcols)
                    if dist <= 3:
                        new_parts.append(f"{best_col} = {right.strip()}")
                        issues.append((1, f"Fixed column '{base}' -> '{best_col}'"))
                    else:
                        new_parts.append(p)
                        issues.append((1, f"Column '{base}' not found in schema"))
            if ", ".join(parts) != ", ".join(new_parts):
                corrected = corrected[: m_upd.start(3)] + ", ".join(new_parts) + corrected[m_upd.end(3) :]

    # RAG: try to find similar SQL statements in repo_sql_files
    try:
        similar = rag_similar_sqls(query, repo_sql_files, top_k=3)
        if similar:
            rewritten = llm_rewrite_sql(corrected, similar) if llm else corrected
            if rewritten and rewritten.strip() != corrected.strip():
                issues.append((1, "LLM suggested rewrite"))
                corrected = rewritten
    except Exception:
        pass

    # final checks for common issues: missing WHERE on DELETE/UPDATE
    up = corrected.upper()
    if up.startswith("DELETE") and "WHERE" not in up:
        issues.append((1, "DELETE without WHERE (will delete all rows)"))
    if up.startswith("UPDATE") and "WHERE" not in up:
        issues.append((1, "UPDATE without WHERE (may update all rows)"))

    return issues, corrected

# -----------------------
# High-level repo analysis
# -----------------------
def analyze_repo_files(repo, repo_name: str, chosen_files: List[str], branch: str = "main") -> List[Tuple[str, List[Tuple[int, str]], Optional[str]]]:
    """
    Analyze list of files from repo. Returns report list with only required outputs.
    Final printed output format per file:
      File: path
      Buggy Line Numbers:
      [n1, n2, ...]
      Buggy Lines:
      <line>: <desc>
      Suggested Fix:
      <code>
    """
    # Prepare repo SQL file cache
    repo_sqls: Dict[str, str] = {}
    try:
        for p in get_all_paths(repo):
            if p.lower().endswith(".sql"):
                content = read_repo_file_text(repo, p, ref=branch)
                if content:
                    repo_sqls[p] = content
    except Exception:
        repo_sqls = {}

    # Schema variables (initialized on-demand)
    schema: Dict[str, List[str]] = {}
    schema_built = False
    schema_choice = None
    db_path_for_schema = None

    outputs: List[Tuple[str, List[Tuple[int, str]], Optional[str]]] = []

    for file_path in chosen_files:
        code = read_repo_file_text(repo, file_path, ref=branch)
        if code is None:
            continue

        # language auto-detection from extension, LLM fallback if unknown
        lang = detect_language_from_extension(file_path)
        if lang == "Unknown" and llm is not None:
            guess = llm_detect_language(code)
            if guess and guess != "Unknown":
                lang = guess

        # extract SQL queries (if any)
        queries = extract_sql_queries(code, language=lang)

        # If SQL found and schema not built -> prompt schema source menu
        if queries and not schema_built:
            schema_choice = prompt_schema_source()
            st = schema_choice.get("type")
            if st == "db_in_repo":
                local = download_file(repo, "db.sqlite3", TEMP_DB)
                if local:
                    schema = build_schema_from_sqlite(local)
                    db_path_for_schema = local
                else:
                    print("Warning: db.sqlite3 not found in repo root.")
            elif st == "sql_in_repo":
                path = schema_choice.get("path")
                if path:
                    local_sql = download_file(repo, path, TEMP_SQL_FILE)
                    if local_sql:
                        dbp = create_db_from_sql_file(local_sql, TEMP_DB)
                        if dbp:
                            schema = build_schema_from_sqlite(dbp)
                            db_path_for_schema = dbp
            elif st == "sql_other_repo":
                owner_repo = schema_choice.get("owner_repo")
                path = schema_choice.get("path")
                if owner_repo and path and gh:
                    try:
                        other_repo = gh.get_repo(owner_repo)
                        local_sql = download_file(other_repo, path, TEMP_SQL_FILE)
                        if local_sql:
                            dbp = create_db_from_sql_file(local_sql, TEMP_DB)
                            if dbp:
                                schema = build_schema_from_sqlite(dbp)
                                db_path_for_schema = dbp
                    except Exception:
                        pass
            elif st == "local_sql":
                path = schema_choice.get("path")
                if path and os.path.exists(path):
                    dbp = create_db_from_sql_file(path, TEMP_DB)
                    if dbp:
                        schema = build_schema_from_sqlite(dbp)
                        db_path_for_schema = dbp
            elif st == "auto-detect":
                local = download_file(repo, "db.sqlite3", TEMP_DB)
                if local:
                    schema = build_schema_from_sqlite(local)
                    db_path_for_schema = local
                else:
                    first_sql = None
                    try:
                        first_sql = next((p for p in get_all_paths(repo) if p.lower().endswith(".sql")), None)
                    except Exception:
                        first_sql = None
                    if first_sql:
                        local_sql = download_file(repo, first_sql, TEMP_SQL_FILE)
                        if local_sql:
                            dbp = create_db_from_sql_file(local_sql, TEMP_DB)
                            if dbp:
                                schema = build_schema_from_sqlite(dbp)
                                db_path_for_schema = dbp
            schema_built = True

            if (not schema) and (st != "failed"):
                want = input(
                    "Schema not found or empty. Do you want to provide an external DB connection string? (y/N): "
                ).strip().lower()
                if want == "y":
                    conn_str = input(
                        "Enter SQLAlchemy connection string (e.g., postgresql://user:pw@host/db): "
                    ).strip()
                    if conn_str:
                        schema = build_schema_from_external_db(conn_str)
                        db_path_for_schema = None

        # 1) LLM bug detector for code (if available)
        buggy_lines: List[int] = []
        suggested_fix_text: Optional[str] = None

        if llm is not None:
            try:
                bug_prompt = PromptTemplate(
                    input_variables=["code", "language"],
                    template=(
                        "You are a {language} bug detection assistant. "
                        "Return only buggy line numbers as a Python list (e.g., [2,5]) or [] if none.\n\n"
                        "Code:\n{code}"
                    ),
                )
                chain = bug_prompt | llm | StrOutputParser()
                raw = chain.invoke({"code": code, "language": lang}).strip()
                try:
                    buggy_lines = ast.literal_eval(raw)
                    if not isinstance(buggy_lines, list):
                        buggy_lines = []
                except Exception:
                    nums = re.findall(r"\d+", raw)
                    buggy_lines = [int(n) for n in nums]
            except Exception:
                buggy_lines = []
        else:
            lines = code.splitlines()
            for i, ln in enumerate(lines, start=1):
                if re.search(r"\$.*_GET\[", ln) or re.search(r"\$.*_POST\[", ln):
                    if re.search(r"\bquery\s*\(|->query\s*\(", " ".join(lines[max(0, i - 5) : i + 3])):
                        buggy_lines.append(i)
                if "DELETE" in ln.upper() or "UPDATE" in ln.upper():
                    if "WHERE" not in ln.upper() and not re.search(r"\bWHERE\b", code, flags=re.I):
                        buggy_lines.append(i)

        # 2) Suggested fixed code via LLM (optional)
        if buggy_lines and llm is not None:
            try:
                fixer_prompt = PromptTemplate(
                    input_variables=["code", "language"],
                    template=(
                        "You are a {language} bug fixer. Return only the corrected code (no explanation).\n\n"
                        "Buggy code:\n{code}"
                    ),
                )
                chainf = fixer_prompt | llm | StrOutputParser()
                fixed = chainf.invoke({"code": code, "language": lang}).strip()
                suggested_fix_text = fixed
            except Exception:
                suggested_fix_text = None

        # 3) SQL validation and corrections
        sql_issues: List[Tuple[int, str]] = []
        sql_corrections: List[str] = []
        if queries:
            for q in queries:
                issues, corrected_q = validate_and_fix_sql(q, schema, repo_sqls, db_path_for_schema or TEMP_DB)
                # map SQL issues to approximate line numbers within original file
                try:
                    idx = code.find(q)
                    if idx >= 0:
                        ln = code[:idx].count("\n") + 1
                        for (num, desc) in issues:
                            sql_issues.append((ln + num - 1, desc))
                    else:
                        for (num, desc) in issues:
                            sql_issues.append((1, desc))
                except Exception:
                    for (num, desc) in issues:
                        sql_issues.append((1, desc))
                if corrected_q and corrected_q.strip() != q.strip():
                    sql_corrections.append(corrected_q)

        # Consolidate
        consolidated_issues: List[Tuple[int, str]] = []
        if buggy_lines:
            for ln in sorted(set(buggy_lines)):
                consolidated_issues.append((ln, "LLM flagged potential bug"))
        for ln, desc in sql_issues:
            consolidated_issues.append((ln, desc))

        if not consolidated_issues:
            outputs.append((file_path, [], None))
            continue

        final_suggested = None
        if suggested_fix_text:
            final_suggested = suggested_fix_text
        elif sql_corrections:
            final_suggested = "\n\n".join(sql_corrections)
        else:
            final_suggested = None

        outputs.append((file_path, consolidated_issues, final_suggested))

    # cleanup temp artifacts
    try:
        if os.path.exists(TEMP_DB):
            os.remove(TEMP_DB)
        if os.path.exists(TEMP_SQL_FILE):
            os.remove(TEMP_SQL_FILE)
    except Exception:
        pass

    return outputs

# -----------------------
# Simple interactive chooser
# -----------------------
def choose_from_list(prompt: str, options: List[str]) -> List[str]:
    print(f"\n{prompt}")
    for i, o in enumerate(options, start=1):
        print(f"{i}. {o}")
    print("a/A. All")
    print("m/M. Multiple (comma-separated numbers or names)")
    choice = input("\nEnter your choice: ").strip()
    if not choice:
        return []
    if choice.lower() == "a":
        return options
    if choice.lower() == "m":
        raw = input("Enter numbers or names (comma-separated): ").split(",")
        selected: List[str] = []
        for v in raw:
            v = v.strip()
            if v.isdigit():
                idx = int(v) - 1
                if 0 <= idx < len(options):
                    selected.append(options[idx])
            else:
                matches = [opt for opt in options if opt.lower() == v.lower()]
                selected.extend(matches)
        return selected
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return [options[idx]]
    matches = [opt for opt in options if opt.lower() == choice.lower()]
    if matches:
        return matches
    print("Invalid choice. Returning empty.")
    return []

# -----------------------
# Main interactive flow
# -----------------------
def main() -> None:
    print("GitHub Hybrid Code+SQL Analyzer (minimal output)\n")
    username = input("Enter GitHub username/org: ").strip()
    if not username:
        print("No username provided. Exiting.")
        return

    # prefer repos_index.yaml if present
    yaml_repos: Dict[str, dict] = {}
    if os.path.exists("repos_index.yaml"):
        try:
            import yaml

            with open("repos_index.yaml", "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            if yaml_data and "accounts" in yaml_data:
                account = next((a for a in yaml_data["accounts"] if a.get("username") == username), None)
                if account:
                    for r in account.get("repos", []):
                        name = r.get("name") if isinstance(r, dict) else r
                        yaml_repos[name] = r
        except Exception:
            yaml_repos = {}

    repo_map: Dict[str, any] = {}
    repo_list = list(yaml_repos.keys())

    # fallback to GitHub listing
    if not repo_list:
        if gh is None:
            print("No GitHub token and no repos_index.yaml. Exiting.")
            return
        try:
            user = gh.get_user(username)
            repos = [r for r in user.get_repos()]
            repo_list = [r.name for r in repos]
            for r in repos:
                repo_map[r.name] = r
        except Exception as e:
            print(f"Failed to list repos for {username}: {e}")
            return

    if not repo_list:
        print(f"No repositories found for {username}. Exiting.")
        return

    chosen_repos = choose_from_list(f"Select repository/repositories under {username}:", repo_list)
    if not chosen_repos:
        print("No repositories chosen. Exiting.")
        return

    results: List[Tuple[str, List[Tuple[int, str]], Optional[str]]] = []
    # concurrency
    with ThreadPoolExecutor(max_workers=min(6, len(chosen_repos))) as ex:
        futures = []
        for repo_name in chosen_repos:
            repo_obj = None
            if repo_name in repo_map:
                repo_obj = repo_map[repo_name]
            else:
                try:
                    repo_obj = gh.get_repo(f"{username}/{repo_name}")
                except Exception:
                    repo_obj = None
            if not repo_obj:
                print(f"Could not access {username}/{repo_name}. Skipping.")
                continue

            all_paths = get_all_paths(repo_obj)
            if not all_paths:
                print(f"No files in {repo_name}. Skipping.")
                continue

            candidates = [
                p
                for p in sorted(all_paths)
                if not p.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe", ".bin"))
            ]
            print(f"\nFiles in {repo_name} (showing up to 300):")
            show = candidates[:300]
            chosen_files = choose_from_list(f"Select file(s) from {repo_name}:", show)
            if not chosen_files:
                print(f"No files chosen for {repo_name}. Skipping.")
                continue

            futures.append(ex.submit(analyze_repo_files, repo_obj, f"{username}/{repo_name}", chosen_files, "main"))

        for fut in as_completed(futures):
            try:
                out = fut.result()
                results.extend(out)
            except Exception as e:
                print(f"Error analyzing: {e}")

    # Print final minimal outputs
    for (file_path, issues, suggested) in results:
        print(f"\nFile: {file_path}")
        if not issues:
            print("No bugs detected.")
            continue
        # New: compact list of buggy line numbers
        only_lines = sorted({ln for (ln, _desc) in issues})
        print("Buggy Line Numbers:")
        print(str(only_lines))
        # Existing: detailed mapping
        print("Buggy Lines:")
        for (ln, desc) in sorted(issues, key=lambda x: x[0]):
            print(f"{ln}: {desc}")
        print("Suggested Fix:")
        if suggested:
            print(suggested)
        else:
            print("No automated fix available.")

    # Optionally save a compact JSON report
    save = input("\nSave compact JSON report of results? (y/N): ").strip().lower()
    if save == "y":
        report_path = os.path.join(REPORTS_DIR, f"compact_report.json")
        compact = []
        for (file_path, issues, suggested) in results:
            compact.append(
                {
                    "file": file_path,
                    "issues": [{"line": ln, "desc": desc} for ln, desc in issues],
                    "suggested": suggested,
                }
            )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, indent=2)
        print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
