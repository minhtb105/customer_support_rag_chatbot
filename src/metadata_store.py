import sqlite3, json, time
from config import META_DB_PATH


def init_db(db_path=META_DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_name TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            updated_at REAL NOT NULL
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_hash TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            chunk_index INTEGER,
            vector_id TEXT,
            extra_meta TEXT,
            FOREIGN KEY(file_name) REFERENCES files(file_name) ON DELETE CASCADE
        )""")
        # index for file lookup
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_name);")
        
# upsert file + chunk rows in a transaction
def upsert_file_and_chunks(file_name, file_hash, chunks: list, db_path=META_DB_PATH):
    """
    chunks: list of dicts { 'chunk_hash':..., 'chunk_index': int, 'vector_id': str|None, 'extra_meta': dict|None }
    """
    now = time.time()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        cur = conn.cursor()
        cur.execute("INSERT INTO files(file_name, file_hash, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(file_name) DO UPDATE SET file_hash=excluded.file_hash, updated_at=excluded.updated_at",
                    (file_name, file_hash, now))
        # upsert chunks
        for ch in chunks:
            extra_json = json.dumps(ch.get("extra_meta") or {})
            cur.execute("""
                INSERT INTO chunks(chunk_hash, file_name, chunk_index, vector_id, extra_meta)
                VALUES(?,?,?,?,?)
                ON CONFLICT(chunk_hash) DO UPDATE SET file_name=excluded.file_name,
                                                     chunk_index=excluded.chunk_index,
                                                     vector_id=excluded.vector_id,
                                                     extra_meta=excluded.extra_meta
            """, (ch["chunk_hash"], file_name, ch.get("chunk_index"), ch.get("vector_id"), extra_json))
        conn.commit()

# get old chunk list for a file
def get_chunk_hashes_for_file(file_name, db_path=META_DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT chunk_hash FROM chunks WHERE file_name=? ORDER BY chunk_index", (file_name,))
        rows = cur.fetchall()
        
        return [r[0] for r in rows]

# find vector_ids to remove for old chunks that changed
def find_vector_ids_for_chunk_hashes(chunk_hashes, db_path=META_DB_PATH):
    if not chunk_hashes:
        return []
    
    q = "SELECT vector_id FROM chunks WHERE chunk_hash IN ({})".format(",".join("?"*len(chunk_hashes)))
    
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(q, tuple(chunk_hashes))
        
        return [r[0] for r in cur.fetchall() if r[0]]
    
def get_file_hash(file_name: str, db_path=META_DB_PATH):
    """Return stored file_hash or None if not found."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_hash FROM files WHERE file_name = ?", (file_name,))
        row = cur.fetchone()
        
        return row[0] if row else None

def delete_chunks_by_hashes(chunk_hashes: list, db_path=META_DB_PATH):
    """Delete chunk rows (by chunk_hash) from chunks table."""
    if not chunk_hashes:
        return
    
    placeholders = ",".join("?" * len(chunk_hashes))
    q = f"DELETE FROM chunks WHERE chunk_hash IN ({placeholders})"
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        cur = conn.cursor()
        cur.execute(q, tuple(chunk_hashes))
        conn.commit()    
