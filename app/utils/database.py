import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import uuid
import psycopg2.extras

# Load environment variables
load_dotenv()

# Database connection details
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

def get_db_connection():
    """
    Create and return a database connection.
    
    Returns:
        connection: A PostgreSQL database connection
    """
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def verify_team_access(team_id, access_code):
    """Verify team access code and return True if valid."""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM app_schema.teams 
            WHERE team_id = %s AND access_code = %s
        """, (team_id, access_code))
        
        result = cur.fetchone()[0] > 0
        
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Error verifying team access: {e}")
        return False

def store_document(team_id, filename, metadata=None, doc_id=None):
    """Store a new document or update an existing one and return its ID."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cur = conn.cursor()
        
        # Convert metadata to JSONB
        if metadata is None:
            metadata = {}
        
        if doc_id:
            # Update existing document
            cur.execute("""
                UPDATE app_schema.documents 
                SET filename = %s, metadata = %s::jsonb
                WHERE doc_id = %s AND team_id = %s
                RETURNING doc_id
            """, (filename, psycopg2.extras.Json(metadata), doc_id, team_id))
        else:
            # Create new document
            doc_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO app_schema.documents 
                (doc_id, team_id, filename, metadata) 
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING doc_id
            """, (doc_id, team_id, filename, psycopg2.extras.Json(metadata)))
        
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return doc_id
    except Exception as e:
        print(f"Error storing document: {e}")
        return None

def store_chunk(doc_id, text, embedding):
    """Store a document chunk with its embedding."""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        chunk_id = str(uuid.uuid4())
        
        cur.execute("""
            INSERT INTO app_schema.chunks 
            (chunk_id, doc_id, text, embedding) 
            VALUES (%s, %s, %s, %s)
        """, (chunk_id, doc_id, text, embedding))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing chunk: {e}")
        return False

def search_similar_chunks(team_id, embedding, limit=5, admin_override=False):
    """Search for similar chunks, filtered by team_id unless admin_override is True."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cur = conn.cursor()
        
        if admin_override:
            query = """
                SELECT c.chunk_id, c.doc_id, d.filename, c.text, d.metadata,
                       1 - (c.embedding <=> %s) as similarity
                FROM app_schema.chunks c
                JOIN app_schema.documents d ON c.doc_id = d.doc_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
            """
            cur.execute(query, (embedding, embedding, limit))
        else:
            query = """
                SELECT c.chunk_id, c.doc_id, d.filename, c.text, d.metadata,
                       1 - (c.embedding <=> %s) as similarity
                FROM app_schema.chunks c
                JOIN app_schema.documents d ON c.doc_id = d.doc_id
                WHERE d.team_id = %s
                ORDER BY c.embedding <=> %s
                LIMIT %s
            """
            cur.execute(query, (embedding, team_id, embedding, limit))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return []

def get_team_documents(team_id):
    """Get all documents for a team."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT doc_id, filename, metadata, created_at 
            FROM app_schema.documents 
            WHERE team_id = %s
            ORDER BY created_at DESC
        """, (team_id,))
        results = cur.fetchall()
        # Convert to list of dicts
        documents = [dict(doc) for doc in results]
        cur.close()
        conn.close()
        return documents
    except Exception as e:
        print(f"Error getting team documents: {e}")
        return []

def delete_document(doc_id, team_id=None):
    """Delete a document and its chunks. If team_id is provided, verify ownership."""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        
        if team_id:
            # Verify ownership before deletion
            cur.execute("""
                DELETE FROM app_schema.documents 
                WHERE doc_id = %s AND team_id = %s
            """, (doc_id, team_id))
        else:
            # Admin deletion
            cur.execute("""
                DELETE FROM app_schema.documents 
                WHERE doc_id = %s
            """, (doc_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False

def get_team_config(team_id):
    """Get team-specific configuration."""
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        
        cur = conn.cursor()
        cur.execute("""
            SELECT configuration 
            FROM app_schema.teams 
            WHERE team_id = %s
        """, (team_id,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return result[0] if result else {}
    except Exception as e:
        print(f"Error getting team config: {e}")
        return {}

def initialize_database():
    """
    Initialize the database with required tables and extensions.
    Also ensures an admin team exists.
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        connection = get_db_connection()
        if not connection:
            return False
        
        connection.autocommit = True
        cursor = connection.cursor()
        
        # Create the pgvector extension if it doesn't exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create the app schema
        cursor.execute("CREATE SCHEMA IF NOT EXISTS app_schema;")
        
        # Create teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_schema.teams (
                team_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                access_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            );
        """)
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_schema.documents (
                doc_id TEXT PRIMARY KEY,
                team_id TEXT NOT NULL REFERENCES app_schema.teams(team_id),
                filename TEXT NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create chunks table with vector support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_schema.chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES app_schema.documents(doc_id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                embedding vector(768),
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index on the embedding vector
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
            ON app_schema.chunks 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        cursor.close()
        connection.close()

        # Ensure admin team exists
        admin_code = os.getenv('ADMIN_CODE', 'admin123')
        admin_team_id = 'admin'
        admin_name = 'Administrators'
        
        # Check if admin team exists
        conn = get_db_connection()
        if not conn:
            return False
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM app_schema.teams WHERE team_id = %s", (admin_team_id,))
        exists = cur.fetchone()[0] > 0
        cur.close()
        conn.close()
        if not exists:
            create_team(admin_team_id, admin_name, admin_code)
        
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def store_embedding(document_id, filename, chunk_text, embedding, metadata=None):
    """
    Store a document chunk and its embedding in the database.
    
    Args:
        document_id (str): Unique identifier for the document
        filename (str): Name of the document file
        chunk_text (str): Text content of the chunk
        embedding (list): Vector embedding of the chunk
        metadata (dict, optional): Additional metadata about the chunk
        
    Returns:
        bool: True if storage was successful, False otherwise
    """
    try:
        connection = get_db_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        cursor.execute(sql.SQL("""
            INSERT INTO document_chunks (document_id, filename, chunk_text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """), (document_id, filename, chunk_text, metadata, embedding))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return True
    except Exception as e:
        print(f"Error storing embedding: {e}")
        return False

def search_similar_chunks(embedding, limit=5):
    """
    Search for similar chunks using vector similarity.
    
    Args:
        embedding (list): Vector embedding to search for
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of similar chunks with their similarity scores
    """
    try:
        connection = get_db_connection()
        if not connection:
            return []
        
        cursor = connection.cursor()
        
        cursor.execute(sql.SQL("""
            SELECT id, document_id, filename, chunk_text, metadata, 
                   1 - (embedding <=> %s) as similarity
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        """), (embedding, embedding, limit))
        
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return results
    except Exception as e:
        print(f"Error searching similar chunks: {e}")
        return []

def create_team(team_id, name, access_code, metadata=None):
    """Create a new team."""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cur = conn.cursor()
        
        # Convert metadata to JSONB
        if metadata is None:
            metadata = {}
        
        cur.execute("""
            INSERT INTO app_schema.teams 
            (team_id, name, access_code, metadata) 
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (team_id) DO UPDATE 
            SET name = EXCLUDED.name,
                access_code = EXCLUDED.access_code,
                metadata = EXCLUDED.metadata
            RETURNING team_id
        """, (team_id, name, access_code, psycopg2.extras.Json(metadata)))
        
        team_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return team_id
    except Exception as e:
        print(f"Error creating team: {e}")
        return None

def initialize_test_data():
    """Initialize test data including a test team."""
    try:
        # Create test team
        test_team = create_team(
            team_id="test",
            name="Test Team",
            access_code="test123"
        )
        if test_team:
            print("Test team created successfully")
        return True
    except Exception as e:
        print(f"Error initializing test data: {e}")
        return False

def get_all_teams():
    """Return a list of all teams (for admin use)."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT team_id, name, access_code, created_at, metadata
            FROM app_schema.teams
            ORDER BY created_at DESC
        """)
        results = cur.fetchall()
        teams = [dict(row) for row in results]
        cur.close()
        conn.close()
        return teams
    except Exception as e:
        print(f"Error getting all teams: {e}")
        return []

def get_all_documents():
    """Get all documents (admin only)."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT doc_id, team_id, filename, metadata, created_at 
            FROM app_schema.documents 
            ORDER BY created_at DESC
        """)
        results = cur.fetchall()
        documents = [dict(doc) for doc in results]
        cur.close()
        conn.close()
        return documents
    except Exception as e:
        print(f"Error getting all documents: {e}")
        return []

def get_document_chunks(doc_id):
    """Get all chunks for a given document ID."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT chunk_id, text, embedding, doc_id
            FROM app_schema.chunks
            WHERE doc_id = %s
            ORDER BY chunk_id
        """, (doc_id,))
        results = cur.fetchall()
        chunks = [dict(row) for row in results]
        cur.close()
        conn.close()
        return chunks
    except Exception as e:
        print(f"Error getting document chunks: {e}")
        return [] 