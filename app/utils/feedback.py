from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from sqlalchemy.engine import Engine

class FeedbackService:
    """Service for handling chat feedback operations."""
    
    def __init__(self, engine: Engine, team_schema: Optional[str] = None):
        """Initialize the feedback service.
        
        Args:
            engine: SQLAlchemy engine instance configured with the correct search path.
            team_schema: Optional team schema to use (consider if this is still needed or handled by engine search path)
        """
        self.engine = engine
        self.team_schema = team_schema
        self._init_tables()
    
    def _init_tables(self):
        """Initialize the feedback tables using the engine."""
        try:
            with self.engine.connect() as connection:
                conn = connection.connection
                with conn.cursor() as cur:
                    if self.team_schema:
                        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.team_schema}")
                    
                    schema_prefix = "app_schema."
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {schema_prefix}feedback (
                            id SERIAL PRIMARY KEY,
                            message_id INTEGER NOT NULL,
                            feedback_type TEXT NOT NULL CHECK (feedback_type IN ('helpful', 'not_helpful', 'issue')),
                            issue_description TEXT,
                            schema_name TEXT,
                            query TEXT,
                            response TEXT,
                            context_used JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB
                        )
                    """)
                    conn.commit()
        except Exception as e:
            print(f"Error initializing feedback tables: {e}")
    
    def store_feedback(
        self,
        message_id: int,
        feedback_type: str,
        query: Optional[str] = None,
        response: Optional[str] = None,
        context_used: Optional[Dict[str, Any]] = None,
        issue_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store feedback in the database.
        
        Args:
            message_id: ID of the message being rated
            feedback_type: Type of feedback ('helpful', 'not_helpful', 'issue')
            query: Optional user query
            response: Optional assistant response
            context_used: Optional context used for the response
            issue_description: Optional issue description for issue reports
            metadata: Optional additional metadata
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            with self.engine.connect() as connection:
                conn = connection.connection
                with conn.cursor() as cur:
                    schema_prefix = "app_schema."
                    cur.execute(f"""
                        INSERT INTO {schema_prefix}feedback (
                            message_id,
                            feedback_type,
                            schema_name,
                            query,
                            response,
                            context_used,
                            issue_description,
                            metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        message_id,
                        feedback_type,
                        self.team_schema,
                        query,
                        response,
                        Json(context_used) if context_used else None,
                        issue_description,
                        Json(metadata) if metadata else None
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return False
    
    def get_feedback_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get feedback statistics.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            Dict containing feedback statistics
        """
        try:
            with self.engine.connect() as connection:
                conn = connection.connection
                with conn.cursor() as cur:
                    schema_prefix = "app_schema."
                    query = f"""
                        SELECT 
                            feedback_type,
                            COUNT(*) as count
                        FROM {schema_prefix}feedback
                        WHERE 1=1
                    """
                    params = []
                    
                    if start_date:
                        query += " AND created_at >= %s"
                        params.append(start_date)
                    
                    if end_date:
                        query += " AND created_at <= %s"
                        params.append(end_date)
                    
                    query += " GROUP BY feedback_type"
                    
                    cur.execute(query, tuple(params))
                    results = cur.fetchall()
                    
                    stats = {
                        'helpful': 0,
                        'not_helpful': 0,
                        'issues': 0,
                        'total': 0
                    }
                    
                    for feedback_type, count in results:
                        if feedback_type == 'issue':
                            stats['issues'] = count
                        else:
                            stats[feedback_type] = count
                        stats['total'] += count
                    
                    return stats
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            return {
                'helpful': 0,
                'not_helpful': 0,
                'issues': 0,
                'total': 0,
                'error': str(e)
            }
    
    def get_recent_issues(
        self,
        limit: int = 10
    ) -> list:
        """Get recent issue reports.
        
        Args:
            limit: Maximum number of issues to return
            
        Returns:
            List of issue reports
        """
        try:
            with self.engine.connect() as connection:
                conn = connection.connection
                with conn.cursor() as cur:
                    schema_prefix = "app_schema."
                    query = f"""
                        SELECT 
                            id,
                            message_id,
                            schema_name,
                            query,
                            issue_description,
                            created_at
                        FROM {schema_prefix}feedback
                        WHERE feedback_type = 'issue'
                        ORDER BY created_at DESC
                        LIMIT %s
                    """
                    
                    cur.execute(query, (limit,))
                    results = cur.fetchall()
                    
                    return [{
                        'id': row[0],
                        'message_id': row[1],
                        'schema_name': row[2],
                        'query': row[3],
                        'description': row[4],
                        'created_at': row[5]
                    } for row in results]
        except Exception as e:
            print(f"Error getting recent issues: {e}")
            return [] 