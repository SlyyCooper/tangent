import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional
import os
from tangent.types import Result


def get_db_config():
    """Get database configuration from environment variables."""
    return {
        "dbname": os.getenv("POSTGRES_DB", "postgres"),
        "user": os.getenv("POSTGRES_USER", "tan"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432")
    }


def execute_query(query: str) -> Result:
    """Execute a query on the PostgreSQL database.
    
    Args:
        query: SQL query to execute
        
    Returns:
        Result object with query results
    """
    conn = None
    cursor = None
    try:
        # Get connection parameters from environment
        db_config = get_db_config()
        
        # Attempt connection
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(query)
        if cursor.description:  # If it's a SELECT query
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
        else:  # For INSERT, UPDATE, DELETE
            result = f"Query affected {cursor.rowcount} rows"
            conn.commit()
        
        return Result(
            value=str(result),
            context_variables={
                "success": True,
                "results": result,
                "rowcount": cursor.rowcount
            }
        )
    except psycopg2.OperationalError as e:
        return Result(
            value=f"Database connection error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e),
                "error_type": "connection"
            }
        )
    except psycopg2.Error as e:
        return Result(
            value=f"Database query error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e),
                "error_type": "query"
            }
        )
    except Exception as e:
        return Result(
            value=f"Unexpected error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e),
                "error_type": "unexpected"
            }
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_table_list() -> Result:
    """Get a list of all tables in the database."""
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """
    return execute_query(query)


def get_table_info(table_name: str) -> Result:
    """Get the structure of a specific table."""
    query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = '{table_name}'
        ORDER BY ordinal_position;
    """
    return execute_query(query) 