"""
Timeline manager module.
This module handles the management of operation timelines.
"""

import os
import json
import logging
from pathlib import Path
import sqlite3
from datetime import datetime

from .operation import Operation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimelineManager:
    """
    Manages timelines for CAD operations.
    """
    
    def __init__(self, database_path=None):
        """
        Initialize the timeline manager.
        
        Args:
            database_path (str, optional): Path to the database file.
                                         If None, a default path will be used.
        """
        # Set up database path
        if database_path is None:
            self.database_dir = Path("../database")
            self.database_dir.mkdir(parents=True, exist_ok=True)
            self.database_path = self.database_dir / "timeline.db"
        else:
            self.database_path = Path(database_path)
            self.database_dir = self.database_path.parent
            self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize timeline
        self.current_timeline = []
        self.current_position = -1
        
        logger.info(f"Timeline manager initialized with database: {self.database_path}")
    
    def _init_database(self):
        """Initialize the SQLite database for storing timelines."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create operations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    operation_type TEXT,
                    parameters TEXT,
                    description TEXT,
                    session_id TEXT
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at REAL,
                    last_modified REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def create_session(self, name=None):
        """
        Create a new session.
        
        Args:
            name (str, optional): Name for the session.
                                If None, a timestamp-based name will be used.
            
        Returns:
            str: ID of the created session.
        """
        import uuid
        
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        
        if name is None:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO sessions (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                (session_id, name, timestamp, timestamp)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created session: {name} (ID: {session_id})")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def add_operation(self, operation, session_id):
        """
        Add an operation to the timeline.
        
        Args:
            operation (Operation): The operation to add.
            session_id (str): The session ID to add the operation to.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Add to database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO operations (id, timestamp, operation_type, parameters, description, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    operation.id, 
                    operation.timestamp, 
                    operation.operation_type, 
                    json.dumps(operation.parameters), 
                    operation.description,
                    session_id
                )
            )
            
            # Update session last_modified timestamp
            cursor.execute(
                "UPDATE sessions SET last_modified = ? WHERE id = ?",
                (datetime.now().timestamp(), session_id)
            )
            
            conn.commit()
            conn.close()
            
            # Add to current timeline
            self.current_timeline.append(operation)
            self.current_position = len(self.current_timeline) - 1
            
            logger.info(f"Added operation to timeline: {operation.description}")
            return True
        except Exception as e:
            logger.error(f"Error adding operation to timeline: {e}")
            return False
    
    def get_operations(self, session_id, limit=None):
        """
        Get operations for a session.
        
        Args:
            session_id (str): The session ID to get operations for.
            limit (int, optional): Maximum number of operations to retrieve.
            
        Returns:
            list: List of operations.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if limit:
                cursor.execute(
                    "SELECT * FROM operations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM operations WHERE session_id = ? ORDER BY timestamp",
                    (session_id,)
                )
            
            operations = []
            for row in cursor.fetchall():
                operation = Operation(
                    operation_type=row[2],
                    parameters=json.loads(row[3]),
                    description=row[4]
                )
                operation.id = row[0]
                operation.timestamp = row[1]
                operations.append(operation)
            
            conn.close()
            
            logger.info(f"Retrieved {len(operations)} operations for session {session_id}")
            return operations
        except Exception as e:
            logger.error(f"Error retrieving operations: {e}")
            return []
    
    def get_sessions(self):
        """
        Get all sessions.
        
        Returns:
            list: List of sessions.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[2],
                    'last_modified': row[3]
                })
            
            conn.close()
            
            logger.info(f"Retrieved {len(sessions)} sessions")
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return []
    
    def load_session(self, session_id):
        """
        Load a session into the current timeline.
        
        Args:
            session_id (str): The session ID to load.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            operations = self.get_operations(session_id)
            
            self.current_timeline = operations
            self.current_position = len(operations) - 1 if operations else -1
            
            logger.info(f"Loaded session {session_id} with {len(operations)} operations")
            return True
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def revert_to(self, position):
        """
        Revert to a specific position in the timeline.
        
        Args:
            position (int): The position to revert to.
            
        Returns:
            list: Operations up to the specified position.
        """
        if not self.current_timeline:
            logger.warning("Cannot revert: Timeline is empty")
            return []
        
        if position < 0 or position >= len(self.current_timeline):
            logger.warning(f"Cannot revert: Position {position} is out of range")
            return []
        
        self.current_position = position
        logger.info(f"Reverted to position {position}: {self.current_timeline[position].description}")
        
        return self.current_timeline[:position + 1]
    
    def move_back(self, steps=1):
        """
        Move back in the timeline.
        
        Args:
            steps (int): Number of steps to move back.
            
        Returns:
            Operation or None: The operation at the new position, or None if out of range.
        """
        new_position = self.current_position - steps
        
        if new_position < -1:
            logger.warning(f"Cannot move back {steps} steps: Out of range")
            return None
        
        self.current_position = new_position
        
        if new_position == -1:
            logger.info("Moved back to beginning of timeline")
            return None
        else:
            logger.info(f"Moved back to position {new_position}: {self.current_timeline[new_position].description}")
            return self.current_timeline[new_position]
    
    def move_forward(self, steps=1):
        """
        Move forward in the timeline.
        
        Args:
            steps (int): Number of steps to move forward.
            
        Returns:
            Operation or None: The operation at the new position, or None if out of range.
        """
        new_position = self.current_position + steps
        
        if new_position >= len(self.current_timeline):
            logger.warning(f"Cannot move forward {steps} steps: Out of range")
            return None
        
        self.current_position = new_position
        logger.info(f"Moved forward to position {new_position}: {self.current_timeline[new_position].description}")
        
        return self.current_timeline[new_position]
    
    def get_current_operation(self):
        """
        Get the current operation.
        
        Returns:
            Operation or None: The current operation, or None if at the beginning of the timeline.
        """
        if self.current_position == -1 or not self.current_timeline:
            return None
        
        return self.current_timeline[self.current_position]
    
    def get_operations_by_type(self, operation_type, session_id):
        """
        Get operations of a specific type.
        
        Args:
            operation_type (str): The operation type to filter by.
            session_id (str): The session ID to get operations for.
            
        Returns:
            list: List of operations of the specified type.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM operations WHERE operation_type = ? AND session_id = ? ORDER BY timestamp",
                (operation_type, session_id)
            )
            
            operations = []
            for row in cursor.fetchall():
                operation = Operation(
                    operation_type=row[2],
                    parameters=json.loads(row[3]),
                    description=row[4]
                )
                operation.id = row[0]
                operation.timestamp = row[1]
                operations.append(operation)
            
            conn.close()
            
            logger.info(f"Retrieved {len(operations)} operations of type {operation_type}")
            return operations
        except Exception as e:
            logger.error(f"Error retrieving operations by type: {e}")
            return []
    
    def search_operations(self, search_term, session_id):
        """
        Search operations by description.
        
        Args:
            search_term (str): The term to search for.
            session_id (str): The session ID to search in.
            
        Returns:
            list: List of matching operations.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM operations WHERE description LIKE ? AND session_id = ? ORDER BY timestamp",
                (f"%{search_term}%", session_id)
            )
            
            operations = []
            for row in cursor.fetchall():
                operation = Operation(
                    operation_type=row[2],
                    parameters=json.loads(row[3]),
                    description=row[4]
                )
                operation.id = row[0]
                operation.timestamp = row[1]
                operations.append(operation)
            
            conn.close()
            
            logger.info(f"Found {len(operations)} operations matching '{search_term}'")
            return operations
        except Exception as e:
            logger.error(f"Error searching operations: {e}")
            return []
