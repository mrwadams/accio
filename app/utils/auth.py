"""Authentication utilities for the RAG chatbot."""
import os
import json
from typing import Optional, Dict, Literal

Role = Literal["admin", "user"]

class AuthResult:
    def __init__(self, role: Role, team: Optional[str] = None):
        self.role = role
        self.team = team

def get_team_codes() -> Dict[str, str]:
    """Get team codes from environment variable."""
    try:
        return json.loads(os.getenv("TEAM_CODES", "{}"))
    except json.JSONDecodeError:
        return {}

def authenticate(code: str) -> Optional[AuthResult]:
    """
    Authenticate a user based on their access code.
    
    Args:
        code: The access code to validate
        
    Returns:
        AuthResult if authentication successful, None otherwise
    """
    TEAM_CODES = get_team_codes()
    ADMIN_CODE = os.getenv("ADMIN_CODE")
    
    if code == ADMIN_CODE:
        return AuthResult(role="admin")
    
    for team, team_code in TEAM_CODES.items():
        if code == team_code:
            return AuthResult(role="user", team=team)
    
    return None 