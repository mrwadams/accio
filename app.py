import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai
import logging

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.utils.database import initialize_database, verify_team_access

# Import the configured LLM service instance
from app.utils.genai_client import configured_llm_service, configured_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Always initialize the database (safe to run multiple times)
initialize_database()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'team_id' not in st.session_state:
    st.session_state.team_id = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# Configure the main page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Authentication handling
def authenticate():
    st.sidebar.title("Authentication")
    team_id = st.sidebar.text_input("Team ID")
    access_code = st.sidebar.text_input("Access Code", type="password")

    if st.sidebar.button("Login"):
        if team_id == "admin":
            admin_code = os.getenv("ADMIN_CODE")
            if access_code == admin_code:
                st.session_state.authenticated = True
                st.session_state.team_id = "admin"
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.sidebar.error("Invalid admin access code")
        elif verify_team_access(team_id, access_code):
            st.session_state.authenticated = True
            st.session_state.team_id = team_id
            st.session_state.is_admin = False
            st.rerun()
        else:
            st.sidebar.error("Invalid team ID or access code")

# Initialize services if authenticated
if st.session_state.authenticated and 'initialized' not in st.session_state:
    try:
        # Services are now initialized in genai_client.py
        # Just assign the pre-configured instances to session state
        st.session_state.llm_service = configured_llm_service
        
        # Assuming EmbeddingService also needs the correct client
        # If EmbeddingService exists and uses the client, update it similarly
        # st.session_state.embedding_service = EmbeddingService(configured_client) 
        # If EmbeddingService doesn't exist or doesn't need the client, remove/comment out

        st.session_state.initialized = True
        logger.info("Services assigned to session state.")
    except Exception as e:
        # Log the full traceback for debugging
        logger.exception(f"Error assigning services to session state: {str(e)}") 
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()

# Handle authentication and page display
if not st.session_state.authenticated:
    authenticate()
else:
    # Sidebar: user info and logout
    with st.sidebar:
        st.markdown(
            f"**Logged in as:** {'🛡️ Admin' if st.session_state.is_admin else f'👥 Team {st.session_state.team_id}'}"
        )
        if st.button("Logout"):
            for key in ['authenticated', 'team_id', 'is_admin', 'initialized']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.markdown("---")

    # Define all pages (only those that exist)
    pages = [
        st.Page("app_pages/00_💬_Chat.py", title="Chat", icon="💬"),
        st.Page("app_pages/01_📄_Document_Management.py", title="Documents", icon="📄"),
        st.Page("app_pages/05_📊_Feedback_Analysis.py", title="Feedback", icon="📊"),
    ]

    # Add admin team management page for admins
    if st.session_state.is_admin:
        pages.insert(0, st.Page("app_pages/10_🛡️_Admin_Team_Management.py", title="Team Management", icon="🛡️"))

    # Filter by role
    if st.session_state.is_admin:
        nav_pages = pages
    else:
        nav_pages = [pages[0], pages[1]]

    # Show navigation and run selected page
    selected_page = st.navigation(nav_pages)
    selected_page.run() 