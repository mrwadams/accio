import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.utils.database import initialize_database, verify_team_access
from app.utils.embeddings import EmbeddingService
from app.utils.llm_service import LLMService

# Load environment variables
load_dotenv()

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
        # Initialize Google GenAI client
        client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Initialize services
        st.session_state.llm_service = LLMService(client)
        st.session_state.embedding_service = EmbeddingService(client)
        
        # Initialize database if needed
        initialize_database()
        
        st.session_state.initialized = True
    except Exception as e:
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