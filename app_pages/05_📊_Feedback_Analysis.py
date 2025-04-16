import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from app.utils.feedback import FeedbackService
from app.utils.embeddings import VectorStore

# Load environment variables
load_dotenv()

def check_admin_auth():
    """Check if user has admin access"""
    if 'admin_authenticated' not in st.session_state:
        admin_key = st.text_input("Enter admin key:", type="password")
        if admin_key == os.getenv("ADMIN_CODE"):
            st.session_state.admin_authenticated = True
        else:
            st.error("Invalid admin key")
            st.stop()
    return st.session_state.admin_authenticated

def main():
    # Check authentication
    if not check_admin_auth():
        return

    st.title("📊 Feedback Analysis")
    st.markdown("""
    Analyze user feedback and reported issues to improve the chatbot's performance.
    """)

    # Initialize services
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if "feedback_service" not in st.session_state:
        st.session_state.feedback_service = FeedbackService(
            db_params=st.session_state.vector_store.db_params
        )

    # Date range selector
    st.header("Filter Data")
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.selectbox(
            "Time Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
            index=1
        )
    
    with col2:
        schema_filter = st.selectbox(
            "Schema",
            ["All Schemas", "default", "team_a", "team_b"],  # Will be dynamic
            index=0
        )

    # Calculate date range
    end_date = datetime.now()
    if date_range == "Last 24 Hours":
        start_date = end_date - timedelta(days=1)
    elif date_range == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif date_range == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    else:  # All Time
        start_date = None
        end_date = None

    # Get feedback statistics
    stats = st.session_state.feedback_service.get_feedback_stats(
        schema_name=None if schema_filter == "All Schemas" else schema_filter,
        start_date=start_date,
        end_date=end_date
    )

    # Display statistics
    st.header("Feedback Overview")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Feedback",
            stats['total'],
            help="Total number of feedback entries"
        )
    
    with col2:
        helpful_pct = (stats['helpful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        st.metric(
            "Helpful Responses",
            f"{stats['helpful']} ({helpful_pct:.1f}%)",
            help="Number of responses marked as helpful"
        )
    
    with col3:
        not_helpful_pct = (stats['not_helpful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        st.metric(
            "Not Helpful",
            f"{stats['not_helpful']} ({not_helpful_pct:.1f}%)",
            help="Number of responses marked as not helpful"
        )
    
    with col4:
        issues_pct = (stats['issues'] / stats['total'] * 100) if stats['total'] > 0 else 0
        st.metric(
            "Issues Reported",
            f"{stats['issues']} ({issues_pct:.1f}%)",
            help="Number of issues reported"
        )

    # Display recent issues
    st.header("Recent Issues")
    issues = st.session_state.feedback_service.get_recent_issues(
        limit=10,
        schema_name=None if schema_filter == "All Schemas" else schema_filter
    )

    if issues:
        for issue in issues:
            with st.expander(f"Issue from {issue['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(f"**Schema:** {issue['schema_name'] or 'default'}")
                st.markdown(f"**Query:** {issue['query']}")
                st.markdown(f"**Description:** {issue['description']}")
                
                # Add action buttons
                col1, col2, spacer = st.columns([1, 1, 4])
                with col1:
                    if st.button("Mark Resolved", key=f"resolve_{issue['id']}"):
                        # TODO: Implement issue resolution logic
                        st.success("Issue marked as resolved")
                with col2:
                    if st.button("View Details", key=f"details_{issue['id']}"):
                        # TODO: Implement detailed view
                        st.info("Detailed view coming soon")
    else:
        st.info("No issues reported in the selected time period.")

    # Footer
    st.markdown("---")
    st.caption("RAG Chatbot Admin Panel - Feedback Analysis")

if __name__ == "__main__":
    main() 