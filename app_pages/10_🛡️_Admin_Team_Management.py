import streamlit as st
import json
from app.utils.database import get_all_teams, create_team

def main():
    if not st.session_state.get("is_admin", False):
        st.error("You do not have admin access to this page.")
        st.stop()

    st.title("🛡️ Admin: Team Management")
    st.markdown("Manage teams, access codes, and metadata.")

    # Add new team form
    st.header("Add New Team")
    with st.form("add_team_form"):
        team_id = st.text_input("Team ID")
        name = st.text_input("Team Name")
        access_code = st.text_input("Access Code")
        metadata_str = st.text_area("Metadata (JSON, optional)", "{}")
        submitted = st.form_submit_button("Add Team")

        if submitted:
            try:
                metadata = json.loads(metadata_str) if metadata_str.strip() else {}
            except Exception as e:
                st.error(f"Invalid JSON for metadata: {e}")
                st.stop()
            result = create_team(team_id, name, access_code, metadata)
            if result:
                st.success(f"Team '{team_id}' added/updated successfully.")
                st.rerun()
            else:
                st.error("Failed to add/update team. Check for errors or duplicate team_id.")

    st.header("All Teams")
    teams = get_all_teams()
    if not teams:
        st.info("No teams found.")
    else:
        for team in teams:
            with st.expander(f"{team['team_id']} - {team['name']}"):
                st.write(f"**Access Code:** {team['access_code']}")
                st.write(f"**Created At:** {team['created_at']}")
                st.json(team['metadata'])

if __name__ == "__main__":
    main() 