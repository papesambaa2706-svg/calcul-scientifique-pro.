import streamlit as st
from laser_simulation import laser_page


def cavity_page():
    """Wrapper page for direct access to the cavity losses module."""
    st.session_state["laser_main_option"] = "🔲 Pertes de Cavité"
    laser_page()
