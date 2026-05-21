import streamlit as st
from laser_simulation import laser_page


def gaussian_page():
    """Wrapper page for direct access to the Gaussian profile module."""
    st.session_state["laser_main_option"] = "📊 Profil Gaussien"
    laser_page()
