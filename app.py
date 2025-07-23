import streamlit as st
from controllers.ui_controller import UIController

if __name__ == "__main__":
    """
    Main entry point for the Streamlit application.

    Responsibilities:
    -----------------
    - Instantiates the UIController which manages the application's UI logic and state.
    - Triggers the Streamlit app run process through the controller's `run` method.

    Usage:
    ------
    Run this script directly to launch the Streamlit frontend.

    Example:
    --------
    python app.py
    """
    ui = UIController()
    ui.run()
