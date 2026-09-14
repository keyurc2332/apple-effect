import streamlit as st
st.set_page_config(page_title="Apple Effect", page_icon="🍎", layout="wide")

try:
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    PROC    = Path("data/processed")
    REPORTS = Path("outputs/reports")

    # Test basic file loading immediately
    test = pd.read_csv(PROC / "absa_aspect_rows.csv")
    st.success(f"✅ Data loaded: {len(test):,} rows")

except Exception as e:
    st.error(f"Startup error: {type(e).__name__}: {e}")
    import traceback
    st.code(traceback.format_exc())
