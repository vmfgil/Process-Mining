import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

# Import your full analysis pipeline (unchanged) from a separate module.
# E.g., put all notebook code into analysis.py with two functions:
#   - run_pre_mining(df_projects, df_tasks, df_resources, df_allocs, df_deps)
#   - run_post_mining(df_projects, df_tasks, df_resources, df_allocs, df_deps)
import analysis  

#───────────────────────────────
# 1. CONFIGURAÇÃO & ESTILO
#───────────────────────────────
st.set_page_config(
    page_title="IT Resource Mgmt Dashboard",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for brand-like look (gradient header, modern font)
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, #root, .viewerBadge_container {
        font-family: 'Inter', sans-serif;
      }
      .css-18e3th9 {
        background: linear-gradient(90deg, #0D47A1, #1976D2);
      }
      .css-1v0mbdj e1fqkh3o4 {
        color: white;
      }
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0D47A1;
      }
      .sidebar .css-1d391kg {
        font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navegação")
page = st.sidebar.radio("", [
    "1. Carregar Dados",
    "2. Executar Análise",
    "3. Resultados"
])

#───────────────────────────────
# 2. SESSION STATE STORAGE
#───────────────────────────────
if "dfs" not in st.session_state:
    st.session_state.dfs = {}

#───────────────────────────────
# 3. PAGE: CARREGAR DADOS
#───────────────────────────────
if page == "1. Carregar Dados":
    st.header("📂 Carregar e Pré-visualizar Dados")
    uploaded = st.file_uploader(
        "Arraste os 5 CSVs ou selecione aqui:",
        type="csv",
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded:
        # map filenames to dfs
        required = {
            "projects.csv": "projects",
            "tasks.csv": "tasks",
            "resources.csv": "resources",
            "resource_allocations.csv": "allocs",
            "dependencies.csv": "deps"
        }
        missing = set(required) - {f.name for f in uploaded}
        if missing:
            st.error(f"Faltam estes ficheiros: {', '.join(missing)}")
        else:
            # read into session_state.dfs
            for f in uploaded:
                key = required[f.name]
                st.session_state.dfs[key] = pd.read_csv(f)
            st.success("📥 Ficheiros carregados com sucesso!")
            # preview heads
            for name, df in st.session_state.dfs.items():
                st.subheader(f"Preview: {name}")
                st.dataframe(df.head(), height=200)

#───────────────────────────────
# 4. PAGE: EXECUTAR ANÁLISE
#───────────────────────────────
if page == "2. Executar Análise":
    st.header("⚙️ Executar Pipeline de Análise")
    if len(st.session_state.dfs) < 5:
        st.warning("Antes de executar, carregue todos os 5 ficheiros na secção “Carregar Dados”.")
    else:
        if st.button("▶️ Executar Análise Completa"):
            with st.spinner("🔄 A correr análises pré-mineração…"):
                analysis.run_pre_mining(
                    st.session_state.dfs["projects"],
                    st.session_state.dfs["tasks"],
                    st.session_state.dfs["resources"],
                    st.session_state.dfs["allocs"],
                    st.session_state.dfs["deps"]
                )
            with st.spinner("🔄 A correr análises pós-mineração…"):
                analysis.run_post_mining(
                    st.session_state.dfs["projects"],
                    st.session_state.dfs["tasks"],
                    st.session_state.dfs["resources"],
                    st.session_state.dfs["allocs"],
                    st.session_state.dfs["deps"]
                )
            st.success("✅ Análise concluída! Veja “Resultados”.")
            st.balloons()

#───────────────────────────────
# 5. PAGE: RESULTADOS
#───────────────────────────────
if page == "3. Resultados
