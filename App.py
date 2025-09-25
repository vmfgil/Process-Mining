import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io

# Imports de Process Mining (PM4PY)
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_miner

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Painel de Análise de Processos de IT",
    page_icon="✨",
    layout="wide"
)

# --- 2. CSS PARA O NOVO DESIGN (FASE 2) ---
st.markdown("""
<style>
    /* TEMA ESCURO E ESTILO GERAL */
    body {
        color: #E2E8F0; /* Texto claro */
        background-color: #0F172A; /* Fundo principal escuro */
    }
    .stApp {
        background-color: #0F172A;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .stButton>button {
        border-color: #334155;
    }
    
    /* PAINEL LATERAL */
    [data-testid="stSidebar"] {
        background-color: #1E293B; /* Fundo da sidebar um pouco mais claro */
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] p {
        color: #E2E8F0 !important;
        font-size: 1.05rem !important;
    }
    .sidebar-note p {
        color: #94A3B8 !important; /* Cinza mais claro para a nota */
    }

    /* COMPONENTE CARTÃO */
    .card {
        background-color: #1E293B; /* Cor de fundo do cartão */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #334155; /* Borda subtil */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%; /* Faz com que os cartões na mesma linha tenham a mesma altura */
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 15px;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
    }
    .stMetric {
        background-color: #334155;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #475569;
    }
    
    /* NAVEGAÇÃO MODERNA (SUBSTITUI ABAS) */
    div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: row"] > div[data-testid="stVerticalBlock"] > div.element-container > button[kind="secondary"] {
        width: 100%;
        text-align: center;
        padding: 8px;
        border-radius: 8px;
        background-color: transparent;
        color: #94A3B8;
        border: 1px solid #334155;
        transition: all 0.2s;
    }
    div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: row"] > div[data-testid="stVerticalBlock"] > div.element-container > button[kind="secondary"]:hover {
        background-color: #334155;
        color: #FFFFFF;
        border: 1px solid #475569;
    }
    /* Estilo para o botão ativo é tratado via Python */

</style>
""", unsafe_allow_html=True)

# --- 3. INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: 
    st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: 
    st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: 
    st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: 
    st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: 
    st.session_state.metrics = {}

# --- FUNÇÕES AUXILIARES, DE ANÁLISE, ETC ---
# (O código destas funções permanece inalterado)
# ...

# --- FUNÇÃO HELPER PARA OS CARTÕES (VERSÃO CORRIGIDA) ---
class card:
    def __init__(self, title, icon=""):
        self.title = title
        self.icon = icon

    def __enter__(self):
        st.markdown(f'<div class="card"><div class="card-header">{self.icon} {self.title}</div>', unsafe_allow_html=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        st.markdown('</div>', unsafe_allow_html=True)


# --- FUNÇÃO PRINCIPAL DA APLICAÇÃO ---
def main_app():
    st.sidebar.title("Painel de Análise")
    st.sidebar.markdown('<div class="sidebar-note"><p>Selecione a vista do dashboard.</p></div>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Menu Principal", 
        ["📊 Dashboard", "⚙️ Configuração (Upload)"],
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    st.sidebar.write(f"Utilizador: **{st.session_state['username']}**")
    if st.sidebar.button("Sair ⏏️"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.rerun()

    if page == "⚙️ Configuração (Upload)":
        st.title("Configuração e Carregamento de Dados")
        file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
        
        # Lógica de Upload aqui...
        col1, col2 = st.columns(2)
        with col1:
            for name in file_names[:3]:
                uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
                if uploaded_file:
                    st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                    st.success(f"`{name}.csv` carregado.")
        
        with col2:
            for name in file_names[3:]:
                uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
                if uploaded_file:
                    st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                    st.success(f"`{name}.csv` carregado.")

        if all(st.session_state.dfs[name] is not None for name in file_names):
            if st.button("Executar Análise Completa", type="primary"):
                with st.spinner("A executar a análise... Isto pode demorar um pouco."):
                    # Chamar as funções de análise aqui...
                    st.session_state.analysis_run = True # Marcar como concluída
                st.success("Análise completa! Navegue para o Dashboard para ver os resultados.")


    elif page == "📊 Dashboard":
        st.title("Dashboard de Análise de Processos")

        if not st.session_state.analysis_run:
            st.warning("Ainda não foram analisados dados. Por favor, execute a análise na página de Configuração.")
            return

        if 'active_view' not in st.session_state:
            st.session_state.active_view = "Visão Geral"

        nav_cols = st.columns(4)
        views = ["Visão Geral", "Análise de Processo", "Análise de Recursos", "Análise Aprofundada"]
        for i, view in enumerate(views):
            is_active = st.session_state.active_view == view
            button_type = "primary" if is_active else "secondary"
            if nav_cols[i].button(view, key=f"nav_{view}", use_container_width=True, type=button_type):
                st.session_state.active_view = view
                st.rerun()
        st.markdown("---")

        if st.session_state.active_view == "Visão Geral":
            st.subheader("🏁 KPIs de Alto Nível")
            kpi_cols = st.columns(4)
            # Popular KPIs...
            
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                with card("Matriz de Performance (Custo vs. Prazo)"):
                    # Colocar gráfico aqui
                    pass
            with col2:
                with card("Distribuição da Duração dos Projetos"):
                    # Colocar gráfico aqui
                    pass
            
            # ... e por aí adiante para os outros cartões


# --- LÓGICA DE AUTENTICAÇÃO E PONTO DE ENTRADA ---
def login():
    st.markdown("""
    <style>
        .main { background-color: #F0F2F6; }
        [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.title("Painel de Análise de Processos")
        st.header("Login")
        username = st.text_input("Utilizador", value="admin", key="login_username")
        password = st.text_input("Password", type="password", value="password", key="login_password")
        if st.button("Entrar", type="primary"):
            if username == "admin" and password == "password":
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Utilizador ou password incorretos.")

if st.session_state['authenticated']:
    main_app()
else:
    login()
