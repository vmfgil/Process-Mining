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
    page_title="Painel de Análise de Processos",
    page_icon="✨",
    layout="wide"
)

# --- 2. CSS PARA O NOVO DESIGN (FASE 2 - VERSÃO FINAL) ---
st.markdown("""
<style>
    /* TEMA ESCURO E ESTILO GERAL */
    body, .stApp {
        background-color: #0F172A;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stDataFrame, .stTable {
        color: #FFFFFF !important;
    }
    .stButton>button {
        border-color: #334155;
    }
    .stTextInput label, .stFileUploader label {
        color: #E2E8F0 !important;
    }
    
    /* PAINEL LATERAL */
    [data-testid="stSidebar"] {
        background-color: #1E2B3A;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] p, [data-testid="stSidebar"] .stButton>button {
        color: #E2E8F0 !important;
    }
    .sidebar-note p {
        color: #94A3B8 !important;
    }

    /* COMPONENTE CARTÃO */
    .card {
        background-color: #1E2B3A;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #334155;
        height: 100%;
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 15px;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
    }
    
    /* ESTILOS DE MÉTRICAS E ALERTAS DE ALTO CONTRASTE */
    .stMetric {
        background-color: #334155;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #475569;
    }
    [data-testid="stAlert"][data-st-alert-type="warning"] p,
    [data-testid="stAlert"][data-st-alert-type="warning"] {
        color: #FBBF24 !important; /* Amarelo claro */
    }
    
    /* NAVEGAÇÃO SECUNDÁRIA (BOTÕES) */
    div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: row"] > div[data-testid="stVerticalBlock"] > div.element-container > button[kind="secondary"] {
        background-color: transparent;
        color: #94A3B8;
        border: 1px solid #334155;
    }
    div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: row"] > div[data-testid="stVerticalBlock"] > div.element-container > button[kind="primary"] {
        background-color: #3B82F6;
        color: #FFFFFF;
        border: 1px solid #3B82F6;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. INICIALIZAÇÃO DO ESTADO DA SESSÃO (CORRIGIDO E COMPLETO) ---
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
# (O código destas funções permanece inalterado e completo no ficheiro)
# ...

# --- FUNÇÃO HELPER PARA OS CARTÕES ---
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
        ["📊 Dashboard", "⚙️ Configuração (upload de dados sobre os processos)"],
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    st.sidebar.write(f"Utilizador: **{st.session_state['username']}**")
    if st.sidebar.button("Sair ⏏️"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.rerun()

    if page == "⚙️ Configuração (upload de dados sobre os processos)":
        with card("Carregamento e Análise de Dados"):
            st.subheader("1. Upload dos Ficheiros de Dados (.csv)")
            file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
            
            col1, col2 = st.columns(2)
            # ... (código de upload) ...

            if all(st.session_state.dfs[name] is not None for name in file_names):
                st.subheader("2. Execução da Análise")
                if st.button("Executar Análise Completa", type="primary", use_container_width=True):
                    with st.spinner("A executar a análise... Isto pode demorar um pouco."):
                        # ... (código que corre a análise) ...
                        st.session_state.analysis_run = True
                    st.success("Análise completa! Navegue para o Dashboard para ver os resultados.")

    elif page == "📊 Dashboard":
        st.title("Dashboard de Análise de Processos")

        if not st.session_state.analysis_run:
            st.warning("Ainda não foram analisados dados. Por favor, execute a análise na página de Configuração.")
            return

        if 'active_view' not in st.session_state:
            st.session_state.active_view = "Visão Geral"

        # ... (código dos botões de navegação) ...

        # --- CONTEÚDO DAS VISTAS ---
        if st.session_state.active_view == "Visão Geral":
            kpi_cols = st.columns(4)
            kpi_data = st.session_state.tables_pre_mining['kpi_data']
            kpi_cols[0].metric(label="Total de Projetos", value=kpi_data['Total de Projetos'])
            # ... (restantes KPIs) ...

            c1, c2 = st.columns(2)
            with c1:
                with card("Matriz de Performance (Custo vs. Prazo)"):
                    st.image(st.session_state.plots_pre_mining['performance_matrix'], use_column_width=True)
            with c2:
                with card("Distribuição da Duração dos Projetos"):
                    st.image(st.session_state.plots_pre_mining['case_durations_boxplot'], use_column_width=True)
            # ... (restante conteúdo da Visão Geral)

        elif st.session_state.active_view == "Análise de Processo":
            # ... (conteúdo da Análise de Processo)

        elif st.session_state.active_view == "Análise de Recursos":
            # ... (conteúdo da Análise de Recursos)

        elif st.session_state.active_view == "Análise Aprofundada":
            # ... (conteúdo da Análise Aprofundada)


# --- LÓGICA DE AUTENTICAÇÃO E PONTO DE ENTRADA ---
def login():
    with card("Login"):
        st.header("Painel de Análise de Processos")
        username = st.text_input("Utilizador", value="admin", key="login_username")
        password = st.text_input("Password", type="password", value="password", key="login_password")
        if st.button("Entrar", type="primary", use_container_width=True):
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
