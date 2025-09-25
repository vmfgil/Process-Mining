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
        font-size: 1.2rem;
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
    .nav-button {
        width: 100%;
        text-align: center;
        padding: 8px;
        border-radius: 8px;
        background-color: transparent;
        color: #94A3B8;
        border: 1px solid #334155;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-button:hover {
        background-color: #334155;
        color: #FFFFFF;
    }
    .nav-button-active {
        background-color: #3B82F6;
        color: #FFFFFF;
        border: 1px solid #3B82F6;
    }

    /* ESCONDER BOTÕES DE RÁDIO PADRÃO PARA CRIAR BOTÕES PERSONALIZADOS */
    [data-testid="stRadio"] > label {
        display: none !important;
    }

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
# ... (outras inicializações de estado permanecem as mesmas)

# --- FUNÇÕES AUXILIARES ---
# ... (funções auxiliares permanecem as mesmas)

# --- FUNÇÕES DE ANÁLISE ---
# ... (funções de análise permanecem as mesmas)

# --- FUNÇÃO HELPER PARA OS CARTÕES (FASE 2) ---
@st.contextmanager
def card(title, icon=""):
    st.markdown(f'<div class="card-header">{icon} {title}</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        yield
        st.markdown('</div>', unsafe_allow_html=True)

# --- FUNÇÃO PRINCIPAL DA APLICAÇÃO ---
def main_app():
    st.sidebar.title("Painel de Análise")
    st.sidebar.markdown('<div class="sidebar-note"><p>Selecione a vista do dashboard.</p></div>', unsafe_allow_html=True)

    # NAVEGAÇÃO PRINCIPAL NA SIDEBAR
    page = st.sidebar.radio(
        "Menu Principal", 
        ["📊 Dashboard", "⚙️ Configuração (Upload)"],
        label_visibility="collapsed" # Esconde o label "Menu Principal"
    )
    
    st.sidebar.divider()
    st.sidebar.write(f"Utilizador: **{st.session_state['username']}**")
    if st.sidebar.button("Sair ⏏️"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.rerun()

    # CONTEÚDO DA PÁGINA
    if page == "⚙️ Configuração (Upload)":
        st.title("Configuração e Carregamento de Dados")
        # ... (código de upload permanece igual)

    elif page == "📊 Dashboard":
        st.title("Dashboard de Análise de Processos")

        if not st.session_state.analysis_run:
            st.warning("Ainda não foram analisados dados. Por favor, execute a análise na página de Configuração.")
            if st.button("Executar Análise"):
                # Lógica para correr análise
                pass # Simplificado por agora
            return

        # NAVEGAÇÃO SECUNDÁRIA (SUBSTITUI AS ABAS)
        if 'active_view' not in st.session_state:
            st.session_state.active_view = "Visão Geral"

        nav_cols = st.columns(4)
        views = ["Visão Geral", "Análise de Processo", "Análise de Recursos", "Análise Aprofundada"]
        for i, view in enumerate(views):
            with nav_cols[i]:
                is_active = st.session_state.active_view == view
                button_class = "nav-button-active" if is_active else "nav-button"
                if st.button(view, key=f"nav_{view}"):
                    st.session_state.active_view = view
                    st.rerun()
        st.markdown("---")


        # VISTA: VISÃO GERAL
        if st.session_state.active_view == "Visão Geral":
            st.subheader("🏁 KPIs de Alto Nível")
            kpi_cols = st.columns(4)
            kpi_data = st.session_state.tables_pre_mining['kpi_data']
            with kpi_cols[0]:
                st.metric(label="Total de Projetos", value=kpi_data['Total de Projetos'])
            with kpi_cols[1]:
                st.metric(label="Total de Tarefas", value=kpi_data['Total de Tarefas'])
            with kpi_cols[2]:
                st.metric(label="Total de Recursos", value=kpi_data['Total de Recursos'])
            with kpi_cols[3]:
                st.metric(label="Duração Média (dias)", value=kpi_data['Duração Média (dias)'])

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                with card("Matriz de Performance (Custo vs. Prazo)"):
                    st.image(st.session_state.plots_pre_mining['performance_matrix'])
            with col2:
                with card("Distribuição da Duração dos Projetos"):
                    st.image(st.session_state.plots_pre_mining['case_durations_boxplot'])

            col3, col4 = st.columns(2)
            with col3:
                 with card("Top 5 Projetos Mais Longos"):
                    st.dataframe(st.session_state.tables_pre_mining['outlier_duration'], use_container_width=True)
            with col4:
                with card("Top 5 Projetos Mais Caros"):
                    st.dataframe(st.session_state.tables_pre_mining['outlier_cost'], use_container_width=True)

        # VISTA: ANÁLISE DE PROCESSO
        elif st.session_state.active_view == "Análise de Processo":
            st.subheader("🔎 Descoberta e Conformidade do Processo")
            
            with card("Modelo de Processo (Inductive Miner)"):
                st.image(st.session_state.plots_post_mining['model_inductive_petrinet'], use_column_width=True)

            with card("Modelo de Processo (Heuristics Miner)"):
                st.image(st.session_state.plots_post_mining['model_heuristic_petrinet'], use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                with card("Métricas de Qualidade (Inductive Miner)"):
                    st.image(st.session_state.plots_post_mining['metrics_inductive'])
            with col2:
                with card("Métricas de Qualidade (Heuristics Miner)"):
                    st.image(st.session_state.plots_post_mining['metrics_heuristic'])
        
        # ... (Outras vistas podem ser adicionadas aqui)

# --- LÓGICA DE AUTENTICAÇÃO E PONTO DE ENTRADA ---
# ... (código de login e ponto de entrada permanece o mesmo)
