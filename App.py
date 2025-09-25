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
    body {
        color: #E2E8F0;
        background-color: #0F172A;
    }
    .stApp {
        background-color: #0F172A;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown p {
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
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] p {
        color: #E2E8F0 !important;
        font-size: 1.05rem !important;
    }
    .sidebar-note p {
        color: #94A3B8 !important;
    }

    /* COMPONENTE CARTÃO */
    .card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    [data-testid="stAlert"][data-st-alert-type="warning"] {
        background-color: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.2);
        color: #FBBF24 !important; /* Amarelo claro */
    }
    [data-testid="stAlert"][data-st-alert-type="warning"] p {
        color: #FBBF24 !important;
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

# --- 3. INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
# ... (outras inicializações) ...

# --- FUNÇÕES AUXILIARES E DE ANÁLISE ---
# ... (Todo o código das funções de análise permanece aqui, inalterado) ...

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
        st.title("Configuração e Carregamento de Dados")
        file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
        
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
                    plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                    st.session_state.plots_pre_mining = plots_pre
                    st.session_state.tables_pre_mining = tables_pre
                    st.session_state.event_log_for_cache = pm4py.convert_to_dataframe(event_log)
                    st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}

                    log_from_df = pm4py.convert_to_event_log(st.session_state.event_log_for_cache)
                    dfs_cache = st.session_state.dfs_for_cache
                    plots_post, metrics = run_post_mining_analysis(log_from_df, dfs_cache['projects'], dfs_cache['tasks_raw'], dfs_cache['resources'], dfs_cache['full_context'])
                    st.session_state.plots_post_mining = plots_post
                    st.session_state.metrics = metrics
                    st.session_state.analysis_run = True
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

        # --- VISTA: VISÃO GERAL ---
        if st.session_state.active_view == "Visão Geral":
            kpi_cols = st.columns(4)
            kpi_data = st.session_state.tables_pre_mining['kpi_data']
            kpi_cols[0].metric(label="Total de Projetos", value=kpi_data['Total de Projetos'])
            kpi_cols[1].metric(label="Total de Tarefas", value=kpi_data['Total de Tarefas'])
            kpi_cols[2].metric(label="Total de Recursos", value=kpi_data['Total de Recursos'])
            kpi_cols[3].metric(label="Duração Média (dias)", value=kpi_data['Duração Média (dias)'])

            c1, c2 = st.columns(2)
            with c1:
                with card("Matriz de Performance (Custo vs. Prazo)"):
                    st.image(st.session_state.plots_pre_mining['performance_matrix'], use_column_width=True)
            with c2:
                with card("Distribuição da Duração dos Projetos"):
                    st.image(st.session_state.plots_pre_mining['case_durations_boxplot'], use_column_width=True)
            
            c3, c4 = st.columns(2)
            with c3:
                 with card("Top 5 Projetos Mais Longos"):
                    st.dataframe(st.session_state.tables_pre_mining['outlier_duration'], use_container_width=True)
            with c4:
                with card("Top 5 Projetos Mais Caros"):
                    st.dataframe(st.session_state.tables_pre_mining['outlier_cost'], use_container_width=True)
        
        # --- VISTA: ANÁLISE DE PROCESSO ---
        elif st.session_state.active_view == "Análise de Processo":
            with card("Modelo de Processo (Inductive Miner)"):
                st.image(st.session_state.plots_post_mining['model_inductive_petrinet'], use_column_width=True)

            with card("Modelo de Processo (Heuristics Miner)"):
                st.image(st.session_state.plots_post_mining['model_heuristic_petrinet'], use_column_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                with card("Métricas de Qualidade (Inductive Miner)"):
                    st.image(st.session_state.plots_post_mining['metrics_inductive'], use_column_width=True)
            with c2:
                with card("Métricas de Qualidade (Heuristics Miner)"):
                    st.image(st.session_state.plots_post_mining['metrics_heuristic'], use_column_width=True)

            with card("Frequência das Variantes de Processo"):
                st.image(st.session_state.plots_pre_mining['variants_frequency'], use_column_width=True)
        
        # --- VISTA: ANÁLISE DE RECURSOS ---
        elif st.session_state.active_view == "Análise de Recursos":
            c1, c2 = st.columns(2)
            with c1:
                with card("Top Recursos por Horas Trabalhadas"):
                    st.image(st.session_state.plots_pre_mining['resource_workload'], use_column_width=True)
            with c2:
                with card("Top Handoffs entre Recursos"):
                    st.image(st.session_state.plots_pre_mining['resource_handoffs'], use_column_width=True)
            
            with card("Heatmap de Esforço (Recurso vs. Atividade)"):
                st.image(st.session_state.plots_pre_mining['resource_activity_matrix'], use_column_width=True)

            with card("Rede Social de Recursos"):
                st.image(st.session_state.plots_post_mining['resource_network_adv'], use_column_width=True)
        
        # --- VISTA: ANÁLISE APROFUNDADA ---
        elif st.session_state.active_view == "Análise Aprofundada":
            c1, c2 = st.columns(2)
            with c1:
                with card("Distribuição do Lead Time"):
                    st.image(st.session_state.plots_pre_mining['lead_time_hist'], use_column_width=True)
            with c2:
                with card("Distribuição do Throughput"):
                    st.image(st.session_state.plots_pre_mining['throughput_hist'], use_column_width=True)

            with card("Heatmap de Performance no Processo (Tempo entre atividades)"):
                st.image(st.session_state.plots_post_mining['performance_heatmap'], use_column_width=True)
            
            with card("Score de Conformidade ao Longo do Tempo"):
                st.image(st.session_state.plots_post_mining['conformance_over_time_plot'], use_column_width=True)


# --- LÓGICA DE AUTENTICAÇÃO E PONTO DE ENTRADA ---
def login():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .stTextInput label { color: #E2E8F0 !important; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
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

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Carregar o resto do código da app aqui, que estava a faltar
# (O código completo das funções de análise, etc., deve estar acima desta secção)

# Ponto de entrada final
if st.session_state['authenticated']:
    main_app()
else:
    login()
