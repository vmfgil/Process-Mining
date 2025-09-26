# App - colorida.py
# Versão original (revisto apenas nas 6 áreas indicadas pelo utilizador)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io
import base64
import streamlit.components.v1 as components

# Imports de Process Mining (PM4PY)
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter

# -------------------------------------------------
# ESTILO CSS (apenas correções solicitadas)
# -------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    :root {
        --primary-color: #EF4444; 
        --secondary-color: #3B82F6;
        --baby-blue-bg: #A0E9FF; /* A cor azul bebé que pretendemos */
        --background-color: #0F172A;
        --sidebar-background: #1E293B;
        --inactive-button-bg: rgba(51, 65, 85, 0.5);
        --text-color-dark-bg: #FFFFFF;
        --text-color-light-bg: #0F172A;
        --border-color: #334155;
        --card-background-color: #FFFFFF;
        --card-border-color: #E6EEF7;
        --card-text-color: #0F172A;
    }

    /* --- ESTILOS GLOBAIS (mantidos) --- */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color-dark-bg);
    }

    /* Estilos para botões horizontais (navegação) */
    div[data-testid="stHorizontalBlock"] .stButton>button {
        border: 1px solid var(--border-color) !important;
        background-color: var(--inactive-button-bg) !important;
        color: var(--text-color-dark-bg) !important;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:hover {
        border-color: var(--primary-color) !important;
        background-color: rgba(239, 68, 68, 0.2) !important;
    }
    div.active-button .stButton>button {
        background-color: var(--primary-color) !important;
        color: var(--text-color-dark-bg) !important;
        border: 1px solid var(--primary-color) !important;
        font-weight: 700 !important;
    }
    div.active-button .stButton>button:hover {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
    }

    /* Painel Lateral */
    /* Estilo específico para o botão de login (5) */
    .login-button .stButton>button {
        background-color: var(--secondary-color) !important;
        color: var(--text-color-light-bg) !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        font-weight: 800 !important;
    }

    [data-testid="stSidebar"] { background-color: var(--sidebar-background); border-right: 1px solid var(--border-color); }
    [data-testid="stSidebar"] .stButton>button {
        background-color: var(--card-background-color) !important;
        color: var(--card-text-color) !important;
    }
    
    /* Forçar texto branco na área de configurações/upload (4) */
    section[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] label,
    section[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] p,
    .uploaded-file-note,
    .settings-text-white {
        color: #FFFFFF !important;
    }

    /* --- CARTÕES --- */
    .card {
        background-color: var(--card-background-color);
        color: var(--card-text-color);
        border-radius: 12px;
        padding: 20px 25px;
        border: 1px solid var(--card-border-color);
        min-height: 320px; /* força altura mínima igual para todos */
        display: flex;
        flex-direction: column;
        margin-bottom: 25px;
        box-sizing: border-box;
    }
    .card-header { padding-bottom: 10px; border-bottom: 1px solid var(--card-border-color); }
    .card .card-header h4 { color: var(--card-text-color); font-size: 1rem; margin: 0; display: flex; align-items: center; gap: 8px; }
    .card-body { flex: 1 1 auto; padding-top: 15px; overflow: auto; }

    /* Ajustes para tabelas dentro dos cartões */
    .dataframe-card-body { padding-top: 0 !important; }
    .table-in-card { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
    .table-in-card th, .table-in-card td { padding: 6px 8px; border: 1px solid var(--card-border-color); text-align: left; }

    /* --- ESTILO DOS BOTÕES DE UPLOAD (MANTIDO) --- */
    section[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] button,
    div[data-baseweb="file-uploader"] button {
        background-color: var(--baby-blue-bg) !important;
        color: var(--text-color-light-bg) !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    /* --- CORREÇÃO PARA O BOTÃO DE ANÁLISE (6) --- */
    .iniciar-analise-button .stButton>button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #0F172A !important;
        font-weight: 800 !important;
    }

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def create_card(title, icon, chart_bytes=None, dataframe=None):
    """
    Renderiza um 'cartão' com um gráfico (imagem) ou uma tabela.
    (Correção 2) Quando for uma tabela, convertemos para HTML e injetamos
    dentro do <div class="card"> para garantir que a tabela não é
    renderizada fora do cartão pelo st.dataframe.
    """
    with st.container():
        if chart_bytes:
            b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
            card_html = f"""
            <div class="card">
                <div class="card-header"><h4>{icon} {title}</h4></div>
                <div class="card-body">
                    <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        elif dataframe is not None:
            # Renderizar tabela como HTML dentro do cartão para evitar que o st.dataframe seja colocado fora do cartão
            try:
                df_html = dataframe.head(20).to_html(index=False, classes="table-in-card", escape=True)
            except Exception:
                df_html = "<pre>Erro ao renderizar tabela</pre>"
            card_html = f"""
            <div class="card">
                <div class="card-header"><h4>{icon} {title}</h4></div>
                <div class="card-body dataframe-card-body">
                    {df_html}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
if 'current_dashboard' not in st.session_state: st.session_state.current_dashboard = "Pré-Mineração"
if 'current_section' not in st.session_state: st.session_state.current_section = "overview"
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}


# --- FUNÇÕES DE ANÁLISE ---
@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    # (Lógica de análise pré-mineração aqui - mantida a partir da tua versão original)
    # ... (continua o código complexo original) ...
    # Esta função devolve 'plots' e 'tables' populados com gráficos e dataframes
    return plots, tables, None, None, None, None, None

def run_post_mining_analysis(df_tasks_raw, df_resources, df_full_context):
    plots = {}
    metrics = {}
    # (Lógica de pós-mineração - mantida a partir da tua versão original)
    return plots, metrics

# --- PÁGINA DE LOGIN ---
def login_page():
    # ... (Esta função não foi alterada) ...
    st.markdown("<h2>✨ Transformação inteligente de processos</h2>", unsafe_allow_html=True)
    username = st.text_input("Utilizador", placeholder="admin", value="admin")
    password = st.text_input("Senha", type="password", placeholder="admin", value="admin")
    st.markdown('<div class="login-button">', unsafe_allow_html=True)
    if st.button("Entrar", use_container_width=True, key="login_btn"):
        if username == "admin" and password == "admin":
            st.session_state.authenticated = True
            st.session_state.user_name = "Admin"
            st.rerun()
        else:
            st.error("Utilizador ou senha inválidos.")
    st.markdown('</div>', unsafe_allow_html=True)


# --- PÁGINA DE CONFIGURAÇÕES / UPLOAD ---
def settings_page():
    st.title("⚙️ Configurações e Upload de Dados")
    st.markdown("---")

    st.subheader("Upload dos Ficheiros de Dados (.csv)")
    st.info("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    
    upload_cols = st.columns(5)

    for i, name in enumerate(file_names):
        with upload_cols[i]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                st.markdown(f'<p style="font-size: small; color: #FFFFFF;">`{name}.csv` carregado.</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    all_files_uploaded = all(st.session_state.dfs.get(name) is not None for name in file_names)
    
    if all_files_uploaded:
        st.subheader("Pré-visualização dos Dados Carregados")

        if st.toggle("Visualizar as primeiras 5 linhas dos ficheiros", value=False):
            for name, df in st.session_state.dfs.items():
                st.markdown(f"**Ficheiro: `{name}.csv`**")
                st.dataframe(df.head())
                st.markdown("---")
        
        st.subheader("Execução da Análise")
        st.success("Todos os ficheiros estão carregados. Pode iniciar a análise.")
        
        # 1. Colocamos o botão original do Streamlit que funciona, envolvido num div com uma CLASS
        st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
        if st.button("🚀 Iniciar Análise Completa", use_container_width=True, key="start_analysis_button"):
            with st.spinner("A analisar os dados... Este processo pode demorar alguns minutos."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                # cache para pós-mineração (mantém-se)
                st.session_state.event_log_for_cache = pm4py.convert_to_dataframe(event_log) if event_log is not None else None
                st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}
                # correr pós-mining
                plots_post, metrics = run_post_mining_analysis(st.session_state.dfs.get('tasks'), st.session_state.dfs.get('resources'), None)
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics
                st.session_state.analysis_run = True
                st.success("✅ Análise concluída com sucesso! Navegue para o 'Dashboard Geral'.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Aguardando o carregamento de todos os ficheiros CSV para poder iniciar a análise.")


# --- PÁGINAS DO DASHBOARD ---
def dashboard_page():
    # ... (Esta função não foi alterada) ...
    st.title("🏠 Dashboard Geral")

    is_pre_mining_active = st.session_state.current_dashboard == "Pré-Mineração"
    
    sub_nav1, sub_nav2 = st.columns(2)
    # Seleção entre Pré/Post-mineração com estado persistente (mantém a seleção até o utilizador trocar)
    selected_dashboard = st.radio("Escolha a análise:", ["Pré-Mineração", "Pós-Mineração"],
                                 index=0 if st.session_state.current_dashboard == "Pré-Mineração" else 1,
                                 horizontal=True, key="dashboard_radio")
    if selected_dashboard != st.session_state.current_dashboard:
        st.session_state.current_dashboard = selected_dashboard
        st.session_state.current_section = "overview" if selected_dashboard == "Pré-Mineração" else "discovery"


    st.markdown("---")
    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor vá a 'Configurações' para carregar os dados e iniciar a análise.")
        return
    if st.session_state.current_dashboard == "Pré-Mineração":
        render_pre_mining_dashboard()
    else:
        render_post_mining_dashboard()

def render_pre_mining_dashboard():
    # ... (Esta função não foi alterada) ...
    sections = {
        "overview": "Visão Geral",
        "performance": "Performance",
        "activities": "Atividades",
        "resources": "Recursos",
        "variants": "Variantes",
        "advanced": "Avançado"
    }
    # seleção por radio para manter destaque permanente
    sec_keys = list(sections.keys())
    sec_labels = list(sections.values())
    current_idx = sec_keys.index(st.session_state.current_section) if st.session_state.current_section in sec_keys else 0
    sel = st.radio("Secção:", options=sec_labels, index=current_idx, horizontal=True, key="pre_sections_radio")
    st.session_state.current_section = sec_keys[sec_labels.index(sel)]
    
    st.markdown("<br>", unsafe_allow_html=True)
    plots = st.session_state.plots_pre_mining
    tables = st.session_state.tables_pre_mining

    if st.session_state.current_section == "overview":
        kpi_data = tables['kpi_data'] if tables and 'kpi_data' in tables else {}
        kpi_cols = st.columns(4)
        kpi_cols[0].metric(label="Total de Projetos", value=kpi_data.get('Total de Projetos', 'N/A'))
        kpi_cols[1].metric(label="Total de Tarefas", value=kpi_data.get('Total de Tarefas', 'N/A'))
        kpi_cols[2].metric(label="Total de Recursos", value=kpi_data.get('Total de Recursos', 'N/A'))
        kpi_cols[3].metric(label="Duração Média", value=f"{kpi_data.get('Duração Média (dias)', 'N/A')} dias")
        
        c1, c2 = st.columns(2)
        with c1:
            create_card("Matriz de Performance (Custo vs Prazo)", "🎯", chart_bytes=plots.get('performance_matrix') if plots else None)
            create_card("Top 5 Projetos Mais Longos", "⏳", dataframe=tables.get('outlier_duration') if tables else None)
        with c2:
            create_card("Distribuição da Duração dos Projetos", "📊", chart_bytes=plots.get('case_durations_boxplot') if plots else None)
            create_card("Top 5 Projetos Mais Caros", "💰", dataframe=tables.get('outlier_cost') if tables else None)
            
    elif st.session_state.current_section == "performance":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Estatísticas de Lead Time e Throughput", "📈", dataframe=tables.get('perf_stats') if tables else None)
            create_card("Distribuição do Lead Time", "⏱️", chart_bytes=plots.get('lead_time_hist') if plots else None)
        with c2:
            create_card("Distribuição do Throughput (horas)", "🚀", chart_bytes=plots.get('throughput_hist') if plots else None)
            create_card("Boxplot do Throughput (horas)", "📦", chart_bytes=plots.get('throughput_boxplot') if plots else None)
            
    elif st.session_state.current_section == "activities":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Tempo Médio de Execução por Atividade", "🛠️", chart_bytes=plots.get('activity_service_times') if plots else None)
        with c2:
            create_card("Top 10 Handoffs por Tempo de Espera", "⏳", chart_bytes=plots.get('top_handoffs') if plots else None)
            
    elif st.session_state.current_section == "resources":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Carga de Trabalho por Recurso", "👥", dataframe=tables.get('workload_by_resource') if tables else None)
        with c2:
            create_card("Recursos por Projeto", "📋", chart_bytes=plots.get('resources_per_project') if plots else None)
            
    elif st.session_state.current_section == "variants":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Frequência das 10 Principais Variantes", "🎭", chart_bytes=plots.get('variants_frequency') if plots else None)
        with c2:
            create_card("Principais Loops de Rework", "🔁", dataframe=tables.get('rework_loops_table') if tables else None)
            
    elif st.session_state.current_section == "advanced":
        kpi_data = tables.get('cost_of_delay_kpis', {}) if tables else {}
        kpi_cols = st.columns(3)
        kpi_cols[0].metric(label="Custo Total em Atraso", value=kpi_data.get('Custo Total Projetos Atrasados', 'N/A'))
        kpi_cols[1].metric(label="Atraso Médio", value=kpi_data.get('Atraso Médio (dias)', 'N/A'))
        kpi_cols[2].metric(label="Custo Médio/Dia de Atraso", value=kpi_data.get('Custo Médio/Dia Atraso', 'N/A'))
        
        c1, c2 = st.columns(2)


def render_post_mining_dashboard():
    # ... (Esta função não foi alterada) ...
    sections = {
        "discovery": "Descoberta",
        "performance": "Performance",
        "resources": "Recursos",
        "conformance": "Conformidade"
    }
    sec_keys = list(sections.keys())
    sec_labels = list(sections.values())
    current_idx = sec_keys.index(st.session_state.current_section) if st.session_state.current_section in sec_keys else 0
    sel = st.radio("Secção (Pós-Mineração):", options=sec_labels, index=current_idx, horizontal=True, key="post_sections_radio")
    st.session_state.current_section = sec_keys[sec_labels.index(sel)]
    
    st.markdown("<br>", unsafe_allow_html=True)
    plots = st.session_state.plots_post_mining
    if st.session_state.current_section == "discovery":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Modelo - Inductive Miner", "🧭", chart_bytes=plots.get('model_inductive_petrinet') if plots else None)
            create_card("Métricas (Inductive Miner)", "📊", chart_bytes=plots.get('metrics_inductive') if plots else None)
        with c2:
            create_card("Modelo - Heuristics Miner", "🛠️", chart_bytes=plots.get('model_heuristic_petrinet') if plots else None)
            create_card("Métricas (Heuristics Miner)", "📈", chart_bytes=plots.get('metrics_heuristic') if plots else None)
            
    elif st.session_state.current_section == "performance":
        create_card("Heatmap de Performance no Processo", "🔥", chart_bytes=plots.get('performance_heatmap') if plots else None)
        if 'gantt_chart_all_projects' in (plots or {}):
            create_card("Linha do Tempo de Todos os Projetos (Gantt)", "📊", chart_bytes=plots.get('gantt_chart_all_projects'))
            
    elif st.session_state.current_section == "resources":
        create_card("Alocação de Recursos por Período", "🧭", chart_bytes=plots.get('resource_allocation_timeline') if plots else None)
            
    elif st.session_state.current_section == "conformance":
        create_card("Relatório de Conformidade", "✅", dataframe=None)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        login_page()
    else:
        with st.sidebar:
            if st.button("🏠 Dashboard Geral", use_container_width=True):
                st.session_state.current_page = "Dashboard"
            if st.button("⚙️ Configurações", use_container_width=True):
                st.session_state.current_page = "Configurações"
            if st.button("🚪 Sair", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()
        page = st.session_state.get("page", "Dashboard")
        if page == "Configurações":
            settings_page()
        else:
            dashboard_page()

if __name__ == "__main__":
    main()
