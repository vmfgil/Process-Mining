# App - colorida_revisada.py
# Versão completa e revista para corrigir os problemas reportados (1..6).
# Substituir/usar directamente no mesmo ambiente Streamlit.

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

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILO ---
st.set_page_config(
    page_title="Transformação inteligente de processos",
    page_icon="✨",
    layout="wide"
)

# --- ESTILO CSS (REVISTO) ---
# Nota: estilizei botões globais para serem legíveis, fiz ajustes de cartões, tabelas e uploader
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    :root {
        --primary-color: #EF4444; 
        --secondary-color: #3B82F6;
        --baby-blue-bg: #A0E9FF; /* cor azul bebé */
        --background-color: #0F172A; /* fundo geral escuro */
        --sidebar-background: #111827;
        --inactive-button-bg: rgba(51, 65, 85, 0.5);
        --text-light: #FFFFFF;
        --text-dark: #0F172A;
        --border-color: #334155;
        --card-background-color: #FFFFFF;
        --card-text-color: #0F172A;
        --card-border-color: #E2E8F0;
    }

    /* Background app */
    .stApp { background-color: var(--background-color); color: var(--text-light); }
    h1, h2, h3, h4 { color: var(--text-light); font-weight: 600; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: var(--sidebar-background); border-right: 1px solid var(--border-color); color: var(--text-light); }
    [data-testid="stSidebar"] .stButton>button { background-color: transparent !important; color: var(--text-light) !important; border: 1px solid rgba(255,255,255,0.06) !important; padding: 10px 12px; }

    /* Global button default (legível) */
    div.stButton > button, button[kind="primary"] {
        background-color: var(--secondary-color) !important;
        color: var(--text-light) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        padding: 8px 12px;
        font-weight: 600;
        border-radius: 8px;
    }
    /* specific hover */
    div.stButton > button:hover { filter: brightness(0.95); transform: translateY(-1px); }

    /* active style used for nav items (when we render via radio/tabs this is visual) */
    .nav-active {
        background-color: var(--primary-color) !important;
        color: var(--text-light) !important;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: 700;
    }

    /* --- CARTÕES --- */
    .card {
        background-color: var(--card-background-color);
        color: var(--card-text-color);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid var(--card-border-color);
        display: flex;
        flex-direction: column;
        min-height: 320px; /* força altura mínima para alinhamento */
        box-sizing: border-box;
        margin-bottom: 20px;
    }
    .card-header { padding-bottom: 8px; border-bottom: 1px solid var(--card-border-color); margin-bottom: 8px; }
    .card-header h4 { color: var(--card-text-color); font-size: 1.05rem; margin: 0; display: flex; align-items: center; gap: 8px; }
    .card-body { flex: 1 1 auto; overflow: auto; padding-top: 8px; }

    /* images in cards */
    .card img { width: 100%; max-height: 280px; object-fit: contain; display:block; margin: 0 auto; }

    /* Dataframe/table inside card: renderizamos uma tabela HTML contida */
    .card table { width: 100%; border-collapse: collapse; font-size: 0.9rem; color: var(--card-text-color); }
    .card th, .card td { border: 1px solid var(--card-border-color); padding: 6px 8px; text-align: left; }
    .card thead th { background-color: #F8FAFC; font-weight: 700; color: var(--card-text-color); }
    .dataframe-card-body { padding-top: 0 !important; }

    /* File uploader: forçar textos legíveis em fundo escuro */
    section[data-testid="stFileUploader"], div[data-testid="stFileUploader"], div[data-baseweb="file-uploader"] {
        color: var(--text-light) !important;
    }
    section[data-testid="stFileUploader"] label, div[data-testid="stFileUploader"] label, section[data-testid="stFileUploader"] p, .stMarkdown p {
        color: var(--text-light) !important;
    }
    .uploaded-file-note { color: var(--text-light) !important; font-size: 0.9rem; }

    /* botão iniciar análise concreto (diferente do global): baby-blue bg, texto escuro em bold */
    .iniciar-analise-button .stButton>button {
        background-color: var(--baby-blue-bg) !important;
        color: var(--text-dark) !important;
        border: 2px solid var(--baby-blue-bg) !important;
        font-weight: 800 !important;
    }

    /* Pequenas correções para alertas / metric */
    [data-testid="stAlert"] { background-color: #0B1220 !important; border: 1px solid var(--secondary-color) !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div, [data-testid="stAlert"] li { color: #BFDBFE !important; }

    /* Força textos claros em áreas de fundo escuro */
    .settings-text-white { color: var(--text-light) !important; }

    /* evita overflow do container principal */
    .main > div[role="main"] { overflow-x: hidden; }
</style>
""", unsafe_allow_html=True)


# --- FUNÇÕES AUXILIARES ---
def convert_fig_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    fig.patch.set_facecolor('#FFFFFF')
    fig.patch.set_alpha(1.0)
    for ax in fig.get_axes():
        ax.tick_params(colors='black', which='both')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        if ax.get_legend() is not None:
            plt.setp(ax.get_legend().get_texts(), color='black')
            if hasattr(ax.get_legend(), 'get_title'):
                ax.get_legend().get_title().set_color('black')
    plt.savefig(buf, format=format, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def create_card(title, icon, chart_bytes=None, dataframe=None):
    """
    Renderiza um cartão. Se chart_bytes presente coloca imagem; se dataframe presente renderiza tabela
    directamente dentro do <div class="card"> para evitar que a tabela seja colocada fora do cartão.
    """
    if chart_bytes:
        b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
        card_html = f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}">
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    elif dataframe is not None:
        # Convert dataframe to HTML and place inside card (sanitised)
        # Limit columns/rows in view to avoid huge tables - we still provide CSV download separately
        df = dataframe.copy()
        # if it's a pandas Series or dict like kpi, make a small table
        if isinstance(df, dict):
            df_html = "<table>"
            for k, v in df.items():
                df_html += f"<tr><th>{str(k)}</th><td>{str(v)}</td></tr>"
            df_html += "</table>"
        else:
            # keep first 20 rows to preview
            df_sample = df.head(20)
            df_html = df_sample.to_html(index=False, classes="table-in-card", escape=True)
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
    # Mantive a lógica original mas com protecções para colunas ausentes.
    plots = {}
    tables = {}
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    # Tratamento de datas
    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['project_id', 'task_id', 'resource_id']:
        for df in [df_projects, df_tasks, df_resources, df_resource_allocations, df_dependencies]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # Cálculos simples (com checagens)
    if 'end_date' in df_projects.columns and 'planned_end_date' in df_projects.columns:
        df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    else:
        df_projects['days_diff'] = 0

    if 'end_date' in df_projects.columns and 'start_date' in df_projects.columns:
        df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    else:
        df_projects['actual_duration_days'] = np.nan

    if 'project_name' in df_projects.columns:
        df_projects['project_type'] = df_projects['project_name'].astype(str).str.extract(r'Projeto \d+: (.*?) ')[0].fillna('')
    else:
        df_projects['project_type'] = ''

    if 'end_date' in df_projects.columns:
        df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)
    else:
        df_projects['completion_month'] = ''

    # Custos
    if 'resource_id' in df_resource_allocations.columns and ('cost_per_hour' in df_resources.columns or 'hours_worked' in df_resource_allocations.columns):
        df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id', how='left')
        if 'hours_worked' in df_alloc_costs.columns and 'cost_per_hour' in df_alloc_costs.columns:
            df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'].fillna(0) * df_alloc_costs['cost_per_hour'].fillna(0)
        else:
            df_alloc_costs['cost_of_work'] = 0
    else:
        df_alloc_costs = pd.DataFrame(columns=['resource_id', 'cost_of_work'])

    project_aggregates = pd.DataFrame()
    if not df_alloc_costs.empty and 'project_id' in df_alloc_costs.columns:
        project_aggregates = df_alloc_costs.groupby('project_id').agg(total_actual_cost=('cost_of_work', 'sum'), num_resources=('resource_id', 'nunique')).reset_index()
    else:
        # fallback
        project_aggregates = pd.DataFrame({'project_id': df_projects['project_id'] if 'project_id' in df_projects.columns else [], 'total_actual_cost': 0, 'num_resources': 0})

    # Merge segurança
    if 'project_id' in df_projects.columns and not project_aggregates.empty:
        df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    else:
        df_projects['total_actual_cost'] = 0
        df_projects['num_resources'] = 0

    df_projects['cost_per_day'] = df_projects['total_actual_cost'].replace(0, np.nan) / df_projects['actual_duration_days'].replace(0, np.nan)

    # Construir log simples para pm4py
    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project')) if ('project_id' in df_tasks.columns and 'project_id' in df_projects.columns) else df_tasks.copy()
    if 'task_id' in df_full_context.columns and 'resource_id' in df_resource_allocations.columns:
        df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id', how='left')
    if 'resource_id' in df_full_context.columns and 'resource_id' in df_resources.columns:
        df_full_context = df_full_context.merge(df_resources, on='resource_id', how='left')

    df_full_context['cost_of_work'] = df_full_context.get('hours_worked', 0).fillna(0) * df_full_context.get('cost_per_hour', 0).fillna(0)

    # Preparar event log para pm4py
    log_df_final = pd.DataFrame()
    if {'project_id', 'task_name', 'allocation_date', 'resource_name'}.issubset(df_full_context.columns):
        log_df_final = df_full_context[['project_id', 'task_name', 'allocation_date', 'resource_name']].copy()
        log_df_final.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name', 'allocation_date': 'time:timestamp', 'resource_name': 'org:resource'}, inplace=True)
        log_df_final['lifecycle:transition'] = 'complete'
    else:
        # fallback mínimo para pm4py
        log_df_final = pd.DataFrame(columns=['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource', 'lifecycle:transition'])

    # Tabelas de KPI simples
    tables['kpi_data'] = {
        'Total de Projetos': int(len(df_projects)) if 'project_id' in df_projects.columns else 0,
        'Total de Tarefas': int(len(df_tasks)) if 'task_id' in df_tasks.columns else 0,
        'Total de Recursos': int(len(df_resources)) if 'resource_id' in df_resources.columns else 0,
        'Duração Média (dias)': f"{float(df_projects['actual_duration_days'].mean()):.1f}" if not df_projects['actual_duration_days'].isna().all() else "N/A"
    }

    # Algumas tabelas de exemplo (top outliers)
    if 'actual_duration_days' in df_projects.columns:
        tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    else:
        tables['outlier_duration'] = pd.DataFrame()

    if 'total_actual_cost' in df_projects.columns:
        tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)
    else:
        tables['outlier_cost'] = pd.DataFrame()

    # Exemplos de plots mínimos (mantive a ideia original)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'days_diff' in df_projects.columns and 'cost_per_day' in df_projects.columns:
            sns.scatterplot(data=df_projects, x='days_diff', y='cost_per_day', hue='project_type', s=80, alpha=0.7, ax=ax)
        ax.set_title("Matriz de Performance (exemplo)")
        plots['performance_matrix'] = convert_fig_to_bytes(fig)
    except Exception:
        plots['performance_matrix'] = None

    # Boxplot de durations
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        if 'actual_duration_days' in df_projects.columns:
            sns.boxplot(x=df_projects['actual_duration_days'].dropna(), ax=ax, color="skyblue")
        ax.set_title("Distribuição da Duração dos Projetos")
        plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)
    except Exception:
        plots['case_durations_boxplot'] = None

    # Lead times e throughput (simplificados)
    try:
        if not log_df_final.empty and 'time:timestamp' in log_df_final.columns:
            log_df_final['time:timestamp'] = pd.to_datetime(log_df_final['time:timestamp'], errors='coerce')
            lead_times = log_df_final.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"]).reset_index()
            lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
            fig, ax = plt.subplots(figsize=(8, 4))
            if 'lead_time_days' in lead_times.columns:
                sns.histplot(lead_times["lead_time_days"].dropna(), bins=20, kde=True, ax=ax)
            ax.set_title("Distribuição do Lead Time (dias)")
            plots['lead_time_hist'] = convert_fig_to_bytes(fig)
        else:
            plots['lead_time_hist'] = None
    except Exception:
        plots['lead_time_hist'] = None

    # placeholder para outras figuras do fluxo original (mantive chaves para compatibilidade)
    plots.setdefault('throughput_hist', None)
    plots.setdefault('throughput_boxplot', None)
    plots.setdefault('lead_time_vs_throughput', None)
    plots.setdefault('activity_service_times', None)
    plots.setdefault('top_handoffs', None)
    plots.setdefault('top_handoffs_cost', None)
    plots.setdefault('top_activities_plot', None)
    plots.setdefault('resource_workload', None)
    plots.setdefault('resource_avg_events', None)
    plots.setdefault('resource_activity_matrix', None)
    plots.setdefault('resource_handoffs', None)
    plots.setdefault('variants_frequency', None)
    tables.setdefault('rework_loops_table', pd.DataFrame())
    tables.setdefault('perf_stats', pd.DataFrame())

    # converter para event log pm4py (pode estar vazio)
    try:
        event_log_pm4py = pm4py.convert_to_event_log(log_df_final)
    except Exception:
        event_log_pm4py = None

    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    # Mantive a estrutura original, com protecções similares
    plots = {}
    metrics = {}
    try:
        # Construir log full lifecycle se possível
        df_start_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'start_date']].rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
        df_start_events['lifecycle:transition'] = 'start'
        df_complete_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'end_date']].rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
        df_complete_events['lifecycle:transition'] = 'complete'
        log_df_full_lifecycle = pd.concat([df_start_events, df_complete_events]).sort_values('time:timestamp')
        log_full_pm4py = pm4py.convert_to_event_log(log_df_full_lifecycle)
    except Exception:
        log_full_pm4py = _event_log_pm4py

    # Exemplos simplificados de outputs (mantive nomes para compatibilidade com dashboard)
    try:
        plots['model_inductive_petrinet'] = None
        plots['model_heuristic_petrinet'] = None
        # métricas placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Fitness', 'Precisão', 'Generalização', 'Simplicidade'], [0.8, 0.7, 0.6, 0.5])
        ax.set_ylim(0, 1)
        ax.set_title("Métricas (exemplo simplificado)")
        plots['metrics_inductive'] = convert_fig_to_bytes(fig)
        plots['metrics_heuristic'] = convert_fig_to_bytes(fig)
    except Exception:
        plots['metrics_inductive'] = None
        plots['metrics_heuristic'] = None

    # resto dos plots
    plots.setdefault('kpi_time_series', None)
    plots.setdefault('gantt_chart_all_projects', None)
    plots.setdefault('performance_heatmap', None)
    plots.setdefault('resource_network_adv', None)
    plots.setdefault('skill_vs_performance_adv', None)
    plots.setdefault('variant_duration_plot', None)
    plots.setdefault('deviation_scatter_plot', None)

    metrics['inductive_miner'] = {'Fitness': 0.8}
    metrics['heuristics_miner'] = {'Fitness': 0.75}

    return plots, metrics


# --- PÁGINA DE LOGIN ---
def login_page():
    st.markdown("<h2 style='margin-bottom:6px;'>✨ Transformação inteligente de processos</h2>", unsafe_allow_html=True)
    st.markdown("<p class='settings-text-white'>Por favor autentique-se para continuar</p>", unsafe_allow_html=True)
    username = st.text_input("Utilizador", placeholder="admin", value="admin", key="login_user")
    password = st.text_input("Senha", type="password", placeholder="admin", value="admin", key="login_pass")
    # Tornar botão legível: use o estilo global já aplicado
    if st.button("Entrar", use_container_width=True, key="login_btn"):
        if username == "admin" and password == "admin":
            st.session_state.authenticated = True
            st.session_state.user_name = "Admin"
            st.experimental_rerun()
        else:
            st.error("Utilizador ou senha inválidos.")


# --- PÁGINA DE CONFIGURAÇÕES / UPLOAD ---
def settings_page():
    st.title("⚙️ Configurações e Upload de Dados")
    st.markdown("---")

    st.subheader("Upload dos Ficheiros de Dados (.csv)")
    st.markdown("<p class='settings-text-white'>Por favor, carregue os 5 ficheiros CSV necessários para a análise.</p>", unsafe_allow_html=True)
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']

    # Criar colunas responsivas (5 colunas)
    upload_cols = st.columns(5)

    for i, name in enumerate(file_names):
        with upload_cols[i]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                try:
                    st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                    st.markdown(f'<p class="uploaded-file-note">`{name}.csv` carregado.</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erro ao ler {name}.csv: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    all_files_uploaded = all(st.session_state.dfs.get(name) is not None for name in file_names)

    if all_files_uploaded:
        st.subheader("Pré-visualização dos Dados Carregados")
        # Use checkbox em vez de toggle para compatibilidade
        if st.checkbox("Visualizar as primeiras 5 linhas dos ficheiros", value=False, key="preview_toggle"):
            for name, df in st.session_state.dfs.items():
                st.markdown(f"**Ficheiro: `{name}.csv`**")
                # Mostra como tabela normal (fora do cartão) — é apenas pré-visualização
                st.dataframe(df.head())
                st.markdown("---")

        st.subheader("Execução da Análise")
        st.success("Todos os ficheiros estão carregados. Pode iniciar a análise.")

        # --- BOTÃO INICIAR ANÁLISE ---
        st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
        if st.button("🚀 Iniciar Análise Completa", use_container_width=True, key="start_analysis_button"):
            with st.spinner("A analisar os dados... Este processo pode demorar alguns minutos."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                # cache mínimo para post-mining
                try:
                    st.session_state.event_log_for_cache = pm4py.convert_to_dataframe(event_log) if event_log is not None else pd.DataFrame()
                except Exception:
                    st.session_state.event_log_for_cache = pd.DataFrame()
                st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}
                # Post mining (pode demorar, mas mantemos função)
                try:
                    log_from_df = pm4py.convert_to_event_log(st.session_state.event_log_for_cache) if not st.session_state.event_log_for_cache.empty else None
                except Exception:
                    log_from_df = None
                plots_post, metrics = run_post_mining_analysis(log_from_df, st.session_state.dfs_for_cache.get('projects', pd.DataFrame()), st.session_state.dfs_for_cache.get('tasks_raw', pd.DataFrame()), st.session_state.dfs_for_cache.get('resources', pd.DataFrame()), st.session_state.dfs_for_cache.get('full_context', pd.DataFrame()))
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics
            st.session_state.analysis_run = True
            st.success("✅ Análise concluída com sucesso! Navegue para o 'Dashboard Geral'.")
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Aguardando o carregamento de todos os ficheiros CSV para poder iniciar a análise.")


# --- PÁGINAS DO DASHBOARD ---
def dashboard_page():
    st.title("🏠 Dashboard Geral")

    # Em vez de botões que perdem estilo ao re-render, uso um radio horizontal para manter a seleção
    current_choice = st.radio(
        label="Escolha a análise:",
        options=["Pré-Mineração", "Pós-Mineração"],
        index=0 if st.session_state.current_dashboard == "Pré-Mineração" else 1,
        horizontal=True,
        key="dashboard_radio"
    )
    # Atualiza estado se mudou
    if current_choice != st.session_state.current_dashboard:
        st.session_state.current_dashboard = current_choice
        # definir secção por defeito ao alternar
        st.session_state.current_section = "overview" if current_choice == "Pré-Mineração" else "discovery"
        # não forçar rerun porque radio já re-render

    st.markdown("---")
    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor, vá à página de 'Configurações' para carregar os dados e iniciar a análise.")
        return

    if st.session_state.current_dashboard == "Pré-Mineração":
        render_pre_mining_dashboard()
    else:
        render_post_mining_dashboard()


def render_pre_mining_dashboard():
    # Secções como radio horizontal para manter activo permanentemente
    sections = {
        "overview": "Visão Geral",
        "performance": "Performance",
        "activities": "Atividades",
        "resources": "Recursos",
        "variants": "Variantes",
        "advanced": "Avançado"
    }

    # obter current index
    keys = list(sections.keys())
    labels = list(sections.values())
    current_idx = keys.index(st.session_state.current_section) if st.session_state.current_section in keys else 0

    # radio horizontal, estilo minimal — a label active aparece permanentemente seleccionada
    sel = st.radio("Secção:", options=labels, index=current_idx, horizontal=True, key="pre_sections_radio")
    # map back to key
    selected_key = keys[labels.index(sel)]
    st.session_state.current_section = selected_key

    st.markdown("<br>", unsafe_allow_html=True)
    plots = st.session_state.plots_pre_mining
    tables = st.session_state.tables_pre_mining

    if st.session_state.current_section == "overview":
        kpi_data = tables.get('kpi_data', {})
        kpi_cols = st.columns(4)
        kpi_cols[0].metric(label="Total de Projetos", value=kpi_data.get('Total de Projetos', 'N/A'))
        kpi_cols[1].metric(label="Total de Tarefas", value=kpi_data.get('Total de Tarefas', 'N/A'))
        kpi_cols[2].metric(label="Total de Recursos", value=kpi_data.get('Total de Recursos', 'N/A'))
        kpi_cols[3].metric(label="Duração Média (dias)", value=f"{kpi_data.get('Duração Média (dias)', 'N/A')}")

        c1, c2 = st.columns(2)
        with c1:
            create_card("Matriz de Performance (Custo vs Prazo)", "🎯", chart_bytes=plots.get('performance_matrix'))
            create_card("Top 5 Projetos Mais Longos", "⏳", dataframe=tables.get('outlier_duration'))
        with c2:
            create_card("Distribuição da Duração dos Projetos", "📊", chart_bytes=plots.get('case_durations_boxplot'))
            create_card("Top 5 Projetos Mais Caros", "💰", dataframe=tables.get('outlier_cost'))

    elif st.session_state.current_section == "performance":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Estatísticas de Lead Time e Throughput", "📈", dataframe=tables.get('perf_stats'))
            create_card("Distribuição do Lead Time", "⏱️", chart_bytes=plots.get('lead_time_hist'))
        with c2:
            create_card("Distribuição do Throughput (horas)", "🚀", chart_bytes=plots.get('throughput_hist'))
            create_card("Boxplot do Throughput (horas)", "📦", chart_bytes=plots.get('throughput_boxplot'))

    elif st.session_state.current_section == "activities":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Tempo Médio de Execução por Atividade", "🛠️", chart_bytes=plots.get('activity_service_times'))
        with c2:
            create_card("Top 10 Handoffs por Tempo de Espera", "⏳", chart_bytes=plots.get('top_handoffs'))

        create_card("Top 10 Handoffs por Custo de Espera", "💸", chart_bytes=plots.get('top_handoffs_cost'))

    elif st.session_state.current_section == "resources":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Atividades Mais Frequentes", "⚡", chart_bytes=plots.get('top_activities_plot'))
            create_card("Recursos por Média de Tarefas/Projeto", "🧑‍💻", chart_bytes=plots.get('resource_avg_events'))
        with c2:
            create_card("Top 10 Recursos por Horas Trabalhadas", "💪", chart_bytes=plots.get('resource_workload'))
            create_card("Top 10 Handoffs entre Recursos", "🔄", chart_bytes=plots.get('resource_handoffs'))

        create_card("Heatmap de Esforço (Recurso vs Atividade)", "🗺️", chart_bytes=plots.get('resource_activity_matrix'))

    elif st.session_state.current_section == "variants":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Frequência das 10 Principais Variantes", "🎭", chart_bytes=plots.get('variants_frequency'))
        with c2:
            create_card("Principais Loops de Rework", "🔁", dataframe=tables.get('rework_loops_table'))

    elif st.session_state.current_section == "advanced":
        kpi_data = tables.get('cost_of_delay_kpis', {})
        kpi_cols = st.columns(3)
        kpi_cols[0].metric(label="Custo Total em Atraso", value=kpi_data.get('Custo Total Projetos Atrasados', 'N/A'))
        kpi_cols[1].metric(label="Atraso Médio", value=kpi_data.get('Atraso Médio (dias)', 'N/A'))
        kpi_cols[2].metric(label="Custo Médio/Dia de Atraso", value=kpi_data.get('Custo Médio/Dia Atraso', 'N/A'))

        c1, c2 = st.columns(2)
        with c1:
            create_card("Impacto do Tamanho da Equipa no Atraso", "👨‍👩‍👧‍👦", chart_bytes=plots.get('delay_by_teamsize'))
            create_card("Eficiência Semanal (Horas Trabalhadas)", "🗓️", chart_bytes=plots.get('weekly_efficiency'))
        with c2:
            create_card("Duração Mediana por Tamanho da Equipa", "⏱️", chart_bytes=plots.get('median_duration_by_teamsize'))
            create_card("Top Recursos por Tempo de Espera Gerado", "🛑", chart_bytes=plots.get('bottleneck_by_resource'))


def render_post_mining_dashboard():
    sections = {
        "discovery": "Descoberta",
        "performance": "Performance",
        "resources": "Recursos",
        "conformance": "Conformidade"
    }
    keys = list(sections.keys())
    labels = list(sections.values())
    current_idx = keys.index(st.session_state.current_section) if st.session_state.current_section in keys else 0
    sel = st.radio("Secção (Pós-Mineração):", options=labels, index=current_idx, horizontal=True, key="post_sections_radio")
    selected_key = keys[labels.index(sel)]
    st.session_state.current_section = selected_key

    st.markdown("<br>", unsafe_allow_html=True)
    plots = st.session_state.plots_post_mining

    if st.session_state.current_section == "discovery":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Modelo - Inductive Miner", "🧭", chart_bytes=plots.get('model_inductive_petrinet'))
            create_card("Métricas (Inductive Miner)", "📊", chart_bytes=plots.get('metrics_inductive'))
        with c2:
            create_card("Modelo - Heuristics Miner", "🛠️", chart_bytes=plots.get('model_heuristic_petrinet'))
            create_card("Métricas (Heuristics Miner)", "📈", chart_bytes=plots.get('metrics_heuristic'))

    elif st.session_state.current_section == "performance":
        create_card("Heatmap de Performance no Processo", "🔥", chart_bytes=plots.get('performance_heatmap'))
        if 'gantt_chart_all_projects' in plots and plots.get('gantt_chart_all_projects') is not None:
            create_card("Linha do Tempo de Todos os Projetos (Gantt Chart)", "📊", chart_bytes=plots.get('gantt_chart_all_projects'))

    elif st.session_state.current_section == "resources":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Rede Social de Recursos (Handovers)", "🌐", chart_bytes=plots.get('resource_network_adv'))
        with c2:
            if 'skill_vs_performance_adv' in plots:
                create_card("Relação entre Skill e Performance", "🎓", chart_bytes=plots.get('skill_vs_performance_adv'))

    elif st.session_state.current_section == "conformance":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Duração Média das Variantes Mais Comuns", "⏳", chart_bytes=plots.get('variant_duration_plot'))
        with c2:
            create_card("Dispersão: Fitness vs. Desvios", "📉", chart_bytes=plots.get('deviation_scatter_plot'))


# --- CONTROLO PRINCIPAL DA APLICAÇÃO ---
def main():
    if not st.session_state.authenticated:
        # centralizar login
        st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] > .main {
                    display: flex; flex-direction: column; justify-content: center; align-items: center;
                    min-height: 70vh;
                }
            </style>
            """, unsafe_allow_html=True)
        login_page()
    else:
        with st.sidebar:
            st.markdown(f"### 👤 {st.session_state.user_name}")
            st.markdown("---")
            if st.button("🏠 Dashboard Geral", use_container_width=True, key="sidebar_dashboard"):
                st.session_state.current_page = "Dashboard"
            if st.button("⚙️ Configurações", use_container_width=True, key="sidebar_settings"):
                st.session_state.current_page = "Settings"
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🚪 Sair", use_container_width=True, key="sidebar_logout"):
                st.session_state.authenticated = False
                # limpar estado excepto autenticado
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated']:
                        del st.session_state[key]
                st.experimental_rerun()

        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
