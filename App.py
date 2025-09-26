import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import io
import base64
import time
import random
from datetime import timedelta

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

# Imports para Análise Exploratória (EDA)
import missingno as msno

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILO ---
st.set_page_config(
    page_title="Transformação Inteligente de Processos",
    page_icon="✨",
    layout="wide"
)

# --- ESTILO CSS REFORMULADO (NOVO ESQUEMA DE CORES) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    
    /* Nova Paleta de Cores Profissional e de Alto Contraste */
    :root {
        --primary-color: #2563EB; /* Azul de Realce (Botões Ativos, Bordas) */
        --secondary-color: #FBBF24; /* Amarelo/Âmbar (Alertas, Destaque) */
        --accent-color: #06B6D4; /* Ciano (Botões de Upload/Análise) */
        
        --background-color: #0A112A; /* Fundo Principal Escuro (Azul Marinho Sólido) */
        --sidebar-background: #111827; /* Fundo da Sidebar Ligeiramente Mais Claro */
        --card-background-color: #1E293B; /* Fundo dos Cartões (Azul Escuro Suave) */
        
        --text-color-dark-bg: #E5E7EB; /* Texto Principal (Branco Sujo) */
        --text-color-light-bg: #0A112A; /* Texto em Elementos Claros */
        --border-color: #374151; /* Cor da Borda/Separador */
        --inactive-button-bg: #374151; /* Fundo de Botões Inativos */
        --metric-value-color: #FBBF24; /* Cor para Valores de Métricas */
    }
    
    .stApp { background-color: var(--background-color); color: var(--text-color-dark-bg); }
    h1, h2, h3 { color: var(--text-color-dark-bg); font-weight: 600; }
    
    [data-testid="stSidebar"] h3 { color: var(--text-color-dark-bg) !important; }

    /* --- ESTILOS PARA BOTÕES DE NAVEGAÇÃO --- */
    div[data-testid="stHorizontalBlock"] .stButton>button {
        border: 1px solid var(--border-color) !important;
        background-color: var(--inactive-button-bg) !important;
        color: var(--text-color-dark-bg) !important;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:hover {
        border-color: var(--primary-color) !important;
        background-color: rgba(37, 99, 235, 0.2) !important; /* Azul com 20% de opacidade */
    }
    div.active-button .stButton>button {
        background-color: var(--primary-color) !important;
        color: var(--text-color-dark-bg) !important;
        border: 1px solid var(--primary-color) !important;
        font-weight: 700 !important;
    }

    /* Painel Lateral */
    [data-testid="stSidebar"] { background-color: var(--sidebar-background); border-right: 1px solid var(--border-color); }
    [data-testid="stSidebar"] .stButton>button {
        background-color: var(--primary-color) !important; /* Botões da sidebar com cor de destaque */
        color: var(--text-color-dark-bg) !important;
    }
    
    /* --- CARTÕES --- */
    .card {
        background-color: var(--card-background-color);
        color: var(--text-color-dark-bg);
        border-radius: 12px;
        padding: 20px 25px;
        border: 1px solid var(--border-color);
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    .card-header { padding-bottom: 10px; border-bottom: 1px solid var(--border-color); }
    .card .card-header h4 { color: var(--text-color-dark-bg); font-size: 1.1rem; margin: 0; display: flex; align-items: center; gap: 8px; }
    .card-body { flex-grow: 1; padding-top: 15px; }
    .dataframe-card-body {
        max-height: 300px;
        overflow-y: auto;
        overflow-x: auto;
        padding: 0;
    }
    
    /* --- BOTÕES DE UPLOAD --- */
    section[data-testid="stFileUploader"] button,
    div[data-baseweb="file-uploader"] button {
        background-color: var(--accent-color) !important;
        color: var(--text-color-light-bg) !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    
    /* --- BOTÃO DE ANÁLISE --- */
    .iniciar-analise-button .stButton>button {
        background-color: var(--secondary-color) !important;
        color: var(--text-color-light-bg) !important;
        border: 2px solid var(--secondary-color) !important;
        font-weight: 700 !important;
    }
    
    /* --- CARTÕES DE MÉTRICAS (KPIs) --- */
    [data-testid="stMetric"] {
        background-color: var(--card-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
    }
    [data-testid="stMetric"] label {
        color: var(--text-color-dark-bg) !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--metric-value-color) !important;
        font-weight: 700;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--text-color-dark-bg) !important;
    }
    
    /* Alertas */
    [data-testid="stAlert"] {
        background-color: #1E293B !important;
        border: 1px solid var(--secondary-color) !important;
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] * { color: var(--text-color-dark-bg) !important; }
    
    .stDataFrame {
        color: var(--text-color-dark-bg) !important;
        background-color: var(--card-background-color) !important;
    }

    .pandas-df-card {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-color-dark-bg);
        font-size: 0.85rem;
    }
    .pandas-df-card th {
        background-color: var(--sidebar-background);
        color: var(--text-color-dark-bg);
        border: 1px solid var(--border-color);
        padding: 8px;
        text-align: left;
    }
    .pandas-df-card td {
        background-color: var(--card-background-color);
        color: var(--text-color-dark-bg);
        border: 1px solid var(--border-color);
        padding: 8px;
    }
    .pandas-df-card tr:nth-child(even) td {
        background-color: #2F394B;
    }
    
    .stTextInput>div>div>input, .stTextInput>div>div>textarea, .stNumberInput>div>div>input {
        background-color: var(--sidebar-background) !important;
        color: var(--text-color-dark-bg) !important;
        border: 1px solid var(--border-color) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- FUNÇÕES AUXILIARES ---
def convert_fig_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    fig.patch.set_facecolor('#1E293B')
    for ax in fig.get_axes():
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#E5E7EB', which='both')
        ax.xaxis.label.set_color('#E5E7EB')
        ax.yaxis.label.set_color('#E5E7EB')
        ax.title.set_color('#E5E7EB')
        if ax.get_legend() is not None:
            plt.setp(ax.get_legend().get_texts(), color='#E5E7EB')
            ax.get_legend().get_frame().set_facecolor('#1E293B')
            ax.get_legend().get_frame().set_edgecolor('#374151')
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

def create_card(title, icon, chart_bytes=None, dataframe=None, use_container_width=False):
    if chart_bytes:
        b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif dataframe is not None:
        df_html = dataframe.to_html(classes=['pandas-df-card'], index=False)
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body dataframe-card-body">
                {df_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
if 'current_section' not in st.session_state: st.session_state.current_section = "visao_geral"
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'rl_analysis_run' not in st.session_state: st.session_state.rl_analysis_run = False
if 'rl_params_expanded' not in st.session_state: st.session_state.rl_params_expanded = True
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'plots_eda' not in st.session_state: st.session_state.plots_eda = {}
if 'tables_eda' not in st.session_state: st.session_state.tables_eda = {}
if 'plots_rl' not in st.session_state: st.session_state.plots_rl = {}
if 'tables_rl' not in st.session_state: st.session_state.tables_rl = {}
if 'logs_rl' not in st.session_state: st.session_state.logs_rl = {}


# --- FUNÇÕES DE ANÁLISE (PROCESS MINING E EDA) ---
# ... (As funções run_pre_mining_analysis, run_post_mining_analysis, e run_eda_analysis permanecem as mesmas da versão anterior)
@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['project_id', 'task_id', 'resource_id']:
        for df in [df_projects, df_tasks, df_resources, df_resource_allocations, df_dependencies]:
            if col in df.columns: df[col] = df[col].astype(str)

    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['project_name'].str.extract(r'Projeto \d+: (.*?) ')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)

    df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
    df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']
    
    project_aggregates = df_alloc_costs.groupby('project_id').agg(total_actual_cost=('cost_of_work', 'sum'), num_resources=('resource_id', 'nunique')).reset_index()
    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)
    
    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

    log_df_final = df_full_context[['project_id', 'task_name', 'allocation_date', 'resource_name']].copy()
    log_df_final.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name', 'allocation_date': 'time:timestamp', 'resource_name': 'org:resource'}, inplace=True)
    log_df_final['lifecycle:transition'] = 'complete'
    event_log_pm4py = pm4py.convert_to_event_log(log_df_final)
    
    tables['kpi_data'] = {
        'Total de Projetos': len(df_projects),
        'Total de Tarefas': len(df_tasks),
        'Total de Recursos': len(df_resources),
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.1f}"
    }
    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.7, ax=ax, palette='viridis'); ax.axhline(0, color='#FBBF24', ls='--'); ax.axvline(0, color='#FBBF24', ls='--'); ax.set_title("Matriz de Performance (PM)")
    plots['performance_matrix'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=df_projects['actual_duration_days'], ax=ax, color="#2563EB"); sns.stripplot(x=df_projects['actual_duration_days'], color="#FBBF24", size=4, jitter=True, alpha=0.7, ax=ax); ax.set_title("Distribuição da Duração dos Projetos (PM)")
    plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)
    
    lead_times = log_df_final.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"]).reset_index()
    lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
    def compute_avg_throughput(group):
        group = group.sort_values("time:timestamp"); deltas = group["time:timestamp"].diff().dropna()
        return deltas.mean().total_seconds() if not deltas.empty else 0
    throughput_per_case = log_df_final.groupby("case:concept:name").apply(compute_avg_throughput).reset_index(name="avg_throughput_seconds")
    throughput_per_case["avg_throughput_hours"] = throughput_per_case["avg_throughput_seconds"] / 3600
    perf_df = pd.merge(lead_times, throughput_per_case, on="case:concept:name")
    tables['perf_stats'] = perf_df[["lead_time_days", "avg_throughput_hours"]].describe()
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["lead_time_days"], bins=20, kde=True, ax=ax, color="#2563EB"); ax.set_title("Distribuição do Lead Time (dias)")
    plots['lead_time_hist'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["avg_throughput_hours"], bins=20, kde=True, color='#06B6D4', ax=ax); ax.set_title("Distribuição do Throughput (horas)")
    plots['throughput_hist'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=perf_df["avg_throughput_hours"], color='#FBBF24', ax=ax); ax.set_title("Boxplot do Throughput (horas)")
    plots['throughput_boxplot'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação Lead Time vs Throughput")
    plots['lead_time_vs_throughput'] = convert_fig_to_bytes(fig)
    
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, hue='task_name', legend=False, palette='coolwarm'); ax.set_title("Tempo Médio de Execução por Atividade")
    plots['activity_service_times'] = convert_fig_to_bytes(fig)
    
    df_handoff = log_df_final.sort_values(['case:concept:name', 'time:timestamp'])
    df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', ax=ax, hue='transition', legend=False, palette='viridis'); ax.set_title("Top 10 Handoffs por Tempo de Espera")
    plots['top_handoffs'] = convert_fig_to_bytes(fig)
    
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, hue='transition', legend=False, palette='magma'); ax.set_title("Top 10 Handoffs por Custo de Espera")
    plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

    activity_counts = df_tasks["task_name"].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='YlGnBu'); ax.set_title("Atividades Mais Frequentes")
    plots['top_activities_plot'] = convert_fig_to_bytes(fig)
    
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, hue='resource_name', legend=False, palette='plasma'); ax.set_title("Top 10 Recursos por Horas Trabalhadas (PM)")
    plots['resource_workload'] = convert_fig_to_bytes(fig)
    
    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()
    resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), ax=ax, hue='resource_name', legend=False, palette='coolwarm'); ax.set_title("Recursos por Média de Tarefas por Projeto")
    plots['resource_avg_events'] = convert_fig_to_bytes(fig)
    
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_activity_matrix_pivot, cmap='Blues', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8}, linewidths=.5, linecolor='#374151'); ax.set_title("Heatmap de Esforço por Recurso e Atividade")
    plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)
    
    handoff_counts = Counter((trace[i]['org:resource'], trace[i+1]['org:resource']) for trace in event_log_pm4py for i in range(len(trace) - 1) if 'org:resource' in trace[i] and 'org:resource' in trace[i+1] and trace[i]['org:resource'] != trace[i+1]['org:resource'])
    df_resource_handoffs = pd.DataFrame(handoff_counts.most_common(10), columns=['Handoff', 'Contagem'])
    df_resource_handoffs['Handoff'] = df_resource_handoffs['Handoff'].apply(lambda x: f"{x[0]} -> {x[1]}")
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs, ax=ax, hue='Handoff', legend=False, palette='rocket'); ax.set_title("Top 10 Handoffs entre Recursos")
    plots['resource_handoffs'] = convert_fig_to_bytes(fig)
    
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', ax=ax, hue='resource_type', legend=False, palette='magma'); ax.set_title("Custo por Tipo de Recurso")
    plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)
    
    variants_df = log_df_final.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='trace')
    variants_df['variant_str'] = variants_df['trace'].apply(lambda x: ' -> '.join(x))
    variant_analysis = variants_df['variant_str'].value_counts().reset_index(name='frequency')
    variant_analysis['percentage'] = (variant_analysis['frequency'] / variant_analysis['frequency'].sum()) * 100
    tables['variants_table'] = variant_analysis.head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='frequency', y='variant_str', data=variant_analysis.head(10), ax=ax, orient='h', hue='variant_str', legend=False, palette='coolwarm'); ax.set_title("Top 10 Variantes de Processo por Frequência")
    plots['variants_frequency'] = convert_fig_to_bytes(fig)
    
    rework_loops = Counter(f"{trace[i]} -> {trace[i+1]} -> {trace[i]}" for trace in variants_df['trace'] for i in range(len(trace) - 2) if trace[i] == trace[i+2] and trace[i] != trace[i+1])
    tables['rework_loops_table'] = pd.DataFrame(rework_loops.most_common(10), columns=['rework_loop', 'frequency'])
    
    delayed_projects = df_projects[df_projects['days_diff'] > 0]
    tables['cost_of_delay_kpis'] = {
        'Custo Total Projetos Atrasados': f"€{delayed_projects['total_actual_cost'].sum():,.2f}",
        'Atraso Médio (dias)': f"{delayed_projects['days_diff'].mean():.1f}",
        'Custo Médio/Dia Atraso': f"€{(delayed_projects.get('total_actual_cost', 0) / delayed_projects['days_diff']).mean():,.2f}"
    }
    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()
    bins = np.linspace(min_res, max_res, 5, dtype=int) if max_res > min_res else [min_res, max_res]
    df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='flare'); ax.set_title("Impacto do Tamanho da Equipa no Atraso (PM)")
    plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)
    
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='crest'); ax.set_title("Duração Mediana por Tamanho da Equipa")
    plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)
    
    df_alloc_costs['day_of_week'] = df_alloc_costs['allocation_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, hue='day_of_week', legend=False, palette='viridis'); ax.set_title("Eficiência Semanal (Horas Trabalhadas)")
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    
    df_tasks_analysis = df_tasks.copy(); df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True); df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates()
    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, hue='resource_name', legend=False, palette='rocket'); ax.set_title("Top 15 Recursos por Tempo Médio de Espera")
    plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)
    
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5)); bottleneck_by_activity.plot(kind='bar', stacked=True, color=['#2563EB', '#FBBF24'], ax=ax); ax.set_title("Gargalos: Tempo de Serviço vs. Espera")
    plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=bottleneck_by_activity.reset_index(), x='service_time_days', y='waiting_time_days', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Tempo de Espera vs. Tempo de Execução")
    plots['wait_vs_service_scatter'] = convert_fig_to_bytes(fig)
    
    df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id', 'completion_month']], on='project_id')
    monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax, color='#06B6D4'); plt.xticks(rotation=45); ax.set_title("Evolução do Tempo Médio de Espera")
    plots['wait_time_evolution'] = convert_fig_to_bytes(fig)
    
    df_perf_full = perf_df.merge(df_projects, left_on='case:concept:name', right_on='project_id')
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_perf_full, x='team_size_bin_dynamic', y='avg_throughput_hours', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='plasma'); ax.set_title("Benchmark de Throughput por Tamanho da Equipa")
    plots['throughput_benchmark_by_teamsize'] = convert_fig_to_bytes(fig)
    
    def get_phase(task_type):
        if task_type in ['Desenvolvimento', 'Correção', 'Revisão', 'Design']: return 'Desenvolvimento & Design'
        if task_type == 'Teste': return 'Teste (QA)'
        if task_type in ['Deploy', 'DBA']: return 'Operações & Deploy'
        return 'Outros'
    df_tasks['phase'] = df_tasks['task_type'].apply(get_phase)
    phase_times = df_tasks.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index()
    phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days
    avg_cycle_time_by_phase = phase_times.groupby('phase')['cycle_time_days'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 4)); avg_cycle_time_by_phase.plot(kind='bar', color=sns.color_palette('tab10'), ax=ax); ax.set_title("Duração Média por Fase do Processo"); plt.xticks(rotation=0)
    plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)
    
    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    # (O código desta função permanece exatamente o mesmo do ficheiro que forneceu)
    plots = {}
    metrics = {}
    
    df_start_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'start_date']].rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_start_events['lifecycle:transition'] = 'start'
    df_complete_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'end_date']].rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_complete_events['lifecycle:transition'] = 'complete'
    log_df_full_lifecycle = pd.concat([df_start_events, df_complete_events]).sort_values('time:timestamp')
    log_full_pm4py = pm4py.convert_to_event_log(log_df_full_lifecycle)

    variants_dict = variants_filter.get_variants(_event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_3_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)
    
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    
    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        fig, ax = plt.subplots(figsize=(8, 4)); barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, hue='Métrica', legend=False, palette='coolwarm')
        for p in barplot.patches: ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', color='#E5E7EB')
        ax.set_title(title); ax.set_ylim(0, 1.05); return fig
        
    metrics_im = {"Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im), "Generalização": generalization_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im), "Simplicidade": simplicity_evaluator.apply(net_im)}
    plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)'))
    metrics['inductive_miner'] = metrics_im

    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    
    metrics_hm = {"Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm), "Generalização": generalization_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm), "Simplicidade": simplicity_evaluator.apply(net_hm)}
    plots['metrics_heuristic'] = convert_fig_to_bytes(plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)'))
    metrics['heuristics_miner'] = metrics_hm
    
    kpi_temporal = _df_projects.groupby('completion_month').agg(avg_lead_time=('actual_duration_days', 'mean'), throughput=('project_id', 'count')).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6)); ax1.plot(kpi_temporal['completion_month'], kpi_temporal['avg_lead_time'], marker='o', color='#2563EB', label='Lead Time'); ax2 = ax1.twinx(); ax2.bar(kpi_temporal['completion_month'], kpi_temporal['throughput'], color='#06B6D4', alpha=0.6, label='Throughput'); fig.suptitle('Séries Temporais de KPIs de Performance')
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9)); ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.label.set_color('#2563EB'); ax2.yaxis.label.set_color('#06B6D4'); ax1.tick_params(axis='y', colors='#2563EB'); ax2.tick_params(axis='y', colors='#06B6D4')
    plots['kpi_time_series'] = convert_fig_to_bytes(fig)
    
    fig_gantt, ax_gantt = plt.subplots(figsize=(20, max(10, len(_df_projects) * 0.4))); all_projects = _df_projects.sort_values('start_date')['project_id'].tolist(); gantt_data = _df_tasks_raw[_df_tasks_raw['project_id'].isin(all_projects)].sort_values(['project_id', 'start_date']); project_y_map = {proj_id: i for i, proj_id in enumerate(all_projects)}; color_map = {task_name: plt.get_cmap('tab10', gantt_data['task_name'].nunique())(i) for i, task_name in enumerate(gantt_data['task_name'].unique())};
    for _, task in gantt_data.iterrows(): ax_gantt.barh(project_y_map[task['project_id']], (task['end_date'] - task['start_date']).days + 1, left=task['start_date'], height=0.6, color=color_map[task['task_name']], edgecolor='#E5E7EB')
    ax_gantt.set_yticks(list(project_y_map.values())); ax_gantt.set_yticklabels([f"Projeto {pid}" for pid in project_y_map.keys()]); ax_gantt.invert_yaxis(); ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); plt.xticks(rotation=45)
    handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]; ax_gantt.legend(handles, color_map.keys(), title='Tipo de Tarefa', bbox_to_anchor=(1.05, 1), loc='upper left'); ax_gantt.set_title('Linha do Tempo de Todos os Projetos (Gantt Chart)'); fig_gantt.tight_layout()
    plots['gantt_chart_all_projects'] = convert_fig_to_bytes(fig_gantt)

    dfg_perf, _, _ = pm4py.discover_performance_dfg(log_full_pm4py)
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=log_full_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)
    
    fig, ax = plt.subplots(figsize=(8, 4)); log_df_full_lifecycle['weekday'] = log_df_full_lifecycle['time:timestamp'].dt.day_name(); weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = log_df_full_lifecycle.groupby('weekday')['case:concept:name'].count().reindex(weekday_order).fillna(0); sns.barplot(x=heatmap_data.index, y=heatmap_data.values, ax=ax, hue=heatmap_data.index, legend=False, palette='coolwarm'); ax.set_title('Ocorrências de Atividades por Dia da Semana'); plt.xticks(rotation=45)
    plots['temporal_heatmap_fixed'] = convert_fig_to_bytes(fig)
    
    log_df_complete = pm4py.convert_to_dataframe(_event_log_pm4py)
    handovers = Counter((log_df_complete.iloc[i]['org:resource'], log_df_complete.iloc[i+1]['org:resource']) for i in range(len(log_df_complete)-1) if log_df_complete.iloc[i]['case:concept:name'] == log_df_complete.iloc[i+1]['case:concept:name'] and log_df_complete.iloc[i]['org:resource'] != log_df_complete.iloc[i+1]['org:resource'])
    fig_net, ax_net = plt.subplots(figsize=(10, 10)); G = nx.DiGraph();
    for (source, target), weight in handovers.items(): G.add_edge(str(source), str(target), weight=weight)
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42); weights = [G[u][v]['weight'] for u,v in G.edges()]; nx.draw(G, pos, with_labels=True, node_color='#2563EB', edge_color='#E5E7EB', width=[w*0.5 for w in weights], ax=ax_net, font_size=10, connectionstyle='arc3,rad=0.1', labels={node: node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net, font_color='#FBBF24'); ax_net.set_title('Rede Social de Recursos (Handover Network)')
    plots['resource_network_adv'] = convert_fig_to_bytes(fig_net)
    
    if 'skill_level' in _df_resources.columns:
        perf_recursos = _df_full_context.groupby('resource_id').agg(total_hours=('hours_worked', 'sum'), total_tasks=('task_id', 'nunique')).reset_index()
        perf_recursos['avg_hours_per_task'] = perf_recursos['total_hours'] / perf_recursos['total_tasks']
        perf_recursos = perf_recursos.merge(_df_resources[['resource_id', 'skill_level', 'resource_name']], on='resource_id')
        fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=perf_recursos, x='skill_level', y='avg_hours_per_task', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação entre Skill e Performance")
        plots['skill_vs_performance_adv'] = convert_fig_to_bytes(fig)
        
        resource_role_counts = _df_full_context.groupby(['resource_name', 'skill_level']).size().reset_index(name='count')
        G_bipartite = nx.Graph(); resources_nodes = resource_role_counts['resource_name'].unique(); roles_nodes = resource_role_counts['skill_level'].unique(); G_bipartite.add_nodes_from(resources_nodes, bipartite=0); G_bipartite.add_nodes_from(roles_nodes, bipartite=1)
        for _, row in resource_role_counts.iterrows(): G_bipartite.add_edge(row['resource_name'], row['skill_level'], weight=row['count'])
        fig, ax = plt.subplots(figsize=(12, 10)); pos = nx.bipartite_layout(G_bipartite, resources_nodes); nx.draw(G_bipartite, pos, with_labels=True, node_color=['#2563EB' if node in resources_nodes else '#FBBF24' for node in G_bipartite.nodes()], node_size=2000, ax=ax, font_size=8, edge_color='#374151', labels={node: node for node in G_bipartite.nodes()})
        edge_labels = nx.get_edge_attributes(G_bipartite, 'weight'); nx.draw_networkx_edge_labels(G_bipartite, pos, edge_labels=edge_labels, ax=ax, font_color='#06B6D4'); ax.set_title('Rede de Recursos por Função')
        plots['resource_network_bipartite'] = convert_fig_to_bytes(fig)

    variants_df = log_df_full_lifecycle.groupby('case:concept:name').agg(variant=('concept:name', lambda x: tuple(x)), start_timestamp=('time:timestamp', 'min'), end_timestamp=('time:timestamp', 'max')).reset_index()
    variants_df['duration_hours'] = (variants_df['end_timestamp'] - variants_df['start_timestamp']).dt.total_seconds() / 3600
    variant_durations = variants_df.groupby('variant').agg(count=('case:concept:name', 'count'), avg_duration_hours=('duration_hours', 'mean')).reset_index().sort_values(by='count', ascending=False).head(10)
    variant_durations['variant_str'] = variant_durations['variant'].apply(lambda x: ' -> '.join([str(i) for i in x][:4]) + '...')
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_duration_hours', y='variant_str', data=variant_durations.astype({'avg_duration_hours':'float'}), ax=ax, hue='variant_str', legend=False, palette='plasma'); ax.set_title('Duração Média das 10 Variantes Mais Comuns'); fig.tight_layout()
    plots['variant_duration_plot'] = convert_fig_to_bytes(fig)

    aligned_traces = alignments_miner.apply(log_full_pm4py, net_im, im_im, fm_im)
    deviations_list = [{'fitness': trace['fitness'], 'deviations': sum(1 for move in trace['alignment'] if '>>' in move[0] or '>>' in move[1])} for trace in aligned_traces if 'fitness' in trace]
    deviations_df = pd.DataFrame(deviations_list)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(x='fitness', y='deviations', data=deviations_df, alpha=0.6, ax=ax, color='#FBBF24'); ax.set_title('Diagrama de Dispersão (Fitness vs. Desvios)'); fig.tight_layout()
    plots['deviation_scatter_plot'] = convert_fig_to_bytes(fig)

    case_fitness_data = [{'project_id': str(trace.attributes['concept:name']), 'fitness': alignment['fitness']} for trace, alignment in zip(log_full_pm4py, aligned_traces) if 'concept:name' in trace.attributes]
    case_fitness_df = pd.DataFrame(case_fitness_data).merge(_df_projects[['project_id', 'end_date']], on='project_id')
    case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str)
    monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax, color='#2563EB'); ax.set_title('Score de Conformidade ao Longo do Tempo'); ax.set_ylim(0, 1.05); ax.tick_params(axis='x', rotation=45); fig.tight_layout()
    plots['conformance_over_time_plot'] = convert_fig_to_bytes(fig)

    kpi_daily = _df_projects.groupby(_df_projects['end_date'].dt.date).agg(avg_cost_per_day=('cost_per_day', 'mean')).reset_index()
    kpi_daily.rename(columns={'end_date': 'completion_date'}, inplace=True)
    kpi_daily['completion_date'] = pd.to_datetime(kpi_daily['completion_date'])
    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=kpi_daily, x='completion_date', y='avg_cost_per_day', ax=ax, color='#FBBF24'); ax.set_title('Custo Médio por Dia ao Longo do Tempo'); fig.tight_layout()
    plots['cost_per_day_time_series'] = convert_fig_to_bytes(fig)

    df_projects_sorted = _df_projects.sort_values(by='end_date'); df_projects_sorted['cumulative_throughput'] = range(1, len(df_projects_sorted) + 1)
    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(x='end_date', y='cumulative_throughput', data=df_projects_sorted, ax=ax, color='#06B6D4'); ax.set_title('Gráfico Acumulado de Throughput'); fig.tight_layout()
    plots['cumulative_throughput_plot'] = convert_fig_to_bytes(fig)
    
    def generate_custom_variants_plot(event_log):
        variants = variants_filter.get_variants(event_log)
        top_variants = sorted(variants.items(), key=lambda item: len(item[1]), reverse=True)[:10]
        variant_sequences = {f"V{i+1} ({len(v)} casos)": [str(a) for a in k] for i, (k, v) in enumerate(top_variants)}
        fig, ax = plt.subplots(figsize=(12, 6)) 
        all_activities = sorted(list(set([act for seq in variant_sequences.values() for act in seq])))
        activity_to_y = {activity: i for i, activity in enumerate(all_activities)}
        
        colors = plt.cm.get_cmap('tab10', len(variant_sequences.keys()))
        for i, (variant_name, sequence) in enumerate(variant_sequences.items()):
            ax.plot(range(len(sequence)), [activity_to_y[activity] for activity in sequence], marker='o', linestyle='-', label=variant_name, color=colors(i))
            
        ax.set_yticks(list(activity_to_y.values()))
        ax.set_yticklabels(list(activity_to_y.keys()))
        ax.set_title('Sequência de Atividades das 10 Variantes Mais Comuns')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        return fig
    plots['custom_variants_sequence_plot'] = convert_fig_to_bytes(generate_custom_variants_plot(log_full_pm4py))
    
    milestones = ['Analise e Design', 'Implementacao da Funcionalidade', 'Execucao de Testes', 'Deploy da Aplicacao']
    df_milestones = _df_tasks_raw[_df_tasks_raw['task_name'].isin(milestones)].copy()
    milestone_pairs = []
    for project_id, group in df_milestones.groupby('project_id'):
        sorted_tasks = group.sort_values('start_date')
        for i in range(len(sorted_tasks) - 1):
            duration = (sorted_tasks.iloc[i+1]['start_date'] - sorted_tasks.iloc[i]['end_date']).total_seconds() / 3600
            if duration >= 0: milestone_pairs.append({'transition': f"{sorted_tasks.iloc[i]['task_name']} -> {sorted_tasks.iloc[i+1]['task_name']}", 'duration_hours': duration})
    df_milestone_pairs = pd.DataFrame(milestone_pairs)
    if not df_milestone_pairs.empty:
        fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_milestone_pairs, x='duration_hours', y='transition', ax=ax, orient='h', hue='transition', legend=False, palette='coolwarm'); ax.set_title('Análise de Tempo entre Marcos do Processo'); fig.tight_layout()
        plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    df_tasks_sorted = _df_tasks_raw.sort_values(['project_id', 'start_date']); df_tasks_sorted['previous_end_date'] = df_tasks_sorted.groupby('project_id')['end_date'].shift(1)
    df_tasks_sorted['waiting_time_days'] = (df_tasks_sorted['start_date'] - df_tasks_sorted['previous_end_date']).dt.total_seconds() / (24 * 3600)
    df_tasks_sorted.loc[df_tasks_sorted['waiting_time_days'] < 0, 'waiting_time_days'] = 0
    df_tasks_sorted['previous_task_name'] = df_tasks_sorted.groupby('project_id')['task_name'].shift(1)
    waiting_times_matrix = df_tasks_sorted.pivot_table(index='previous_task_name', columns='task_name', values='waiting_time_days', aggfunc='mean').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(waiting_times_matrix * 24, cmap='Blues', annot=True, fmt='.1f', ax=ax, annot_kws={"size": 8}, linewidths=.5, linecolor='#374151'); ax.set_title('Matriz de Tempo de Espera entre Atividades (horas)'); fig.tight_layout()
    plots['waiting_time_matrix_plot'] = convert_fig_to_bytes(fig)
    
    resource_efficiency = _df_full_context.groupby('resource_name').agg(total_hours_worked=('hours_worked', 'sum'), total_tasks_completed=('task_name', 'count')).reset_index()
    resource_efficiency['avg_hours_per_task'] = resource_efficiency['total_hours_worked'] / resource_efficiency['total_tasks_completed']
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=resource_efficiency.sort_values(by='avg_hours_per_task'), x='avg_hours_per_task', y='resource_name', orient='h', ax=ax, hue='resource_name', legend=False, palette='viridis'); ax.set_title('Métricas de Eficiência Individual por Recurso'); fig.tight_layout()
    plots['resource_efficiency_plot'] = convert_fig_to_bytes(fig)

    df_tasks_sorted['sojourn_time_hours'] = df_tasks_sorted['waiting_time_days'] * 24
    waiting_time_by_task = df_tasks_sorted.groupby('task_name')['sojourn_time_hours'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=waiting_time_by_task.sort_values(by='sojourn_time_hours', ascending=False), x='sojourn_time_hours', y='task_name', ax=ax, hue='task_name', legend=False, palette='magma'); ax.set_title('Tempo Médio de Espera por Atividade'); fig.tight_layout()
    plots['avg_waiting_time_by_activity_plot'] = convert_fig_to_bytes(fig)
    
    return plots, metrics

# --- NOVA FUNÇÃO DE ANÁLISE (EDA) ---
@st.cache_data
def run_eda_analysis(dfs):
    plots = {}
    tables = {}
    
    # --- Pré-processamento e Feature Engineering (da célula 6 do notebook) ---
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')

    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['project_name'].str.extract(r'Projeto \d+: (.*?) ')
    df_tasks['task_duration_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.days
    df_projects['completion_quarter'] = df_projects['end_date'].dt.to_period('Q')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M')

    df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
    df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']

    dep_counts = df_dependencies.groupby('project_id').size().reset_index(name='dependency_count')
    task_counts = df_tasks.groupby('project_id').size().reset_index(name='task_count')
    project_complexity = pd.merge(dep_counts, task_counts, on='project_id', how='outer').fillna(0)
    project_complexity['complexity_ratio'] = (project_complexity['dependency_count'] / project_complexity['task_count']).fillna(0)

    project_aggregates = df_alloc_costs.groupby('project_id').agg(
        total_actual_cost=('cost_of_work', 'sum'),
        avg_hourly_rate=('cost_per_hour', 'mean'),
        num_resources=('resource_id', 'nunique')
    ).reset_index()

    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects = df_projects.merge(project_complexity, on='project_id', how='left')
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)

    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
    
    tables['stats_table'] = df_projects[['actual_duration_days', 'days_diff', 'budget_impact', 'total_actual_cost', 'cost_diff', 'num_resources', 'avg_hourly_rate']].describe().round(2)

    # --- Geração dos Gráficos (da célula 6 do notebook) ---
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_projects, x='project_status', ax=ax, palette='viridis'); ax.set_title('Distribuição do Status dos Projetos')
    plots['plot_01'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='days_diff', kde=True, color='salmon', ax=ax); ax.set_title('Diferença entre Data Real e Planeada')
    plots['plot_03'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(15, 8)); df_projects_sorted = df_projects.sort_values('budget_impact', ascending=False); sns.barplot(data=df_projects_sorted, x='project_name', y='budget_impact', color='lightblue', label='Orçamento', ax=ax); sns.barplot(data=df_projects_sorted, x='project_name', y='total_actual_cost', color='salmon', alpha=0.8, label='Custo Real', ax=ax); ax.tick_params(axis='x', rotation=90); ax.legend(); ax.set_title('Custo Real vs. Orçamento por Projeto')
    plots['plot_04'] = convert_fig_to_bytes(fig)
    
    df_projects_q = df_projects.dropna(subset=['completion_quarter']).copy()
    df_projects_q['completion_quarter'] = df_projects_q['completion_quarter'].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_projects_q, x='completion_quarter', y='days_diff', ax=ax, palette='coolwarm'); ax.set_title('Performance de Prazos por Trimestre')
    plots['plot_05'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_projects_q.groupby('completion_quarter')['total_actual_cost'].mean().reset_index(), x='completion_quarter', y='total_actual_cost', ax=ax, palette='viridis'); ax.set_title('Custo Médio dos Projetos por Trimestre')
    plots['plot_06'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_projects_q.groupby('completion_quarter')['num_resources'].mean().reset_index(), x='completion_quarter', y='num_resources', ax=ax, palette='crest'); ax.set_title('Nº Médio de Recursos por Projeto a Cada Trimestre')
    plots['plot_07'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_tasks, y='task_type', order=df_tasks['task_type'].value_counts().index, ax=ax, palette='crest'); ax.set_title('Distribuição de Tarefas por Tipo')
    plots['plot_08'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_tasks, x='priority', ax=ax, palette='magma'); ax.set_title('Distribuição de Tarefas por Prioridade')
    plots['plot_09'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_tasks, x='task_duration_days', kde=True, color='indigo', ax=ax); ax.set_title('Distribuição da Duração das Tarefas')
    plots['plot_10'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_tasks.sort_values('task_duration_days', ascending=False).head(10), x='task_duration_days', y='task_name', ax=ax, palette='rocket'); ax.set_title('Top 10 Tarefas Específicas Mais Demoradas')
    plots['plot_11'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_resources, x='resource_type', ax=ax, palette='cubehelix'); ax.set_title('Distribuição de Recursos por Tipo')
    plots['plot_12'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_full_context.groupby('resource_name')['days_diff'].mean().sort_values(ascending=False).reset_index().head(20), y='resource_name', x='days_diff', ax=ax, palette='cividis'); ax.set_title('Atraso Médio por Recurso')
    plots['plot_14'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='cost_per_day', kde=True, color='teal', ax=ax); ax.set_title('Distribuição do Custo por Dia (Eficiência)')
    plots['plot_16'] = convert_fig_to_bytes(fig)
    
    df_projects['budget_bin'] = pd.cut(df_projects['budget_impact'], bins=4)
    data_plot_17 = df_full_context.merge(df_projects[['project_id', 'budget_bin']], on='project_id').groupby(['budget_bin', 'resource_type'], observed=False)['cost_of_work'].sum().unstack()
    fig = data_plot_17.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8)).get_figure()
    plots['plot_17'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='total_actual_cost', y='days_diff', color='crimson', ax=ax); ax.set_title('Custo Real vs. Atraso')
    plots['plot_18'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='avg_hourly_rate', y='days_diff', color='olivedrab', ax=ax); ax.set_title('Rate Horário Médio vs. Atraso')
    plots['plot_19'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='num_resources', y='total_actual_cost', color='darkorange', ax=ax); ax.set_title('Nº de Recursos vs. Custo Total')
    plots['plot_20'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(x=df_projects['budget_bin'], y=df_projects['days_diff'], ax=ax, palette='pastel'); ax.set_title('Atraso por Faixa de Orçamento')
    plots['plot_22'] = convert_fig_to_bytes(fig)
    
    df_skill_delay = df_full_context[['skill_level', 'project_id', 'days_diff']].drop_duplicates().dropna()
    if not df_skill_delay.empty:
        fig, ax = plt.subplots(figsize=(10, 6)); sns.violinplot(data=df_skill_delay, x='skill_level', y='days_diff', ax=ax, palette='muted'); ax.set_title('Atraso por Nível de Competência')
        plots['plot_23'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='complexity_ratio', kde=True, color='darkslateblue', ax=ax); ax.set_title('Distribuição da Complexidade dos Projetos')
    plots['plot_24'] = convert_fig_to_bytes(fig)
    
    predecessor_counts = df_dependencies.merge(df_tasks, left_on='task_id_predecessor', right_on='task_id')['task_type'].value_counts()
    successor_counts = df_dependencies.merge(df_tasks, left_on='task_id_successor', right_on='task_id')['task_type'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)); predecessor_counts.plot(kind='bar', color=sns.color_palette("Paired")[1], title='Mais Comuns como Predecessoras', ax=ax1); successor_counts.plot(kind='bar', color=sns.color_palette("Paired")[3], title='Mais Comuns como Sucessoras', ax=ax2);
    plots['plot_25'] = convert_fig_to_bytes(fig)

    PROJECT_ID_EXAMPLE = "25"
    project_deps = df_dependencies[df_dependencies['project_id'] == PROJECT_ID_EXAMPLE]
    if not project_deps.empty:
        G = nx.from_pandas_edgelist(project_deps, 'task_id_predecessor', 'task_id_successor', create_using=nx.DiGraph()); pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(14, 9)); nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, ax=ax); ax.set_title(f'Grafo de Dependências: Projeto {PROJECT_ID_EXAMPLE}')
        plots['plot_26'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='complexity_ratio', y='days_diff', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax); ax.set_title('Relação entre Complexidade e Atraso')
    plots['plot_27'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='dependency_count', y='cost_diff', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax); ax.set_title('Relação entre Dependências e Desvio de Custo')
    plots['plot_28'] = convert_fig_to_bytes(fig)

    df_numeric = df_full_context[['budget_impact', 'total_actual_cost', 'days_diff', 'skill_level', 'cost_per_hour', 'priority']].dropna()
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax); ax.set_title('Matriz de Correlação')
    plots['plot_29'] = convert_fig_to_bytes(fig)
    
    monthly_kpis = df_projects.groupby('completion_month').agg(mean_days_diff=('days_diff', 'mean'), mean_cost_diff=('cost_diff', 'mean'), completed_projects=('project_id', 'count'), mean_duration=('actual_duration_days', 'mean')).reset_index()
    monthly_kpis['completion_month'] = monthly_kpis['completion_month'].astype(str)
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10));
    ax1.plot(monthly_kpis['completion_month'], monthly_kpis['mean_days_diff'], marker='o', color='royalblue'); ax1.set_title('Atraso Médio Mensal'); ax1.grid(True)
    ax2.plot(monthly_kpis['completion_month'], monthly_kpis['mean_cost_diff'], marker='o', color='firebrick'); ax2.set_title('Desvio de Custo Médio Mensal'); ax2.grid(True)
    plots['plot_30'] = convert_fig_to_bytes(fig)
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10));
    ax1.bar(monthly_kpis['completion_month'], monthly_kpis['completed_projects'], color='seagreen'); ax1.set_title('Nº de Projetos Concluídos por Mês'); ax1.grid(True)
    ax2.plot(monthly_kpis['completion_month'], monthly_kpis['mean_duration'], marker='o', color='purple'); ax2.set_title('Duração Média dos Projetos Concluídos'); ax2.grid(True)
    plots['plot_31'] = convert_fig_to_bytes(fig)

    return plots, tables

# --- NOVA FUNÇÃO DE ANÁLISE (REINFORCEMENT LEARNING) ---
#@st.cache_data # Removido para permitir interatividade e barra de progresso
def run_rl_analysis(dfs, project_id_to_simulate, num_episodes, reward_config, progress_bar, status_text):
    # Dicionários para guardar os resultados
    plots = {}
    tables = {}
    logs = {}

    # --- Carregamento e Pré-processamento dos dados (da célula de RL) ---
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations, df_dependencies]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    def calculate_business_days(start, end):
        return np.busday_count(start.date(), end.date())

    df_projects['planned_duration_days'] = df_projects.apply(lambda row: calculate_business_days(row['start_date'], row['planned_end_date']), axis=1)
    df_projects['total_duration_days'] = df_projects.apply(lambda row: calculate_business_days(row['start_date'], row['end_date']), axis=1)

    df_real_costs = (df_resource_allocations.merge(df_resources[['resource_id', 'cost_per_hour']], on='resource_id')
                     .assign(cost=lambda df: df.hours_worked * df.cost_per_hour)
                     .groupby('project_id')['cost'].sum().rename('actual_historical_cost').reset_index())
    df_projects = df_projects.merge(df_real_costs, on='project_id', how='left').fillna({'actual_historical_cost': 0})

    # --- AMBIENTE E AGENTE (CLASSES) ---
    class ProjectManagementEnv:
        def __init__(self, df_tasks, df_resources, df_dependencies, min_progress_for_next_phase=0.7, reward_config=None):
            self.rewards = reward_config
            self.df_tasks = df_tasks
            self.df_resources = df_resources
            self.df_dependencies = df_dependencies
            self.resource_types = sorted(self.df_resources['resource_type'].unique().tolist())
            self.task_types = sorted(self.df_tasks['task_type'].unique().tolist())
            self.resources_by_type = {rt: self.df_resources[self.df_resources['resource_type'] == rt] for rt in self.resource_types}
            self.all_actions = self._generate_all_actions()
            self.min_progress_for_next_phase = min_progress_for_next_phase
            self.reset(df_projects.iloc[0]['project_id'])

        def _generate_all_actions(self):
            actions = set()
            for res_type in self.resource_types:
                actions.add((res_type, 'idle'))
                for task_type in self.task_types:
                    actions.add((res_type, task_type))
            return tuple(sorted(list(actions)))

        def reset(self, project_id):
            self.current_project_id = project_id
            project_info = df_projects.loc[df_projects['project_id'] == project_id].iloc[0]
            self.current_cost = 0.0
            self.day_count = 0
            self.current_date = project_info['start_date']
            self.episode_logs = []
            project_tasks = self.df_tasks[self.df_tasks['project_id'] == project_id].sort_values('task_id')
            self.tasks_to_do_count = len(project_tasks)
            self.total_estimated_budget = project_info['actual_historical_cost']
            self.total_estimated_effort = project_tasks[['estimated_effort_dev', 'estimated_effort_qa', 'estimated_effort_ops', 'estimated_effort_dba']].sum().sum()
            project_dependencies = self.df_dependencies[self.df_dependencies['project_id'] == project_id]
            self.task_dependencies = {row['task_id_successor']: row['task_id_predecessor'] for _, row in project_dependencies.iterrows()}
            self.tasks_state = {
                task['task_id']: {'status': 'Pendente', 'progress_dev': 0.0, 'progress_qa': 0.0, 'progress_ops': 0.0, 'progress_dba': 0.0,
                                  'estimated_effort_dev': task['estimated_effort_dev'], 'estimated_effort_qa': task['estimated_effort_qa'],
                                  'estimated_effort_ops': task['estimated_effort_ops'], 'estimated_effort_dba': task['estimated_effort_dba'],
                                  'priority': task['priority'], 'task_type': task['task_type']}
                for _, task in project_tasks.iterrows()
            }
            return self.get_state()

        def get_state(self):
            progress_total = sum(d.get('progress_dev',0) + d.get('progress_qa',0) + d.get('progress_ops',0) + d.get('progress_dba',0) for d in self.tasks_state.values())
            progress_ratio = progress_total / self.total_estimated_effort if self.total_estimated_effort > 0 else 1.0
            budget_ratio = self.current_cost / self.total_estimated_budget if self.total_estimated_budget > 0 else 0.0
            project_info = df_projects.loc[df_projects['project_id'] == self.current_project_id].iloc[0]
            time_ratio = self.day_count / project_info['total_duration_days'] if project_info['total_duration_days'] > 0 else 0.0
            pending_tasks = sum(1 for t in self.tasks_state.values() if t['status'] != 'Concluída')
            return (int(progress_ratio * 10), int(budget_ratio * 10), int(time_ratio * 10), pending_tasks)

        def get_possible_actions_for_state(self):
            possible_actions = set()
            for res_type in self.resource_types:
                has_eligible_task_for_type = False
                for task_type in self.task_types:
                    if any(t_data['task_type'] == task_type and self._is_task_eligible(t_id, res_type) for t_id, t_data in self.tasks_state.items()):
                        possible_actions.add((res_type, task_type))
                        has_eligible_task_for_type = True
                if not has_eligible_task_for_type:
                    possible_actions.add((res_type, 'idle'))
            return list(possible_actions)

        def _is_task_eligible(self, task_id, res_type):
            task_data = self.tasks_state[task_id]
            if task_data['status'] == 'Concluída': return False
            pred_id = self.task_dependencies.get(task_id)
            if pred_id and self.tasks_state.get(pred_id, {}).get('status') != 'Concluída': return False
            effort_map = {'Desenvolvedor': 'dev', 'Testador': 'qa', 'Eng. DevOps': 'ops', 'DBA': 'dba'}
            phase = effort_map.get(res_type)
            if not phase or task_data[f'progress_{phase}'] >= task_data[f'estimated_effort_{phase}']: return False
            dependencies = {'qa': 'dev', 'ops': 'qa', 'dba': 'dev'}
            if phase in dependencies:
                dep_phase = dependencies[phase]
                if task_data[f'estimated_effort_{dep_phase}'] > 0:
                    progress_ratio = task_data[f'progress_{dep_phase}'] / task_data[f'estimated_effort_{dep_phase}']
                    if progress_ratio < self.min_progress_for_next_phase: return False
            return True

        def step(self, action_set):
            if self.current_date.weekday() >= 5:
                self.current_date += timedelta(days=1)
                daily_cost = 0; reward_from_tasks = 0
            else:
                daily_cost = 0; reward_from_tasks = 0; resources_used_today = set()
                for res_type, task_type in action_set:
                    if task_type == "idle":
                        reward_from_tasks -= self.rewards['idle_penalty']; continue
                    available_resources = self.resources_by_type[res_type][~self.resources_by_type[res_type]['resource_id'].isin(resources_used_today)]
                    if available_resources.empty: continue
                    res_info = available_resources.sample(1).iloc[0]
                    eligible_tasks = [tid for tid, tdata in self.tasks_state.items() if tdata['task_type'] == task_type and self._is_task_eligible(tid, res_type)]
                    if not eligible_tasks: continue
                    resources_used_today.add(res_info['resource_id'])
                    eligible_tasks.sort(key=lambda tid: self.tasks_state[tid]['priority'], reverse=True)
                    task_id_to_work = eligible_tasks[0]
                    task_data = self.tasks_state[task_id_to_work]
                    effort_type = {'Desenvolvedor': 'dev', 'Testador': 'qa', 'Eng. DevOps': 'ops', 'DBA': 'dba'}[res_type]
                    remaining_effort = task_data[f'estimated_effort_{effort_type}'] - task_data[f'progress_{effort_type}']
                    hours_to_work = min(res_info['daily_capacity'], remaining_effort)
                    cost_today = hours_to_work * res_info['cost_per_hour']
                    daily_cost += cost_today
                    self.episode_logs.append({'day': self.day_count, 'resource_id': res_info['resource_id'], 'resource_type': res_type, 'task_id': task_id_to_work, 'hours_worked': hours_to_work, 'daily_cost': cost_today, 'action': f'Work on {task_type}'})
                    if task_data['status'] == 'Pendente': task_data['status'] = 'Em Andamento'
                    task_data[f'progress_{effort_type}'] += hours_to_work
                    if (task_data['progress_dev'] >= task_data['estimated_effort_dev'] and task_data['progress_qa'] >= task_data['estimated_effort_qa'] and
                        task_data['progress_ops'] >= task_data['estimated_effort_ops'] and task_data['progress_dba'] >= task_data['estimated_effort_dba']):
                        task_data['status'] = 'Concluída'; reward_from_tasks += task_data['priority'] * self.rewards['priority_task_bonus_factor']

                self.current_cost += daily_cost
                self.current_date += timedelta(days=1)
                if self.current_date.weekday() < 5: self.day_count += 1

            project_is_done = all(t['status'] == 'Concluída' for t in self.tasks_state.values())
            total_reward = reward_from_tasks - self.rewards['daily_time_penalty']
            if project_is_done:
                project_info = df_projects.loc[df_projects['project_id'] == self.current_project_id].iloc[0]
                time_diff = project_info['total_duration_days'] - self.day_count
                total_reward += self.rewards['completion_base']
                total_reward += time_diff * self.rewards['per_day_early_bonus'] if time_diff >= 0 else time_diff * self.rewards['per_day_late_penalty']
                total_reward -= self.current_cost * self.rewards['cost_impact_factor']
            return total_reward, project_is_done

    class QLearningAgent:
        def __init__(self, actions, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
            self.actions = actions
            self.action_to_index = {action: i for i, action in enumerate(actions)}
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
            self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
            self.epsilon_decay, self.min_epsilon = epsilon_decay, min_epsilon
            self.epsilon_history, self.episode_rewards, self.episode_durations, self.episode_costs = [], [], [], []

        def choose_action(self, state, possible_actions):
            if not possible_actions: return None
            if random.uniform(0, 1) < self.epsilon: return random.choice(possible_actions)
            else:
                q_values = {action: self.q_table[state][self.action_to_index[action]] for action in possible_actions if action in self.action_to_index}
                if not q_values: return random.choice(possible_actions)
                max_q = max(q_values.values()); best_actions = sorted([action for action, q_val in q_values.items() if q_val == max_q])
                return best_actions[0]

        def update_q_table(self, state, action, reward, next_state):
            action_index = self.action_to_index.get(action)
            if action_index is None: return
            old_value = self.q_table[state][action_index]
            next_max = np.max(self.q_table[next_state]) if next_state in self.q_table else 0.0
            new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
            self.q_table[state][action_index] = new_value

        def decay_epsilon(self):
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)
    
    # --- EXECUÇÃO PRINCIPAL DO RL ---
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    
    df_projects_train = df_projects.sample(frac=0.8, random_state=SEED)
    df_projects_test = df_projects.drop(df_projects_train.index)

    env = ProjectManagementEnv(df_tasks, df_resources, df_dependencies, reward_config=reward_config)
    agent = QLearningAgent(actions=env.all_actions)
    
    time_per_episode = 0.06 
    
    for episode in range(num_episodes):
        project_id = df_projects_train.sample(1, random_state=episode).iloc[0]['project_id']
        state = env.reset(project_id)
        episode_reward, done = 0, False
        calendar_day = 0
        while not done and calendar_day < 1000:
            possible_actions = env.get_possible_actions_for_state()
            action_set = set()
            for res_type in env.resource_types:
                actions_for_res = [a for a in possible_actions if a[0] == res_type]
                if actions_for_res:
                    chosen_action = agent.choose_action(state, actions_for_res)
                    if chosen_action: action_set.add(chosen_action)
            reward, done = env.step(action_set)
            next_state = env.get_state()
            for action in action_set: agent.update_q_table(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            calendar_day += 1
            if env.day_count > 730: break
        agent.decay_epsilon()
        agent.episode_rewards.append(episode_reward)
        agent.episode_durations.append(env.day_count)
        agent.episode_costs.append(env.current_cost)
        
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        remaining_time = (num_episodes - (episode + 1)) * time_per_episode
        status_text.info(f"A treinar... Episódio {episode + 1}/{num_episodes}. Tempo estimado restante: {remaining_time:.0f} segundos.")
    
    status_text.success("Treino e simulação concluídos!")
    
    # --- FUNÇÕES DE PLOT E TABULAÇÃO ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    rewards, durations, costs, epsilon_history = agent.episode_rewards, agent.episode_durations, agent.episode_costs, agent.epsilon_history
    axes[0, 0].plot(rewards, alpha=0.6); axes[0, 0].plot(pd.Series(rewards).rolling(50).mean(), lw=2, label='Média Móvel (50 ep)')
    axes[0, 0].set_title('Recompensa por Episódio'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(durations, alpha=0.6); axes[0, 1].plot(pd.Series(durations).rolling(50).mean(), lw=2, label='Média Móvel (50 ep)')
    axes[0, 1].set_title('Duração por Episódio'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(epsilon_history); axes[1, 0].set_title('Decaimento do Epsilon'); axes[1, 0].grid(True)
    axes[1, 1].plot(costs, alpha=0.6); axes[1, 1].plot(pd.Series(costs).rolling(50).mean(), lw=2, label='Média Móvel (50 ep)')
    axes[1, 1].set_title('Custo por Episódio'); axes[1, 1].legend(); axes[1, 1].grid(True)
    fig.tight_layout()
    plots['training_metrics'] = convert_fig_to_bytes(fig)

    def evaluate_agent(agent, env, df_projects_to_evaluate):
        agent.epsilon = 0
        results = []
        for _, prj_info in df_projects_to_evaluate.iterrows():
            state = env.reset(prj_info['project_id'])
            done = False; calendar_day = 0
            while not done and calendar_day < 1000:
                possible_actions = env.get_possible_actions_for_state()
                action_set = set()
                for res_type in env.resource_types:
                    actions_for_res = [a for a in possible_actions if a[0] == res_type]
                    if actions_for_res:
                        chosen_action = agent.choose_action(state, actions_for_res)
                        if chosen_action: action_set.add(chosen_action)
                _, done = env.step(action_set)
                state = env.get_state(); calendar_day += 1
                if env.day_count > 730: break
            results.append({'project_id': prj_info['project_id'], 'simulated_duration': env.day_count, 'simulated_cost': env.current_cost,
                            'real_duration': prj_info['total_duration_days'], 'real_cost': prj_info['actual_historical_cost']})
        return pd.DataFrame(results)

    test_results_df = evaluate_agent(agent, env, df_projects_test)
    df_plot_test = test_results_df.sort_values(by='real_duration').reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    index_test = np.arange(len(df_plot_test)); bar_width = 0.35
    axes[0].bar(index_test - bar_width/2, df_plot_test['real_duration'], bar_width, label='Real', color='orangered'); axes[0].bar(index_test + bar_width/2, df_plot_test['simulated_duration'], bar_width, label='Simulado (RL)', color='dodgerblue')
    axes[0].set_title('Duração do Projeto (Teste)'); axes[0].set_xticks(index_test); axes[0].set_xticklabels(df_plot_test['project_id'], rotation=45, ha="right"); axes[0].legend()
    axes[1].bar(index_test - bar_width/2, df_plot_test['real_cost'], bar_width, label='Real', color='orangered'); axes[1].bar(index_test + bar_width/2, df_plot_test['simulated_cost'], bar_width, label='Simulado (RL)', color='dodgerblue')
    axes[1].set_title('Custo do Projeto (Teste)'); axes[1].set_xticks(index_test); axes[1].set_xticklabels(df_plot_test['project_id'], rotation=45, ha="right"); axes[1].legend()
    plots['evaluation_comparison_test'] = convert_fig_to_bytes(fig)

    all_results_df = evaluate_agent(agent, env, df_projects)
    df_plot_all = all_results_df.sort_values(by='real_duration').reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    index_all = np.arange(len(df_plot_all));
    axes[0].bar(index_all - bar_width/2, df_plot_all['real_duration'], bar_width, label='Real', color='orangered'); axes[0].bar(index_all + bar_width/2, df_plot_all['simulated_duration'], bar_width, label='Simulado (RL)', color='dodgerblue')
    axes[0].set_title('Duração do Projeto (Todos)'); axes[0].set_xticks(index_all); axes[0].set_xticklabels(df_plot_all['project_id'], rotation=90); axes[0].legend()
    axes[1].bar(index_all - bar_width/2, df_plot_all['real_cost'], bar_width, label='Real', color='orangered'); axes[1].bar(index_all + bar_width/2, df_plot_all['simulated_cost'], bar_width, label='Simulado (RL)', color='dodgerblue')
    axes[1].set_title('Custo do Projeto (Todos)'); axes[1].set_xticks(index_all); axes[1].set_xticklabels(df_plot_all['project_id'], rotation=90); axes[1].legend()
    plots['evaluation_comparison_all'] = convert_fig_to_bytes(fig)

    def get_global_performance_df(results_df):
        real_duration = results_df['real_duration'].sum(); sim_duration = results_df['simulated_duration'].sum()
        real_cost = results_df['real_cost'].sum(); sim_cost = results_df['simulated_cost'].sum()
        dur_improv = real_duration - sim_duration; cost_improv = real_cost - sim_cost
        dur_improv_perc = (dur_improv / real_duration) * 100 if real_duration > 0 else 0
        cost_improv_perc = (cost_improv / real_cost) * 100 if real_cost > 0 else 0
        perf_data = {'Métrica': ['Duração Total (dias úteis)', 'Custo Total (€)'],
                     'Real (Histórico)': [f"{real_duration:.0f}", f"€{real_cost:,.2f}"],
                     'Simulado (RL)': [f"{sim_duration:.0f}", f"€{sim_cost:,.2f}"],
                     'Melhoria': [f"{dur_improv:.0f} ({dur_improv_perc:.1f}%)", f"€{cost_improv:,.2f} ({cost_improv_perc:.1f}%)"]}
        return pd.DataFrame(perf_data)
        
    tables['global_performance_test'] = get_global_performance_df(test_results_df)
    tables['global_performance_all'] = get_global_performance_df(all_results_df)

    # Simulação detalhada
    agent.epsilon = 0; state = env.reset(project_id_to_simulate)
    done = False; calendar_day = 0
    while not done and calendar_day < 1000:
        possible_actions = env.get_possible_actions_for_state()
        action_set = set()
        for res_type in env.resource_types:
            actions_for_res = [a for a in possible_actions if a[0] == res_type]
            if actions_for_res:
                action = agent.choose_action(state, actions_for_res)
                if action: action_set.add(action)
        _, done = env.step(action_set)
        state = env.get_state(); calendar_day += 1
        if env.day_count > 730: break
    simulated_log = pd.DataFrame(env.episode_logs)
    sim_duration, sim_cost = env.day_count, env.current_cost

    project_info = df_projects.loc[df_projects['project_id'] == project_id_to_simulate].iloc[0]
    real_duration, real_cost = project_info['total_duration_days'], project_info['actual_historical_cost']
    
    tables['project_summary'] = pd.DataFrame({
        'Métrica': ['Duração (dias úteis)', 'Custo (€)'],
        'Real (Histórico)': [real_duration, real_cost],
        'Simulado (RL)': [sim_duration, sim_cost]
    })
    
    # Gráfico de comparação detalhada
    project_start_date = df_projects.loc[df_projects['project_id'] == project_id_to_simulate, 'start_date'].iloc[0]
    real_allocations = df_resource_allocations[df_resource_allocations['project_id'] == project_id_to_simulate].copy()
    real_allocations['day'] = real_allocations.apply(lambda row: np.busday_count(project_start_date.date(), row['allocation_date'].date()), axis=1)
    
    total_estimated_effort = env.total_estimated_effort
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    max_day_sim = simulated_log['day'].max() if not simulated_log.empty else 0
    max_day_plot = int(max(max_day_sim, real_duration)); day_range = pd.RangeIndex(start=0, stop=max_day_plot + 1, name='day')
    sim_daily_cost = simulated_log.groupby('day')['daily_cost'].sum()
    sim_cumulative_cost = sim_daily_cost.reindex(day_range, fill_value=0).cumsum()
    real_log_merged = real_allocations.merge(df_resources[['resource_id', 'cost_per_hour']], on='resource_id', how='left')
    real_log_merged['daily_cost'] = real_log_merged['hours_worked'] * real_log_merged['cost_per_hour']
    real_daily_cost = real_log_merged.groupby('day')['daily_cost'].sum()
    real_cumulative_cost = real_daily_cost.reindex(day_range, fill_value=0).cumsum()
    axes[0].plot(sim_cumulative_cost.index, sim_cumulative_cost.values, label='Custo Simulado', marker='o', linestyle='--', color='b')
    axes[0].plot(real_cumulative_cost.index, real_cumulative_cost.values, label='Custo Real', marker='x', linestyle='-', color='r')
    axes[0].axvline(x=real_duration, color='k', linestyle=':', label=f'Fim Real ({real_duration} dias úteis)'); axes[0].set_title('Custo Acumulado'); axes[0].legend(); axes[0].grid(True)
    sim_daily_progress = simulated_log.groupby('day')['hours_worked'].sum()
    sim_cumulative_progress = sim_daily_progress.reindex(day_range, fill_value=0).cumsum()
    real_daily_progress = real_log_merged.groupby('day')['hours_worked'].sum()
    real_cumulative_progress = real_daily_progress.reindex(day_range, fill_value=0).cumsum()
    axes[1].plot(sim_cumulative_progress.index, sim_cumulative_progress.values, label='Progresso Simulado', marker='o', linestyle='--', color='b')
    axes[1].plot(real_cumulative_progress.index, real_cumulative_progress.values, label='Progresso Real', marker='x', linestyle='-', color='r')
    axes[1].axhline(y=total_estimated_effort, color='g', linestyle='-.', label='Esforço Total Estimado'); axes[1].axvline(x=real_duration, color='k', linestyle=':', label=f'Fim Real ({real_duration} dias úteis)'); axes[1].set_title('Progresso Acumulado'); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    plots['project_detailed_comparison'] = convert_fig_to_bytes(fig)
    
    return plots, tables, logs

# --- PÁGINA DE LOGIN ---
def login_page():
    st.markdown("<h2>✨ Transformação Inteligente de Processos</h2>", unsafe_allow_html=True)
    username = st.text_input("Utilizador", placeholder="Vasco", value="Vasco")
    password = st.text_input("Senha", type="password", placeholder="1234", value="1234")
    if st.button("Entrar", use_container_width=True):
        if username == "Vasco" and password == "1234":
            st.session_state.authenticated = True
            st.session_state.user_name = "Vasco"
            st.session_state.show_welcome_message = True
            st.rerun()
        else:
            st.error("Utilizador ou senha inválidos.")


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
                st.markdown(f'<p style="font-size: small; color: #06B6D4;">`{name}.csv` carregado.</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    all_files_uploaded = all(st.session_state.dfs.get(name) is not None for name in file_names)
    
    if all_files_uploaded:
        if st.toggle("Visualizar as primeiras 5 linhas dos ficheiros", value=False):
            for name, df in st.session_state.dfs.items():
                st.markdown(f"**Ficheiro: `{name}.csv`**")
                st.dataframe(df.head())
        
        st.subheader("Execução da Análise")
        st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
        if st.button("🚀 Iniciar Análise Inicial (PM & EDA)", use_container_width=True):
            with st.spinner("A executar a análise... Este processo pode demorar alguns minutos."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                log_from_df = pm4py.convert_to_event_log(pm4py.convert_to_dataframe(event_log))
                plots_post, metrics = run_post_mining_analysis(log_from_df, df_p, df_t, df_r, df_fc)
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics
                plots_eda, tables_eda = run_eda_analysis(st.session_state.dfs)
                st.session_state.plots_eda = plots_eda
                st.session_state.tables_eda = tables_eda

            st.session_state.analysis_run = True
            st.success("✅ Análise concluída! Navegue para o 'Dashboard Geral' ou para a página de 'Reinforcement Learning'.")
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Aguardando o carregamento de todos os ficheiros CSV para poder iniciar a análise.")


# --- PÁGINA DO DASHBOARD ---
def dashboard_page():
    # (O código desta função permanece exatamente o mesmo do ficheiro que forneceu)
    st.title("🏠 Dashboard Geral de Análise de Processos")

    if st.session_state.get('show_welcome_message', False):
        st.success(f"Bem-vindo, {st.session_state.user_name}!")
        st.session_state.show_welcome_message = False

    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Vá à página de 'Configurações' para carregar os dados e iniciar.")
        return
        
    sections = {
        "visao_geral": "1. Visão Geral e Custos",
        "performance": "2. Performance e Prazos",
        "recursos": "3. Recursos e Equipa",
        "gargalos": "4. Handoffs e Espera",
        "fluxo": "5. Fluxo e Conformidade"
    }
    
    nav_cols = st.columns(len(sections))
    for i, (key, name) in enumerate(sections.items()):
        with nav_cols[i]:
            st.markdown(f'<div class="{"active-button" if st.session_state.current_section == key else ""}">', unsafe_allow_html=True)
            if st.button(name, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_section = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-top: 25px; margin-bottom: 25px; border-color: var(--border-color);'>", unsafe_allow_html=True)
    
    plots_pre = st.session_state.plots_pre_mining
    tables_pre = st.session_state.tables_pre_mining
    plots_post = st.session_state.plots_post_mining
    plots_eda = st.session_state.plots_eda
    tables_eda = st.session_state.tables_eda

    if st.session_state.current_section == "visao_geral":
        st.subheader("1. Visão Geral e Custos")
        kpi_data = tables_pre.get('kpi_data', {})
        kpi_cols = st.columns(4)
        kpi_cols[0].metric(label="Total de Projetos", value=kpi_data.get('Total de Projetos'))
        kpi_cols[1].metric(label="Total de Tarefas", value=kpi_data.get('Total de Tarefas'))
        kpi_cols[2].metric(label="Total de Recursos", value=kpi_data.get('Total de Recursos'))
        kpi_cols[3].metric(label="Duração Média", value=f"{kpi_data.get('Duração Média (dias)')} dias")
        
        kpi_delay_data = tables_pre.get('cost_of_delay_kpis', {})
        kpi_cols_2 = st.columns(3)
        kpi_cols_2[0].metric(label="Custo Total em Atraso", value=kpi_delay_data.get('Custo Total Projetos Atrasados', 'N/A'))
        kpi_cols_2[1].metric(label="Atraso Médio (dias)", value=kpi_delay_data.get('Atraso Médio (dias)', 'N/A'))
        kpi_cols_2[2].metric(label="Custo Médio/Dia de Atraso", value=kpi_delay_data.get('Custo Médio/Dia Atraso', 'N/A'))
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            create_card("Matriz de Performance (Custo vs Prazo) (PM)", "🎯", chart_bytes=plots_pre.get('performance_matrix'))
            create_card("Top 5 Projetos Mais Caros", "💰", dataframe=tables_pre.get('outlier_cost'))
            create_card("Séries Temporais de KPIs de Performance", "📈", chart_bytes=plots_post.get('kpi_time_series'))
            create_card("Distribuição do Status dos Projetos", "📊", chart_bytes=plots_eda.get('plot_01'))
            create_card("Custo Médio dos Projetos por Trimestre", "💶", chart_bytes=plots_eda.get('plot_06'))
            create_card("Alocação de Custos por Orçamento e Recurso", "💰", chart_bytes=plots_eda.get('plot_17'))
        with c2:
            create_card("Custo por Tipo de Recurso", "💶", chart_bytes=plots_pre.get('cost_by_resource_type'))
            create_card("Top 5 Projetos Mais Longos", "⏳", dataframe=tables_pre.get('outlier_duration'))
            create_card("Custo Médio por Dia ao Longo do Tempo", "💸", chart_bytes=plots_post.get('cost_per_day_time_series'))
            create_card("Custo Real vs. Orçamento por Projeto", "💳", chart_bytes=plots_eda.get('plot_04'))
            create_card("Distribuição do Custo por Dia (Eficiência)", "💡", chart_bytes=plots_eda.get('plot_16'))
            create_card("Evolução do Volume e Tamanho dos Projetos", "📈", chart_bytes=plots_eda.get('plot_31'))

    elif st.session_state.current_section == "performance":
        st.subheader("2. Performance e Prazos")
        c1, c2 = st.columns(2)
        with c1:
            create_card("Relação Lead Time vs Throughput", "🔗", chart_bytes=plots_pre.get('lead_time_vs_throughput'))
            create_card("Distribuição do Lead Time", "⏱️", chart_bytes=plots_pre.get('lead_time_hist'))
            create_card("Distribuição da Duração dos Projetos (PM)", "📊", chart_bytes=plots_pre.get('case_durations_boxplot'))
            create_card("Gráfico Acumulado de Throughput", "📈", chart_bytes=plots_post.get('cumulative_throughput_plot'))
            create_card("Performance de Prazos por Trimestre", "📉", chart_bytes=plots_eda.get('plot_05'))
        with c2:
            create_card("Duração Média por Fase do Processo", "🗂️", chart_bytes=plots_pre.get('cycle_time_breakdown'))
            create_card("Distribuição do Throughput (horas)", "🚀", chart_bytes=plots_pre.get('throughput_hist'))
            create_card("Boxplot do Throughput (horas)", "📦", chart_bytes=plots_pre.get('throughput_boxplot'))
            create_card("Atividades por Dia da Semana", "🗓️", chart_bytes=plots_post.get('temporal_heatmap_fixed'))
            create_card("Evolução da Performance (Prazo e Custo)", "📈", chart_bytes=plots_eda.get('plot_30'))
        
        c3, c4 = st.columns(2)
        with c3:
             create_card("Diferença entre Data Real e Planeada", "🗓️", chart_bytes=plots_eda.get('plot_03'))
        with c4:
            create_card("Estatísticas de Performance", "📈", dataframe=tables_pre.get('perf_stats'))
            
        create_card("Linha do Tempo de Todos os Projetos (Gantt Chart)", "📊", chart_bytes=plots_post.get('gantt_chart_all_projects'))

    elif st.session_state.current_section == "recursos":
        st.subheader("3. Recursos e Equipa")
        c1, c2 = st.columns(2)
        with c1:
            create_card("Distribuição de Recursos por Tipo", "🔧", chart_bytes=plots_eda.get('plot_12'))
            create_card("Recursos por Média de Tarefas/Projeto", "🧑‍💻", chart_bytes=plots_pre.get('resource_avg_events'))
            create_card("Eficiência Semanal (Horas Trabalhadas)", "🗓️", chart_bytes=plots_pre.get('weekly_efficiency'))
            create_card("Impacto do Tamanho da Equipa no Atraso (PM)", "👨‍👩‍👧‍👦", chart_bytes=plots_pre.get('delay_by_teamsize'))
            create_card("Benchmark de Throughput por Equipa", "🏆", chart_bytes=plots_pre.get('throughput_benchmark_by_teamsize'))
            create_card("Atraso por Nível de Competência", "🎓", chart_bytes=plots_eda.get('plot_23'))
        with c2:
            create_card("Top 10 Recursos por Horas Trabalhadas (PM)", "💪", chart_bytes=plots_pre.get('resource_workload'))
            create_card("Top 10 Handoffs entre Recursos", "🔄", chart_bytes=plots_pre.get('resource_handoffs'))
            create_card("Métricas de Eficiência Individual por Recurso", "🎯", chart_bytes=plots_post.get('resource_efficiency_plot'))
            create_card("Duração Mediana por Tamanho da Equipa", "⏱️", chart_bytes=plots_pre.get('median_duration_by_teamsize'))
            create_card("Nº Médio de Recursos por Projeto a Cada Trimestre", "👥", chart_bytes=plots_eda.get('plot_07'))
            create_card("Atraso Médio por Recurso", "⏳", chart_bytes=plots_eda.get('plot_14'))
        
        col_skill, col_bipartite = st.columns(2)
        with col_skill:
            if 'skill_vs_performance_adv' in plots_post:
                create_card("Relação entre Skill e Performance", "🎓", chart_bytes=plots_post.get('skill_vs_performance_adv'))
        with col_bipartite:
            if 'resource_network_bipartite' in plots_post:
                create_card("Rede de Recursos por Função", "🔗", chart_bytes=plots_post.get('resource_network_bipartite'))

        create_card("Rede Social de Recursos (Handovers)", "🌐", chart_bytes=plots_post.get('resource_network_adv'))
        
        create_card("Heatmap de Esforço (Recurso vs Atividade)", "🗺️", chart_bytes=plots_pre.get('resource_activity_matrix'))

    elif st.session_state.current_section == "gargalos":
        st.subheader("4. Handoffs e Espera")
        create_card("Heatmap de Performance no Processo (Gargalos)", "🔥", chart_bytes=plots_post.get('performance_heatmap'))
        
        c1, c2 = st.columns(2)
        with c1:
            create_card("Atividades Mais Frequentes", "⚡", chart_bytes=plots_pre.get('top_activities_plot'))
            create_card("Gargalos: Tempo de Serviço vs. Espera", "🚦", chart_bytes=plots_pre.get('service_vs_wait_stacked'))
            create_card("Top 10 Handoffs por Custo de Espera", "💸", chart_bytes=plots_pre.get('top_handoffs_cost'))
            create_card("Top Recursos por Tempo de Espera Gerado", "🛑", chart_bytes=plots_pre.get('bottleneck_by_resource'))
            create_card("Custo Real vs. Atraso", "💰", chart_bytes=plots_eda.get('plot_18'))
            create_card("Nº de Recursos vs. Custo Total", "👥", chart_bytes=plots_eda.get('plot_20'))
            create_card("Matriz de Correlação", "🔗", chart_bytes=plots_eda.get('plot_29'))
        with c2:
            create_card("Tempo Médio de Execução por Atividade", "🛠️", chart_bytes=plots_pre.get('activity_service_times'))
            create_card("Espera vs. Execução (Dispersão)", "🔍", chart_bytes=plots_pre.get('wait_vs_service_scatter'))
            create_card("Evolução do Tempo Médio de Espera", "📈", chart_bytes=plots_pre.get('wait_time_evolution'))
            create_card("Top 10 Handoffs por Tempo de Espera", "⏳", chart_bytes=plots_pre.get('top_handoffs'))
            create_card("Rate Horário Médio vs. Atraso", "⏰", chart_bytes=plots_eda.get('plot_19'))
            create_card("Atraso por Faixa de Orçamento", "📊", chart_bytes=plots_eda.get('plot_22'))

        c3, c4 = st.columns(2)
        with c3:
            if 'milestone_time_analysis_plot' in plots_post:
                create_card("Análise de Tempo entre Marcos do Processo", "🚩", chart_bytes=plots_post.get('milestone_time_analysis_plot'))
        with c4:
             create_card("Tempo Médio de Espera por Atividade", "⏱️", chart_bytes=plots_post.get('avg_waiting_time_by_activity_plot'))
        
        create_card("Matriz de Tempo de Espera entre Atividades (horas)", "⏳", chart_bytes=plots_post.get('waiting_time_matrix_plot'))

    elif st.session_state.current_section == "fluxo":
        st.subheader("5. Fluxo e Conformidade")

        create_card("Modelo - Inductive Miner", "🧭", chart_bytes=plots_post.get('model_inductive_petrinet'))
        create_card("Modelo - Heuristics Miner", "🛠️", chart_bytes=plots_post.get('model_heuristic_petrinet'))

        c1, c2 = st.columns(2)
        with c1:
            create_card("Métricas (Inductive Miner)", "📊", chart_bytes=plots_post.get('metrics_inductive'))
        with c2:
            create_card("Métricas (Heuristics Miner)", "📈", chart_bytes=plots_post.get('metrics_heuristic'))
        
        create_card("Sequência de Atividades das 10 Variantes Mais Comuns", "🎶", chart_bytes=plots_post.get('custom_variants_sequence_plot'))
        
        c3, c4 = st.columns(2)
        with c3:
            create_card("Duração Média das Variantes Mais Comuns", "⏳", chart_bytes=plots_post.get('variant_duration_plot'))
            create_card("Frequência das 10 Principais Variantes", "🎭", dataframe=tables_pre.get('variants_table'))
            create_card("Distribuição de Tarefas por Tipo", "📋", chart_bytes=plots_eda.get('plot_08'))
            create_card("Distribuição da Duração das Tarefas", "⏳", chart_bytes=plots_eda.get('plot_10'))
            create_card("Centralidade dos Tipos de Tarefa", "🎯", chart_bytes=plots_eda.get('plot_25'))
            create_card("Relação entre Dependências e Desvio de Custo", "💸", chart_bytes=plots_eda.get('plot_28'))
        with c4:
            create_card("Score de Conformidade ao Longo do Tempo", "📉", chart_bytes=plots_post.get('conformance_over_time_plot'))
            create_card("Principais Loops de Rework", "🔁", dataframe=tables_pre.get('rework_loops_table'))
            create_card("Distribuição de Tarefas por Prioridade", "🥇", chart_bytes=plots_eda.get('plot_09'))
            create_card("Top 10 Tarefas Específicas Mais Demoradas", "🕒", chart_bytes=plots_eda.get('plot_11'))
            create_card("Distribuição da Complexidade dos Projetos", "🕸️", chart_bytes=plots_eda.get('plot_24'))
            create_card("Relação entre Complexidade e Atraso", "🔗", chart_bytes=plots_eda.get('plot_27'))

        c5, c6 = st.columns(2)
        with c5:
             create_card("Top 10 Variantes de Processo por Frequência", "📊", chart_bytes=plots_pre.get('variants_frequency'))
        with c6:
            create_card("Grafo de Dependências: Projeto 25", "📈", chart_bytes=plots_eda.get('plot_26'))

# --- NOVA PÁGINA (REINFORCEMENT LEARNING) ---
def rl_page():
    st.title("🤖 Simulação com Reinforcement Learning")

    if not st.session_state.analysis_run:
        st.warning("É necessário executar a análise inicial primeiro. Vá à página de 'Configurações' para carregar os dados.")
        return

    st.info("Esta secção permite treinar um agente de IA para otimizar a gestão de projetos, com base nos dados históricos. Pode ajustar os parâmetros para testar diferentes cenários.")

    # --- Parâmetros de Entrada ---
    with st.expander("⚙️ Parâmetros da Simulação", expanded=st.session_state.rl_params_expanded):
        st.subheader("Parâmetros Gerais")
        project_ids = st.session_state.dfs['projects']['project_id'].unique()
        
        c1, c2 = st.columns(2)
        with c1:
            project_id_to_simulate = st.selectbox(
                "Selecione o ID do Projeto para Simulação Detalhada",
                options=project_ids,
                index=list(project_ids).index("25") if "25" in project_ids else 0
            )
        with c2:
            num_episodes = st.number_input("Número de Episódios de Treino", min_value=100, max_value=10000, value=1000, step=100)

        st.markdown("<h6>Parâmetros de Recompensa e Penalização do Agente</h6>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            cost_impact_factor = st.number_input("Fator de Impacto do Custo", value=1.0)
            daily_time_penalty = st.number_input("Penalização Diária por Tempo", value=20.0)
            idle_penalty = st.number_input("Penalização por Inatividade", value=10.0)
        with rc2:
            per_day_early_bonus = st.number_input("Bónus por Dia de Adiantamento", value=500.0)
            completion_base = st.number_input("Recompensa Base por Conclusão", value=5000.0)
            per_day_late_penalty = st.number_input("Penalização por Dia de Atraso", value=1500.0)
        with rc3:
            priority_task_bonus_factor = st.number_input("Bónus por Tarefa Prioritária", value=500)
            pending_task_penalty_factor = st.number_input("Penalização por Tarefa Pendente", value=20)
        
        reward_config = {
            'cost_impact_factor': cost_impact_factor, 'daily_time_penalty': daily_time_penalty, 'idle_penalty': idle_penalty,
            'per_day_early_bonus': per_day_early_bonus, 'completion_base': completion_base, 'per_day_late_penalty': per_day_late_penalty,
            'priority_task_bonus_factor': priority_task_bonus_factor, 'pending_task_penalty_factor': pending_task_penalty_factor
        }

    status_container = st.empty()

    if st.button("▶️ Iniciar Treino e Simulação do Agente", use_container_width=True):
        st.session_state.rl_params_expanded = False
        
        progress_bar = st.progress(0)
        status_text = status_container.info("A iniciar o treino do agente de RL...")

        plots_rl, tables_rl, logs_rl = run_rl_analysis(
            st.session_state.dfs, 
            project_id_to_simulate, 
            num_episodes, 
            reward_config,
            progress_bar,
            status_text
        )
        st.session_state.plots_rl = plots_rl
        st.session_state.tables_rl = tables_rl
        st.session_state.logs_rl = logs_rl
        st.session_state.rl_analysis_run = True
        st.rerun()

    
    if st.session_state.rl_analysis_run:
        st.markdown("---")
        st.subheader("Resultados da Simulação")
        
        plots_rl = st.session_state.plots_rl
        tables_rl = st.session_state.tables_rl
        
        st.markdown(f"<h4>Análise Detalhada da Simulação (Projeto {st.session_state.get('project_id_simulated')})</h4>", unsafe_allow_html=True)
        
        summary_df = tables_rl.get('project_summary')
        if summary_df is not None:
            real_duration = summary_df.loc[summary_df['Métrica'] == 'Duração (dias úteis)', 'Real (Histórico)'].iloc[0]
            sim_duration = summary_df.loc[summary_df['Métrica'] == 'Duração (dias úteis)', 'Simulado (RL)'].iloc[0]
            real_cost = summary_df.loc[summary_df['Métrica'] == 'Custo (€)', 'Real (Histórico)'].iloc[0]
            sim_cost = summary_df.loc[summary_df['Métrica'] == 'Custo (€)', 'Simulado (RL)'].iloc[0]
            
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(label="Duração (dias úteis)", value=f"{sim_duration:.0f}", delta=f"{sim_duration - real_duration:.0f} vs Real")
            with metric_cols[1]:
                st.metric(label="Custo (€)", value=f"{sim_cost:,.2f}", delta=f"{sim_cost - real_cost:,.2f} vs Real")

        st.markdown("<h4>Desempenho Global</h4>", unsafe_allow_html=True)
        res1, res2 = st.columns(2)
        with res1:
            create_card("Performance Global (Conjunto de Teste)", "📊", dataframe=tables_rl.get('global_performance_test'))
        with res2:
            create_card("Performance Global (Todos os Projetos)", "📈", dataframe=tables_rl.get('global_performance_all'))

        st.markdown("<h4>Métricas de Treinamento do Agente</h4>", unsafe_allow_html=True)
        create_card("Evolução do Treino", "🤖", chart_bytes=plots_rl.get('training_metrics'))
        
        st.markdown("<h4>Comparação de Desempenho (Simulado vs. Real)</h4>", unsafe_allow_html=True)
        create_card("Comparação do Desempenho (Conjunto de Teste)", "🎯", chart_bytes=plots_rl.get('evaluation_comparison_test'))
        create_card("Comparação do Desempenho (Todos os Projetos)", "🌍", chart_bytes=plots_rl.get('evaluation_comparison_all'))
        
        create_card(f"Comparação Detalhada (Projeto {st.session_state.get('project_id_simulated')})", "🔍", chart_bytes=plots_rl.get('project_detailed_comparison'))

# --- CONTROLO PRINCIPAL DA APLICAÇÃO ---
def main():
    if not st.session_state.authenticated:
        st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] > .main {
                display: flex; flex-direction: column; justify-content: center; align-items: center;
            }
        </style>
        """, unsafe_allow_html=True)
        login_page()
    else:
        with st.sidebar:
            st.markdown(f"### 👤 {st.session_state.get('user_name', 'Admin')}")
            st.markdown("---")
            if st.button("🏠 Dashboard Geral", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.rerun()
            if st.button("🤖 Reinforcement Learning", use_container_width=True):
                st.session_state.current_page = "RL"
                st.rerun()
            if st.button("⚙️ Configurações", use_container_width=True):
                st.session_state.current_page = "Settings"
                st.rerun()
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🚪 Sair", use_container_width=True):
                st.session_state.authenticated = False
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated']: del st.session_state[key]
                st.rerun()
                
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "RL":
            rl_page()
        elif st.session_state.current_page == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
