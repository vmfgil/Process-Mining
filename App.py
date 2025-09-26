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
import os
import time

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
        /* Adicionar altura máxima e scroll interno para o corpo do cartão que contém o dataframe */
    .dataframe-card-body {
        max-height: 300px; /* Defina a altura máxima desejada para a caixa da tabela */
        overflow-y: auto; /* Adicionar scroll vertical */
        overflow-x: auto; /* Adicionar scroll horizontal (se a tabela for larga) */
        padding: 0; /* Remover padding padrão para evitar barra de scroll dupla */
    }
    
    /* --- BOTÕES DE UPLOAD --- */
    section[data-testid="stFileUploader"] button,
    div[data-baseweb="file-uploader"] button {
        background-color: var(--accent-color) !important; /* Ciano */
        color: var(--text-color-light-bg) !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    
    /* --- BOTÃO DE ANÁLISE --- */
    .iniciar-analise-button .stButton>button {
        background-color: var(--secondary-color) !important; /* Amarelo */
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
        color: var(--text-color-dark-bg) !important; /* Label da métrica */
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--metric-value-color) !important; /* Valor da métrica (Âmbar) */
        font-weight: 700;
    }
    
    /* Alertas */
    [data-testid="stAlert"] {
        background-color: #1E293B !important; /* Fundo ligeiramente mais claro */
        border: 1px solid var(--secondary-color) !important; /* Borda de destaque (Amarelo) */
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] * { color: var(--text-color-dark-bg) !important; }
    
/* Melhorar legibilidade de dataframes NATIVOS do Streamlit */
    .stDataFrame {
        color: var(--text-color-dark-bg) !important;
        background-color: var(--card-background-color) !important;
    }

    /* Adicionar estilos para o DataFrame HTML gerado pela correção */
    .pandas-df-card {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-color-dark-bg);
        font-size: 0.85rem;
    }
    .pandas-df-card th {
        background-color: var(--sidebar-background); /* Fundo da sidebar */
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
        background-color: #2F394B; /* Linhas pares ligeiramente mais escuras */
    }
    
    .stTextInput>div>div>input, .stTextInput>div>div>textarea {
        background-color: var(--sidebar-background) !important;
        color: var(--text-color-dark-bg) !important;
        border: 1px solid var(--border-color) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- FUNÇÕES AUXILIARES ---
def convert_fig_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    # Cores do gráfico para combinar com o fundo escuro
    fig.patch.set_facecolor('#1E293B') # Cor de fundo dos cartões
    for ax in fig.get_axes():
        ax.set_facecolor('#1E293B') # Fundo do eixo
        ax.tick_params(colors='#E5E7EB', which='both') # Cor dos ticks
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
    # Os gráficos Graphviz (Petri Nets, DFG) são mais difíceis de estilizar diretamente,
    # mas o PM4PY tenta renderizá-los com cores default.
    return io.BytesIO(gviz.pipe(format=format))

def create_card(title, icon, chart_bytes=None, dataframe=None):
    # Esta função será substituída por display_chart_or_table na nova lógica do dashboard
    if chart_bytes:
        if isinstance(chart_bytes, io.BytesIO):
             b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
        else: # Assumir que é bytes puros
             b64_image = base64.b64encode(chart_bytes).decode()
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif dataframe is not None:
        # CONVERTER O DATAFRAME PARA HTML E APLICAR UMA CLASSE PARA ESTILOS
        df_html = dataframe.to_html(classes=['pandas-df-card'], index=False)
        
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body dataframe-card-body">
                {df_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_chart_or_table(title, icon, chart_data):
    # Função auxiliar para o novo dashboard (similar a create_card, mas ajustada para o st.expander)
    st.markdown(f"#### {icon} {title}")
    
    if isinstance(chart_data, pd.DataFrame):
        # Exibe DataFrames usando a lógica original de create_card para garantir o estilo
        df_html = chart_data.to_html(classes=['pandas-df-card'], index=False)
        st.markdown(f"""
        <div class="card-body dataframe-card-body">
            {df_html}
        </div>
        """, unsafe_allow_html=True)
    elif isinstance(chart_data, dict):
        # Lógica especial para KPIs (mantida)
        kpis = chart_data
        cols = st.columns(len(kpis))
        for i, (k, v) in enumerate(kpis.items()):
            cols[i].metric(label=k, value=v, delta=None) # Valor formatado já vem do run_pre_mining_analysis
    elif chart_data:
        # Assumindo que são dados binários (PNG, SVG) ou BytesIO
        if isinstance(chart_data, io.BytesIO):
             chart_bytes = chart_data.getvalue()
        else: # Assumir que é bytes puros
             chart_bytes = chart_data
        
        b64_img = base64.b64encode(chart_bytes).decode('utf-8')
        st.markdown(f'<div class="card-body"><img src="data:image/png;base64,{b64_img}" alt="{title}" style="max-width: 100%; height: auto;"/></div>', unsafe_allow_html=True)
    else:
        st.info("Conteúdo da análise não disponível.")
        
        
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


# --- FUNÇÕES DE ANÁLISE (DO SCRIPT ORIGINAL) ---
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
    
    # Análise 1: Sumário Estatístico do Log (log_summary_table)
    tables['log_summary_table'] = pd.DataFrame({
        'Métrica': ['Total de Casos', 'Total de Eventos', 'Período (dias)'],
        'Valor': [
            len(df_projects),
            len(log_df_final),
            (log_df_final['time:timestamp'].max() - log_df_final['time:timestamp'].min()).days
        ]
    })
    
    # Análise 2: KPIs de Portfólio (kpi_data)
    tables['kpi_data'] = {
        'Total de Projetos': str(len(df_projects)),
        'Desvio Médio Duração (dias)': f"{df_projects['days_diff'].mean():.1f}",
        'Desvio Médio Custo (€)': f"€{df_projects['cost_diff'].mean():,.2f}",
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.1f}"
    }
    
    # Análise 3, 4: Projetos Outlier por Duração (outlier_duration) e Custo (outlier_cost)
    tables['outlier_duration'] = df_projects[['project_name', 'actual_duration_days', 'days_diff']].sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects[['project_name', 'total_actual_cost', 'cost_diff']].sort_values('total_actual_cost', ascending=False).head(5)
    
    # Análise 5, 6: Métricas de Atraso e Custo de Desvio (cost_of_delay_kpis)
    delayed_projects = df_projects[df_projects['days_diff'] > 0]
    tables['cost_of_delay_kpis'] = {
        'Custo Total Projetos Atrasados': f"€{delayed_projects['total_actual_cost'].sum():,.2f}",
        'Atraso Médio (dias)': f"{delayed_projects['days_diff'].mean():.1f}",
        'Custo Médio/Dia Atraso': f"€{(delayed_projects.get('total_actual_cost', 0) / delayed_projects['days_diff']).mean():,.2f}"
    }

    # Análise 7: Custo de Desvio por Categoria (cost_of_delay_breakdown)
    tables['cost_of_delay_breakdown'] = df_projects.groupby('project_type')['cost_diff'].sum().reset_index().sort_values('cost_diff', ascending=False)
    
    # Análise 8: Custo por Tipo de Recurso (cost_by_resource_type)
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', ax=ax, hue='resource_type', legend=False, palette='magma'); ax.set_title("Custo por Tipo de Recurso"); ax.set_xlabel("Custo Total (€)"); ax.set_ylabel("Tipo de Recurso")
    plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)
    
    # Análise 9: Matriz de Performance (performance_matrix)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.7, ax=ax, palette='viridis'); ax.axhline(0, color='#FBBF24', ls='--'); ax.axvline(0, color='#FBBF24', ls='--'); ax.set_title("Matriz de Performance (Desvio Duração vs Custo)"); ax.set_xlabel("Desvio Duração (dias)"); ax.set_ylabel("Desvio Custo (€)")
    plots['performance_matrix'] = convert_fig_to_bytes(fig)
    
    # Cálculos de Performance (para várias análises)
    lead_times = log_df_final.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"]).reset_index()
    lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
    def compute_avg_throughput(group):
        group = group.sort_values("time:timestamp"); deltas = group["time:timestamp"].diff().dropna()
        return deltas.mean().total_seconds() if not deltas.empty else 0
    throughput_per_case = log_df_final.groupby("case:concept:name").apply(compute_avg_throughput).reset_index(name="avg_throughput_seconds")
    throughput_per_case["avg_throughput_hours"] = throughput_per_case["avg_throughput_seconds"] / 3600
    perf_df = pd.merge(lead_times, throughput_per_case, on="case:concept:name")
    
    # Análise 10: Estatísticas Descritivas (Tempo) (perf_stats)
    tables['perf_stats'] = perf_df[["lead_time_days", "avg_throughput_hours"]].describe()

    # Análise 11: Distribuição Temporal de Eventos/Casos (event_over_time_plot)
    monthly_events = log_df_final.set_index('time:timestamp').resample('M')['case:concept:name'].count()
    fig, ax = plt.subplots(figsize=(10, 5)); monthly_events.plot(kind='line', marker='o', color='#06B6D4', ax=ax); ax.set_title("Distribuição Temporal de Eventos"); ax.set_xlabel("Mês"); ax.set_ylabel("Contagem de Eventos")
    plots['event_over_time_plot'] = convert_fig_to_bytes(fig)
    
    # Análise 12: Evolução do Lead Time/Throughput (throughput_vs_lead_time_over_time)
    df_perf_full_temp = perf_df.merge(log_df_final.groupby('case:concept:name')['time:timestamp'].min().reset_index(name='start_date'), on='case:concept:name')
    monthly_perf = df_perf_full_temp.set_index('start_date').resample('M')[['lead_time_days', 'avg_throughput_hours']].mean()
    fig, ax = plt.subplots(figsize=(10, 5)); monthly_perf['lead_time_days'].plot(kind='line', marker='o', color='#2563EB', ax=ax, label='Lead Time Médio (dias)'); ax2 = ax.twinx(); monthly_perf['avg_throughput_hours'].plot(kind='line', marker='x', color='#FBBF24', ax=ax2, label='Throughput Médio (horas)'); ax.set_title("Evolução Lead Time vs Throughput (Mensal)")
    plots['throughput_vs_lead_time_over_time'] = convert_fig_to_bytes(fig)
    
    # Análise 13: Distribuição da Duração dos Projetos (case_durations_boxplot)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=df_projects['actual_duration_days'], ax=ax, color="#2563EB"); sns.stripplot(x=df_projects['actual_duration_days'], color="#FBBF24", size=4, jitter=True, alpha=0.7, ax=ax); ax.set_title("Distribuição da Duração dos Projetos (dias)")
    plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)
    
    # Análise 14: Distribuição do Lead Time (lead_time_hist)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["lead_time_days"], bins=20, kde=True, ax=ax, color="#2563EB"); ax.set_title("Distribuição do Lead Time (dias)")
    plots['lead_time_hist'] = convert_fig_to_bytes(fig)
    
    # Análise 15: Distribuição do Throughput (throughput_hist)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["avg_throughput_hours"], bins=20, kde=True, color='#06B6D4', ax=ax); ax.set_title("Distribuição do Throughput (horas)")
    plots['throughput_hist'] = convert_fig_to_bytes(fig)
    
    # Análise 16: Relação Lead Time vs Throughput (lead_time_vs_throughput)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação Lead Time vs Throughput"); ax.set_xlabel("Throughput Médio (horas)"); ax.set_ylabel("Lead Time (dias)")
    plots['lead_time_vs_throughput'] = convert_fig_to_bytes(fig)
    
    # Análise 17: Duração Média por Fase do Processo (cycle_time_breakdown)
    def get_phase(task_type): # Função auxiliar do seu código original
        if task_type in ['Desenvolvimento', 'Correção', 'Revisão', 'Design']: return 'Desenvolvimento & Design'
        if task_type == 'Teste': return 'Teste (QA)'
        if task_type in ['Deploy', 'DBA']: return 'Operações & Deploy'
        return 'Outros'
    df_tasks['phase'] = df_tasks['task_type'].apply(get_phase)
    phase_times = df_tasks.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index()
    phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days
    avg_cycle_time_by_phase = phase_times.groupby('phase')['cycle_time_days'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=avg_cycle_time_by_phase, x='phase', y='cycle_time_days', ax=ax, hue='phase', legend=False, palette='viridis'); ax.set_title("Duração Média por Fase do Processo"); ax.set_xlabel("Fase"); ax.set_ylabel("Duração Média (dias)"); plt.xticks(rotation=45)
    plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)
    
    # Análise 18: Tempo de Início e Fim dos Casos (case_start_end_scatter)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.scatter(df_projects['start_date'], df_projects['end_date'], s=20, color='#06B6D4'); ax.plot(df_projects['start_date'], df_projects['start_date'] + pd.Timedelta(days=df_projects['actual_duration_days'].median()), linestyle='--', color='#FBBF24', label='Duração Mediana'); ax.set_title("Tempo de Início vs. Fim dos Projetos"); ax.set_xlabel("Data de Início"); ax.set_ylabel("Data de Fim"); plt.xticks(rotation=45)
    plots['case_start_end_scatter'] = convert_fig_to_bytes(fig)

    # Análise 19: Top 10 Recursos por Horas Trabalhadas (resource_workload)
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, hue='resource_name', legend=False, palette='plasma'); ax.set_title("Top 10 Recursos por Horas Trabalhadas"); ax.set_xlabel("Horas Trabalhadas"); ax.set_ylabel("Recurso")
    plots['resource_workload'] = convert_fig_to_bytes(fig)
    
    # Análise 20: Recursos por Média de Tarefas/Projeto (resource_avg_events)
    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()
    resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), ax=ax, hue='resource_name', legend=False, palette='coolwarm'); ax.set_title("Recursos por Média de Tarefas por Projeto"); ax.set_xlabel("Média de Tarefas/Projeto"); ax.set_ylabel("Recurso")
    plots['resource_avg_events'] = convert_fig_to_bytes(fig)
    
    # Análise 21: Eficiência Semanal (weekly_efficiency)
    df_alloc_costs['day_of_week'] = df_alloc_costs['allocation_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, hue='day_of_week', legend=False, palette='viridis'); ax.set_title("Eficiência Semanal (Horas Trabalhadas)"); ax.set_xlabel("Dia da Semana"); ax.set_ylabel("Horas Trabalhadas"); plt.xticks(rotation=45)
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    
    # Análise 22: Heatmap de Esforço (Recurso vs Atividade) (resource_activity_matrix)
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_activity_matrix_pivot, cmap='Blues', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8}, linewidths=.5, linecolor='#374151'); ax.set_title("Heatmap de Esforço (Recurso vs Atividade)")
    plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)
    
    # Análise 23: Top 10 Handoffs entre Recursos (resource_handoffs)
    handoff_counts = Counter((trace[i]['org:resource'], trace[i+1]['org:resource']) for trace in event_log_pm4py for i in range(len(trace) - 1) if 'org:resource' in trace[i] and 'org:resource' in trace[i+1] and trace[i]['org:resource'] != trace[i+1]['org:resource'])
    df_resource_handoffs = pd.DataFrame(handoff_counts.most_common(10), columns=['Handoff', 'Contagem'])
    df_resource_handoffs['Handoff'] = df_resource_handoffs['Handoff'].apply(lambda x: f"{x[0]} -> {x[1]}")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs, ax=ax, hue='Handoff', legend=False, palette='rocket'); ax.set_title("Top 10 Handoffs entre Recursos")
    plots['resource_handoffs'] = convert_fig_to_bytes(fig)
    
    # Análise 24: Análise de Redes Sociais (resource_activity_social_network) - Usando uma métrica de rede simples
    dfg_res = dfg_discovery.apply(pm4py.filter_log(event_log_pm4py, {'case_id': log_df_final['case:concept:name'].unique().tolist()}), variant=dfg_discovery.Variants.FREQUENCY, parameters={'resource_key': 'org:resource'})
    g_res = nx.DiGraph(); [g_res.add_edge(n[0][0], n[0][1], weight=n[1]) for n in dfg_res.items()]
    pos = nx.spring_layout(g_res, k=0.15, iterations=20); edge_weights = [d['weight'] for (u, v, d) in g_res.edges(data=True)]
    fig, ax = plt.subplots(figsize=(10, 8)); nx.draw_networkx_nodes(g_res, pos, node_size=3000, node_color='#2563EB', alpha=0.9); nx.draw_networkx_edges(g_res, pos, edgelist=g_res.edges(), width=[w / max(edge_weights) * 5 for w in edge_weights], edge_color='#FBBF24', arrowsize=20, alpha=0.7); nx.draw_networkx_labels(g_res, pos, font_size=10, font_family='sans-serif', font_color='#0A112A')
    ax.set_title("Análise de Redes Sociais (Handoffs de Recursos)")
    plots['resource_activity_social_network'] = convert_fig_to_bytes(fig)
    
    # Análise 25: Heatmap de Mistura de Casos/Recurso (resource_case_mix_heatmap)
    resource_case_mix = df_full_context.groupby(['resource_name', 'project_id'])['hours_worked'].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_case_mix, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Horas Trabalhadas'}); ax.set_title("Heatmap de Mistura de Casos por Recurso")
    plots['resource_case_mix_heatmap'] = convert_fig_to_bytes(fig)
    
    # Análise 26: Duração Mediana por Tamanho da Equipa (median_duration_by_teamsize)
    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()
    bins = np.linspace(min_res, max_res, 5, dtype=int) if max_res > min_res else [min_res, max_res + 1]
    df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='crest'); ax.set_title("Duração Mediana por Tamanho da Equipa"); ax.set_xlabel("Tamanho da Equipa (Binned)"); ax.set_ylabel("Duração Mediana (dias)")
    plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)
    
    # Análise 27: Impacto do Tamanho da Equipa no Atraso (delay_by_teamsize)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='flare'); ax.set_title("Impacto do Tamanho da Equipa no Atraso"); ax.set_xlabel("Tamanho da Equipa (Binned)"); ax.set_ylabel("Desvio de Prazo (dias)")
    plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)
    
    # Análise 28: Benchmark de Throughput por Equipa (throughput_benchmark_by_teamsize)
    df_perf_full = perf_df.merge(df_projects, left_on='case:concept:name', right_on='project_id')
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_perf_full, x='team_size_bin_dynamic', y='avg_throughput_hours', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='plasma'); ax.set_title("Benchmark de Throughput por Tamanho da Equipa"); ax.set_xlabel("Tamanho da Equipa (Binned)"); ax.set_ylabel("Throughput Médio (horas)")
    plots['throughput_benchmark_by_teamsize'] = convert_fig_to_bytes(fig)
    
    # Análise 29: Fluxo de Gargalos (DFG) (bottleneck_flow_dfg)
    # DFG de Frequência
    dfg_freq = dfg_discovery.apply(event_log_pm4py)
    # DFG de Performance (tempo)
    dfg_perf = dfg_discovery.apply(event_log_pm4py, variant=dfg_discovery.Variants.PERFORMANCE)
    # Geração da visualização DFG de Performance (bottlenecks)
    parameters = {dfg_visualizer.Variants.PERFORMANCE.value.Parameters.FORMAT: "png", dfg_visualizer.Variants.PERFORMANCE.value.Parameters.ACTIVITY_KEY: "concept:name"}
    gviz_dfg_perf = dfg_visualizer.apply(dfg_perf, log=event_log_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE, parameters=parameters)
    plots['bottleneck_flow_dfg'] = convert_gviz_to_bytes(gviz_dfg_perf, format='png')
    
    # Análise 30: Gargalos: Tempo de Serviço vs. Espera (service_vs_wait_stacked)
    df_tasks_analysis = df_tasks.copy(); df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True); df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    df_tasks_analysis['task_type'] = df_tasks_analysis['task_name'].str.extract(r'\[(.*?)\]') # Se a task_name contiver o tipo
    bottleneck_by_activity = df_tasks_analysis.groupby('task_name')[['service_time_days', 'waiting_time_days']].mean()
    fig, ax = plt.subplots(figsize=(10, 6)); bottleneck_by_activity.head(10).plot(kind='bar', stacked=True, color=['#2563EB', '#FBBF24'], ax=ax); ax.set_title("Gargalos: Tempo de Serviço vs. Espera (Top 10 Atividades)"); ax.set_ylabel("Tempo Médio (dias)"); plt.xticks(rotation=45)
    plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)
    
    # Análise 31: Tempo Médio de Execução por Atividade (activity_service_times)
    service_times = df_tasks_analysis.groupby('task_name')['service_time_days'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, hue='task_name', legend=False, palette='coolwarm'); ax.set_title("Tempo Médio de Execução por Atividade"); ax.set_xlabel("Tempo de Serviço (dias)")
    plots['activity_service_times'] = convert_fig_to_bytes(fig)
    
    # Análise 32: Evolução do Tempo Médio de Espera (Mensal) (wait_time_evolution)
    df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id', 'completion_month']], on='project_id')
    monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax, color='#06B6D4'); plt.xticks(rotation=45); ax.set_title("Evolução do Tempo Médio de Espera"); ax.set_xlabel("Mês"); ax.set_ylabel("Tempo de Espera (dias)")
    plots['wait_time_evolution'] = convert_fig_to_bytes(fig)
    
    # Análise 33: Espera vs. Execução (Dispersão) (wait_vs_service_scatter)
    wait_service_df = df_tasks_analysis.groupby('task_name')[['service_time_days', 'waiting_time_days']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=wait_service_df, x='service_time_days', y='waiting_time_days', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Tempo de Espera vs. Tempo de Execução (Atividade)"); ax.set_xlabel("Tempo de Serviço Médio (dias)"); ax.set_ylabel("Tempo de Espera Médio (dias)")
    plots['wait_vs_service_scatter'] = convert_fig_to_bytes(fig)

    # Análise 34, 35: Top 10 Handoffs por Tempo de Espera (top_handoffs) e Custo de Espera (top_handoffs_cost)
    df_handoff = log_df_final.sort_values(['case:concept:name', 'time:timestamp'])
    df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', ax=ax, hue='transition', legend=False, palette='viridis'); ax.set_title("Top 10 Handoffs por Tempo de Espera"); ax.set_xlabel("Tempo de Espera Médio (dias)")
    plots['top_handoffs'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, hue='transition', legend=False, palette='magma'); ax.set_title("Top 10 Handoffs por Custo de Espera"); ax.set_xlabel("Custo Estimado de Espera (€)")
    plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

    # Análise 36: Top 15 Recursos por Tempo de Espera (bottleneck_by_resource)
    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates(subset=['task_id', 'resource_name'])
    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, hue='resource_name', legend=False, palette='rocket'); ax.set_title("Top 15 Recursos por Tempo Médio de Espera"); ax.set_xlabel("Tempo de Espera Médio (dias)")
    plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)
    
    # Análise 37: Atividades Mais Frequentes (top_activities_plot)
    activity_counts = df_tasks["task_name"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='YlGnBu'); ax.set_title("Atividades Mais Frequentes"); ax.set_xlabel("Contagem"); ax.set_ylabel("Atividade")
    plots['top_activities_plot'] = convert_fig_to_bytes(fig)

    # Análise 38: Tempo de Início e Fim das Atividades (activity_start_end_scatter)
    df_tasks_time = df_tasks.dropna(subset=['start_date', 'end_date'])
    df_tasks_time['duration'] = (df_tasks_time['end_date'] - df_tasks_time['start_date']).dt.total_seconds() / 3600
    fig, ax = plt.subplots(figsize=(10, 6)); ax.scatter(df_tasks_time['start_date'], df_tasks_time['end_date'], s=df_tasks_time['duration']/df_tasks_time['duration'].mean() * 50, alpha=0.6, color='#FBBF24'); ax.set_title("Tempo de Início e Fim das Atividades"); ax.set_xlabel("Data de Início"); ax.set_ylabel("Data de Fim"); plt.xticks(rotation=45)
    plots['activity_start_end_scatter'] = convert_fig_to_bytes(fig)

    # Análise 39: Análise de Marcos do Processo (milestone_time_analysis_plot)
    milestone_tasks = df_tasks[df_tasks['task_type'] == 'Marcos'].copy()
    milestone_tasks['time_to_complete'] = (milestone_tasks['end_date'] - milestone_tasks.groupby('project_id')['start_date'].transform('min')).dt.days
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(x='task_name', y='time_to_complete', data=milestone_tasks, ax=ax, hue='task_name', legend=False, palette='cool'); ax.set_title("Análise de Marcos do Processo (Tempo até à Conclusão)"); ax.set_xlabel("Marco"); ax.set_ylabel("Tempo (dias)"); plt.xticks(rotation=45)
    plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    # Análise 40: Modelo de Processo (DFG) (process_flow_dfg)
    parameters = {dfg_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png", dfg_visualizer.Variants.FREQUENCY.value.Parameters.ACTIVITY_KEY: "concept:name"}
    gviz_dfg_freq = dfg_visualizer.apply(dfg_freq, log=event_log_pm4py, variant=dfg_visualizer.Variants.FREQUENCY, parameters=parameters)
    plots['process_flow_dfg'] = convert_gviz_to_bytes(gviz_dfg_freq, format='png')
    
    # Análise 41, 42: Top 10 Variantes de Processo (variants_table) e Frequência de Variantes (variants_frequency)
    variants = variants_filter.get_variants(event_log_pm4py)
    df_variants = pd.DataFrame([{'variant_str': ' -> '.join(v), 'frequency': len(cases)} for v, cases in variants.items()])
    df_variants['percentage'] = (df_variants['frequency'] / df_variants['frequency'].sum()) * 100
    tables['variants_table'] = df_variants.sort_values('frequency', ascending=False).head(10).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='percentage', y='variant_str', data=tables['variants_table'], ax=ax, orient='h', hue='variant_str', legend=False, palette='coolwarm'); ax.set_title("Top 10 Variantes de Processo por Frequência (%)"); ax.set_xlabel("Frequência (%)")
    plots['variants_frequency'] = convert_fig_to_bytes(fig)
    
    # Análise 43: Retrabalho e Loops Mais Frequentes (rework_loops_table)
    rework_loops = Counter(f"{trace[i]} -> {trace[i+1]} -> {trace[i]}" for trace in df_variants['variant_str'].str.split(' -> ') for i in range(len(trace) - 2) if trace[i] == trace[i+2] and trace[i] != trace[i+1])
    tables['rework_loops_table'] = pd.DataFrame(rework_loops.most_common(10), columns=['rework_loop', 'frequency'])
    
    # Descoberta de Modelos (Inductive Miner)
    net_im, initial_marking_im, final_marking_im = inductive_miner.apply(event_log_pm4py)
    
    # Análise 44: Modelo Inductive Miner (Rede de Petri) (model_inductive_petrinet)
    gviz_im = pn_visualizer.apply(net_im, initial_marking_im, final_marking_im, variant=pn_visualizer.Variants.WO_DECORATION, parameters={"format": "png"})
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im, format='png')
    
    # Análise 45: Métricas de Qualidade (Inductive Miner) (metrics_inductive)
    fitness_im = replay_fitness_evaluator.apply(event_log_pm4py, net_im, initial_marking_im, final_marking_im)
    precision_im = precision_evaluator.apply(event_log_pm4py, net_im, initial_marking_im, final_marking_im)
    generalization_im = generalization_evaluator.apply(event_log_pm4py, net_im, initial_marking_im, final_marking_im)
    simplicity_im = simplicity_evaluator.apply(net_im)
    tables['metrics_inductive'] = pd.DataFrame({
        'Métrica': ['Fitness', 'Precisão', 'Generalização', 'Simplicidade'],
        'Valor': [f"{fitness_im['log_fitness']:.4f}", f"{precision_im:.4f}", f"{generalization_im:.4f}", f"{simplicity_im:.4f}"]
    })
    
    # Descoberta de Modelos (Heuristics Miner)
    net_hm, initial_marking_hm, final_marking_hm = heuristics_miner.apply(event_log_pm4py)
    
    # Análise 46: Modelo Heuristics Miner (Rede de Petri) (model_heuristic_petrinet)
    gviz_hm = pn_visualizer.apply(net_hm, initial_marking_hm, final_marking_hm, variant=pn_visualizer.Variants.WO_DECORATION, parameters={"format": "png"})
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm, format='png')
    
    # Análise 47: Métricas de Qualidade (Heuristics Miner) (metrics_heuristic)
    fitness_hm = replay_fitness_evaluator.apply(event_log_pm4py, net_hm, initial_marking_hm, final_marking_hm)
    precision_hm = precision_evaluator.apply(event_log_pm4py, net_hm, initial_marking_hm, final_marking_hm)
    generalization_hm = generalization_evaluator.apply(event_log_pm4py, net_hm, initial_marking_hm, final_marking_hm)
    simplicity_hm = simplicity_evaluator.apply(net_hm)
    tables['metrics_heuristic'] = pd.DataFrame({
        'Métrica': ['Fitness', 'Precisão', 'Generalização', 'Simplicidade'],
        'Valor': [f"{fitness_hm['log_fitness']:.4f}", f"{precision_hm:.4f}", f"{generalization_hm:.4f}", f"{simplicity_hm:.4f}"]
    })
    
    # Análise de Conformidade
    aligned_traces = alignments_miner.apply_log(event_log_pm4py, net_im, initial_marking_im, final_marking_im)
    
    # Análise 48: Distribuição de Fitness de Alinhamento (alignment_fitness_hist)
    fitness_values = [trace['fitness'] for trace in aligned_traces]
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(fitness_values, bins=20, kde=True, ax=ax, color='#FBBF24'); ax.set_title("Distribuição de Fitness de Alinhamento")
    plots['alignment_fitness_hist'] = convert_fig_to_bytes(fig)
    
    # Análise 49: Sumário de Alinhamento (alignment_summary_table)
    total_cost = sum(trace['cost'] for trace in aligned_traces)
    avg_fitness = np.mean(fitness_values)
    tables['alignment_summary_table'] = pd.DataFrame({
        'Métrica': ['Casos Alinhados', 'Fitness Médio', 'Custo Total de Desvio'],
        'Valor': [len(aligned_traces), f"{avg_fitness:.4f}", f"€{total_cost:,.2f}"]
    })

    # Análise 50: Alinhamento por Traço/Caso (trace_alignment_table)
    tables['trace_alignment_table'] = pd.DataFrame([{
        'Case ID': trace['trace_key'],
        'Fitness': f"{trace['fitness']:.4f}",
        'Custo': f"€{trace['cost']:.2f}",
        'Desvios': len(trace['replays'])
    } for trace in aligned_traces]).sort_values('Fitness').head(10) # Mostrar os 10 piores
    
    # Consolidação das saídas
    plots.update(tables)
    return plots

@st.cache_data
def run_post_mining_analysis(log, net, im, fm):
    # Esta função está definida no original, mas é assumida como opcional para a análise principal.
    # A sua implementação foi mantida vazia para este exemplo, mas no seu código original deve estar preenchida.
    plots = {}
    return plots
    
    
# --- ESTRUTURA DE NAVEGAÇÃO REVISADA (NOVA SECÇÃO) ---
NEW_DASHBOARD_STRUCTURE = {
    "1. Visão Geral e Custos": [
        ("Sumário Estatístico do Log", "📊", 'log_summary_table'),
        ("KPIs de Portfólio", "⭐", 'kpi_data'),
        ("Matriz de Performance (Duração vs Custo)", "📈", 'performance_matrix'),
        ("Projetos Outlier por Duração", "📉", 'outlier_duration'),
        ("Projetos Outlier por Custo", "💸", 'outlier_cost'),
        ("Métricas de Atraso e Custo de Desvio", "⏰", 'cost_of_delay_kpis'),
        ("Custo de Desvio por Categoria", "💰", 'cost_of_delay_breakdown'),
        ("Custo por Tipo de Recurso", "🏷️", 'cost_by_resource_type'),
    ],
    "2. Performance": [
        ("Estatísticas Descritivas (Tempo)", "⏳", 'perf_stats'),
        ("Distribuição Temporal de Eventos/Casos", "🗓️", 'event_over_time_plot'),
        ("Evolução do Lead Time/Throughput (Temporal)", "🚀", 'throughput_vs_lead_time_over_time'),
        ("Distribuição da Duração dos Projetos", "📦", 'case_durations_boxplot'),
        ("Distribuição do Lead Time", "⏱️", 'lead_time_hist'),
        ("Distribuição do Throughput", "💨", 'throughput_hist'), 
        ("Relação Lead Time vs Throughput", "↔️", 'lead_time_vs_throughput'),
        ("Duração Média por Fase do Processo", "🔀", 'cycle_time_breakdown'),
        ("Tempo de Início e Fim dos Casos", "📍", 'case_start_end_scatter'),
    ],
    "3. Recursos e Equipa": [
        ("Top 10 Recursos por Horas Trabalhadas", "👷", 'resource_workload'),
        ("Recursos por Média de Tarefas/Projeto", "👤", 'resource_avg_events'),
        ("Eficiência Semanal", "📅", 'weekly_efficiency'),
        ("Heatmap de Esforço (Recurso vs Atividade)", "🔥", 'resource_activity_matrix'),
        ("Top 10 Handoffs entre Recursos", "🤝", 'resource_handoffs'),
        ("Análise de Redes Sociais", "🌐", 'resource_activity_social_network'),
        ("Heatmap de Mistura de Casos/Recurso", "🧪", 'resource_case_mix_heatmap'),
        ("Duração Mediana por Tamanho da Equipa", "📏", 'median_duration_by_teamsize'),
        ("Impacto do Tamanho da Equipa no Atraso", "🐢", 'delay_by_teamsize'),
        ("Benchmark de Throughput por Equipa", "🏁", 'throughput_benchmark_by_teamsize'),
    ],
    "4. Gargalos e Espera": [
        ("Fluxo de Gargalos (DFG)", "🚧", 'bottleneck_flow_dfg'),
        ("Gargalos: Tempo de Serviço vs. Espera", "🛑", 'service_vs_wait_stacked'),
        ("Tempo Médio de Execução por Atividade", "🔨", 'activity_service_times'),
        ("Evolução do Tempo Médio de Espera (Mensal)", "🗓️", 'wait_time_evolution'),
        ("Espera vs. Execução (Dispersão)", "🔍", 'wait_vs_service_scatter'),
        ("Top 10 Handoffs por Tempo de Espera", "➡️", 'top_handoffs'),
        ("Top 10 Handoffs por Custo de Espera", "💵", 'top_handoffs_cost'),
        ("Top 15 Recursos por Tempo de Espera", "👤", 'bottleneck_by_resource'),
        ("Atividades Mais Frequentes", "🔢", 'top_activities_plot'),
        ("Tempo de Início e Fim das Atividades", "⌚", 'activity_start_end_scatter'),
        ("Análise de Marcos do Processo", "🚩", 'milestone_time_analysis_plot'),
    ],
    "5. Fluxo e Conformidade": [
        ("Modelo de Processo (DFG)", "🗺️", 'process_flow_dfg'),
        ("Top 10 Variantes de Processo", "👯", 'variants_table'),
        ("Frequência de Variantes", "📊", 'variants_frequency'),
        ("Retrabalho e Loops Mais Frequentes", "🔁", 'rework_loops_table'),
        ("Distribuição de Fitness de Alinhamento", "✅", 'alignment_fitness_hist'),
        ("Sumário de Alinhamento", "📝", 'alignment_summary_table'),
        ("Alinhamento por Traço/Caso", "📄", 'trace_alignment_table'),
        ("Modelo Inductive Miner (Rede de Petri)", "💠", 'model_inductive_petrinet'),
        ("Métricas de Qualidade (Inductive Miner)", "⭐", 'metrics_inductive'),
        ("Modelo Heuristics Miner (Rede de Petri)", "💎", 'model_heuristic_petrinet'),
        ("Métricas de Qualidade (Heuristics Miner)", "🌟", 'metrics_heuristic'),
    ]
}


# --- FUNÇÕES DE PÁGINA (MANTIDAS) ---
def login_page():
    st.title("🔐 Autenticação")
    st.markdown("---")
    
    st.markdown("""
        <div style="background-color: var(--card-background-color); padding: 30px; border-radius: 12px; max-width: 400px; margin: auto;">
            <h3 style="text-align: center; margin-bottom: 20px;">Acesso ao Dashboard</h3>
    """, unsafe_allow_html=True)
    
    username = st.text_input("Nome de Utilizador", key="username_input")
    password = st.text_input("Palavra-passe", type="password", key="password_input")

    st.markdown("""</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Entrar", type="primary", use_container_width=True, key="login_button"):
        if username == "admin" and password == "1234":
            st.session_state.authenticated = True
            st.session_state.user_name = username
            st.session_state.current_page = "Dashboard"
            st.rerun()
        else:
            st.error("Nome de utilizador ou palavra-passe incorretos.")

def settings_page():
    st.title("⚙️ Configurações e Carregamento de Dados")
    st.markdown("---")
    
    # Área de Upload de Ficheiros
    st.subheader("1. Carregar Dados")
    st.markdown("Carregue os seus ficheiros CSV de Portfólio de Projetos. Os nomes das colunas esperadas são: `project_id`, `task_id`, `resource_id`, `start_date`, `end_date`, `cost`, etc.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uploaded_projects = st.file_uploader("Ficheiro de Projetos (projects.csv)", type=['csv'], key="projects_upload")
        if uploaded_projects:
             st.session_state.dfs['projects'] = pd.read_csv(uploaded_projects)
             st.success(f"Projetos carregados: {len(st.session_state.dfs['projects'])} linhas.")
        
    with col2:
        uploaded_tasks = st.file_uploader("Ficheiro de Tarefas (tasks.csv)", type=['csv'], key="tasks_upload")
        if uploaded_tasks:
             st.session_state.dfs['tasks'] = pd.read_csv(uploaded_tasks)
             st.success(f"Tarefas carregadas: {len(st.session_state.dfs['tasks'])} linhas.")
        
    with col3:
        uploaded_resources = st.file_uploader("Ficheiro de Recursos (resources.csv)", type=['csv'], key="resources_upload")
        if uploaded_resources:
             st.session_state.dfs['resources'] = pd.read_csv(uploaded_resources)
             st.success(f"Recursos carregados: {len(st.session_state.dfs['resources'])} linhas.")

    uploaded_allocations = st.file_uploader("Alocações de Recursos (resource_allocations.csv)", type=['csv'], key="allocations_upload")
    if uploaded_allocations:
         st.session_state.dfs['resource_allocations'] = pd.read_csv(uploaded_allocations)
         st.success(f"Alocações carregadas: {len(st.session_state.dfs['resource_allocations'])} linhas.")
         
    uploaded_dependencies = st.file_uploader("Dependências (dependencies.csv) (Opcional)", type=['csv'], key="dependencies_upload")
    if uploaded_dependencies:
         st.session_state.dfs['dependencies'] = pd.read_csv(uploaded_dependencies)
         st.info(f"Dependências carregadas: {len(st.session_state.dfs['dependencies'])} linhas.")
    else:
        st.session_state.dfs['dependencies'] = pd.DataFrame() # DataFrame vazio se não houver dependências

    st.markdown("---")
    st.subheader("2. Executar Análise")
    
    if all(st.session_state.dfs[key] is not None for key in ['projects', 'tasks', 'resources', 'resource_allocations']):
        
        st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
        if st.button("Processar Dados e Gerar Dashboard", type="secondary", use_container_width=True):
            with st.spinner('A processar dados e gerar 50 análises...'):
                try:
                    # Chamar a função de análise principal
                    st.session_state.plots_full = run_pre_mining_analysis(st.session_state.dfs)
                    
                    st.session_state.analysis_run = True
                    st.success("Dados processados com sucesso! O Dashboard está pronto.")
                    st.session_state.current_page = "Dashboard"
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro durante o processamento de dados ou análise: {e}")
                    st.exception(e)
                    st.session_state.analysis_run = False
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Por favor, carregue os ficheiros essenciais (Projetos, Tarefas, Recursos, Alocações) para iniciar a análise.")

def dashboard_page():
    # Verifica se os dados estão carregados
    if 'plots_full' not in st.session_state or not st.session_state.analysis_run:
        st.title("Transformação Inteligente de Processos")
        st.error("Por favor, carregue e processe os dados na secção 'Configurações' para aceder ao Dashboard.")
        return

    st.title("🏠 Dashboard Geral")
    plots = st.session_state.plots_full
    
    # --- 1. SELEÇÃO DO GRUPO PRINCIPAL (Nova Navegação) ---
    group_keys = list(NEW_DASHBOARD_STRUCTURE.keys())
    
    # Permite ao utilizador selecionar um dos 5 grupos funcionais
    selected_group_name = st.selectbox(
        "Escolha a Área de Análise Funcional:",
        options=group_keys,
        key='selected_analysis_group'
    )
    
    # --- 2. EXIBIÇÃO SEQUENCIAL DAS ANÁLISES DO GRUPO SELECIONADO ---
    st.markdown("---")
    st.markdown(f"## {selected_group_name}")
    st.markdown("Clique em cada análise para expandir o gráfico ou a tabela.")
    
    analysis_list = NEW_DASHBOARD_STRUCTURE[selected_group_name]
    
    # Itera sobre a lista de análises do grupo selecionado, na ordem aprovada
    for title, icon, key in analysis_list:
        
        # Otimização do espaço vertical com Expander
        with st.expander(f"{icon} **{title}**", expanded=False):
            
            # Tenta obter o gráfico/tabela gerado
            chart_data = plots.get(key)
            
            if chart_data is not None:
                # Usa a função utilitária para exibir
                display_chart_or_table(title=title, icon=icon, chart_data=chart_data)
            else:
                # Trata casos em que a chave não existe no dicionário 'plots'
                st.warning(f"Análise '{title}' (Chave: '{key}') não encontrada ou não gerada. Verifique a fase de processamento dos dados.")


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
        elif st.session_state.current_page == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
