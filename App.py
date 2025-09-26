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
if 'tables_post_mining' not in st.session_state: st.session_state.tables_post_mining = {} 
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'event_log_df' not in st.session_state: st.session_state.event_log_df = None
if 'user_name' not in st.session_state: st.session_state.user_name = 'Admin'


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
    
    tables['kpi_data'] = {
        'Total de Projetos': len(df_projects),
        'Total de Tarefas': len(df_tasks),
        'Total de Recursos': len(df_resources),
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.1f}"
    }
    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)
    
    # Reformulação das Cores dos Gráficos:
    
    # Gráfico 1: Matriz de Performance
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.7, ax=ax, palette='viridis'); ax.axhline(0, color='#FBBF24', ls='--'); ax.axvline(0, color='#FBBF24', ls='--'); ax.set_title("Matriz de Performance")
    plots['performance_matrix'] = convert_fig_to_bytes(fig)
    
    # Gráfico 2: Distribuição da Duração dos Projetos
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=df_projects['actual_duration_days'], ax=ax, color="#2563EB"); sns.stripplot(x=df_projects['actual_duration_days'], color="#FBBF24", size=4, jitter=True, alpha=0.7, ax=ax); ax.set_title("Distribuição da Duração dos Projetos")
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
    
    # Gráfico 3: Distribuição do Lead Time
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["lead_time_days"], bins=20, kde=True, ax=ax, color="#2563EB"); ax.set_title("Distribuição do Lead Time (dias)")
    plots['lead_time_hist'] = convert_fig_to_bytes(fig)
    
    # Gráfico 4: Distribuição do Throughput
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["avg_throughput_hours"], bins=20, kde=True, color='#06B6D4', ax=ax); ax.set_title("Distribuição do Throughput (horas)")
    plots['throughput_hist'] = convert_fig_to_bytes(fig)
    
    # Gráfico 5: Boxplot do Throughput
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=perf_df["avg_throughput_hours"], color='#FBBF24', ax=ax); ax.set_title("Boxplot do Throughput (horas)")
    plots['throughput_boxplot'] = convert_fig_to_bytes(fig)
    
    # Gráfico 6: Relação Lead Time vs Throughput
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação Lead Time vs Throughput")
    plots['lead_time_vs_throughput'] = convert_fig_to_bytes(fig)
    
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    
    # Gráfico 7: Tempo Médio de Execução por Atividade
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, hue='task_name', legend=False, palette='coolwarm'); ax.set_title("Tempo Médio de Execução por Atividade")
    plots['activity_service_times'] = convert_fig_to_bytes(fig)
    
    df_handoff = log_df_final.sort_values(['case:concept:name', 'time:timestamp'])
    df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
    
    # Gráfico 8: Top 10 Handoffs por Tempo de Espera
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', ax=ax, hue='transition', legend=False, palette='viridis'); ax.set_title("Top 10 Handoffs por Tempo de Espera")
    plots['top_handoffs'] = convert_fig_to_bytes(fig)
    
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()
    
    # Gráfico 9: Top 10 Handoffs por Custo de Espera
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, hue='transition', legend=False, palette='magma'); ax.set_title("Top 10 Handoffs por Custo de Espera")
    plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

    activity_counts = df_tasks["task_name"].value_counts()
    
    # Gráfico 10: Atividades Mais Frequentes
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='YlGnBu'); ax.set_title("Atividades Mais Frequentes")
    plots['top_activities_plot'] = convert_fig_to_bytes(fig)
    
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    
    # Gráfico 11: Top 10 Recursos por Horas Trabalhadas
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, hue='resource_name', legend=False, palette='plasma'); ax.set_title("Top 10 Recursos por Horas Trabalhadas")
    plots['resource_workload'] = convert_fig_to_bytes(fig)
    
    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()
    resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
    
    # Gráfico 12: Recursos por Média de Tarefas por Projeto
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), ax=ax, hue='resource_name', legend=False, palette='coolwarm'); ax.set_title("Recursos por Média de Tarefas por Projeto")
    plots['resource_avg_events'] = convert_fig_to_bytes(fig)
    
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    
    # Gráfico 13: Heatmap de Esforço por Recurso e Atividade
    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_activity_matrix_pivot, cmap='Blues', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8}, linewidths=.5, linecolor='#374151'); ax.set_title("Heatmap de Esforço por Recurso e Atividade")
    plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)
    
    handoff_counts = Counter((trace[i]['org:resource'], trace[i+1]['org:resource']) for trace in event_log_pm4py for i in range(len(trace) - 1) if 'org:resource' in trace[i] and 'org:resource' in trace[i+1] and trace[i]['org:resource'] != trace[i+1]['org:resource'])
    df_resource_handoffs = pd.DataFrame(handoff_counts.most_common(10), columns=['Handoff', 'Contagem'])
    df_resource_handoffs['Handoff'] = df_resource_handoffs['Handoff'].apply(lambda x: f"{x[0]} -> {x[1]}")
    
    # Gráfico 14: Top 10 Handoffs entre Recursos
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs, ax=ax, hue='Handoff', legend=False, palette='rocket'); ax.set_title("Top 10 Handoffs entre Recursos")
    plots['resource_handoffs'] = convert_fig_to_bytes(fig)
    
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    
    # Gráfico 15: Custo por Tipo de Recurso
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', ax=ax, hue='resource_type', legend=False, palette='magma'); ax.set_title("Custo por Tipo de Recurso")
    plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)
    
    variants_df = log_df_final.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='trace')
    variants_df['variant_str'] = variants_df['trace'].apply(lambda x: ' -> '.join(x))
    variant_analysis = variants_df['variant_str'].value_counts().reset_index(name='frequency')
    variant_analysis['percentage'] = (variant_analysis['frequency'] / variant_analysis['frequency'].sum()) * 100
    tables['variants_table'] = variant_analysis.head(10)
    
    # Gráfico 16: Top 10 Variantes de Processo por Frequência
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
    
    # Gráfico 17: Impacto do Tamanho da Equipa no Atraso
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='flare'); ax.set_title("Impacto do Tamanho da Equipa no Atraso")
    plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)
    
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    
    # Gráfico 18: Duração Mediana por Tamanho da Equipa
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='crest'); ax.set_title("Duração Mediana por Tamanho da Equipa")
    plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)
    
    df_alloc_costs['day_of_week'] = df_alloc_costs['allocation_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Gráfico 19: Eficiência Semanal (Horas Trabalhadas)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, hue='day_of_week', legend=False, palette='viridis'); ax.set_title("Eficiência Semanal (Horas Trabalhadas)"); plt.xticks(rotation=45)
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    
    df_tasks_analysis = df_tasks.copy(); df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True); df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates()
    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
    
    # Gráfico 20: Top 15 Recursos por Tempo Médio de Espera
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, hue='resource_name', legend=False, palette='rocket'); ax.set_title("Top 15 Recursos por Tempo Médio de Espera")
    plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)
    
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    
    # Gráfico 21: Gargalos: Tempo de Serviço vs. Espera
    fig, ax = plt.subplots(figsize=(8, 5)); bottleneck_by_activity.plot(kind='bar', stacked=True, color=['#2563EB', '#FBBF24'], ax=ax); ax.set_title("Gargalos: Tempo de Serviço vs. Espera")
    plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)
    
    # Gráfico 22: Espera vs. Execução (Dispersão)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=bottleneck_by_activity.reset_index(), x='service_time_days', y='waiting_time_days', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Tempo de Espera vs. Tempo de Execução")
    plots['wait_vs_service_scatter'] = convert_fig_to_bytes(fig)
    
    df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id', 'completion_month']], on='project_id')
    monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()
    
    # Gráfico 23: Evolução do Tempo Médio de Espera
    fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax, color='#06B6D4'); plt.xticks(rotation=45); ax.set_title("Evolução do Tempo Médio de Espera")
    plots['wait_time_evolution'] = convert_fig_to_bytes(fig)
    
    df_perf_full = perf_df.merge(df_projects, left_on='case:concept:name', right_on='project_id')
    
    # Gráfico 24: Benchmark de Throughput por Tamanho da Equipa
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
    
    # Gráfico 25: Duração Média por Fase do Processo
    fig, ax = plt.subplots(figsize=(8, 4)); avg_cycle_time_by_phase.plot(kind='bar', color=sns.color_palette('tab10'), ax=ax); ax.set_title("Duração Média por Fase do Processo"); plt.xticks(rotation=0)
    plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)
    
    # Gráfico 26: Análise de Desvios de Tempo de Marcos do Processo
    # Assumindo que df_projects tem 'days_diff' > 0 para atrasos
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(df_projects['days_diff'], bins=30, kde=True, ax=ax, color='#FBBF24'); ax.set_title("Desvios de Prazo (Atraso em Dias)")
    plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    plots = {}
    metrics = {}
    tables = {}
    
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

    # Inductive Miner
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)

    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        fig, ax = plt.subplots(figsize=(8, 4)); 
        barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, hue='Métrica', legend=False, palette='coolwarm')
        for p in barplot.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', color='#E5E7EB')
        ax.set_title(title); ax.set_ylim(0, 1.05); 
        return fig

    metrics_im = {
        "Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0),
        "Precisão": precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im),
        "Generalização": generalization_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im),
        "Simplicidade": simplicity_evaluator.apply(net_im)
    }
    plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)'))
    metrics['inductive_miner'] = metrics_im

    # Heuristics Miner
    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    
    metrics_hm = {
        "Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0),
        "Precisão": precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm),
        "Generalização": generalization_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm),
        "Simplicidade": simplicity_evaluator.apply(net_hm)
    }
    plots['metrics_heuristic'] = convert_fig_to_bytes(plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)'))
    metrics['heuristic_miner'] = metrics_hm

    # DFG de Performance
    dfg_perf_gviz = dfg_visualizer.apply(dfg_discovery.apply(_event_log_pm4py, parameters={'activity_key': 'concept:name', 'timestamp_key': 'time:timestamp', 'case_id_key': 'case:concept:name', 'measure': 'performance'}))
    plots['dfg_performance_visualization'] = convert_gviz_to_bytes(dfg_perf_gviz)
    
    # DFG de Frequência (usado para variants_dfg_visualization)
    dfg_freq_gviz = dfg_visualizer.apply(dfg_discovery.apply(_event_log_pm4py))
    plots['variants_dfg_visualization'] = convert_gviz_to_bytes(dfg_freq_gviz)

    # Social Network
    social_network_gviz = pm4py.viz.social_network.visualizer.apply(pm4py.discover_social_network(_event_log_pm4py))
    plots['social_network_visualization'] = convert_gviz_to_bytes(social_network_gviz)

    # Conformance (Alignments)
    alignments = alignments_miner.apply(_event_log_pm4py, net_im, im_im, fm_im, parameters={alignments_miner.Variants.VERSION_TOKEN_BASED.value.Parameters.RET_COMP_STATS: True})
    
    # Resumo da Conformidade
    df_alignments = pd.DataFrame([{
        'Case ID': trace['case:concept:name'],
        'Fitness': trace['fitness'],
        'Custo (Modelo vs Real)': trace['cost'],
        'É Desvio?': trace['cost'] > 0
    } for trace in alignments])
    
    alignments_summary = df_alignments.groupby('É Desvio?').agg(
        Num_Casos=('Case ID', 'count'),
        Fitness_Media=('Fitness', 'mean'),
        Custo_Total=('Custo (Modelo vs Real)', 'sum')
    ).reset_index()
    tables['alignments_summary'] = alignments_summary
    
    def plot_conformance_hist(df_alignments):
        fig, ax = plt.subplots(figsize=(8, 5)); 
        sns.histplot(df_alignments['Fitness'], bins=20, kde=True, ax=ax, color="#06B6D4"); 
        ax.set_title("Distribuição do Fitness de Conformidade");
        return fig
    
    plots['conformance_alignments_visualization'] = convert_fig_to_bytes(plot_conformance_hist(df_alignments))
    
    return plots, tables, metrics

# --- PÁGINA DE CONFIGURAÇÕES (Lógica reconstruída para garantir integridade) ---
def settings_page():
    st.title("⚙️ Configurações e Carregamento de Dados")
    st.markdown("Faça o upload dos 5 ficheiros CSV necessários para a análise.")
    
    # Lógica de upload completa
    df_files = {}
    required_files = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    
    for filename in required_files:
        uploaded_file = st.file_uploader(f"Upload: {filename}.csv", type=['csv'], key=f'file_{filename}')
        if uploaded_file is not None:
            try:
                df_files[filename] = pd.read_csv(uploaded_file)
                st.session_state.dfs[filename] = df_files[filename]
                st.success(f"{filename}.csv carregado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao carregar {filename}.csv: {e}")

    all_files_loaded = all(st.session_state.dfs[f] is not None for f in required_files)
    
    st.markdown("---")

    if st.button("▶️ Iniciar Análise de Dados", disabled=not all_files_loaded, use_container_width=True, key='run_analysis', help="Carregue todos os 5 ficheiros CSV para iniciar."):
        with st.spinner('A processar dados e a executar algoritmos de Mineração de Processos...'):
            try:
                plots_pre, tables_pre, event_log, df_projects, df_tasks, df_resources, df_full_context = run_pre_mining_analysis(st.session_state.dfs)
                plots_post, tables_post, metrics_post = run_post_mining_analysis(event_log, df_projects, df_tasks, df_resources, df_full_context)
                
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                st.session_state.plots_post_mining = plots_post
                st.session_state.tables_post_mining = tables_post
                st.session_state.metrics = metrics_post
                st.session_state.event_log_df = pm4py.convert_to_dataframe(event_log) # Para a tabela no dashboard
                st.session_state.analysis_run = True
                st.success("Análise concluída com sucesso! Vá para o Dashboard Geral.")
                
            except Exception as e:
                st.error(f"Ocorreu um erro durante a análise: {e}")
                st.session_state.analysis_run = False
                
    if st.session_state.analysis_run:
        st.success("Análise de dados pronta. Volte ao Dashboard Geral.")


# --- PÁGINA LOGIN (Lógica reconstruída para garantir integridade) ---
def login_page():
    st.title("🔒 Login")
    
    # Controlo de layout para centralizar o login
    col_login, col_empty = st.columns([1, 2])
    
    with col_login:
        username = st.text_input("Utilizador", value="admin")
        password = st.text_input("Palavra-passe", type="password", value="admin")
        
        if st.button("Entrar", use_container_width=True):
            # Credenciais hardcoded como no código original
            if username == "admin" and password == "admin":
                st.session_state.authenticated = True
                st.session_state.user_name = username
                st.session_state.current_page = "Dashboard"
                st.rerun()
            else:
                st.error("Credenciais Inválidas")


# --- PÁGINA DASHBOARD (COM AS ALTERAÇÕES SOLICITADAS) ---
def dashboard_page():
    st.title("📊 Dashboard de Análise de Processos")
    st.markdown("Uma visão abrangente dos resultados da Mineração de Processos e da Análise Exploratória.")
    
    if not st.session_state.analysis_run:
        st.info("Por favor, carregue os dados e inicie a análise nas Configurações.")
        return

    plots_pre = st.session_state.plots_pre_mining
    tables_pre = st.session_state.tables_pre_mining
    plots_post = st.session_state.plots_post_mining
    tables_post = st.session_state.tables_post_mining
    metrics_post = st.session_state.metrics
    
    # ----------------------------------------------------
    # --- ALTERAÇÃO APLICADA: REORGANIZAÇÃO DAS ABAS ---
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Visão Geral e Custos",
        "⚡ Performance",
        "🧑‍💻 Recursos",
        "🚦 Gargalos e Espera",
        "✅ Fluxo e Conformidade"
    ])
    # ----------------------------------------------------
    # --- FIM DA ALTERAÇÃO NA DEFINIÇÃO DAS ABAS ---
    # ----------------------------------------------------

    # --- 1. VISÃO GERAL E CUSTOS ---
    with tab1:
        st.header("Visão Geral e Custos (Análise Exploratória)")
        
        # 1.1. KPIs (Linha 1)
        col1, col2, col3, col4 = st.columns(4)
        kpi_data = tables_pre.get('kpi_data', {})
        col1.metric("Total de Projetos", kpi_data.get('Total de Projetos', 'N/A'))
        col2.metric("Total de Tarefas", kpi_data.get('Total de Tarefas', 'N/A'))
        col3.metric("Total de Recursos", kpi_data.get('Total de Recursos', 'N/A'))
        col4.metric("Duração Média", f"{kpi_data.get('Duração Média (dias)', 'N/A')} dias")

        st.markdown("---")

        # 1.2. KPIs de Custo e Atraso (Linha 2)
        col_c1, col_c2, col_c3 = st.columns(3)
        cost_delay_kpis = tables_pre.get('cost_of_delay_kpis', {})
        col_c1.metric("Custo Total Projetos Atrasados", cost_delay_kpis.get('Custo Total Projetos Atrasados', 'N/A'))
        col_c2.metric("Atraso Médio", f"{cost_delay_kpis.get('Atraso Médio (dias)', 'N/A')} dias")
        col_c3.metric("Custo Médio/Dia Atraso", cost_delay_kpis.get('Custo Médio/Dia Atraso', 'N/A'))
        
        st.markdown("---")
        
        # 1.3. Gráficos Principais e Outliers
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            create_card("Matriz de Performance (Desvio de Custo vs. Tempo)", "⚖️", chart_bytes=plots_pre.get('performance_matrix'))
            create_card("Custo por Tipo de Recurso", "💸", chart_bytes=plots_pre.get('cost_by_resource_type'))
        
        with col_g2:
            create_card("Distribuição da Duração dos Projetos", "📦", chart_bytes=plots_pre.get('case_durations_boxplot'))
            
        st.markdown("---")
        
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            create_card("Top 10 Projetos Outliers de Duração", "⏰", dataframe=tables_pre.get('outlier_duration'))
        with col_g4:
            create_card("Top 10 Projetos Outliers de Custo", "💰", dataframe=tables_pre.get('outlier_cost'))

    # --- 2. PERFORMANCE ---
    with tab2:
        st.header("Análise de Performance Temporal e Eficiência")

        # 2.1. Gráficos de Distribuição de Tempo
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            create_card("Distribuição do Lead Time (dias)", "⏱️", chart_bytes=plots_pre.get('lead_time_hist'))
            create_card("Distribuição do Throughput (horas)", "💨", chart_bytes=plots_pre.get('throughput_hist'))
            
        with col_p2:
            create_card("Relação Lead Time vs Throughput", "📈", chart_bytes=plots_pre.get('lead_time_vs_throughput'))
            create_card("Boxplot do Throughput (horas)", "📦", chart_bytes=plots_pre.get('throughput_boxplot'))
            
        st.subheader("Performance do Fluxo (Process Mining)")
        create_card("DFG de Performance (Tempo Médio)", "🗺️", chart_bytes=plots_post.get('dfg_performance_visualization'))

        st.subheader("Performance por Atividade e Estrutura")
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            create_card("Tempo Médio de Execução por Atividade", "⏳", chart_bytes=plots_pre.get('activity_service_times'))
            create_card("Duração Média por Fase do Processo", "🗂️", chart_bytes=plots_pre.get('cycle_time_breakdown'))
        with col_p4:
            create_card("Atividades Mais Frequentes", "🔢", chart_bytes=plots_pre.get('top_activities_plot'))
            
        st.subheader("Impacto Organizacional na Performance")
        col_p5, col_p6 = st.columns(2)
        with col_p5:
            create_card("Impacto do Tamanho da Equipa no Atraso", "🧑‍🤝‍🧑", chart_bytes=plots_pre.get('delay_by_teamsize'))
            create_card("Benchmark de Throughput por Tamanho da Equipa", "🚀", chart_bytes=plots_pre.get('throughput_benchmark_by_teamsize'))
        with col_p6:
            create_card("Duração Mediana por Tamanho da Equipa", "⏱️", chart_bytes=plots_pre.get('median_duration_by_teamsize'))

    # --- 3. RECURSOS ---
    with tab3:
        st.header("Análise de Carga de Trabalho e Interação de Recursos")

        # 3.1. Carga de Trabalho e Utilização
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            create_card("Top 10 Recursos por Horas Trabalhadas", "💪", chart_bytes=plots_pre.get('resource_workload'))
        with col_r2:
            create_card("Recursos por Média de Tarefas por Projeto", "🎯", chart_bytes=plots_pre.get('resource_avg_events'))

        st.subheader("Matriz de Esforço")
        create_card("Heatmap de Esforço por Recurso e Atividade", "🔥", chart_bytes=plots_pre.get('resource_activity_matrix'))

        st.subheader("Interação e Colaboração (Social Mining)")
        col_r3, col_r4 = st.columns(2)
        with col_r3:
            create_card("Rede Social de Recursos (Handoffs)", "🌐", chart_bytes=plots_post.get('social_network_visualization'))
        with col_r4:
            create_card("Top 10 Handoffs entre Recursos (Frequência)", "🔄", chart_bytes=plots_pre.get('resource_handoffs'))
        
        st.subheader("Eficiência e Produtividade")
        create_card("Eficiência Semanal (Horas Trabalhadas)", "📅", chart_bytes=plots_pre.get('weekly_efficiency'))

    # --- 4. GARGALOS E ESPERA ---
    with tab4:
        st.header("Identificação de Tempos de Espera e Estrangulamentos")

        # 4.1. Análise de Espera e Serviço
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            create_card("Gargalos: Tempo de Serviço vs. Espera (Stacked)", "🧱", chart_bytes=plots_pre.get('service_vs_wait_stacked'))
            create_card("Evolução do Tempo Médio de Espera (Mensal)", "📉", chart_bytes=plots_pre.get('wait_time_evolution'))
        with col_g2:
            create_card("Top 15 Recursos por Tempo Médio de Espera", "⏳", chart_bytes=plots_pre.get('bottleneck_by_resource'))
            create_card("Tempo de Espera vs. Tempo de Execução (Dispersão)", "🔍", chart_bytes=plots_pre.get('wait_vs_service_scatter'))

        st.subheader("Análise de Transições e Custo da Espera")
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            create_card("Top 10 Handoffs por Tempo de Espera", "🛑", chart_bytes=plots_pre.get('top_handoffs'))
            create_card("Top 10 Loops de Retrabalho (Rework Loops)", "🔁", dataframe=tables_pre.get('rework_loops_table'))
        with col_g4:
            create_card("Top 10 Handoffs por Custo de Espera", "💲", chart_bytes=plots_pre.get('top_handoffs_cost'))
            create_card("Análise de Desvios de Tempo de Marcos do Processo", "🚩", chart_bytes=plots_pre.get('milestone_time_analysis_plot'))

    # --- 5. FLUXO E CONFORMIDADE ---
    with tab5:
        st.header("Modelos de Processo e Avaliação de Qualidade (Process Mining)")
        
        # 5.1. Métricas de Qualidade (Fitness, Precision, etc.)
        st.subheader("Métricas de Qualidade do Modelo")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            create_card("Qualidade do Modelo (Inductive Miner)", "⭐", chart_bytes=plots_post.get('metrics_inductive'))
        with col_m2:
            create_card("Qualidade do Modelo (Heuristics Miner)", "⭐", chart_bytes=plots_post.get('metrics_heuristic'))
        
        # 5.2. Visualização do Modelo
        st.subheader("Redes de Petri Descobertas")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            create_card("Rede de Petri Descoberta (Inductive Miner)", "🗺️", chart_bytes=plots_post.get('model_inductive_petrinet'))
        with col_v2:
            create_card("Rede de Petri Descoberta (Heuristics Miner)", "🗺️", chart_bytes=plots_post.get('model_heuristic_petrinet'))
            
        st.subheader("Variantes e Conformidade")
        col_v3, col_v4 = st.columns(2)
        with col_v3:
            create_card("Top 10 Variantes de Processo por Frequência", "🔄", chart_bytes=plots_pre.get('variants_frequency'))
            create_card("Tabela Top 10 Variantes", "📜", dataframe=tables_pre.get('variants_table'))
        with col_v4:
            create_card("DFG de Frequência (Análise de Fluxo)", "📈", chart_bytes=plots_post.get('variants_dfg_visualization'))
            
        st.subheader("Análise de Conformidade (Alignments)")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            create_card("Conformidade (Desvios do Modelo)", "⚠️", chart_bytes=plots_post.get('conformance_alignments_visualization'))
        with col_c2:
            create_card("Resumo dos Desvios (Alignments)", "📝", dataframe=tables_post.get('alignments_summary'))
        
        st.subheader("Log de Eventos (Base de Dados)")
        # Verifica se o log_df_final existe no session state
        if st.session_state.event_log_df is not None:
            create_card("Log de Eventos (Primeiras 100 Linhas)", "🪵", dataframe=st.session_state.event_log_df.head(100))
        else:
            st.warning("O Log de Eventos ainda não foi processado.")


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
            
if __name__ == '__main__':
    main()
