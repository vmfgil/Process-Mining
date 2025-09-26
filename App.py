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
        color: var(--text-color-light-bg) !important; /* ALTERADO para Azul Escuro */
        font-weight: 700 !important; /* ALTERADO para BOLD */
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:hover {
        border-color: var(--primary-color) !important;
        background-color: rgba(37, 99, 235, 0.2) !important; /* Azul com 20% de opacidade */
    }
    div.active-button .stButton>button {
        background-color: var(--primary-color) !important;
        color: var(--text-color-light-bg) !important; /* ALTERADO para Azul Escuro */
        border: 1px solid var(--primary-color) !important;
        font-weight: 700 !important;
    }

    /* Painel Lateral */
    [data-testid="stSidebar"] { background-color: var(--sidebar-background); border-right: 1px solid var(--border-color); }
    [data-testid="stSidebar"] .stButton>button {
        background-color: var(--primary-color) !important; /* Botões da sidebar com cor de destaque */
        color: var(--text-color-light-bg) !important; /* ALTERADO para Azul Escuro */
        font-weight: 700 !important; /* ADICIONADO BOLD */
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
        font-weight: 700 !important; /* ALTERADO para BOLD */
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
        # Este método preserva o estilo básico do DataFrame do Pandas,
        # mas precisa de estilos CSS adicionais para o modo escuro, que já tem no seu código.
        # Adicionamos a classe 'pandas-df-card' para controlo de estilo.
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
    
    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    plots = {}
    metrics = {}
    
    # Preparar log para conformidade (incluindo start/complete)
    df_start_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'start_date']].rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_start_events['lifecycle:transition'] = 'start'
    df_complete_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'end_date']].rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_complete_events['lifecycle:transition'] = 'complete'
    log_df_full_lifecycle = pd.concat([df_start_events, df_complete_events]).sort_values('time:timestamp')
    log_full_pm4py = pm4py.convert_to_event_log(log_df_full_lifecycle)
    
    # 1. Mineração de Processos (Inductive Miner - Top 3 Variantes)
    variants_dict = variants_filter.get_variants(_event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_3_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)
    
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    
    # Função para plotar métricas de qualidade
    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        fig, ax = plt.subplots(figsize=(8, 4));
        barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, hue='Métrica', legend=False, palette='coolwarm')
        
        # Adicionar rótulos de valor
        for p in barplot.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        color='#E5E7EB')
                        
        ax.set_title(title); 
        ax.set_ylim(0, 1.05);
        return fig

    # Avaliação do Inductive Miner
    metrics_im = {
        "Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0),
        "Precisão": precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im),
        "Generalização": generalization_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im),
        "Simplicidade": simplicity_evaluator.apply(net_im)
    }
    plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)'))
    metrics['inductive_miner'] = metrics_im

    # 2. Mineração Heurística (Heuristics Miner)
    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristics_petrinet'] = convert_gviz_to_bytes(gviz_hm)

    # Avaliação do Heuristics Miner
    metrics_hm = {
        "Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0),
        "Precisão": precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm),
        "Generalização": generalization_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm),
        "Simplicidade": simplicity_evaluator.apply(net_hm)
    }
    plots['metrics_heuristics'] = convert_fig_to_bytes(plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)'))
    metrics['heuristics_miner'] = metrics_hm
    
    # 3. Descoberta do DFG (Directly Follows Graph)
    dfg = dfg_discovery.apply(_event_log_pm4py)
    gviz_dfg = dfg_visualizer.apply(dfg, log=_event_log_pm4py, variant=dfg_visualizer.Variants.FREQUENCY)
    plots['model_dfg_frequency'] = convert_gviz_to_bytes(gviz_dfg)
    
    # 4. Análise de Conformidade: Desvios
    # Vamos usar o modelo mais fitness (Inductive Miner) para a conformidade.
    alignment_results = alignments_miner.apply(log_full_pm4py, net_im, im_im, fm_im, parameters={alignments_miner.Variants.VERSION_STATE_EQUIVALENCE.value.Parameters.ACTIVITY_KEY: "concept:name"})
    
    # Métrica: Desvios (Deviations)
    deviations = [a for a in alignment_results if a['cost'] > 0]
    deviation_traces = len(deviations)
    conformance_percentage = (1 - (deviation_traces / len(log_full_pm4py))) * 100
    
    # Análise dos Desvios (Top 5)
    df_deviations = pd.DataFrame(deviations)
    if not df_deviations.empty:
        df_deviations['case_id'] = df_deviations.apply(lambda x: x['trace'][0]['case:concept:name'], axis=1)
        df_deviations = df_deviations.merge(_df_projects[['project_id', 'project_name']], left_on='case_id', right_on='project_id', how='left')
        
        # Simplificar o cálculo do desvio: o PM4PY não dá uma descrição simples do desvio.
        # Vamos contar o número de passos de modelo e passos de log nos desvios.
        df_deviations['moves_on_model'] = df_deviations['alignment'].apply(lambda x: sum(1 for (m, l) in x if m is not None and l is None))
        df_deviations['moves_on_log'] = df_deviations['alignment'].apply(lambda x: sum(1 for (m, l) in x if m is None and l is not None))
        
        deviation_summary = df_deviations.sort_values('cost', ascending=False).head(5)[['case_id', 'project_name', 'cost', 'moves_on_model', 'moves_on_log']]
    else:
        deviation_summary = pd.DataFrame({'case_id': ['Nenhum'], 'project_name': ['Nenhum'], 'cost': [0], 'moves_on_model': [0], 'moves_on_log': [0]})

    # 5. Análise de Recursos - Resource Pooling e Handoffs (Detalhe)
    # Já fizemos alguns no pre-mining, mas vamos focar em métricas.
    
    # Taxa de transferência de trabalho (Handoffs)
    df_full_context = _df_full_context.copy()
    df_full_context['resource_transition'] = df_full_context.groupby('project_id')['resource_name'].shift(1)
    df_handoffs = df_full_context[df_full_context['resource_name'] != df_full_context['resource_transition']].dropna(subset=['resource_transition'])
    handoff_rate = len(df_handoffs) / len(df_full_context) if len(df_full_context) > 0 else 0
    
    # 6. Análise de Custo por Caminho (Path Cost Analysis)
    # Não há custo detalhado por atividade, mas podemos usar o custo por recurso.
    
    # Tempo de ciclo por fase (já calculado no pre-mining, vamos usá-lo para plotar)
    df_tasks_raw = _df_tasks_raw.copy()
    def get_phase(task_type):
        if task_type in ['Desenvolvimento', 'Correção', 'Revisão', 'Design']: return 'Desenvolvimento & Design'
        if task_type == 'Teste': return 'Teste (QA)'
        if task_type in ['Deploy', 'DBA']: return 'Operações & Deploy'
        return 'Outros'
    df_tasks_raw['phase'] = df_tasks_raw['task_type'].apply(get_phase)
    phase_times = df_tasks_raw.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index()
    phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days
    
    # Gráfico: Boxplot do Tempo de Ciclo por Fase
    fig, ax = plt.subplots(figsize=(8, 5));
    sns.boxplot(data=phase_times, x='phase', y='cycle_time_days', ax=ax, hue='phase', legend=False, palette='viridis');
    ax.set_title("Boxplot do Tempo de Ciclo por Fase");
    plt.xticks(rotation=45);
    plots['phase_cycle_time_boxplot'] = convert_fig_to_bytes(fig)
    
    # 7. Análise de Milestones e Marcos do Processo (Simulação de Gating)
    # Analisar o tempo médio que leva para atingir um marco (ex: 50% das atividades)
    df_tasks_raw['task_status'] = df_tasks_raw['status'].apply(lambda x: 1 if x == 'complete' else 0)
    task_counts = df_tasks_raw.groupby('project_id')['task_id'].transform('count')
    df_tasks_raw['completion_ratio'] = df_tasks_raw.groupby('project_id')['task_status'].cumsum() / task_counts
    
    milestones = [0.5, 0.9] # 50% e 90%
    milestone_times = []
    
    for project_id, group in df_tasks_raw.sort_values('end_date').groupby('project_id'):
        for m in milestones:
            milestone_task = group[group['completion_ratio'] >= m].head(1)
            if not milestone_task.empty:
                start_date = group['start_date'].min()
                end_date = milestone_task['end_date'].iloc[0]
                duration = (end_date - start_date).total_seconds() / (24*3600)
                milestone_times.append({'project_id': project_id, 'milestone': f'{int(m*100)}%', 'duration_days': duration})

    df_milestone_times = pd.DataFrame(milestone_times)
    
    # Gráfico: Duração para Atingir Marcos
    fig, ax = plt.subplots(figsize=(8, 5));
    sns.boxplot(data=df_milestone_times, x='milestone', y='duration_days', ax=ax, hue='milestone', legend=False, palette='coolwarm');
    ax.set_title("Duração para Atingir Marcos do Processo (dias)");
    plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    metrics['conformance'] = {
        'Desvios': f"{deviation_traces} / {len(log_full_pm4py)}",
        'Conformidade (%)': f"{conformance_percentage:.2f}",
        'Taxa Handoffs': f"{handoff_rate:.2%}"
    }

    return plots, metrics, deviation_summary, df_milestone_times


# --- FUNÇÕES DE PÁGINAS ---

def login_page():
    # ... (código da login_page) ...
    st.title("Bem-vindo ao 📊 Process Intelligence Platform ✨")
    st.markdown("""
    <div style="padding: 20px; border: 1px solid var(--primary-color); border-radius: 8px; background-color: var(--card-background-color); max-width: 400px;">
        <h3 style="color: var(--text-color-dark-bg); text-align: center;">Acesso à Plataforma</h3>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f'<div style="max-width: 400px; margin: 0 auto;">', unsafe_allow_html=True)
        
        # Campos de login
        username = st.text_input("Nome de Utilizador", key="login_user")
        password = st.text_input("Palavra-passe", type="password", key="login_pass")

        # Botão de Login
        if st.button("Entrar", key="login_button", use_container_width=True):
            if username.lower() == "admin" and password == "admin":
                st.session_state.authenticated = True
                st.session_state.user_name = "Admin"
                st.session_state.current_page = "Dashboard"
                st.rerun()
            else:
                st.error("Nome de utilizador ou palavra-passe incorretos.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)


def dashboard_page():
    # ... (código da dashboard_page) ...
    st.title("Dashboard de Processos ✨")

    # Colunas para os botões de navegação da dashboard
    col1, col2, col3 = st.columns([1, 1, 1])

    # Funções de seleção de dashboard
    def set_dashboard(name):
        st.session_state.current_dashboard = name
        st.session_state.current_section = "overview" # Resetar a secção
        
    with col1:
        # Botão 'Pré-Mineração'
        active_class = "active-button" if st.session_state.current_dashboard == "Pré-Mineração" else ""
        st.markdown(f"""
        <div class="{active_class}">
            {st.button("1. Pré-Mineração & Análise Descritiva", key="btn_pre_mining", on_click=set_dashboard, args=("Pré-Mineração", ), use_container_width=True)}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Botão 'Pós-Mineração'
        active_class = "active-button" if st.session_state.current_dashboard == "Pós-Mineração" else ""
        st.markdown(f"""
        <div class="{active_class}">
            {st.button("2. Descoberta & Conformidade", key="btn_post_mining", on_click=set_dashboard, args=("Pós-Mineração", ), use_container_width=True)}
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Botão 'Recomendação'
        active_class = "active-button" if st.session_state.current_dashboard == "Recomendação" else ""
        st.markdown(f"""
        <div class="{active_class}">
            {st.button("3. Otimização & Recomendação", key="btn_recommendation", on_click=set_dashboard, args=("Recomendação", ), use_container_width=True)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- Carregamento de Dados ---
    if st.session_state.dfs['projects'] is None:
        st.subheader("Carregamento de Dados")
        st.info("Por favor, carregue os ficheiros CSV/Excel para iniciar a análise.")
        
        with st.expander("Instruções e Formato Esperado"):
            st.markdown("""
            **Formato de Dados Esperado (Obrigatório para PM):**
            Para uma análise de Process Mining completa, são necessários os seguintes ficheiros com as colunas essenciais:

            * **`tasks`** (Atividades/Eventos):
                * `task_id` (ID do Evento/Tarefa)
                * `project_id` (ID do Case)
                * `task_name` (Nome da Atividade - *concept:name*)
                * `start_date`, `end_date` (Timestamps de Início/Fim - *time:timestamp*)
                * `task_type`, `status` (Atributos adicionais)
            * **`resource_allocations`** (Alocação de Recursos):
                * `task_id` (ID do Evento/Tarefa)
                * `resource_id` (ID do Recurso - *org:resource*)
                * `hours_worked`, `allocation_date`

            Ficheiros de contexto adicionais (`projects`, `resources`, `dependencies`) são usados para enriquecer a análise.
            """)

        col_files = st.columns(5)
        file_keys = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
        file_names = {
            'projects': 'Projetos', 
            'tasks': 'Tarefas (Eventos)', 
            'resources': 'Recursos', 
            'resource_allocations': 'Alocações', 
            'dependencies': 'Dependências'
        }
        
        for i, key in enumerate(file_keys):
            with col_files[i]:
                uploaded_file = st.file_uploader(f"Carregar {file_names[key]}", type=['csv', 'xlsx'], key=f'file_uploader_{key}')
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        st.session_state.dfs[key] = df
                        st.success(f"Ficheiro '{file_names[key]}' carregado!")
                    except Exception as e:
                        st.error(f"Erro ao carregar ficheiro {file_names[key]}: {e}")

        # Botão para iniciar análise
        required_files = ['projects', 'tasks', 'resources', 'resource_allocations']
        all_required_loaded = all(st.session_state.dfs[key] is not None for key in required_files)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="iniciar-analise-button" style="text-align: center;">', unsafe_allow_html=True)
        if st.button("▶️ Iniciar Análise", use_container_width=True, disabled=not all_required_loaded):
            if all_required_loaded:
                st.session_state.analysis_run = True
                st.toast("Análise de Pré-Mineração a ser executada...")
                
                # Executar a pré-mineração
                try:
                    plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context = run_pre_mining_analysis(st.session_state.dfs)
                    st.session_state.plots_pre_mining = plots
                    st.session_state.tables_pre_mining = tables
                    st.session_state.event_log_pm4py = event_log_pm4py
                    st.session_state.df_projects = df_projects
                    st.session_state.df_tasks = df_tasks
                    st.session_state.df_resources = df_resources
                    st.session_state.df_full_context = df_full_context
                    
                    st.success("Análise de Pré-Mineração concluída com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro durante a análise de Pré-Mineração: {e}")
                    st.session_state.analysis_run = False
            else:
                st.warning("Carregue todos os ficheiros obrigatórios (Projetos, Tarefas, Recursos, Alocações) para iniciar.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif not st.session_state.analysis_run:
        # Se os ficheiros estão carregados mas a análise não foi corrida (após um rerun, por exemplo)
        st.subheader("Análise Pendente")
        st.warning("Os dados estão carregados, mas a análise de pré-mineração ainda não foi executada. Clique em 'Iniciar Análise'.")

    # --- Navegação e Visualização ---
    elif st.session_state.analysis_run:
        if st.session_state.current_dashboard == "Pré-Mineração":
            pre_mining_dashboard()
        elif st.session_state.current_dashboard == "Pós-Mineração":
            post_mining_dashboard()
        elif st.session_state.current_dashboard == "Recomendação":
            recommendation_dashboard()


def pre_mining_dashboard():
    # ... (código da pre_mining_dashboard) ...
    st.subheader("Dashboard de Pré-Mineração e Análise Descritiva 📊")
    
    # 1. KPIs
    st.markdown("### 🔑 Indicadores Chave de Performance (KPIs)")
    kpis = st.session_state.tables_pre_mining.get('kpi_data', {})
    col_kpi = st.columns(4)
    with col_kpi[0]: st.metric("Total de Projetos", kpis.get('Total de Projetos', '-'))
    with col_kpi[1]: st.metric("Total de Tarefas", kpis.get('Total de Tarefas', '-'))
    with col_kpi[2]: st.metric("Total de Recursos", kpis.get('Total de Recursos', '-'))
    with col_kpi[3]: st.metric("Duração Média (dias)", f"{kpis.get('Duração Média (dias)', '-')} dias")
    
    st.markdown("---")
    
    # 2. Navegação Secundária
    st.markdown("### 🔍 Secções de Análise")
    sec1, sec2, sec3, sec4, sec5 = st.columns(5)
    
    sections = {
        "overview": "Visão Geral & KPIs",
        "performance": "Performance & Tempo",
        "workload": "Carga de Trabalho & Recursos",
        "variants": "Variantes & Loops",
        "bottlenecks": "Gargalos & Atrasos"
    }

    # Helper para criar botões de secção
    def section_button(col, section_key, section_name):
        with col:
            active_class = "active-button" if st.session_state.current_section == section_key else ""
            st.markdown(f"""
            <div class="{active_class}">
                {st.button(section_name, key=f"btn_sec_{section_key}", on_click=lambda: st.session_state.update(current_section=section_key), use_container_width=True)}
            </div>
            """, unsafe_allow_html=True)

    section_button(sec1, "overview", "Visão Geral & KPIs")
    section_button(sec2, "performance", "Performance & Tempo")
    section_button(sec3, "workload", "Carga de Trabalho & Recursos")
    section_button(sec4, "variants", "Variantes & Loops")
    section_button(sec5, "bottlenecks", "Gargalos & Atrasos")

    st.markdown("<br>", unsafe_allow_html=True)
    
    plots = st.session_state.plots_pre_mining
    tables = st.session_state.tables_pre_mining
    
    # --- CONTEÚDO DA SECÇÃO ---
    if st.session_state.current_section == "overview":
        st.subheader("Visão Geral e Distribuições")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            create_card("Matriz de Performance (Atraso vs. Custo)", "📈", chart_bytes=plots.get('performance_matrix'))
        with col_g2:
            create_card("Distribuição da Duração dos Projetos", "⏳", chart_bytes=plots.get('case_durations_boxplot'))
            
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            create_card("Top 5 Projetos de Maior Duração", "🐢", dataframe=tables.get('outlier_duration')[['project_name', 'actual_duration_days', 'days_diff']])
        with col_t2:
            create_card("Top 5 Projetos de Maior Custo", "💰", dataframe=tables.get('outlier_cost')[['project_name', 'total_actual_cost', 'cost_diff']])
            
        st.markdown("---")
        st.subheader("Estatísticas de Desempenho do Log (Lead Time & Throughput)")
        st.dataframe(tables.get('perf_stats'), use_container_width=True)

    elif st.session_state.current_section == "performance":
        st.subheader("Análise Detalhada de Performance e Tempo")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            create_card("Distribuição do Lead Time (Tempo de ponta a ponta)", "⏱️", chart_bytes=plots.get('lead_time_hist'))
        with col_p2:
            create_card("Relação Lead Time vs Throughput (Velocidade vs Duração)", "🔗", chart_bytes=plots.get('lead_time_vs_throughput'))
            
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            create_card("Tempo Médio de Execução por Atividade", "🛠️", chart_bytes=plots.get('activity_service_times'))
        with col_p4:
            create_card("Duração Média por Fase do Processo", "⚙️", chart_bytes=plots.get('cycle_time_breakdown'))

    elif st.session_state.current_section == "workload":
        st.subheader("Análise de Carga de Trabalho e Recursos")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            create_card("Top 10 Recursos por Horas Trabalhadas", "🏋️", chart_bytes=plots.get('resource_workload'))
        with col_w2:
            create_card("Recursos por Média de Tarefas por Projeto", "🎯", chart_bytes=plots.get('resource_avg_events'))
            
        col_w3, col_w4 = st.columns(2)
        with col_w3:
            create_card("Heatmap de Esforço por Recurso e Atividade", "🔥", chart_bytes=plots.get('resource_activity_matrix'))
        with col_w4:
            create_card("Top 10 Handoffs entre Recursos (Frequência)", "🤝", chart_bytes=plots.get('resource_handoffs'))
            
        st.markdown("---")
        create_card("Custo por Tipo de Recurso", "💸", chart_bytes=plots.get('cost_by_resource_type'))

    elif st.session_state.current_section == "variants":
        st.subheader("Análise de Variantes e Repetição de Trabalho")
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            create_card("Top 10 Variantes de Processo (Caminhos)", "🗺️", chart_bytes=plots.get('variants_frequency'))
        with col_v2:
            create_card("Top 10 Repetições de Trabalho (Rework Loops)", "🔄", dataframe=tables.get('rework_loops_table'))
            
        st.markdown("---")
        create_card("Tabela de Top 10 Variantes", "📜", dataframe=tables.get('variants_table'))

    elif st.session_state.current_section == "bottlenecks":
        st.subheader("Análise de Gargalos e Atrasos")
        
        st.markdown("#### Métricas de Atraso e Custo")
        kpi_delay = st.session_state.tables_pre_mining.get('cost_of_delay_kpis', {})
        col_delay = st.columns(3)
        with col_delay[0]: st.metric("Custo Total Projetos Atrasados", kpi_delay.get('Custo Total Projetos Atrasados', '-'))
        with col_delay[1]: st.metric("Atraso Médio (dias)", kpi_delay.get('Atraso Médio (dias)', '-'))
        with col_delay[2]: st.metric("Custo Médio/Dia Atraso", kpi_delay.get('Custo Médio/Dia Atraso', '-'))
        
        st.markdown("#### Análise de Espera e Execução")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            create_card("Top 15 Recursos por Tempo Médio de Espera", "🛑", chart_bytes=plots.get('bottleneck_by_resource'))
        with col_b2:
            create_card("Gargalos: Tempo de Serviço vs. Espera por Tipo", "⚖️", chart_bytes=plots.get('service_vs_wait_stacked'))
            
        col_b3, col_b4 = st.columns(2)
        with col_b3:
            create_card("Top 10 Handoffs por Custo Estimado de Espera", "💸", chart_bytes=plots.get('top_handoffs_cost'))
        with col_b4:
            create_card("Evolução do Tempo Médio de Espera", "📉", chart_bytes=plots.get('wait_time_evolution'))
            
        st.markdown("#### Impacto da Equipa")
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            create_card("Impacto do Tamanho da Equipa no Atraso", "🧑‍🤝‍🧑", chart_bytes=plots.get('delay_by_teamsize'))
        with col_i2:
            create_card("Benchmark de Throughput por Tamanho da Equipa", "🚀", chart_bytes=plots.get('throughput_benchmark_by_teamsize'))


def post_mining_dashboard():
    # ... (código da post_mining_dashboard) ...
    st.subheader("Dashboard de Descoberta e Conformidade ⚙️")

    # Botão para executar a análise de pós-mineração se ainda não foi executada
    if 'plots_post_mining' not in st.session_state or not st.session_state.plots_post_mining:
        st.warning("A análise de Pós-Mineração (Descoberta e Conformidade) ainda não foi executada. Pode demorar alguns segundos.")
        if st.button("▶️ Executar Pós-Mineração"):
            st.toast("Análise de Pós-Mineração a ser executada...")
            try:
                plots, metrics, deviation_summary, df_milestone_times = run_post_mining_analysis(
                    st.session_state.event_log_pm4py, 
                    st.session_state.df_projects, 
                    st.session_state.dfs['tasks'], # Usar o DF bruto para eventos start/complete
                    st.session_state.df_resources, 
                    st.session_state.df_full_context
                )
                st.session_state.plots_post_mining = plots
                st.session_state.metrics = metrics
                st.session_state.deviation_summary = deviation_summary
                st.session_state.df_milestone_times = df_milestone_times
                st.success("Análise de Pós-Mineração concluída!")
                st.rerun()
            except Exception as e:
                st.error(f"Erro durante a análise de Pós-Mineração: {e}")
                
        return # Sair da função se a análise não foi executada

    plots = st.session_state.plots_post_mining
    metrics = st.session_state.metrics
    
    # 1. Descoberta de Processos
    st.markdown("### 🗺️ Modelos de Processo Descobertos (Top 3 Variantes)")
    col_im, col_dfg = st.columns(2)
    
    with col_im:
        create_card("Modelo de Rede de Petri (Inductive Miner)", "🌳", chart_bytes=plots.get('model_inductive_petrinet'))
        
    with col_dfg:
        create_card("Grafo de Fluxo de Dependência (DFG - Frequência)", "➡️", chart_bytes=plots.get('model_dfg_frequency'))
        
    st.markdown("---")
    
    # 2. Qualidade do Modelo
    st.markdown("### 📈 Métricas de Qualidade do Modelo")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        create_card("Métricas de Qualidade (Inductive Miner)", "✅", chart_bytes=plots.get('metrics_inductive'))
        
    with col_m2:
        create_card("Métricas de Qualidade (Heuristics Miner)", "⚖️", chart_bytes=plots.get('metrics_heuristics'))

    st.markdown("---")
    
    # 3. Análise de Conformidade e Desvios
    st.markdown("### ❌ Análise de Conformidade e Desvios")
    
    metrics_conf = metrics.get('conformance', {})
    col_conf = st.columns(3)
    with col_conf[0]: st.metric("Conformidade Total", f"{metrics_conf.get('Conformidade (%)', '-')} %")
    with col_conf[1]: st.metric("Taxa de Handoffs", metrics_conf.get('Taxa Handoffs', '-'))
    with col_conf[2]: st.metric("Rácio de Desvios", metrics_conf.get('Desvios', '-'))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabela de Desvios
    create_card("Top 5 Casos Mais Desviantes", "⚠️", dataframe=st.session_state.deviation_summary)

    st.markdown("---")
    
    # 4. Análise de Performance Avançada
    st.markdown("### ⏱️ Performance por Fase e Marcos do Processo")
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        create_card("Boxplot do Tempo de Ciclo por Fase do Processo", "📦", chart_bytes=plots.get('phase_cycle_time_boxplot'))
    with col_p2:
        create_card("Duração para Atingir Milestones/Marcos do Processo", "🚩", chart_bytes=plots.get('milestone_time_analysis_plot'))


def recommendation_dashboard():
    # ... (código da recommendation_dashboard) ...
    st.subheader("Dashboard de Otimização e Recomendação ✨")
    
    st.info("Esta seção seria dedicada a modelos preditivos, simulações de 'what-if' e sugestões de otimização de processo. (Em desenvolvimento)")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Simulação de Otimização")
    st.warning("A otimização de processo requer modelos de simulação (como SimPy ou ProM). Os dados atuais permitem análises descritivas, mas a simulação não está implementada.")
    
    st.markdown("---")
    
    st.markdown("### 💡 Sugestões de Melhoria (Baseadas na Análise Descritiva)")
    
    suggestions = []
    
    # Sugestão 1: Bottlenecks (Recursos com maior tempo de espera)
    if 'bottleneck_by_resource' in st.session_state.plots_pre_mining:
        df_bottleneck = st.session_state.tables_pre_mining.get('bottleneck_by_resource')
        if not df_bottleneck.empty and df_bottleneck['waiting_time_days'].max() > 1:
            top_resource = df_bottleneck.iloc[0]['resource_name']
            max_wait = df_bottleneck.iloc[0]['waiting_time_days']
            suggestions.append(f"**Foco no Recurso**: O recurso '{top_resource}' apresenta o maior tempo médio de espera ({max_wait:.1f} dias). Isso sugere um potencial gargalo ou sobrecarga de trabalho. Recomenda-se balancear a carga ou aumentar a capacidade deste recurso.")
            
    # Sugestão 2: Rework Loops (Processos com alta repetição)
    if 'rework_loops_table' in st.session_state.tables_pre_mining:
        df_rework = st.session_state.tables_pre_mining.get('rework_loops_table')
        if not df_rework.empty and df_rework.iloc[0]['frequency'] > 5: # Um limiar arbitrário
            top_rework = df_rework.iloc[0]['rework_loop']
            freq = df_rework.iloc[0]['frequency']
            suggestions.append(f"**Redução de Rework**: O loop de retrabalho mais frequente é '{top_rework}' (ocorre {freq} vezes). Investigar a causa-raiz deste retrabalho (falha de qualidade, comunicação, ou dependência) pode reduzir o tempo de ciclo.")

    # Sugestão 3: Conformidade
    if 'conformance' in st.session_state.metrics:
        conf_metric = st.session_state.metrics.get('conformance', {})
        conf_pct = float(conf_metric.get('Conformidade (%)', '100').replace(' %', ''))
        if conf_pct < 95:
             suggestions.append(f"**Melhoria de Conformidade**: A conformidade é de apenas {conf_pct:.2f}%. Isso indica que {100 - conf_pct:.2f}% dos casos não seguem o processo-modelo. Analisar os Top 5 casos desviantes (tabela na aba anterior) para entender as exceções e padronizar o processo.")

    if suggestions:
        for s in suggestions:
            st.markdown(f"**-** {s}")
    else:
        st.info("Com base nos dados atuais, não foram detetadas grandes anomalias que exijam sugestões imediatas. Por favor, execute a análise de Pós-Mineração para mais detalhes.")
        

def settings_page():
    # ... (código da settings_page) ...
    st.title("⚙️ Configurações da Plataforma")
    st.markdown("---")
    
    st.subheader("Gestão de Dados")
    
    if st.button("Limpar Dados Carregados e Resultados da Análise", use_container_width=True, type="primary"):
        keys_to_delete = ['dfs', 'analysis_run', 'plots_pre_mining', 'plots_post_mining', 'tables_pre_mining', 'metrics', 'event_log_pm4py', 'df_projects', 'df_tasks', 'df_resources', 'df_full_context', 'deviation_summary', 'df_milestone_times']
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        # Resetar o estado dfs
        st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
        st.session_state.analysis_run = False
        st.session_state.current_dashboard = "Pré-Mineração"
        st.success("Todos os dados carregados e resultados de análise foram limpos. Redirecionando para o Dashboard.")
        st.rerun()

    st.markdown("---")
    
    st.subheader("Tema e Estilo")
    st.info("O tema da aplicação é gerido por CSS injetado e variáveis `--root` e não pode ser alterado dinamicamente através desta interface.")

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
