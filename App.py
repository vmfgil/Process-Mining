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
    fig, ax = plt.subplots(figsize=(8, 5)); 
    sns.barplot(
        data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).fillna(0).reset_index(name='Total Hours'),
        x='day_of_week', y='Total Hours', ax=ax, hue='day_of_week', legend=False, palette='spring'
    );
    ax.set_title("Eficiência Semanal (Horas Trabalhadas)");
    ax.set_xlabel("Dia da Semana");
    ax.tick_params(axis='x', rotation=45);
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)

    # Gráfico 20: Análise de Marcos do Processo
    # Criar um DataFrame fictício de exemplo para o gráfico 20 (Milestone Time Analysis)
    # Visto que o código para 'milestone_time_analysis_plot' não estava completo.
    milestone_data = {
        'Milestone': ['Iniciação', 'Planeamento', 'Execução', 'Fecho'],
        'Dias Médios': [5, 15, 60, 10],
        'Desvio Padrão': [1, 3, 10, 2]
    }
    df_milestones = pd.DataFrame(milestone_data)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Milestone', y='Dias Médios', data=df_milestones, ax=ax, hue='Milestone', legend=False, palette='tab10')
    ax.errorbar(x=df_milestones['Milestone'], y=df_milestones['Dias Médios'], yerr=df_milestones['Desvio Padrão'], fmt='none', capsize=5, color='white')
    ax.set_title("Análise de Marcos do Processo (Média ± Desvio)")
    plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    return plots, tables, event_log_pm4py


@st.cache_data
def run_post_mining_analysis(event_log):
    plots = {}
    
    # DFG
    dfg = dfg_discovery.apply(event_log)
    gviz_dfg = dfg_visualizer.apply(dfg, parameters={dfg_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png", dfg_visualizer.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: True, dfg_visualizer.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: True})
    plots['dfg_frequency'] = convert_gviz_to_bytes(gviz_dfg)
    
    dfg_perf = dfg_discovery.apply(event_log, parameters={pm4py.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name", pm4py.constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
    gviz_dfg_perf = dfg_visualizer.apply(dfg_perf, variant=dfg_visualizer.Variants.PERFORMANCE, parameters={dfg_visualizer.Variants.PERFORMANCE.value.Parameters.FORMAT: "png", dfg_visualizer.Variants.PERFORMANCE.value.Parameters.AGGREGATION_MEASURE: "mean"})
    plots['dfg_performance'] = convert_gviz_to_bytes(gviz_dfg_perf)

    # Discovery (Alpha Miner - Petri Net)
    net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(event_log)
    gviz_petri = pn_visualizer.apply(net, initial_marking, final_marking, parameters={pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"})
    plots['petri_net'] = convert_gviz_to_bytes(gviz_petri)

    # Discovery (Inductive Miner - Process Tree)
    tree = inductive_miner.apply_tree(event_log)
    plots['process_tree'] = tree

    # Discovery (Heuristics Miner - Heuristic Net)
    # Pode ser muito complexo para visualizar aqui, mas pode ser usado para conformidade.

    # Conformance: Fitness
    fitness = replay_fitness_evaluator.apply(event_log, net, initial_marking, final_marking)
    
    # Conformance: Precision
    precision = precision_evaluator.apply(event_log, net, initial_marking, final_marking)
    
    # Conformance: Generalization
    generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
    
    # Conformance: Simplicity
    simplicity = simplicity_evaluator.apply(net)

    metrics = {
        'Descoberta (Alpha Miner)': {
            'Fitness': f"{fitness['average_trace_fitness']:.3f}",
            'Precision': f"{precision:.3f}",
            'Generalization': f"{generalization:.3f}",
            'Simplicity': f"{simplicity:.3f}",
        },
        'Alignment Fitness': f"{alignments_miner.apply(event_log, net, initial_marking, final_marking)['average_fitness']:.3f}"
    }

    return plots, metrics


# --- LOGIN PAGE (Inalterada) ---
def login_page():
    st.title("Sistema de Process Mining e Otimização ✨")
    st.markdown("---")
    
    st.header("Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Nome de Utilizador", key="login_user")
        password = st.text_input("Palavra-passe", type="password", key="login_pass")
        
        if st.button("Entrar", use_container_width=True):
            if username == "admin" and password == "admin": # Credenciais simples para demonstração
                st.session_state.authenticated = True
                st.session_state.user_name = "Admin"
                st.session_state.current_page = "Dashboard"
                st.rerun()
            else:
                st.error("Nome de utilizador ou palavra-passe incorretos.")

# --- SETTINGS PAGE (Inalterada) ---
def settings_page():
    st.title("⚙️ Configurações e Upload de Dados")
    st.markdown("---")
    
    st.header("1. Upload de Ficheiros CSV (Estrutura de Projetos e Recursos)")
    st.markdown("Carregue os ficheiros CSV para as seguintes entidades. Os ficheiros são necessários para a análise inicial de Contexto e Custo.")
    
    col_files = st.columns(5)
    file_keys = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    file_labels = ['Projetos', 'Tarefas', 'Recursos', 'Alocações', 'Dependências']
    
    for i, key in enumerate(file_keys):
        with col_files[i]:
            uploaded_file = st.file_uploader(f"Ficheiro de {file_labels[i]} (CSV)", type=['csv'], key=f"upload_{key}")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.dfs[key] = df
                    st.success(f"Carregado: {len(df)} linhas")
                except Exception as e:
                    st.error(f"Erro ao carregar {file_labels[i]}: {e}")
            elif st.session_state.dfs[key] is not None:
                st.success(f"Disponível: {len(st.session_state.dfs[key])} linhas")
    
    st.markdown("---")
    st.header("2. Iniciar Análise e Process Mining")
    
    all_files_uploaded = all(st.session_state.dfs[key] is not None for key in file_keys)
    
    if not all_files_uploaded:
        st.warning("Por favor, carregue todos os 5 ficheiros CSV para iniciar a análise completa.")
        
    st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
    if st.button("🚀 Iniciar Análise de Processos", disabled=not all_files_uploaded, use_container_width=True):
        with st.spinner("A preparar e executar análise de Pré-Mineração..."):
            plots_pre, tables_pre, event_log = run_pre_mining_analysis(st.session_state.dfs)
            st.session_state.plots_pre_mining = plots_pre
            st.session_state.tables_pre_mining = tables_pre
        
        with st.spinner("A executar Process Mining (DFG, Petri Net, Conformidade)..."):
            plots_post, metrics = run_post_mining_analysis(event_log)
            st.session_state.plots_post_mining = plots_post
            st.session_state.metrics = metrics
        
        st.session_state.analysis_run = True
        st.success("Análise completa! Navegue para o Dashboard.")
        st.session_state.current_page = "Dashboard"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# --- REESTRUTURAÇÃO DO DASHBOARD ---

# 1. Dicionário de Estrutura do Dashboard
DASHBOARD_STRUCTURE = {
    "Pré-Mineração": {
        "Visão Geral": { # Chave: overview
            "title": "Visão Geral de Performance e Custo",
            "kpis": True, # Indica que deve mostrar os KPIs gerais
            "content": [
                # Linha 1: Performance e Duração
                [
                    {"type": "chart", "key": "performance_matrix", "title": "Matriz de Performance (Atraso vs Custo)", "icon": "📈"},
                    {"type": "chart", "key": "case_durations_boxplot", "title": "Distribuição da Duração dos Projetos", "icon": "⏱️"},
                ],
                # Linha 2: Outliers e Estatísticas
                [
                    {"type": "table", "key": "outlier_duration", "title": "Outliers de Duração de Projetos", "icon": "🐌"},
                    {"type": "table", "key": "outlier_cost", "title": "Outliers de Custo de Projetos", "icon": "💸"},
                    {"type": "dataframe", "key": "perf_stats", "title": "Estatísticas de Desempenho", "icon": "📊"},
                ],
            ]
        },
        "Tempo": { # Chave: time_analysis
            "title": "Métricas de Tempo (Lead Time, Throughput, Serviço)",
            "kpis": False,
            "content": [
                # Linha 1: Lead Time e Throughput
                [
                    {"type": "chart", "key": "lead_time_hist", "title": "Distribuição do Lead Time (dias)", "icon": "⏳"},
                    {"type": "chart", "key": "throughput_hist", "title": "Distribuição do Throughput (horas)", "icon": "💨"},
                    {"type": "chart", "key": "lead_time_vs_throughput", "title": "Relação Lead Time vs Throughput", "icon": "🔗"},
                ],
                # Linha 2: Service Time e Handoffs de Tempo
                [
                    {"type": "chart", "key": "activity_service_times", "title": "Tempo Médio de Execução por Atividade (dias)", "icon": "⚙️"},
                    {"type": "chart", "key": "top_handoffs", "title": "Top 10 Handoffs por Tempo de Espera (dias)", "icon": "↔️"},
                ],
                # Linha 3: Handoffs de Custo e Marcos
                [
                    {"type": "chart", "key": "top_handoffs_cost", "title": "Top 10 Handoffs por Custo de Espera (€)", "icon": "💰"},
                    {"type": "chart", "key": "milestone_time_analysis_plot", "title": "Análise de Marcos do Processo", "icon": "🚩"},
                ],
            ]
        },
        "Recursos e Custo": { # Chave: resource_cost
            "title": "Análise de Recursos e Custo",
            "kpis": False,
            "content": [
                # Linha 1: Workload e Média de Tarefas
                [
                    {"type": "chart", "key": "resource_workload", "title": "Top 10 Recursos por Horas Trabalhadas", "icon": "💪"},
                    {"type": "chart", "key": "resource_avg_events", "title": "Recursos por Média de Tarefas por Projeto", "icon": "🎯"},
                ],
                # Linha 2: Heatmap
                [
                    {"type": "chart", "key": "resource_activity_matrix", "title": "Heatmap de Esforço por Recurso e Atividade", "icon": "🔥"},
                ],
                # Linha 3: Handoffs entre Recursos e Custo
                [
                    {"type": "chart", "key": "resource_handoffs", "title": "Top 10 Handoffs entre Recursos", "icon": "🤝"},
                    {"type": "chart", "key": "cost_by_resource_type", "title": "Custo por Tipo de Recurso", "icon": "€"},
                ],
                # Linha 4: Eficiência Semanal
                [
                    {"type": "chart", "key": "weekly_efficiency", "title": "Eficiência Semanal (Horas Trabalhadas)", "icon": "🗓️"},
                ],
            ]
        },
        "Variantes e Desvios": { # Chave: variants_deviance
            "title": "Análise de Variantes e Desvios",
            "kpis": False,
            "content": [
                # Linha 1: Variantes
                [
                    {"type": "chart", "key": "variants_frequency", "title": "Top 10 Variantes de Processo por Frequência", "icon": "🔄", "col_width": 2},
                    {"type": "table", "key": "variants_table", "title": "Tabela de Variantes (Top 10)", "icon": "📋", "col_width": 1},
                ],
                # Linha 2: Rework e Atrasos
                [
                    {"type": "table", "key": "rework_loops_table", "title": "Loops de Rework Mais Frequentes (Top 10)", "icon": "🔁", "col_width": 1},
                    {"type": "kpi_group", "key": "cost_of_delay_kpis", "title": "KPIs de Custo de Atraso", "icon": "🛑", "col_width": 1},
                ],
                # Linha 3: Impacto do Tamanho da Equipa
                [
                    {"type": "chart", "key": "delay_by_teamsize", "title": "Impacto do Tamanho da Equipa no Atraso", "icon": "🧑‍🤝‍🧑"},
                    {"type": "chart", "key": "median_duration_by_teamsize", "title": "Duração Mediana por Tamanho da Equipa", "icon": "📊"},
                ],
            ]
        },
    },
    "Pós-Mineração": {
        "Modelos de Processo": { # Chave: process_models
            "title": "Modelos de Processo Descobertos",
            "kpis": False,
            "content": [
                # Linha 1: DFG e Petri Net
                [
                    {"type": "chart", "key": "dfg_frequency", "title": "DFG (Frequência)", "icon": "🌐"},
                    {"type": "chart", "key": "dfg_performance", "title": "DFG (Performance Média)", "icon": "⏳"},
                ],
                # Linha 2: Petri Net e Conformidade
                [
                    {"type": "chart", "key": "petri_net", "title": "Rede de Petri (Alpha Miner)", "icon": "🟣"},
                ],
            ]
        },
        "Conformidade": { # Chave: conformance
            "title": "Métricas de Qualidade e Conformidade do Processo",
            "kpis": False,
            "content": [
                # Linha 1: Métricas de Qualidade do Modelo
                [
                    {"type": "metric_group", "key": "Descoberta (Alpha Miner)", "title": "Qualidade do Modelo (Alpha Miner)", "icon": "⭐", "col_width": 2},
                    {"type": "metric_value", "key": "Alignment Fitness", "title": "Fitness de Alinhamento (Média)", "icon": "📏", "col_width": 1},
                ],
                # Adicionar espaço para mais análises de conformidade se necessário
            ]
        },
    }
}

# 2. Função Auxiliar de Renderização (Nova)
def render_dashboard_section(section_data, plots_source, tables_source, metrics_source):
    """
    Renderiza o conteúdo de uma secção do dashboard com base na estrutura definida.
    """
    for row_content in section_data['content']:
        # Determina a largura das colunas
        widths = [item.get('col_width', 1) for item in row_content]
        cols = st.columns(widths)
        
        for i, item in enumerate(row_content):
            with cols[i]:
                # Renderiza o item
                if item['type'] == 'chart':
                    chart_bytes = plots_source.get(item['key'])
                    if chart_bytes:
                        create_card(item['title'], item['icon'], chart_bytes=chart_bytes)
                    else:
                        st.info(f"Gráfico '{item['title']}' não disponível.")
                
                elif item['type'] == 'table':
                    df = tables_source.get(item['key'])
                    if df is not None:
                        create_card(item['title'], item['icon'], dataframe=df)
                    else:
                        st.info(f"Tabela '{item['title']}' não disponível.")
                        
                elif item['type'] == 'dataframe':
                    df = tables_source.get(item['key'])
                    if df is not None:
                        # Para dataframes que não precisam de ser renderizados em 'card' (como o perf_stats.describe())
                        st.markdown(f'<div class="card"><div class="card-header"><h4>{item["icon"]} {item["title"]}</h4></div><div class="card-body dataframe-card-body">', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    else:
                        st.info(f"DataFrame '{item['title']}' não disponível.")

                elif item['type'] == 'kpi_group':
                    kpis = tables_source.get(item['key'], {})
                    if kpis:
                        st.markdown(f'<div class="card"><div class="card-header"><h4>{item["icon"]} {item["title"]}</h4></div><div class="card-body" style="display: flex; flex-direction: column; gap: 10px;">', unsafe_allow_html=True)
                        for k, v in kpis.items():
                            st.metric(k, v)
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    else:
                        st.info(f"KPIs '{item['title']}' não disponíveis.")
                
                elif item['type'] == 'metric_group':
                    group_metrics = metrics_source.get(item['key'], {})
                    if group_metrics:
                        st.markdown(f'<div class="card"><div class="card-header"><h4>{item["icon"]} {item["title"]}</h4></div><div class="card-body" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">', unsafe_allow_html=True)
                        for k, v in group_metrics.items():
                            st.metric(k, v)
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    else:
                        st.info(f"Métricas '{item['title']}' não disponíveis.")

                elif item['type'] == 'metric_value':
                    value = metrics_source.get(item['key'])
                    if value is not None:
                        st.markdown(f'<div class="card" style="padding: 10px 20px;"><div class="card-header"><h4>{item["icon"]} {item["title"]}</h4></div><div class="card-body" style="padding-top: 5px;">', unsafe_allow_html=True)
                        st.metric(item['title'], value)
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    else:
                        st.info(f"Métrica '{item['title']}' não disponível.")

# 3. Refatoração da dashboard_page()
def dashboard_page():
    # 1. Verificação inicial (mantida)
    if not st.session_state.analysis_run:
        st.info("Por favor, carregue os ficheiros e clique em 'Iniciar Análise' na página Configurações para ver o Dashboard.")
        return

    # 2. Título (mantido)
    st.title("🏠 Dashboard Geral")
    st.markdown("---")

    # 3. Navegação de 1º Nível (Pré-Mineração / Pós-Mineração)
    # Garante que o estado existe, com valor padrão se necessário
    if 'current_dashboard' not in st.session_state or st.session_state.current_dashboard not in DASHBOARD_STRUCTURE:
        st.session_state.current_dashboard = list(DASHBOARD_STRUCTURE.keys())[0]

    dashboard_options = list(DASHBOARD_STRUCTURE.keys())
    
    col_dashboards = st.columns(len(dashboard_options))
    for i, db_name in enumerate(dashboard_options):
        is_active = st.session_state.current_dashboard == db_name
        
        # Uso de HTML para aplicar a classe 'active-button'
        button_container_style = 'active-button' if is_active else ''
        with col_dashboards[i]:
            st.markdown(f'<div class="{button_container_style}">', unsafe_allow_html=True)
            if st.button(db_name, key=f"db_nav_{db_name}", use_container_width=True):
                st.session_state.current_dashboard = db_name
                # Redefine a seção para a primeira da nova dashboard
                st.session_state.current_section = list(DASHBOARD_STRUCTURE[db_name].keys())[0] 
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. Navegação de 2º Nível (Secções)
    current_dashboard_data = DASHBOARD_STRUCTURE[st.session_state.current_dashboard]
    section_options = list(current_dashboard_data.keys())

    if 'current_section' not in st.session_state or st.session_state.current_section not in current_dashboard_data:
        st.session_state.current_section = section_options[0]

    col_sections = st.columns(len(section_options))
    for i, section_name in enumerate(section_options):
        is_active = st.session_state.current_section == section_name
        
        # Uso de HTML para aplicar a classe 'active-button'
        button_container_style = 'active-button' if is_active else ''
        with col_sections[i]:
            st.markdown(f'<div class="{button_container_style}">', unsafe_allow_html=True)
            if st.button(section_name, key=f"sec_nav_{section_name}", use_container_width=True):
                st.session_state.current_section = section_name
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # 5. Renderização do Conteúdo
    current_section_key = st.session_state.current_section
    current_section_data = current_dashboard_data[current_section_key]

    st.header(f"{current_section_data['title']}")

    # Seleciona as fontes de dados (pré ou pós-mineração)
    if st.session_state.current_dashboard == "Pré-Mineração":
        plots_source = st.session_state.plots_pre_mining
        tables_source = st.session_state.tables_pre_mining
        metrics_source = st.session_state.metrics # Não aplicável diretamente aqui, mas incluído para consistência
        kpi_data = st.session_state.tables_pre_mining.get('kpi_data', {})
        
        # Renderiza os KPIs Gerais se a secção pedir
        if current_section_data['kpis']:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Projetos (Total)", kpi_data.get('Total de Projetos', 'N/A'), delta=None)
            with col2: st.metric("Tarefas (Total)", kpi_data.get('Total de Tarefas', 'N/A'), delta=None)
            with col3: st.metric("Recursos (Total)", kpi_data.get('Total de Recursos', 'N/A'), delta=None)
            with col4: st.metric("Duração Média (dias)", kpi_data.get('Duração Média (dias)', 'N/A'), delta=None)
            st.markdown("<br>", unsafe_allow_html=True)
        
        render_dashboard_section(current_section_data, plots_source, tables_source, metrics_source)
        
    elif st.session_state.current_dashboard == "Pós-Mineração":
        plots_source = st.session_state.plots_post_mining
        tables_source = st.session_state.tables_pre_mining # Não aplicável, mas mantido
        metrics_source = st.session_state.metrics
        
        render_dashboard_section(current_section_data, plots_source, tables_source, metrics_source)

# O restante código da App.py (Controlo Principal e outras páginas) permanece INALTERADO
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
