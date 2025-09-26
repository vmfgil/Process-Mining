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
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, hue='day_of_week', legend=False, palette='viridis'); ax.set_title("Eficiência Semanal (Horas Trabalhadas)"); plt.xticks(rotation=45)
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    
    df_tasks_analysis = df_tasks.copy(); df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True); df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].clip(lower=0)
    
    # Gráfico 20: Tempo de Espera entre Tarefas (Série Temporal)
    waiting_time_series = df_tasks_analysis.set_index('start_date').resample('M')['waiting_time_days'].mean().dropna()
    fig, ax = plt.subplots(figsize=(12, 5)); waiting_time_series.plot(kind='line', ax=ax, color='#2563EB'); ax.set_title("Tempo de Espera Médio entre Tarefas (Mensal)"); ax.set_xlabel("Mês"); ax.set_ylabel("Dias"); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plots['waiting_time_series'] = convert_fig_to_bytes(fig)
    
    # Gráfico 21: Service Time das Tarefas (Série Temporal)
    service_time_series = df_tasks_analysis.set_index('start_date').resample('M')['service_time_days'].mean().dropna()
    fig, ax = plt.subplots(figsize=(12, 5)); service_time_series.plot(kind='line', ax=ax, color='#06B6D4'); ax.set_title("Service Time Médio das Tarefas (Mensal)"); ax.set_xlabel("Mês"); ax.set_ylabel("Dias"); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plots['service_time_series'] = convert_fig_to_bytes(fig)
    
    # Gráfico 22: Atividade Mensal
    monthly_activity = df_projects['completion_month'].value_counts().sort_index().reset_index()
    monthly_activity.columns = ['month', 'count']
    fig, ax = plt.subplots(figsize=(12, 5)); sns.barplot(data=monthly_activity, x='month', y='count', ax=ax, hue='month', legend=False, palette='rocket'); ax.set_title("Projetos Concluídos por Mês"); ax.set_xlabel("Mês"); ax.set_ylabel("Nº de Projetos")
    plots['monthly_activity_plot'] = convert_fig_to_bytes(fig)

    st.session_state.plots_pre_mining = plots
    st.session_state.tables_pre_mining = tables
    st.session_state.event_log_pm4py = event_log_pm4py
    
    return event_log_pm4py, plots, tables

@st.cache_data
def run_post_mining_analysis(event_log_pm4py):
    plots = {}
    metrics = {}

    # Descoberta de Modelos
    dfg = dfg_discovery.apply(event_log_pm4py)
    net, initial_marking, final_marking = inductive_miner.apply(event_log_pm4py)
    # net_heuristics, initial_marking_heuristics, final_marking_heuristics = heuristics_miner.apply(event_log_pm4py)

    # Visualizações
    dfg_gviz = dfg_visualizer.apply(dfg, log=event_log_pm4py, variant=dfg_visualizer.Variants.FREQUENCY)
    plots['dfg_frequency'] = convert_gviz_to_bytes(dfg_gviz)

    parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
    net_gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters, variant=pn_visualizer.Variants.FREQUENCY, log=event_log_pm4py)
    plots['petri_net_frequency'] = convert_gviz_to_bytes(net_gviz)

    # Conformidade
    fitness = replay_fitness_evaluator.apply(event_log_pm4py, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    precision = precision_evaluator.apply(event_log_pm4py, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    generalization = generalization_evaluator.apply(event_log_pm4py, net, initial_marking, final_marking)
    simplicity = simplicity_evaluator.apply(net)

    metrics['fitness'] = f"{fitness['average_trace_fitness'] * 100:.2f}%"
    metrics['precision'] = f"{precision:.2f}"
    metrics['generalization'] = f"{generalization:.2f}"
    metrics['simplicity'] = f"{simplicity:.2f}"
    
    # Análise de Variantes para a página de Variantes
    variants = variants_filter.get_variants(event_log_pm4py)
    variants_count = {str(k): len(v) for k, v in variants.items()}
    df_variants = pd.DataFrame(variants_count.items(), columns=['Variant', 'Frequency'])
    df_variants['Percentage'] = (df_variants['Frequency'] / df_variants['Frequency'].sum()) * 100
    df_variants = df_variants.sort_values(by='Frequency', ascending=False)
    
    # Gráfico de Variantes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Frequency', y=df_variants['Variant'].head(10).astype(str), data=df_variants.head(10), ax=ax, hue='Variant', legend=False, palette='viridis')
    ax.set_title('Top 10 Variantes de Processo (PM4PY)')
    plots['pm4py_variants_plot'] = convert_fig_to_bytes(fig)
    
    # Análise de Conformidade por Caso (Alinhamentos)
    aligned_traces = alignments_miner.apply_log(event_log_pm4py, net, initial_marking, final_marking)
    cost_data = []
    for trace in aligned_traces:
        if 'alignment_cost' in trace:
            cost_data.append(trace['alignment_cost'])
    
    if cost_data:
        df_costs = pd.DataFrame({'Cost': cost_data})
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_costs['Cost'], bins=20, kde=True, ax=ax, color='#FBBF24')
        ax.set_title('Distribuição dos Custos de Alinhamento')
        plots['alignment_costs_hist'] = convert_fig_to_bytes(fig)
    else:
        plots['alignment_costs_hist'] = None


    st.session_state.plots_post_mining = plots
    st.session_state.metrics = metrics
    st.session_state.tables_post_mining = {'variants_table_pm4py': df_variants}
    
    return plots, metrics

# --- PÁGINAS GERAIS ---
def login_page():
    st.title("Início de Sessão")
    st.markdown("---")
    
    with st.form("login_form"):
        username = st.text_input("Nome de Utilizador")
        password = st.text_input("Palavra-passe", type="password")
        submitted = st.form_submit_button("Entrar")
        
        if submitted:
            # Lógica de autenticação simples
            if username == "admin" and password == "1234":
                st.session_state.authenticated = True
                st.session_state.user_name = "Administrador"
                st.session_state.current_page = "Dashboard"
                st.rerun()
            else:
                st.error("Credenciais inválidas. Tente 'admin' e '1234'.")

def upload_page(from_dashboard=False):
    st.title("⚙️ Configuração & Upload de Dados")
    st.markdown("---")
    
    st.markdown("""
    ### 📂 Estrutura de Ficheiros Requerida
    Por favor, carregue os ficheiros CSV com os seguintes nomes para análise:
    - **`projects.csv`**
    - **`tasks.csv`**
    - **`resources.csv`**
    - **`resource_allocations.csv`**
    - **`dependencies.csv`**
    """)
    
    col1, col2 = st.columns([1, 2])
    
    required_files = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    
    with col1:
        st.subheader("Carregar Ficheiros")
        uploaded_files = st.file_uploader(
            "Selecione todos os ficheiros CSV",
            type="csv",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    file_key = file.name.replace('.csv', '')
                    if file_key in st.session_state.dfs:
                        st.session_state.dfs[file_key] = df
                        st.success(f"Ficheiro **`{file.name}`** carregado com sucesso.")
                    else:
                        st.warning(f"Ficheiro **`{file.name}`** carregado, mas a chave '{file_key}' não é esperada.")
                except Exception as e:
                    st.error(f"Erro ao ler {file.name}: {e}")
    
    with col2:
        st.subheader("Estado Atual")
        all_loaded = True
        for key in required_files:
            icon = "✅" if st.session_state.dfs[key] is not None else "❌"
            st.markdown(f"{icon} **`{key}.csv`** carregado: {'Sim' if st.session_state.dfs[key] is not None else 'Não'}")
            if st.session_state.dfs[key] is None:
                all_loaded = False
                
        if all_loaded:
            st.success("Todos os ficheiros necessários foram carregados.")
            
            # Botão de Análise de Grande Destaque
            st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
            if st.button("🚀 Iniciar Análise de Dados e Processos", key="run_analysis_btn", use_container_width=True):
                st.info("A correr as análises de Pré-Mineração. Por favor, aguarde...")
                
                # --- EXECUÇÃO DAS ANÁLISES ---
                try:
                    # 1. Pré-Mineração
                    event_log, plots_pre, tables_pre = run_pre_mining_analysis(st.session_state.dfs)
                    st.session_state.plots_pre_mining = plots_pre
                    st.session_state.tables_pre_mining = tables_pre
                    st.session_state.event_log_pm4py = event_log
                    
                    # 2. Process Mining (pode demorar um pouco mais)
                    plots_post, metrics = run_post_mining_analysis(event_log)
                    st.session_state.plots_post_mining = plots_post
                    st.session_state.metrics = metrics
                    
                    st.session_state.analysis_run = True
                    st.session_state.current_page = "Dashboard"
                    st.success("Análise concluída com sucesso! Redirecionando para o Dashboard.")
                    st.rerun()
                except Exception as e:
                    st.session_state.analysis_run = False
                    st.error(f"Erro durante a execução da análise: {e}")
                    
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Se chamado do dashboard, permite voltar ao dashboard se não há ficheiros, mas análise já correu
        if from_dashboard and st.session_state.analysis_run:
            if st.button("Voltar ao Dashboard", key="back_to_dash"):
                st.session_state.current_page = "Dashboard"
                st.rerun()

def settings_page():
    st.title("⚙️ Configurações da Aplicação")
    st.markdown("---")
    
    st.subheader("Estado da Sessão")
    st.json({k: v if k not in ['dfs', 'event_log_pm4py', 'plots_pre_mining', 'plots_post_mining'] else f"Conteúdo de {k}" for k, v in st.session_state.items()})

    if st.button("Limpar Dados de Análise e Sessão"):
        for key in list(st.session_state.keys()):
            if key not in ['authenticated', 'user_name']: 
                del st.session_state[key]
        st.session_state.current_page = "Dashboard" # Irá para o upload na próxima iteração
        st.session_state.current_dashboard = "Pré-Mineração"
        st.session_state.current_section = "overview"
        st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
        st.session_state.analysis_run = False
        st.rerun()

# --- PÁGINAS DE PROCESS MINING ---
def petri_net_page(plots):
    st.markdown("### Rede de Petri Descoberta")
    
    net_bytes = plots.get('petri_net_frequency')
    
    if net_bytes:
        b64_image = base64.b64encode(net_bytes.getvalue()).decode()
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>🌐 Rede de Petri</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Modelo de Rede de Petri indisponível. Corra a análise de Process Mining primeiro.")

def dfg_page(plots):
    st.markdown("### Grafo de Fluxo Direto (DFG)")

    dfg_bytes = plots.get('dfg_frequency')
    
    if dfg_bytes:
        b64_image = base64.b64encode(dfg_bytes.getvalue()).decode()
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>🗺️ DFG - Frequência</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Modelo DFG indisponível. Corra a análise de Process Mining primeiro.")

def variants_page(plots):
    st.markdown("### Análise de Variantes de Processo")
    
    col1, col2 = st.columns(2)
    
    # Tabela de Variantes
    df_variants = st.session_state.tables_post_mining.get('variants_table_pm4py')
    if df_variants is not None:
        with col1:
            create_card("Top Variantes de Processo", "📜", dataframe=df_variants.head(15).reset_index(drop=True))
        
        # Gráfico de Variantes
        with col2:
            chart_bytes = plots.get('pm4py_variants_plot')
            if chart_bytes:
                 create_card("Frequência das Top Variantes", "📊", chart_bytes=chart_bytes)
    else:
        st.warning("Dados de variantes indisponíveis. Corra a análise de Process Mining primeiro.")


def conformance_page(plots, tables, metrics):
    st.markdown("### Métricas de Conformidade (Qualidade do Modelo)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Fitness (Ajuste)", metrics.get('fitness', 'N/A'), help="A capacidade do modelo de explicar o log de eventos.")
    with col2: st.metric("Precisão", metrics.get('precision', 'N/A'), help="Medida de quão restrito é o modelo (evita comportamentos invisíveis).")
    with col3: st.metric("Generalização", metrics.get('generalization', 'N/A'), help="Medida da capacidade do modelo de aceitar traços ligeiramente diferentes do log.")
    with col4: st.metric("Simplicidade", metrics.get('simplicity', 'N/A'), help="Quão simples é o modelo (menor número de nós e arcos).")
    
    st.markdown("---")
    st.markdown("### Custos de Alinhamento")

    cost_bytes = plots.get('alignment_costs_hist')
    
    if cost_bytes:
        st.columns([1, 0.1])[0].markdown(create_card("Distribuição dos Custos de Alinhamento", "💰", chart_bytes=cost_bytes), unsafe_allow_html=True)
    else:
        st.info("Não foi possível calcular ou visualizar o histograma dos custos de alinhamento. Isto pode ocorrer se o algoritmo de alinhamento não tiver dados suficientes ou se houver erros de cálculo.")


# --- 3. ESTRUTURA DO DASHBOARD (NOVO) ---
DASHBOARD_STRUCTURE = {
    "Pré-Mineração": {
        "overview": {"icon": "📊", "title": "Visão Geral"},
        "case_performance": {"icon": "⏱️", "title": "Performance do Caso"},
        "activity_performance": {"icon": "⚙️", "title": "Performance da Atividade"},
        "resource_analysis": {"icon": "👥", "title": "Análise de Recursos"},
        "variability_analysis": {"icon": "🔄", "title": "Variabilidade e Rework"},
        "financial_impact": {"icon": "💰", "title": "Impacto Financeiro"},
        "team_dynamics": {"icon": "🤝", "title": "Dinâmicas de Equipa"},
        "time_series": {"icon": "📅", "title": "Séries Temporais"},
    },
    "Process Mining": {
        "petri_net": {"icon": "🌐", "title": "Rede de Petri"},
        "dfg": {"icon": "🗺️", "title": "DFG"},
        "variants": {"icon": "🔗", "title": "Variantes de Processo"},
        "conformance": {"icon": "✅", "title": "Conformidade"},
    }
}


# --- FUNÇÃO AUXILIAR PARA RENDERIZAR SECÇÕES (NOVO) ---
def render_dashboard_section(section_key, plots, tables, metrics):
    """Renderiza o conteúdo específico de uma secção do dashboard."""
    
    # Renderizar a secção correta com base no estado e na estrutura
    if st.session_state.current_dashboard == "Pré-Mineração":
        if section_key == "overview":
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total de Projetos", tables.get('kpi_data', {}).get('Total de Projetos'))
            with col2: st.metric("Total de Tarefas", tables.get('kpi_data', {}).get('Total de Tarefas'))
            with col3: st.metric("Total de Recursos", tables.get('kpi_data', {}).get('Total de Recursos'))
            with col4: st.metric("Duração Média (dias)", tables.get('kpi_data', {}).get('Duração Média (dias)'))
            
            st.markdown("### Matriz de Performance (Desvio de Tempo vs. Custo)")
            st.columns([1, 0.1])[0].markdown(create_card("Performance de Projetos", "🎯", chart_bytes=plots.get('performance_matrix')), unsafe_allow_html=True)
            
            col_dur, col_cost = st.columns(2)
            with col_dur: create_card("Top 5 Projetos por Maior Duração", "⏳", dataframe=tables.get('outlier_duration'))
            with col_cost: create_card("Top 5 Projetos por Maior Custo", "💸", dataframe=tables.get('outlier_cost'))

        elif section_key == "case_performance":
            st.markdown("### Distribuições de Tempo de Caso")
            col1, col2 = st.columns(2)
            with col1: create_card("Distribuição da Duração dos Projetos", "📦", chart_bytes=plots.get('case_durations_boxplot'))
            with col2: create_card("Distribuição do Lead Time (dias)", "📈", chart_bytes=plots.get('lead_time_hist'))

            st.markdown("### Análise de Throughput e Correlação")
            col3, col4 = st.columns(2)
            with col3: create_card("Distribuição do Throughput (horas)", "⏳", chart_bytes=plots.get('throughput_hist'))
            with col4: create_card("Boxplot do Throughput", "📊", chart_bytes=plots.get('throughput_boxplot'))
            
            st.columns([1, 0.1])[0].markdown(create_card("Relação Lead Time vs Throughput", "🔀", chart_bytes=plots.get('lead_time_vs_throughput')), unsafe_allow_html=True)
            
        elif section_key == "activity_performance":
            st.markdown("### Tempos de Execução e Handoffs (Espera)")
            col1, col2 = st.columns(2)
            with col1: create_card("Tempo Médio de Execução por Atividade (Top 10)", "⏱️", chart_bytes=plots.get('activity_service_times'))
            with col2: create_card("Top 10 Handoffs por Tempo de Espera", "⏸️", chart_bytes=plots.get('top_handoffs'))
            
            st.columns([1, 0.1])[0].markdown(create_card("Estatísticas Detalhadas de Performance", "📝", dataframe=tables.get('perf_stats')), unsafe_allow_html=True)
            
        elif section_key == "resource_analysis":
            st.markdown("### Carga de Trabalho e Especialização de Recursos")
            col1, col2 = st.columns(2)
            with col1: create_card("Top 10 Recursos por Horas Trabalhadas", "🏋️", chart_bytes=plots.get('resource_workload'))
            with col2: create_card("Recursos por Média de Tarefas/Projeto (Top 10)", "🧠", chart_bytes=plots.get('resource_avg_events'))
            
            st.markdown("### Matriz de Esforço e Transições")
            st.columns([1, 0.1])[0].markdown(create_card("Heatmap de Esforço por Recurso e Atividade", "🔥", chart_bytes=plots.get('resource_activity_matrix')), unsafe_allow_html=True)
            
            st.columns([1, 0.1])[0].markdown(create_card("Top 10 Handoffs entre Recursos", "➡️", chart_bytes=plots.get('resource_handoffs')), unsafe_allow_html=True)

        elif section_key == "variability_analysis":
            st.markdown("### Descoberta de Variantes e Rework")
            col1, col2 = st.columns(2)
            with col1: create_card("Top 10 Variantes de Processo por Frequência", "🧩", chart_bytes=plots.get('variants_frequency'))
            with col2: create_card("Atividades Mais Frequentes (Top 10)", "🎯", chart_bytes=plots.get('top_activities_plot'))
            
            st.markdown("### Rework (Loops) e Variantes em Tabela")
            col3, col4 = st.columns(2)
            with col3: create_card("Top 10 Loops de Rework", "🔁", dataframe=tables.get('rework_loops_table'))
            with col4: create_card("Tabela de Top 10 Variantes", "📜", dataframe=tables.get('variants_table'))
            
        elif section_key == "financial_impact":
            st.markdown("### Análise de Custo por Recurso e Impacto do Atraso")
            col1, col2 = st.columns(2)
            with col1: create_card("Custo por Tipo de Recurso", "💸", chart_bytes=plots.get('cost_by_resource_type'))
            with col2: create_card("Top 10 Handoffs por Custo de Espera Estimado", "🛑", chart_bytes=plots.get('top_handoffs_cost'))

            st.markdown("### KPIs de Atraso e Custo")
            kpi_data = tables.get('cost_of_delay_kpis', {})
            col3, col4, col5 = st.columns(3)
            with col3: st.metric("Custo Total Projetos Atrasados", kpi_data.get('Custo Total Projetos Atrasados', 'N/A'))
            with col4: st.metric("Atraso Médio (dias)", kpi_data.get('Atraso Médio (dias)', 'N/A'))
            with col5: st.metric("Custo Médio/Dia Atraso", kpi_data.get('Custo Médio/Dia Atraso', 'N/A'))
            
        elif section_key == "team_dynamics":
            st.markdown("### Impacto da Dinâmica de Equipa")
            col1, col2 = st.columns(2)
            with col1: create_card("Impacto do Tamanho da Equipa no Atraso", "📊", chart_bytes=plots.get('delay_by_teamsize'))
            with col2: create_card("Duração Mediana por Tamanho da Equipa", "📏", chart_bytes=plots.get('median_duration_by_teamsize'))

        elif section_key == "time_series":
            st.markdown("### Análise de Produtividade ao Longo do Tempo")
            col1, col2 = st.columns(2)
            with col1: create_card("Eficiência Semanal (Horas Trabalhadas)", "📆", chart_bytes=plots.get('weekly_efficiency'))
            with col2: create_card("Atividade Mensal (Projetos Concluídos)", "📅", chart_bytes=plots.get('monthly_activity_plot'))
            
            st.markdown("### Tempo de Espera e Service Time ao Longo do Tempo")
            col3, col4 = st.columns(2)
            with col3: create_card("Tempo de Espera entre Tarefas (Série Temporal)", "⏳", chart_bytes=plots.get('waiting_time_series'))
            with col4: create_card("Service Time das Tarefas (Série Temporal)", "⏱️", chart_bytes=plots.get('service_time_series'))
            
    elif st.session_state.current_dashboard == "Process Mining":
        if section_key == "petri_net":
            petri_net_page(plots)
        elif section_key == "dfg":
            dfg_page(plots)
        elif section_key == "variants":
            variants_page(plots)
        elif section_key == "conformance":
            conformance_page(plots, tables, metrics)
            
# --- FIM DA FUNÇÃO AUXILIAR ---

# --- PÁGINA DASHBOARD (MODIFICADA) ---
def dashboard_page():
    st.title("✨ Dashboard de Process Mining")
    
    # 1. Escolha entre Pré-Mineração e Process Mining
    dashboard_options = list(DASHBOARD_STRUCTURE.keys())
    
    # Renderizar botões de navegação primária (horizontal)
    cols = st.columns(len(dashboard_options))
    
    for i, option in enumerate(dashboard_options):
        is_active = st.session_state.current_dashboard == option
        button_class = "active-button" if is_active else ""
        
        with cols[i]:
            # Usar HTML para envolver o botão e aplicar o estilo `active-button` (solução alternativa, mas manter o st.button para o clique)
            if st.button(option, key=f"dash_tab_{option}", use_container_width=True):
                st.session_state.current_dashboard = option
                st.session_state.current_section = list(DASHBOARD_STRUCTURE[option].keys())[0] # Resetar secção
                st.rerun()

    st.markdown("---")

    if not st.session_state.analysis_run:
        # Se os ficheiros estão carregados, mas a análise não correu.
        if all(st.session_state.dfs.values()):
            upload_page(True) # Reutilizar o componente de upload para o botão de análise
        else:
            # Se não há ficheiros, vai para a página de upload
            st.session_state.current_page = "Upload"
            st.rerun()
        return

    # 2. Escolha da Secção (Botões da sub-navegação)
    current_dashboard_structure = DASHBOARD_STRUCTURE.get(st.session_state.current_dashboard, {})
    section_keys = list(current_dashboard_structure.keys())
    
    if not section_keys:
        st.error("Estrutura do Dashboard não encontrada.")
        return

    # Renderizar os botões de navegação secundária
    section_cols = st.columns(len(section_keys))
    
    for i, key in enumerate(section_keys):
        section_info = current_dashboard_structure[key]
        section_title = section_info["title"]
        section_icon = section_info["icon"]
        is_active = st.session_state.current_section == key
        button_class = "active-button" if is_active else ""
        
        with section_cols[i]:
            if st.button(f"{section_icon} {section_title}", key=f"sec_tab_{key}", use_container_width=True, help=f"Ver secção de {section_title}"):
                st.session_state.current_section = key
                st.rerun()
                
    st.markdown("---")

    # 3. Renderizar Conteúdo da Secção
    
    current_section_key = st.session_state.current_section

    # Títulos da secção atual
    if current_section_key in current_dashboard_structure:
        info = current_dashboard_structure[current_section_key]
        st.markdown(f"## {info['icon']} {info['title']}")
        
    plots = st.session_state.plots_pre_mining if st.session_state.current_dashboard == "Pré-Mineração" else st.session_state.plots_post_mining
    tables = st.session_state.tables_pre_mining if st.session_state.current_dashboard == "Pré-Mineração" else st.session_state.tables_post_mining
    metrics = st.session_state.metrics if st.session_state.current_dashboard == "Process Mining" else {}
    
    # Chama a função auxiliar para renderizar o conteúdo real
    render_dashboard_section(current_section_key, plots, tables, metrics)


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
            
            # Navegação principal
            nav_buttons = {
                "Dashboard": "🏠 Dashboard Geral",
                "Settings": "⚙️ Configurações"
            }

            for page, label in nav_buttons.items():
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()

            st.markdown("<br><br>---<br><br>", unsafe_allow_html=True) # Separador visual

            # Botão de Sair
            if st.button("🚪 Sair", use_container_width=True):
                st.session_state.authenticated = False
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated']: del st.session_state[key]
                st.rerun()
                
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "Settings":
            settings_page()
        else:
            dashboard_page() # Default

if __name__ == '__main__':
    main()
