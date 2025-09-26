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


# --- FUNÇÕES DE ANÁLISE ---
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
    
    # Criação do Log Completo (Start/Complete)
    df_start_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'start_date']].rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_start_events['lifecycle:transition'] = 'start'
    
    df_complete_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'end_date']].rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_complete_events['lifecycle:transition'] = 'complete'
    
    # Adicionar recurso. A associação recurso->tarefa é feita no full context
    df_full_context_for_merge = _df_full_context[['task_id', 'resource_name']].drop_duplicates()
    
    df_start_events = df_start_events.merge(df_full_context_for_merge, on='task_id', how='left').rename(columns={'resource_name': 'org:resource'})
    df_complete_events = df_complete_events.merge(df_full_context_for_merge, on='task_id', how='left').rename(columns={'resource_name': 'org:resource'})
    
    df_log = pd.concat([df_start_events, df_complete_events]).sort_values(['case:concept:name', 'time:timestamp'])
    
    # Filtra eventos sem carimbo de data/hora válido
    df_log.dropna(subset=['time:timestamp'], inplace=True)
    
    full_event_log = pm4py.convert_to_event_log(df_log)
    
    # 1. Descoberta do Processo (DFG, Heuristics e Petri Net)
    
    # --- DFG (Directly-Follows Graph) ---
    dfg = dfg_discovery.apply(full_event_log)
    gviz_dfg = dfg_visualizer.apply(dfg, log=full_event_log, variant=dfg_visualizer.Variants.FREQUENCY)
    plots['dfg_frequency'] = convert_gviz_to_bytes(gviz_dfg, format='svg') # SVG para maior qualidade e zoom
    
    dfg_perf = dfg_discovery.apply(full_event_log, parameters={dfg_discovery.Variants.PERFORMANCE.value.Parameters.ACTIVITY_KEY: "concept:name", dfg_discovery.Variants.PERFORMANCE.value.Parameters.TIMESTAMP_KEY: "time:timestamp"})
    gviz_dfg_perf = dfg_visualizer.apply(dfg_perf, log=full_event_log, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['dfg_performance'] = convert_gviz_to_bytes(gviz_dfg_perf, format='svg')
    
    # --- Descoberta de Petri Net (Inductive Miner) ---
    net, initial_marking, final_marking = inductive_miner.apply(full_event_log, parameters={inductive_miner.Variants.IMf.value.Parameters.ACTIVITY_KEY: "concept:name"})
    gviz_pn = pn_visualizer.apply(net, initial_marking, final_marking, parameters={pn_visualizer.Variants.FREQUENCY.value.Parameters.ACTIVITY_KEY: "concept:name"})
    plots['petri_net_frequency'] = convert_gviz_to_bytes(gviz_pn, format='svg')
    
    # --- Heuristics Miner (para visualizar Loops/Caminhos Mais Comuns) ---
    heu_net, heu_map = heuristics_miner.apply(full_event_log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.ACTIVITY_KEY: "concept:name"})
    gviz_heu_net = pn_visualizer.apply(heu_net, heu_map, heu_map, parameters={pn_visualizer.Variants.FREQUENCY.value.Parameters.ACTIVITY_KEY: "concept:name"})
    plots['heuristics_net'] = convert_gviz_to_bytes(gviz_heu_net, format='svg')
    
    # 2. Conformidade
    
    # Avaliação de Fitness e Precisão (contra a Petri Net)
    fitness = replay_fitness_evaluator.apply(full_event_log, net, initial_marking, final_marking, parameters={pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name"})
    precision = precision_evaluator.apply(full_event_log, net, initial_marking, final_marking, parameters={pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name"})
    generalization = generalization_evaluator.apply(full_event_log, net, initial_marking, final_marking, parameters={pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name"})
    simplicity = simplicity_evaluator.apply(net)
    
    metrics['conformance_metrics'] = {
        'Fitness (Ajuste)': f"{fitness['average_trace_fitness']:.3f}",
        'Precision (Precisão)': f"{precision:.3f}",
        'Generalization (Generalização)': f"{generalization:.3f}",
        'Simplicity (Simplicidade)': f"{simplicity:.3f}"
    }
    
    # Mapeamento de Atividades
    activities = pm4py.get_event_attribute_values(full_event_log, "concept:name")
    activities_df = pd.DataFrame(list(activities.items()), columns=['Activity', 'Frequency'])
    
    # 3. Análise de Desvio (Apenas para o caso de Conformance)
    
    # Alignment Analysis
    aligned_traces = alignments_miner.apply_log(full_event_log, net, initial_marking, final_marking, parameters={pm4py.util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name"})
    
    # Média de custo de desvio
    mean_cost = np.mean([trace['cost'] for trace in aligned_traces])
    
    # Desvios mais comuns (movimentos fora do modelo)
    deviations = []
    for trace in aligned_traces:
        for move in trace['alignment']:
            if move[0] is not None and move[1] is None: # Move no Log (Skip no Modelo) - Desvio!
                deviations.append(f"Move no Log (Activity: {move[0]})")
            elif move[0] is None and move[1] is not None: # Move no Modelo (Skip no Log) - Desvio!
                deviations.append(f"Move no Modelo (Transition: {move[1]})")
                
    deviation_counts = Counter(deviations)
    df_deviations = pd.DataFrame(deviation_counts.most_common(10), columns=['Desvio', 'Frequência'])
    
    metrics['deviation_metrics'] = {
        'Custo Médio de Desvio (Trilha)': f"{mean_cost:.2f}",
        'Total de Desvios Detectados': len(deviations)
    }
    
    # Gráfico 26: Top 10 Desvios de Conformidade
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_deviations, y='Desvio', x='Frequência', ax=ax, hue='Desvio', legend=False, palette='coolwarm'); ax.set_title("Top 10 Desvios de Conformidade")
    plots['top_deviations'] = convert_fig_to_bytes(fig)
    
    # 4. Análise de Desempenho no Modelo (DFG Performance)
    # Já foi feito no DFG acima.
    
    # Mapeamento de Casos por Variante (reutilizando pre-mining)
    variants_df = pm4py.get_variants(full_event_log)
    df_variants_counts = pd.DataFrame(list(variants_df.items()), columns=['Variant', 'Traces'])
    df_variants_counts['Frequency'] = df_variants_counts['Traces'].apply(len)
    df_variants_counts.sort_values(by='Frequency', ascending=False, inplace=True)
    df_variants_counts['Variant_Activities'] = df_variants_counts['Variant'].apply(lambda x: x.split(','))
    df_variants_counts['Activities_Count'] = df_variants_counts['Variant_Activities'].apply(len)
    
    # Duração média por variante
    variant_performance = pm4py.get_event_attribute_values(full_event_log, "case:concept:name")
    
    # Necessário converter para um log mapeado (case_id -> log)
    # É mais fácil calcular fora do PM4PY a partir do DF log.
    
    # Calcular a duração de ponta a ponta (Lead Time) para cada variante
    lead_time_df = df_log.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max']).reset_index()
    lead_time_df['duration'] = (lead_time_df['max'] - lead_time_df['min']).dt.total_seconds() / 3600
    
    # Mapear cada caso à sua variante
    case_to_variant = {k: ','.join(v) for k, v in pm4py.get_variants(full_event_log, "concept:name").items()}
    lead_time_df['Variant'] = lead_time_df['case:concept:name'].map(case_to_variant)
    
    avg_duration_by_variant = lead_time_df.groupby('Variant')['duration'].mean().sort_values(ascending=False).reset_index().head(10)
    avg_duration_by_variant['Variant'] = avg_duration_by_variant['Variant'].str.replace(',', ' -> ')
    
    # Gráfico 27: Top 10 Variantes por Duração Média (Lead Time)
    fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='duration', y='Variant', data=avg_duration_by_variant, ax=ax, hue='Variant', legend=False, palette='magma'); ax.set_title("Top 10 Variantes por Duração Média (Horas)")
    plots['variant_lead_time'] = convert_fig_to_bytes(fig)
    
    # 5. Análise de Atividades (Baseado em Frequência)
    activities_counts = pm4py.get_event_attribute_values(full_event_log, "concept:name")
    df_activities_counts = pd.DataFrame(list(activities_counts.items()), columns=['Activity', 'Frequency']).sort_values('Frequency', ascending=False).head(10)
    
    # Média de ocorrências por caso (para normalizar)
    num_cases = pm4py.get_trace_attribute_values(full_event_log, "case:concept:name")
    df_activities_counts['Avg_Per_Case'] = df_activities_counts['Frequency'] / len(num_cases)
    
    # Gráfico 28: Média de Ocorrências por Atividade (por caso)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_activities_counts.sort_values('Avg_Per_Case', ascending=False), y='Activity', x='Avg_Per_Case', ax=ax, hue='Activity', legend=False, palette='flare'); ax.set_title("Média de Ocorrências por Atividade (por Caso)")
    plots['activity_avg_per_case'] = convert_fig_to_bytes(fig)
    
    # 6. Análise de Recursos (Baseado em Workload e Handoffs - já está no pre-mining)
    
    return plots, metrics

# --- FUNÇÕES DE NAVEGAÇÃO E LAYOUT ---
def login_page():
    st.title("🔐 Login")
    with st.form("login_form"):
        st.text_input("Username", key="login_user")
        st.text_input("Password", type="password", key="login_pass")
        submitted = st.form_submit_button("Login")

        if submitted:
            if st.session_state.login_user == "admin" and st.session_state.login_pass == "admin":
                st.session_state.authenticated = True
                st.session_state.user_name = "Administrador"
                st.rerun()
            else:
                st.error("Credenciais inválidas.")

def settings_page():
    st.title("⚙️ Configurações e Upload de Dados")
    
    file_ids = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    file_names = {
        'projects': 'projects.csv (Projetos)',
        'tasks': 'tasks.csv (Tarefas)',
        'resources': 'resources.csv (Recursos)',
        'resource_allocations': 'resource_allocations.csv (Alocações)',
        'dependencies': 'dependencies.csv (Dependências)'
    }

    uploaded_files = {}
    
    for id in file_ids:
        uploaded_files[id] = st.file_uploader(f"Carregar {file_names[id]}", type=['csv'], key=f'uploader_{id}')
    
    all_files_uploaded = all(uploaded_files.values())
    
    if all_files_uploaded:
        st.success("Todos os 5 ficheiros CSV foram carregados com sucesso. Pronto para análise!")
        
        if st.button("Iniciar Análise", key='start_analysis_btn', use_container_width=True):
            with st.spinner('A carregar dados e a executar a análise inicial...'):
                try:
                    dfs = {}
                    for id, file in uploaded_files.items():
                        # Usar io.StringIO para ler o conteúdo do arquivo
                        dfs[id] = pd.read_csv(io.StringIO(file.getvalue().decode('utf-8')))
                    
                    st.session_state.dfs = dfs
                    
                    # Run pre-mining analysis
                    plots_pre, tables_pre, event_log, df_projects, df_tasks, df_resources, df_full_context = run_pre_mining_analysis(dfs)
                    
                    # Run post-mining analysis
                    plots_post, metrics_post = run_post_mining_analysis(event_log, df_projects, df_tasks, df_resources, df_full_context)
                    
                    st.session_state.plots_pre_mining = plots_pre
                    st.session_state.tables_pre_mining = tables_pre
                    st.session_state.plots_post_mining = plots_post
                    st.session_state.metrics = metrics_post
                    st.session_state.analysis_run = True
                    st.success("Análise concluída com sucesso!")
                    st.session_state.current_page = "Dashboard"
                    st.rerun()

                except Exception as e:
                    st.error(f"Ocorreu um erro durante o processamento ou análise: {e}")
                    st.session_state.analysis_run = False
    else:
        st.warning("Por favor, carregue os 5 ficheiros CSV para iniciar a análise.")

def dashboard_page():
    st.title("📊 Dashboard de Transformação Inteligente de Processos")

    if not st.session_state.analysis_run:
        st.info("Por favor, vá a **Configurações** para carregar os ficheiros e iniciar a análise.")
        return

    plots = st.session_state.plots_pre_mining
    plots_post = st.session_state.plots_post_mining
    tables = st.session_state.tables_pre_mining
    metrics = st.session_state.metrics

    # Navegação entre Pré-Mineração e Pós-Mineração
    col_nav1, col_nav2 = st.columns([1, 1])
    with col_nav1:
        if st.button("Pré-Mineração (Dados Estruturais)", use_container_width=True, help="Análise de dados estruturais de projetos e recursos"):
            st.session_state.current_dashboard = "Pré-Mineração"
            st.rerun()
    with col_nav2:
        if st.button("Pós-Mineração (Process Mining)", use_container_width=True, help="Análise de fluxo, conformidade e desempenho do processo"):
            st.session_state.current_dashboard = "Pós-Mineração"
            st.rerun()
    
    st.markdown("---")
    
    # Reorganização em 5 abas para o Dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Visão Geral e Custos",
        "⚡ Performance",
        "👥 Recursos",
        "⏳ Gargalos e Espera",
        "🌐 Fluxo e Conformidade"
    ])

    if st.session_state.current_dashboard == "Pré-Mineração":
        st.subheader("Análise Pré-Mineração: Estrutura e Métricas de Projeto")

        # ----------------------------------------------------
        # TAB 1: VISÃO GERAL E CUSTOS (KPIs, Matriz, Custos)
        # ----------------------------------------------------
        with tab1:
            st.subheader("Métricas Chave de Projeto")
            kpi_data = tables.get('kpi_data', {})
            colk1, colk2, colk3, colk4 = st.columns(4)
            colk1.metric("Projetos", kpi_data.get('Total de Projetos', '-'))
            colk2.metric("Tarefas", kpi_data.get('Total de Tarefas', '-'))
            colk3.metric("Recursos", kpi_data.get('Total de Recursos', '-'))
            colk4.metric("Duração Média", kpi_data.get('Duração Média (dias)', '-'), delta="dias")

            st.markdown("---")
            st.subheader("Análise Financeira e de Duração")
            
            col1, col2 = st.columns(2)
            with col1:
                # Gráfico 1: Matriz de Performance
                create_card("Matriz de Performance (Atraso vs Custo)", "📈", chart_bytes=plots.get('performance_matrix'))
            with col2:
                # Tabela: Top 5 Projetos com Maior Custo
                create_card("Top 5 Projetos por Custo Total", "💸", dataframe=tables.get('outlier_cost'))

            col3, col4 = st.columns(2)
            with col3:
                # Gráfico 15: Custo por Tipo de Recurso
                create_card("Custo Total por Tipo de Recurso", "💰", chart_bytes=plots.get('cost_by_resource_type'))
            with col4:
                # Tabela: Top 5 Projetos com Maior Duração
                create_card("Top 5 Projetos por Duração Real", "⏱️", dataframe=tables.get('outlier_duration'))

            st.markdown("---")
            st.subheader("Custo e Impacto do Atraso")
            kpi_delay = tables.get('cost_of_delay_kpis', {})
            cold1, cold2, cold3 = st.columns(3)
            cold1.metric("Custo Total Atrasos", kpi_delay.get('Custo Total Projetos Atrasados', '-'))
            cold2.metric("Atraso Médio", kpi_delay.get('Atraso Médio (dias)', '-'), delta="dias")
            cold3.metric("Custo/Dia de Atraso", kpi_delay.get('Custo Médio/Dia Atraso', '-'))

        # ----------------------------------------------------
        # TAB 2: PERFORMANCE (Duração, Lead Time, Throughput)
        # ----------------------------------------------------
        with tab2:
            st.subheader("Métricas de Duração e Tempo de Ciclo")
            
            col1, col2 = st.columns(2)
            with col1:
                # Gráfico 2: Distribuição da Duração dos Projetos
                create_card("Distribuição da Duração dos Projetos (dias)", "📊", chart_bytes=plots.get('case_durations_boxplot'))
            with col2:
                # Gráfico 25: Duração Média por Fase do Processo
                create_card("Duração Média por Fase do Processo", "🔄", chart_bytes=plots.get('cycle_time_breakdown'))
                
            st.markdown("---")
            st.subheader("Análise de Lead Time e Throughput")
            
            col3, col4 = st.columns(2)
            with col3:
                # Gráfico 3: Distribuição do Lead Time
                create_card("Distribuição do Lead Time (dias)", "📈", chart_bytes=plots.get('lead_time_hist'))
            with col4:
                # Gráfico 4: Distribuição do Throughput (Horas)
                create_card("Distribuição do Throughput (horas)", "📉", chart_bytes=plots.get('throughput_hist'))
                
            col5, col6 = st.columns(2)
            with col5:
                # Gráfico 6: Relação Lead Time vs Throughput
                create_card("Relação Lead Time vs Throughput", "🎯", chart_bytes=plots.get('lead_time_vs_throughput'))
            with col6:
                # Gráfico 24: Benchmark de Throughput por Tamanho da Equipa
                create_card("Benchmark de Throughput por Tamanho da Equipa", "⚖️", chart_bytes=plots.get('throughput_benchmark_by_teamsize'))


        # ----------------------------------------------------
        # TAB 3: RECURSOS (Workload, Handoffs, Impacto da Equipa)
        # ----------------------------------------------------
        with tab3:
            st.subheader("Análise de Workload e Eficiência de Recursos")
            
            col1, col2 = st.columns(2)
            with col1:
                # Gráfico 11: Top 10 Recursos por Horas Trabalhadas
                create_card("Top 10 Recursos por Horas Trabalhadas", "💪", chart_bytes=plots.get('resource_workload'))
            with col2:
                # Gráfico 13: Heatmap de Esforço por Recurso e Atividade
                create_card("Heatmap de Esforço (Recurso vs. Atividade)", "🔥", chart_bytes=plots.get('resource_activity_matrix'))

            st.markdown("---")
            st.subheader("Interação e Colaboração")
            
            col3, col4 = st.columns(2)
            with col3:
                # Gráfico 14: Top 10 Handoffs entre Recursos
                create_card("Top 10 Handoffs entre Recursos (Frequência)", "🤝", chart_bytes=plots.get('resource_handoffs'))
            with col4:
                # Gráfico 12: Recursos por Média de Tarefas por Projeto
                create_card("Recursos por Média de Tarefas por Projeto", "📚", chart_bytes=plots.get('resource_avg_events'))
                
            st.markdown("---")
            st.subheader("Impacto do Tamanho da Equipa")
            
            col5, col6 = st.columns(2)
            with col5:
                # Gráfico 17: Impacto do Tamanho da Equipa no Atraso
                create_card("Impacto do Tamanho da Equipa no Atraso", "🐢", chart_bytes=plots.get('delay_by_teamsize'))
            with col6:
                # Gráfico 18: Duração Mediana por Tamanho da Equipa
                create_card("Duração Mediana por Tamanho da Equipa", "📏", chart_bytes=plots.get('median_duration_by_teamsize'))
            
            # Gráfico 19: Eficiência Semanal (Horas Trabalhadas)
            st.markdown("---")
            create_card("Eficiência Semanal (Horas Trabalhadas)", "📅", chart_bytes=plots.get('weekly_efficiency'))


        # ----------------------------------------------------
        # TAB 4: GARGALOS E ESPERA (Handoffs de Atividades, Wait Time)
        # ----------------------------------------------------
        with tab4:
            st.subheader("Análise de Tempo de Espera e Service Time")

            col1, col2 = st.columns(2)
            with col1:
                # Gráfico 21: Gargalos: Tempo de Serviço vs. Espera
                create_card("Gargalos: Tempo de Serviço vs. Espera (Atividade)", "🚧", chart_bytes=plots.get('service_vs_wait_stacked'))
            with col2:
                # Gráfico 20: Top 15 Recursos por Tempo Médio de Espera
                create_card("Top 15 Recursos por Tempo Médio de Espera", "🛑", chart_bytes=plots.get('bottleneck_by_resource'))
            
            st.markdown("---")
            st.subheader("Transições e Custo da Inatividade")

            col3, col4 = st.columns(2)
            with col3:
                # Gráfico 8: Top 10 Handoffs por Tempo de Espera
                create_card("Top 10 Handoffs Atividade (Tempo de Espera)", "🔗", chart_bytes=plots.get('top_handoffs'))
            with col4:
                # Gráfico 9: Top 10 Handoffs por Custo de Espera
                create_card("Top 10 Handoffs Atividade (Custo de Espera)", "💰", chart_bytes=plots.get('top_handoffs_cost'))
            
            st.markdown("---")
            st.subheader("Evolução Temporal")
            # Gráfico 23: Evolução do Tempo Médio de Espera
            create_card("Evolução do Tempo Médio de Espera", "📉", chart_bytes=plots.get('wait_time_evolution'))


        # ----------------------------------------------------
        # TAB 5: FLUXO E CONFORMIDADE (Variantes, Rework)
        # ----------------------------------------------------
        with tab5:
            st.subheader("Análise de Variantes de Processo")

            col1, col2 = st.columns(2)
            with col1:
                # Tabela: Top 10 Variantes
                create_card("Top 10 Variantes de Processo (Tabela)", "📋", dataframe=tables.get('variants_table'))
            with col2:
                # Gráfico 16: Top 10 Variantes de Processo por Frequência
                create_card("Top 10 Variantes por Frequência", "📈", chart_bytes=plots.get('variants_frequency'))

            st.markdown("---")
            st.subheader("Rework e Atividades Comuns")

            col3, col4 = st.columns(2)
            with col3:
                # Tabela: Top 10 Rework Loops
                create_card("Top 10 Rework Loops Detetados", "🔁", dataframe=tables.get('rework_loops_table'))
            with col4:
                # Gráfico 10: Atividades Mais Frequentes
                create_card("Top 10 Atividades Mais Frequentes", "📌", chart_bytes=plots.get('top_activities_plot'))

    elif st.session_state.current_dashboard == "Pós-Mineração":
        st.subheader("Análise Pós-Mineração: Descoberta e Conformidade do Processo")
        
        # ----------------------------------------------------
        # TAB 1: VISÃO GERAL E CUSTOS (Fluxo de Alto Nível)
        # ----------------------------------------------------
        with tab1:
            st.subheader("Métricas de Desempenho e Modelo")
            
            colk1, colk2 = st.columns(2)
            with colk1:
                # Gráfico 27: Top 10 Variantes por Duração Média (Lead Time)
                create_card("Top 10 Variantes por Duração Média (Horas)", "⏳", chart_bytes=plots_post.get('variant_lead_time'))
            with colk2:
                # DFG de Frequência
                create_card("DFG (Directly-Follows Graph) - Frequência", "🔗", chart_bytes=plots_post.get('dfg_frequency'))
                
            st.markdown("---")
            st.subheader("Modelos de Processo Descobertos")
            # Petri Net (Modelo Formal)
            create_card("Rede de Petri (Modelo Descoberto)", "🗺️", chart_bytes=plots_post.get('petri_net_frequency'))
        
        # ----------------------------------------------------
        # TAB 2: PERFORMANCE (DFG de Performance)
        # ----------------------------------------------------
        with tab2:
            st.subheader("Análise de Performance no Fluxo")
            # DFG de Performance
            create_card("DFG (Directly-Follows Graph) - Desempenho (Tempo)", "⏱️", chart_bytes=plots_post.get('dfg_performance'))

        # ----------------------------------------------------
        # TAB 3: RECURSOS (Heuristics Net)
        # ----------------------------------------------------
        with tab3:
            st.subheader("Visão Heurística do Processo")
            # Heuristics Net
            create_card("Heuristics Net (Fluxo Mais Comum e Loops)", "🧠", chart_bytes=plots_post.get('heuristics_net'))
        
        # ----------------------------------------------------
        # TAB 4: GARGALOS E ESPERA (Desvios mais comuns)
        # ----------------------------------------------------
        with tab4:
            st.subheader("Análise Detalhada de Conformidade e Desvios")
            
            conformance_metrics = metrics.get('conformance_metrics', {})
            colc1, colc2, colc3, colc4 = st.columns(4)
            colc1.metric("Fitness (Ajuste)", conformance_metrics.get('Fitness (Ajuste)', '-'))
            colc2.metric("Precision (Precisão)", conformance_metrics.get('Precision (Precisão)', '-'))
            colc3.metric("Generalization", conformance_metrics.get('Generalization (Generalização)', '-'))
            colc4.metric("Simplicity", conformance_metrics.get('Simplicity (Simplicidade)', '-'))
            
            st.markdown("---")
            st.subheader("Análise de Desvios (Mapeamento de Alinhamento)")
            
            col_dev1, col_dev2 = st.columns(2)
            with col_dev1:
                # Gráfico 26: Top 10 Desvios de Conformidade
                create_card("Top 10 Desvios de Conformidade Detetados", "🚫", chart_bytes=plots_post.get('top_deviations'))
            with col_dev2:
                # Métricas de Desvio
                dev_metrics = metrics.get('deviation_metrics', {})
                st.metric("Custo Médio de Desvio (por Trilha)", dev_metrics.get('Custo Médio de Desvio (Trilha)', '-'))
                st.metric("Total de Desvios", dev_metrics.get('Total de Desvios Detectados', '-'))

        # ----------------------------------------------------
        # TAB 5: FLUXO E CONFORMIDADE (Atividades por Caso)
        # ----------------------------------------------------
        with tab5:
            st.subheader("Frequência de Atividades Normalizada")
            # Gráfico 28: Média de Ocorrências por Atividade (por caso)
            create_card("Média de Ocorrências por Atividade (por Caso)", "🔢", chart_bytes=plots_post.get('activity_avg_per_case'))


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
