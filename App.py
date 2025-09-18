import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io

# Imports específicos de Process Mining (PM4PY)
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
    page_title="Painel de Análise de Processos de IT",
    page_icon="✨",
    layout="wide"
)

# Estilo CSS para replicar a estética da app de referência
st.markdown("""
<style>
    /* --- Main Layout & Background --- */
    .stApp {
        background-color: #F0F2F6; /* Cinza claro de fundo */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* --- Sidebar Style --- */
    [data-testid="stSidebar"] {
        background-color: #0F172A; /* Azul escuro da referência */
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .st-emotion-cache-17l5j99 {
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1g6goon { /* Sidebar radio options */
        color: #cbd5e1;
    }

    /* --- Card Style --- */
    .card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.05);
        transition: 0.3s;
    }
    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
    }

    /* --- Typography & Titles --- */
    h1 {
        color: #1E293B;
        font-weight: 600;
    }
    h2 {
        color: #1E293B;
        font-weight: 600;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h3 {
        color: #334155;
        font-weight: 600;
    }
    h4 { /* Usado para títulos dentro dos cartões */
        color: #1E293B;
        font-weight: 600;
        margin-bottom: 1rem;
    }


    /* --- Component Styling --- */
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 15px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #3B82F6;
        font-weight: bold;
        border-bottom: 3px solid #3B82F6;
    }

    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        color: #1E293B;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÃO AUXILIAR PARA CONVERTER GRÁFICOS ---
def convert_fig_to_bytes(fig, format='png'):
    """Converte uma figura Matplotlib/Seaborn para bytes para ser usada com st.image."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig) # Fecha a figura para libertar memória
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    """Converte um objeto Graphviz (da PM4PY) para bytes."""
    return io.BytesIO(gviz.pipe(format=format))


# --- 2. INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'dfs' not in st.session_state:
    st.session_state.dfs = {
        'projects': None, 'tasks': None, 'resources': None,
        'resource_allocations': None, 'dependencies': None
    }
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state:
    st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state:
    st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state:
    st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}


# --- 3. FUNÇÕES DE ANÁLISE (MODIFICADAS PARA RETORNAR IMAGENS) ---
@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    
    # ... (Cópia e Pré-processamento inicial - SEM ALTERAÇÕES) ...
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()
    for df in [df_projects, df_tasks]:
        for col in ['start_date', 'end_date', 'planned_end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'allocation_date' in df_resource_allocations.columns:
        df_resource_allocations['allocation_date'] = pd.to_datetime(df_resource_allocations['allocation_date'], errors='coerce')
    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['project_name'].str.extract(r'Projeto \d+: (.*?) ')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)
    df_projects['completion_quarter'] = df_projects['end_date'].dt.to_period('Q').astype(str)
    df_tasks['task_duration_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.days
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
    allocations_to_merge = df_resource_allocations.drop(columns=['project_id'], errors='ignore')
    df_full_context = df_full_context.merge(allocations_to_merge, on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']


    # --- GERAÇÃO DE GRÁFICOS E TABELAS (MODIFICADO) ---
    st.session_state.kpi_data = {
        'total_projects': df_projects['project_id'].nunique(),
        'total_tasks': len(df_tasks),
        'avg_duration': df_projects['actual_duration_days'].mean(),
        'total_cost': df_projects['total_actual_cost'].sum()
    }
    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)

    # 1. Matriz de Performance
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.7, palette='bright', ax=ax)
    ax.axhline(0, color='black', linestyle='--', lw=1); ax.axvline(0, color='black', linestyle='--', lw=1)
    ax.set_title('Matriz de Performance: Prazo vs. Orçamento'); ax.set_xlabel('Desvio de Prazo (dias)'); ax.set_ylabel('Desvio de Custo (€)')
    plots['performance_matrix'] = convert_fig_to_bytes(fig)
    
    # 2. Distribuição da Duração
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df_projects['actual_duration_days'], color='skyblue', ax=ax)
    sns.stripplot(x=df_projects['actual_duration_days'], color='blue', size=4, jitter=True, alpha=0.5, ax=ax)
    ax.set_title('Distribuição da Duração dos Projetos (Lead Time)'); ax.set_xlabel('Duração (dias)')
    plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)

    # Lógica de handoff
    log_df_final = df_tasks[['project_id', 'task_id', 'task_name', 'end_date']].copy()
    log_df_final.rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True)
    df_handoff_analysis = log_df_final.copy()
    df_handoff_analysis.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_handoff_analysis['previous_activity_end_time'] = df_handoff_analysis.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff_analysis['handoff_time_days'] = (df_handoff_analysis['time:timestamp'] - df_handoff_analysis['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff_analysis['previous_activity'] = df_handoff_analysis.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff_analysis.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')

    # 3. Top 10 Handoffs
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', hue='transition', palette='magma', legend=False, ax=ax)
    ax.set_title('Top 10 Transições com Maior Tempo de Espera'); ax.set_xlabel('Tempo Médio de Espera (dias)'); ax.set_ylabel('')
    plots['top_handoffs'] = convert_fig_to_bytes(fig)

    # 4. Custo de Espera Estimado
    avg_project_cost_per_day = df_projects['cost_per_day'].mean()
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * avg_project_cost_per_day
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', hue='transition', palette='Reds_r', legend=False, ax=ax)
    ax.set_title('Top 10 Transições por Custo de Espera Estimado'); ax.set_xlabel('Custo Estimado (€)'); ax.set_ylabel('')
    plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)
    
    # 5. Tempo Médio de Execução por Atividade
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), hue='task_name', palette='viridis', legend=False, ax=ax)
    ax.set_title('Tempo Médio de Execução por Atividade'); ax.set_xlabel('Tempo de Execução (dias)'); ax.set_ylabel('')
    plots['activity_service_times'] = convert_fig_to_bytes(fig)
        
    # 6. Atividades Mais Frequentes
    activity_counts = df_tasks["task_name"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax)
    ax.set_title('Atividades Mais Frequentes'); ax.set_xlabel('Contagem')
    plots['top_activities_plot'] = convert_fig_to_bytes(fig)

    # 7. Top 10 Recursos por Horas
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), hue='resource_name', palette='plasma', legend=False, ax=ax)
    ax.set_title('Top 10 Recursos por Horas Trabalhadas'); ax.set_xlabel('Horas Trabalhadas'); ax.set_ylabel('')
    plots['resource_workload'] = convert_fig_to_bytes(fig)
    
    # ... (O restante da função segue o mesmo padrão de conversão para bytes) ...
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8)) # Heatmaps precisam de mais espaço
    sns.heatmap(resource_activity_matrix_pivot, cmap='YlGnBu', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8})
    ax.set_title('Heatmap de Esforço (Horas) por Recurso e Atividade')
    plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)
    
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', hue='resource_type', palette='magma', legend=False, ax=ax)
    ax.set_title('Custo por Tipo de Recurso'); ax.set_xlabel('Custo Total (€)'); ax.set_ylabel('')
    plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)

    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()
    bins = np.linspace(min_res, max_res, 5, dtype=int) if max_res > min_res else [min_res, max_res]
    df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', hue='team_size_bin_dynamic', palette='flare', legend=False, ax=ax)
    ax.set_title('Impacto do Tamanho da Equipa no Atraso'); ax.set_xlabel('Tamanho da Equipa'); ax.set_ylabel('Desvio de Prazo (dias)')
    plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)
    
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', hue='team_size_bin_dynamic', palette='crest', legend=False, ax=ax)
    ax.set_title('Duração Mediana por Tamanho da Equipa'); ax.set_xlabel('Tamanho da Equipa'); ax.set_ylabel('Duração Mediana (dias)')
    plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)
    
    df_alloc_costs['day_of_week'] = pd.to_datetime(df_alloc_costs['allocation_date']).dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', hue='day_of_week', palette='plasma', legend=False, ax=ax)
    ax.set_title('Eficiência Semanal (Horas Trabalhadas)'); ax.set_xlabel(''); ax.set_ylabel('Total de Horas')
    plt.xticks(rotation=45)
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    
    df_tasks_analysis = df_tasks.copy()
    df_tasks_analysis['service_time_days'] = (pd.to_datetime(df_tasks_analysis['end_date']) - pd.to_datetime(df_tasks_analysis['start_date'])).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True)
    df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (pd.to_datetime(df_tasks_analysis['start_date']) - pd.to_datetime(df_tasks_analysis['previous_task_end'])).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    bottleneck_by_activity.plot(kind='bar', stacked=True, color=['royalblue', 'crimson'], ax=ax)
    ax.set_title('Análise de Gargalos: Tempo de Serviço vs. Espera'); ax.set_ylabel('Dias'); ax.set_xlabel('Tipo de Tarefa'); plt.xticks(rotation=45)
    plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)
    
    # Reconstruindo log para handoff de recursos
    df_start_events = df_tasks[['project_id', 'task_id', 'task_name', 'start_date']].copy()
    df_start_events.rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True)
    df_start_events['lifecycle:transition'] = 'start'
    df_complete_events = df_tasks[['project_id', 'task_id', 'task_name', 'end_date']].copy()
    df_complete_events.rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True)
    df_complete_events['lifecycle:transition'] = 'complete'
    log_df_temp = pd.concat([df_start_events, df_complete_events], ignore_index=True)
    resource_mapping = df_resource_allocations.groupby('task_id')['resource_id'].apply(list).reset_index()
    log_df_temp = log_df_temp.merge(resource_mapping, on='task_id', how='left').explode('resource_id')
    log_df_temp = log_df_temp.merge(df_resources[['resource_id', 'resource_name']], on='resource_id', how='left')
    log_df_temp.rename(columns={'resource_name': 'org:resource'}, inplace=True)
    log_df_pm4py = log_converter.apply(log_df_temp)
    handoff_counts = {}
    for trace in log_df_pm4py:
        resources = [event['org:resource'] for event in trace if 'org:resource' in event and pd.notna(event['org:resource'])]
        for i in range(len(resources) - 1):
            pair = (resources[i], resources[i+1])
            if resources[i] != resources[i+1]:
                handoff_counts[pair] = handoff_counts.get(pair, 0) + 1
    df_resource_handoffs = pd.DataFrame([{'De': k[0], 'Para': k[1], 'Contagem': v} for k, v in handoff_counts.items()]).sort_values('Contagem', ascending=False)
    df_rh_typed = df_resource_handoffs.merge(df_resources[['resource_name', 'resource_type']], left_on='De', right_on='resource_name').merge(df_resources[['resource_name', 'resource_type']], left_on='Para', right_on='resource_name', suffixes=('_de', '_para'))
    handoff_matrix = df_rh_typed.groupby(['resource_type_de', 'resource_type_para'])['Contagem'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(handoff_matrix, annot=True, fmt=".0f", cmap="BuPu", ax=ax)
    ax.set_title('Matriz de Handoffs por Tipo de Equipa')
    plots['handoff_matrix_by_type'] = convert_fig_to_bytes(fig)
    
    return plots, tables, df_full_context, log_df_pm4py

@st.cache_data
def run_post_mining_analysis(log_df_pm4py_cached, dfs_cached):
    plots = {}
    metrics = {}
    
    # ... (Reconstituição e garantias de tipo - SEM ALTERAÇÕES) ...
    log_df_final = log_df_pm4py_cached
    df_projects = dfs_cached['projects'].copy()
    df_tasks_raw = dfs_cached['tasks'].copy()
    df_resources = dfs_cached['resources'].copy()
    log_df_final['case:concept:name'] = log_df_final['case:concept:name'].astype(str)
    df_tasks_raw['start_date'] = pd.to_datetime(df_tasks_raw['start_date'])
    df_tasks_raw['end_date'] = pd.to_datetime(df_tasks_raw['end_date'])
    df_projects['start_date'] = pd.to_datetime(df_projects['start_date'])
    df_projects['end_date'] = pd.to_datetime(df_projects['end_date'])
    df_projects['project_id'] = df_projects['project_id'].astype(str)
    df_tasks_raw['project_id'] = df_tasks_raw['project_id'].astype(str)
    event_log_pm4py = pm4py.convert_to_event_log(log_df_final)
    
    # 1. Descoberta de Modelos (MODIFICADO)
    variants_dict = variants_filter.get_variants(event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_3_variants = variants_filter.apply(event_log_pm4py, top_variant_names)

    # Inductive Miner
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    fitness_im = replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im)
    precision_im = precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im)
    metrics['inductive_miner'] = {"Fitness": fitness_im.get('average_trace_fitness', 0), "Precisão": precision_im}

    # Heuristics Miner
    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    fitness_hm = replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm)
    precision_hm = precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm)
    metrics['heuristics_miner'] = {"Fitness": fitness_hm.get('average_trace_fitness', 0), "Precisão": precision_hm}

    # 2. Gantt Chart (Não gerado por ser muito grande e lento, mantemos a lógica caso queira reativar)
    # plots['gantt_chart_all_projects'] = ...
    
    # 3. Heatmap de Performance (MODIFICADO)
    dfg_perf, _, _ = pm4py.discover_performance_dfg(event_log_pm4py)
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=event_log_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)
    
    # 4. Rede Social de Recursos (MODIFICADO)
    handovers = {}
    for _, group in log_df_final.groupby('case:concept:name'):
        resources = group.sort_values('time:timestamp')['org:resource'].tolist()
        for i in range(len(resources) - 1):
            if resources[i] != resources[i+1]:
                pair = (resources[i], resources[i+1])
                handovers[pair] = handovers.get(pair, 0) + 1
    fig_net, ax_net = plt.subplots(figsize=(10, 10))
    G = nx.DiGraph()
    for (source, target), weight in handovers.items(): G.add_edge(source, target, weight=weight)
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, edge_color='gray', width=[w*0.5 for w in weights], ax=ax_net, font_size=9, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net, font_size=8)
    ax_net.set_title('Rede Social de Recursos (Handover Network)')
    plots['resource_network_adv'] = convert_fig_to_bytes(fig_net)
    
    # ... (O restante da função segue o mesmo padrão de conversão) ...
    variants_df = log_df_final.groupby('case:concept:name').agg(
        variant=('concept:name', lambda x: tuple(x)),
        start_timestamp=('time:timestamp', 'min'),
        end_timestamp=('time:timestamp', 'max')
    ).reset_index()
    variants_df['duration_hours'] = (variants_df['end_timestamp'] - variants_df['start_timestamp']).dt.total_seconds() / 3600
    variant_durations = variants_df.groupby('variant').agg(
        count=('case:concept:name', 'count'),
        avg_duration_hours=('duration_hours', 'mean')
    ).reset_index().sort_values(by='count', ascending=False).head(10)
    variant_durations['variant_str'] = variant_durations['variant'].apply(lambda x: ' -> '.join([str(i) for i in x][:4]) + '...') # Limitar texto
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='avg_duration_hours', y='variant_str', data=variant_durations.astype({'avg_duration_hours':'float'}), palette='plasma', ax=ax, hue='variant_str', legend=False)
    ax.set_title('Duração Média das 10 Variantes Mais Comuns'); ax.set_xlabel('Duração Média (horas)'); ax.set_ylabel('')
    plots['variant_duration_plot'] = convert_fig_to_bytes(fig)
    
    aligned_traces = alignments_miner.apply(event_log_pm4py, net_im, im_im, fm_im)
    case_fitness_data = [{'project_id': str(trace.attributes['concept:name']), 'fitness': alignment['fitness']} for trace, alignment in zip(event_log_pm4py, aligned_traces)]
    case_fitness_df = pd.DataFrame(case_fitness_data)
    case_fitness_df = case_fitness_df.merge(df_projects[['project_id', 'end_date']], on='project_id')
    case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str)
    monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax)
    ax.set_title('Score de Conformidade ao Longo do Tempo'); ax.set_ylim(0, 1.05); plt.xticks(rotation=45)
    plots['conformance_over_time_plot'] = convert_fig_to_bytes(fig)
    
    milestones = ['Analise e Design', 'Implementacao da Funcionalidade', 'Execucao de Testes', 'Deploy da Aplicacao']
    df_milestones = df_tasks_raw[df_tasks_raw['task_name'].isin(milestones)].copy()
    milestone_pairs = []
    for project_id, group in df_milestones.groupby('project_id'):
        sorted_tasks = group.sort_values('start_date')
        for i in range(len(sorted_tasks) - 1):
            start_task, end_task = sorted_tasks.iloc[i], sorted_tasks.iloc[i+1]
            duration = (end_task['start_date'] - start_task['end_date']).total_seconds() / 3600
            if duration >= 0: milestone_pairs.append({'transition': str(start_task['task_name']) + ' -> ' + str(end_task['task_name']), 'duration_hours': duration})
    df_milestone_pairs = pd.DataFrame(milestone_pairs)
    if not df_milestone_pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_milestone_pairs, x='duration_hours', y='transition', ax=ax, hue='transition', legend=False, orient='h')
        ax.set_title('Análise de Tempo entre Marcos do Processo'); ax.set_xlabel('Duração (horas)'); ax.set_ylabel('')
        plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    return plots, metrics

# --- 4. LAYOUT DA APLICAÇÃO (SIDEBAR E PÁGINAS) ---

st.sidebar.title("Painel de Análise de Processos")
st.sidebar.markdown("Navegue pelas secções da aplicação.")
page = st.sidebar.radio("Selecione a Página", ["Upload de Ficheiros", "Executar Análise", "Resultados da Análise"], label_visibility="hidden")
file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']

if page == "Upload de Ficheiros":
    st.header("1. Upload dos Ficheiros de Dados (.csv)")
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")
    cols = st.columns(3)
    for i, name in enumerate(file_names):
        with cols[i % 3]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                st.success(f"`{name}.csv` carregado.")
    
    if all(st.session_state.dfs[name] is not None for name in file_names):
        st.subheader("Pré-visualização dos Dados Carregados")
        for name, df in st.session_state.dfs.items():
            with st.expander(f"Visualizar as primeiras 5 linhas de `{name}.csv`"):
                st.dataframe(df.head())

elif page == "Executar Análise":
    st.header("2. Execução da Análise de Processos")
    if not all(st.session_state.dfs[name] is not None for name in file_names):
        st.warning("Por favor, carregue todos os 5 ficheiros CSV na página de 'Upload' antes de continuar.")
    else:
        st.info("Todos os ficheiros estão carregados. Clique no botão abaixo para iniciar a análise completa.")
        if st.button("🚀 Iniciar Análise Completa"):
            with st.spinner("A executar a análise pré-mineração... Isto pode demorar um momento."):
                plots_pre, tables_pre, _, log_df_pm4py = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                st.session_state.log_df_for_cache = pm4py.convert_to_dataframe(log_df_pm4py)
                st.session_state.dfs_for_cache = {k: v.copy() for k, v in st.session_state.dfs.items() if k in ['projects', 'tasks', 'resources']}

            with st.spinner("A executar a análise de Process Mining... Esta é a parte mais demorada."):
                plots_post, metrics = run_post_mining_analysis(st.session_state.log_df_for_cache, st.session_state.dfs_for_cache)
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics

            st.session_state.analysis_run = True
            st.success("✅ Análise concluída com sucesso! Navegue para a página de 'Resultados da Análise' para explorar os insights.")
            st.balloons()

elif page == "Resultados da Análise":
    st.header("Resultados da Análise de Processos")

    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor, vá à página 'Executar Análise' e inicie o processo.")
    else:
        tab1, tab2 = st.tabs(["📊 Análise Geral e Performance", "⛏️ Descoberta e Conformidade de Processos"])

        with tab1:
            st.markdown("### Métricas Chave do Processo")
            kpi = st.session_state.kpi_data
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total de Projetos", f"{kpi['total_projects']}", "Casos")
            c2.metric("Total de Tarefas", f"{kpi['total_tasks']}", "Eventos")
            c3.metric("Duração Média", f"{kpi['avg_duration']:.1f} dias", "Lead Time")
            c4.metric("Custo Total", f"€ {kpi['total_cost']:,.0f}".replace(",", " "))

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("<h4>Matriz de Performance (Prazo vs. Custo)</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.plots_pre_mining['performance_matrix'], use_container_width=True)
            with c2:
                with st.container(border=True):
                    st.markdown("<h4>Distribuição da Duração dos Projetos</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.plots_pre_mining['case_durations_boxplot'], use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("<h4>Top 5 Projetos Mais Longos</h4>", unsafe_allow_html=True)
                    st.dataframe(st.session_state.tables_pre_mining['outlier_duration'])
            with c2:
                with st.container(border=True):
                    st.markdown("<h4>Top 5 Projetos Mais Caros</h4>", unsafe_allow_html=True)
                    st.dataframe(st.session_state.tables_pre_mining['outlier_cost'])
            
            with st.expander("Análise de Atividades, Esperas e Recursos"):
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("<h4>Transições com Maior Tempo de Espera</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_pre_mining['top_handoffs'], use_container_width=True)
                    with st.container(border=True):
                        st.markdown("<h4>Atividades Mais Frequentes</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_pre_mining['top_activities_plot'], use_container_width=True)
                with c2:
                    with st.container(border=True):
                        st.markdown("<h4>Transições com Maior Custo de Espera</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_pre_mining['top_handoffs_cost'], use_container_width=True)
                    with st.container(border=True):
                        st.markdown("<h4>Atividades com Maior Tempo de Execução</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_pre_mining['activity_service_times'], use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("<h4>Recursos com Mais Horas Trabalhadas</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_pre_mining['resource_workload'], use_container_width=True)
                with c2:
                    with st.container(border=True):
                        st.markdown("<h4>Custo por Tipo de Recurso</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_pre_mining['cost_by_resource_type'], use_container_width=True)
                
                with st.container(border=True):
                    st.markdown("<h4>Heatmap de Esforço (Horas) por Recurso e Atividade</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.plots_pre_mining['resource_activity_matrix'], use_container_width=True)

        with tab2:
            st.markdown("### Descoberta e Análise de Conformidade")
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("<h4>Modelo de Processo (Inductive Miner)</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.plots_post_mining['model_inductive_petrinet'], use_container_width=True)
                    st.markdown("<h6>Métricas de Qualidade</h6>", unsafe_allow_html=True)
                    st.json(st.session_state.metrics['inductive_miner'])
            with c2:
                with st.container(border=True):
                    st.markdown("<h4>Modelo de Processo (Heuristics Miner)</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.plots_post_mining['model_heuristic_petrinet'], use_container_width=True)
                    st.markdown("<h6>Métricas de Qualidade</h6>", unsafe_allow_html=True)
                    st.json(st.session_state.metrics['heuristics_miner'])

            with st.container(border=True):
                st.markdown("<h4>Heatmap de Performance (Tempo Médio entre Atividades)</h4>", unsafe_allow_html=True)
                st.image(st.session_state.plots_post_mining['performance_heatmap'], use_container_width=True)

            with st.expander("Análise Detalhada de Variantes e Recursos"):
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown("<h4>Duração das Variantes Mais Comuns</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_post_mining['variant_duration_plot'], use_container_width=True)
                    if 'milestone_time_analysis_plot' in st.session_state.plots_post_mining:
                        with st.container(border=True):
                            st.markdown("<h4>Tempo entre Marcos do Processo</h4>", unsafe_allow_html=True)
                            st.image(st.session_state.plots_post_mining['milestone_time_analysis_plot'], use_container_width=True)

                with c2:
                    with st.container(border=True):
                        st.markdown("<h4>Conformidade do Processo ao Longo do Tempo</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_post_mining['conformance_over_time_plot'], use_container_width=True)
                    with st.container(border=True):
                        st.markdown("<h4>Rede Social de Colaboração (Handovers)</h4>", unsafe_allow_html=True)
                        st.image(st.session_state.plots_post_mining['resource_network_adv'], use_container_width=True)
