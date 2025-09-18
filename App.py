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
    page_icon="📊",
    layout="wide"
)

# Estilo CSS para replicar a estética da app de referência
st.markdown("""
<style>
    /* Cor de fundo principal e da barra lateral */
    .stApp, .css-1d391kg {
        background-color: #F0F2F6; /* Um cinza claro para o fundo */
    }
    .css-163ttbj { /* Conteúdo principal */
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Estilo da barra lateral */
    .css-18e3th9 {
        background-color: #0F172A; /* Azul escuro */
    }
    .css-1d391kg a {
        color: #FFFFFF; /* Links na barra lateral */
    }

    /* Títulos e Cabeçalhos */
    h1, h2, h3 {
        color: #1E293B; /* Azul mais escuro para texto */
    }
    h2 {
        border-bottom: 2px solid #3B82F6; /* Linha azul abaixo do H2 */
        padding-bottom: 10px;
    }
    h3 {
        color: #334155;
    }

    /* Botões */
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }

    /* Abas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: transparent;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
        color: #3B82F6;
        font-weight: bold;
	}

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        color: #1E293B;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
# Usado para armazenar os dados e resultados entre as interações do usuário

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


# --- 3. FUNÇÕES DE ANÁLISE (ADAPTADAS DO NOTEBOOK) ---
# Envolvemos a lógica do notebook em funções para serem chamadas pelo Streamlit

# Usamos o cache do Streamlit para evitar re-execuções pesadas
@st.cache_data
def run_pre_mining_analysis(dfs):
    """
    Executa todas as análises da penúltima célula do notebook.
    Retorna dicionários com os plots e tabelas gerados.
    """
    plots = {}
    tables = {}
    
    # --- Cópia e Pré-processamento inicial ---
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    # Conversões de data
    for df in [df_projects, df_tasks]:
        for col in ['start_date', 'end_date', 'planned_end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'allocation_date' in df_resource_allocations.columns:
        df_resource_allocations['allocation_date'] = pd.to_datetime(df_resource_allocations['allocation_date'], errors='coerce')

    # Engenharia de funcionalidades
    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['project_name'].str.extract(r'Projeto \d+: (.*?) ')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)
    df_projects['completion_quarter'] = df_projects['end_date'].dt.to_period('Q').astype(str)
    
    df_tasks['task_duration_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.days

    # Agregações e custos
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

    # DataFrame Unificado
    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    allocations_to_merge = df_resource_allocations.drop(columns=['project_id'], errors='ignore')
    df_full_context = df_full_context.merge(allocations_to_merge, on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

    # --- GERAÇÃO DE GRÁFICOS E TABELAS ---

    # KPIs de Alto Nível
    monthly_throughput = df_projects.groupby(df_projects['end_date'].dt.to_period('M').astype(str)).size().mean()
    kpi_data = {
        'Métrica': ['Total de Projetos (Casos)', 'Total de Tarefas', 'Total de Eventos no Log', 'Total de Recursos Únicos', 'Duração Média dos Projetos (dias)', 'Produtividade Média (Projetos/Mês)'],
        'Valor': [df_projects['project_id'].nunique(), len(df_tasks), len(df_alloc_costs), df_resources['resource_id'].nunique(), df_projects['actual_duration_days'].mean(), monthly_throughput]
    }
    tables['kpi_df'] = pd.DataFrame(kpi_data)

    # Outliers
    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)

    # 1. Matriz de Performance
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=100, alpha=0.7, palette='bright', ax=ax)
    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.axvline(0, color='black', linestyle='--', lw=1)
    ax.set_title('Matriz de Performance: Prazo vs. Orçamento')
    plots['performance_matrix'] = fig
    
    # ... (Continue para todos os gráficos da célula 2 do notebook) ...
    
    # 2. Distribuição da Duração dos Projetos
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x=df_projects['actual_duration_days'], color='skyblue', ax=ax)
    sns.stripplot(x=df_projects['actual_duration_days'], color='blue', size=4, jitter=True, alpha=0.5, ax=ax)
    ax.set_title('Distribuição da Duração dos Projetos (Lead Time)')
    plots['case_durations_boxplot'] = fig

    # Lógica de handoff e service times
    log_df_final = df_tasks[['project_id', 'task_id', 'task_name', 'end_date']].copy()
    log_df_final.rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True)

    df_handoff_analysis = log_df_final.copy()
    # ***** LINHA CORRIGIDA AQUI *****
    df_handoff_analysis.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_handoff_analysis['previous_activity_end_time'] = df_handoff_analysis.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff_analysis['handoff_time_days'] = (df_handoff_analysis['time:timestamp'] - df_handoff_analysis['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff_analysis['previous_activity'] = df_handoff_analysis.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff_analysis.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')

    # 3. Top 10 Handoffs
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', hue='transition', palette='magma', legend=False, ax=ax)
    ax.set_title('Top 10 Transições com Maior Tempo de Espera (Handoff)')
    plots['top_handoffs'] = fig

    # 4. Custo de Espera Estimado
    avg_project_cost_per_day = df_projects['cost_per_day'].mean()
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * avg_project_cost_per_day
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', hue='transition', palette='Reds_r', legend=False, ax=ax)
    ax.set_title('Top 10 Transições por Custo de Espera Estimado')
    plots['top_handoffs_cost'] = fig
    
    # 5. Tempo Médio de Execução por Atividade
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), hue='task_name', palette='viridis', legend=False, ax=ax)
    ax.set_title('Tempo Médio de Execução por Atividade')
    plots['activity_service_times'] = fig
    
    # 6. Atividades Mais Frequentes
    activity_counts = df_tasks["task_name"].value_counts()
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax)
    ax.set_title('Atividades Mais Frequentes')
    plots['top_activities_plot'] = fig
    
    # 7. Top 10 Recursos por Horas Trabalhadas
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), hue='resource_name', palette='plasma', legend=False, ax=ax)
    ax.set_title('Top 10 Recursos por Horas Trabalhadas')
    plots['resource_workload'] = fig
    
    # 8. Heatmap de Esforço
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(resource_activity_matrix_pivot, cmap='YlGnBu', annot=True, fmt=".0f", ax=ax)
    ax.set_title('Heatmap de Esforço (Horas) por Recurso e Atividade')
    plots['resource_activity_matrix'] = fig
    
    # 9. Custo por Tipo de Recurso
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', hue='resource_type', palette='magma', legend=False, ax=ax)
    ax.set_title('Principais Direcionadores de Custo por Tipo de Recurso')
    plots['cost_by_resource_type'] = fig

    # 10. Impacto do Tamanho da Equipa no Atraso
    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()
    bins = np.linspace(min_res, max_res, 5, dtype=int) if max_res > min_res else [min_res, max_res]
    df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', hue='team_size_bin_dynamic', palette='flare', legend=False, ax=ax)
    ax.set_title('Impacto do Tamanho da Equipa no Atraso')
    plots['delay_by_teamsize'] = fig
    
    # 11. Benchmark de Duração Mediana por Tamanho da Equipa
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', hue='team_size_bin_dynamic', palette='crest', legend=False, ax=ax)
    ax.set_title('Benchmark de Duração Mediana por Tamanho da Equipa')
    plots['median_duration_by_teamsize'] = fig
    
    # 12. Análise de Eficiência Semanal
    df_alloc_costs['day_of_week'] = pd.to_datetime(df_alloc_costs['allocation_date']).dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', hue='day_of_week', palette='plasma', legend=False, ax=ax)
    ax.set_title('Análise de Eficiência Semanal')
    plots['weekly_efficiency'] = fig
    
    # 13. Análise de Gargalos (Tempo de Serviço vs. Tempo de Espera)
    df_tasks_analysis = df_tasks.copy()
    df_tasks_analysis['service_time_days'] = (pd.to_datetime(df_tasks_analysis['end_date']) - pd.to_datetime(df_tasks_analysis['start_date'])).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True)
    df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (pd.to_datetime(df_tasks_analysis['start_date']) - pd.to_datetime(df_tasks_analysis['previous_task_end'])).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    fig, ax = plt.subplots(figsize=(12, 7))
    bottleneck_by_activity.plot(kind='bar', stacked=True, color=['royalblue', 'crimson'], ax=ax)
    ax.set_ylabel('Dias')
    ax.set_xlabel('Tipo de Tarefa')
    ax.tick_params(axis='x', rotation=45)
    ax.set_title('Análise de Gargalos (Tempo de Serviço vs. Tempo de Espera)')
    plots['service_vs_wait_stacked'] = fig
    
    # 14. Matriz de Handoffs por Tipo de Equipa
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
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(handoff_matrix, annot=True, fmt=".0f", cmap="BuPu", ax=ax)
    ax.set_title('Matriz de Handoffs por Tipo de Equipa')
    plots['handoff_matrix_by_type'] = fig

    return plots, tables, df_full_context, log_df_pm4py # Retorna os DFs processados

@st.cache_data
def run_post_mining_analysis(log_df_pm4py_cached, dfs_cached):
    """
    Executa todas as análises da última célula do notebook.
    Retorna dicionários com os plots, tabelas e métricas.
    """
    plots = {}
    metrics = {}
    
    # Reconstituir objetos pm4py a partir de dataframes serializáveis
    log_df_final = log_df_pm4py_cached
    df_projects = dfs_cached['projects'].copy()
    df_tasks_raw = dfs_cached['tasks'].copy() # Usar o raw para certos cálculos
    df_resources = dfs_cached['resources'].copy()

    # Garantias de tipo de dados
    log_df_final['case:concept:name'] = log_df_final['case:concept:name'].astype(str)
    df_tasks_raw['start_date'] = pd.to_datetime(df_tasks_raw['start_date'])
    df_tasks_raw['end_date'] = pd.to_datetime(df_tasks_raw['end_date'])
    df_projects['start_date'] = pd.to_datetime(df_projects['start_date'])
    df_projects['end_date'] = pd.to_datetime(df_projects['end_date'])
    df_projects['project_id'] = df_projects['project_id'].astype(str)
    df_tasks_raw['project_id'] = df_tasks_raw['project_id'].astype(str)
    
    event_log_pm4py = pm4py.convert_to_event_log(log_df_final)

    # 1. Descoberta de Modelos
    variants_dict = variants_filter.get_variants(event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_3_variants = variants_filter.apply(event_log_pm4py, top_variant_names)

    # Inductive Miner
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = gviz_im
    
    fitness_im = replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    precision_im = precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    metrics['inductive_miner'] = {"Fitness": fitness_im.get('average_trace_fitness', 0), "Precisão": precision_im}

    # Heuristics Miner
    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = gviz_hm
    
    fitness_hm = replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    precision_hm = precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    metrics['heuristics_miner'] = {"Fitness": fitness_hm.get('average_trace_fitness', 0), "Precisão": precision_hm}

    # 2. Gantt Chart
    fig, ax_gantt = plt.subplots(figsize=(20, max(10, len(df_projects) * 0.4)))
    all_projects = df_projects.sort_values('start_date')['project_id'].tolist()
    gantt_data = df_tasks_raw[df_tasks_raw['project_id'].isin(all_projects)].sort_values(['project_id', 'start_date'])
    project_y_map = {proj_id: i for i, proj_id in enumerate(all_projects)}
    task_colors = plt.get_cmap('viridis', gantt_data['task_name'].nunique())
    color_map = {task_name: task_colors(i) for i, task_name in enumerate(gantt_data['task_name'].unique())}

    for _, task in gantt_data.iterrows():
        y_pos = project_y_map[task['project_id']]
        ax_gantt.barh(y_pos, (task['end_date'] - task['start_date']).days + 1, left=task['start_date'], height=0.6, color=color_map[task['task_name']], edgecolor='black')
    
    ax_gantt.set_yticks(list(project_y_map.values()))
    ax_gantt.set_yticklabels([f"Projeto {pid}" for pid in project_y_map.keys()])
    ax_gantt.invert_yaxis()
    ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]
    ax_gantt.legend(handles, color_map.keys(), title='Tipo de Tarefa', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_gantt.set_title('Linha do Tempo de Todos os Projetos (Gantt Chart)')
    fig.tight_layout()
    plots['gantt_chart_all_projects'] = fig
    
    # 3. Heatmap de Performance
    dfg_perf, _, _ = pm4py.discover_performance_dfg(event_log_pm4py)
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=event_log_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = gviz_dfg
    
    # ***** INÍCIO DA SECÇÃO CORRIGIDA *****
    # 4. Rede Social de Recursos
    handovers = {}
    for _, group in log_df_final.groupby('case:concept:name'):
        resources = group.sort_values('time:timestamp')['org:resource'].tolist()
        for i in range(len(resources) - 1):
            if resources[i] != resources[i+1]:
                pair = (resources[i], resources[i+1])
                handovers[pair] = handovers.get(pair, 0) + 1
    
    fig_net, ax_net = plt.subplots(figsize=(14, 14))
    G = nx.DiGraph()
    for (source, target), weight in handovers.items(): G.add_edge(source, target, weight=weight)
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='gray', width=[w*0.5 for w in weights], ax=ax_net, font_size=10, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net)
    ax_net.set_title('Rede Social de Recursos (Handover Network)')
    
    # Converter a figura para bytes em vez de guardar o objeto
    img_buf = io.BytesIO()
    fig_net.savefig(img_buf, format='png', bbox_inches='tight')
    plots['resource_network_adv'] = img_buf
    plt.close(fig_net) # Fechar a figura para libertar memória
    # ***** FIM DA SECÇÃO CORRIGIDA *****
    
    # 5. Gráfico de Variantes de Processo com Duração Média
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
    variant_durations['variant_str'] = variant_durations['variant'].apply(lambda x: ' -> '.join([str(i) for i in x]))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='avg_duration_hours', y='variant_str', data=variant_durations.astype({'avg_duration_hours':'float'}), palette='plasma', ax=ax, hue='variant_str', legend=False)
    ax.set_title('Duração Média das 10 Variantes de Processo Mais Comuns')
    fig.tight_layout()
    plots['variant_duration_plot'] = fig
    
    # 6. Score de Conformidade ao Longo do Tempo
    aligned_traces = alignments_miner.apply(event_log_pm4py, net_im, im_im, fm_im)
    case_fitness_data = [{'project_id': str(trace.attributes['concept:name']), 'fitness': alignment['fitness']} for trace, alignment in zip(event_log_pm4py, aligned_traces)]
    case_fitness_df = pd.DataFrame(case_fitness_data)
    case_fitness_df = case_fitness_df.merge(df_projects[['project_id', 'end_date']], on='project_id')
    case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str)
    monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax)
    ax.set_title('Score de Conformidade ao Longo do Tempo')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45); fig.tight_layout()
    plots['conformance_over_time_plot'] = fig
    
    # 7. Análise de Tempo entre Marcos
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
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(data=df_milestone_pairs, x='duration_hours', y='transition', ax=ax, hue='transition', legend=False, orient='h')
        ax.set_title('Análise de Tempo entre Marcos do Processo'); fig.tight_layout()
        plots['milestone_time_analysis_plot'] = fig

    # 8. Rede de Recursos por Função (Grafo Bipartido)
    if 'skill_level' in df_resources.columns:
        resource_roles = df_resources[['resource_name', 'skill_level']].rename(columns={'skill_level':'org:role'})
        log_df_bipartite = log_df_final.merge(resource_roles, left_on='org:resource', right_on='resource_name', how='left')
        
        resource_role_counts = log_df_bipartite.groupby(['org:resource', 'org:role']).size().reset_index(name='count')
        G_bipartite = nx.Graph()
        resources_nodes = resource_role_counts['org:resource'].unique()
        roles_nodes = resource_role_counts['org:role'].unique()
        G_bipartite.add_nodes_from(resources_nodes, bipartite=0)
        G_bipartite.add_nodes_from(roles_nodes, bipartite=1)
        for _, row in resource_role_counts.iterrows():
            G_bipartite.add_edge(row['org:resource'], row['org:role'], weight=row['count'])
        
        fig, ax = plt.subplots(figsize=(15, 12))
        pos = nx.bipartite_layout(G_bipartite, resources_nodes, align='vertical')
        nx.draw(G_bipartite, pos, with_labels=True, node_color=['skyblue' if node in resources_nodes else 'lightgreen' for node in G_bipartite], 
                node_size=2000, ax=ax, font_size=8)
        edge_labels = nx.get_edge_attributes(G_bipartite, 'weight')
        nx.draw_networkx_edge_labels(G_bipartite, pos, edge_labels=edge_labels, ax=ax)
        ax.set_title('Rede de Recursos por Função (Grafo Bipartido)')
        
        # Converter para bytes
        img_buf_bipartite = io.BytesIO()
        fig.savefig(img_buf_bipartite, format='png', bbox_inches='tight')
        plots['resource_network_bipartite'] = img_buf_bipartite
        plt.close(fig)
    
    return plots, metrics

# --- 4. LAYOUT DA APLICAÇÃO (SIDEBAR E PÁGINAS) ---

st.sidebar.title("Painel de Análise de Processos")
st.sidebar.markdown("Navegue pelas secções da aplicação.")

page = st.sidebar.radio("Selecione a Página", ["Upload de Ficheiros", "Executar Análise", "Resultados da Análise"])

# ***** DEFINE file_names HERE SO ALL PAGES CAN ACCESS IT *****
file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']

# --- PÁGINA 1: UPLOAD ---
if page == "Upload de Ficheiros":
    st.header("1. Upload dos Ficheiros de Dados (.csv)")
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")

    # The list is now defined globally, no need to define it here again.
    
    cols = st.columns(3)
    for i, name in enumerate(file_names):
        with cols[i % 3]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.dfs[name] = df
                    st.success(f"`{name}.csv` carregado.")
                except Exception as e:
                    st.error(f"Erro ao ler o ficheiro {name}.csv: {e}")

    # Pré-visualização dos dados
    all_files_uploaded = all(st.session_state.dfs[name] is not None for name in file_names)
    
    if all_files_uploaded:
        st.subheader("Pré-visualização dos Dados Carregados")
        for name, df in st.session_state.dfs.items():
            with st.expander(f"Visualizar as primeiras 5 linhas de `{name}.csv`"):
                st.dataframe(df.head())

# --- PÁGINA 2: EXECUÇÃO ---
elif page == "Executar Análise":
    st.header("2. Execução da Análise de Processos")
    all_files_uploaded = all(st.session_state.dfs[name] is not None for name in file_names)

    if not all_files_uploaded:
        st.warning("Por favor, carregue todos os 5 ficheiros CSV na página de 'Upload' antes de continuar.")
    else:
        st.info("Todos os ficheiros estão carregados. Clique no botão abaixo para iniciar a análise completa.")
        if st.button("🚀 Iniciar Análise Completa", use_container_width=True):
            with st.spinner("A executar a análise pré-mineração... Isto pode demorar um momento."):
                plots_pre, tables_pre, df_full, log_df_pm4py = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                # Guardar versões serializáveis para o cache do post-mining
                st.session_state.log_df_for_cache = pm4py.convert_to_dataframe(log_df_pm4py)
                st.session_state.dfs_for_cache = {
                    'projects': st.session_state.dfs['projects'].copy(),
                    'tasks': st.session_state.dfs['tasks'].copy(),
                    'resources': st.session_state.dfs['resources'].copy()
                }

            with st.spinner("A executar a análise de Process Mining... Esta é a parte mais demorada."):
                plots_post, metrics = run_post_mining_analysis(st.session_state.log_df_for_cache, st.session_state.dfs_for_cache)
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics

            st.session_state.analysis_run = True
            st.success("✅ Análise concluída com sucesso! Navegue para a página de 'Resultados da Análise' para explorar os insights.")
            st.balloons()

# --- PÁGINA 3: RESULTADOS ---
elif page == "Resultados da Análise":
    st.header("3. Resultados da Análise")

    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor, vá à página 'Executar Análise' e inicie o processo.")
    else:
        tab1, tab2 = st.tabs(["📊 Análise Pré-Mineração", "⛏️ Análise de Process Mining (Pós-Mineração)"])

        # Aba de Pré-Mineração
        with tab1:
            st.subheader("Análise Exploratória e de Performance")
            
            # Subseção: KPIs e Análise de Casos
            with st.expander(" KPIs de Alto Nível e Análise de Casos", expanded=True):
                st.markdown("#### Painel de KPIs")
                st.table(st.session_state.tables_pre_mining['kpi_df'].set_index('Métrica'))
                st.pyplot(st.session_state.plots_pre_mining['performance_matrix'])
                st.pyplot(st.session_state.plots_pre_mining['case_durations_boxplot'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Top 5 Projetos Outliers por Duração")
                    st.dataframe(st.session_state.tables_pre_mining['outlier_duration'])
                with col2:
                    st.markdown("##### Top 5 Projetos Outliers por Custo")
                    st.dataframe(st.session_state.tables_pre_mining['outlier_cost'])

            # Subseção: Análise de Atividades e Handoffs
            with st.expander("Análise de Atividades e Handoffs"):
                st.pyplot(st.session_state.plots_pre_mining['activity_service_times'])
                st.pyplot(st.session_state.plots_pre_mining['top_handoffs'])
                st.pyplot(st.session_state.plots_pre_mining['top_handoffs_cost'])
                st.pyplot(st.session_state.plots_pre_mining['top_activities_plot'])

            # Subseção: Análise Organizacional (Recursos)
            with st.expander("Análise Organizacional (Recursos)"):
                st.pyplot(st.session_state.plots_pre_mining['resource_workload'])
                st.pyplot(st.session_state.plots_pre_mining['cost_by_resource_type'])
                st.pyplot(st.session_state.plots_pre_mining['resource_activity_matrix'])
                st.pyplot(st.session_state.plots_pre_mining['handoff_matrix_by_type'])

            # Subseção: Análise Aprofundada (Causa-Raiz, Financeira e Benchmarking)
            with st.expander("Análise Aprofundada (Causa-Raiz e Benchmarking)"):
                st.pyplot(st.session_state.plots_pre_mining['delay_by_teamsize'])
                st.pyplot(st.session_state.plots_pre_mining['median_duration_by_teamsize'])
                st.pyplot(st.session_state.plots_pre_mining['weekly_efficiency'])
                st.pyplot(st.session_state.plots_pre_mining['service_vs_wait_stacked'])

        # Aba de Pós-Mineração (Process Mining)
        with tab2:
            st.subheader("Descoberta, Conformidade e Análise Aprofundada de Processos")

            with st.expander("Descoberta e Avaliação de Modelos de Processo", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Modelo de Processo (Inductive Miner)")
                    st.graphviz_chart(st.session_state.plots_post_mining['model_inductive_petrinet'])
                    st.markdown("##### Métricas de Qualidade")
                    st.json(st.session_state.metrics['inductive_miner'])
                with col2:
                    st.markdown("#### Modelo de Processo (Heuristics Miner)")
                    st.graphviz_chart(st.session_state.plots_post_mining['model_heuristic_petrinet'])
                    st.markdown("##### Métricas de Qualidade")
                    st.json(st.session_state.metrics['heuristics_miner'])

            with st.expander("Análise de Performance e Tempo de Ciclo"):
                st.pyplot(st.session_state.plots_post_mining['gantt_chart_all_projects'])
                st.markdown("#### Heatmap de Performance no Processo")
                st.graphviz_chart(st.session_state.plots_post_mining['performance_heatmap'])
                st.pyplot(st.session_state.plots_post_mining['variant_duration_plot'])
                if 'milestone_time_analysis_plot' in st.session_state.plots_post_mining:
                    st.pyplot(st.session_state.plots_post_mining['milestone_time_analysis_plot'])


            with st.expander("Análise de Conformidade e Variantes"):
                st.pyplot(st.session_state.plots_post_mining['conformance_over_time_plot'])
                # Adicionar aqui outros gráficos de variantes se existirem

            with st.expander("Análise de Recursos e Colaboração"):
                st.image(st.session_state.plots_post_mining['resource_network_adv'])
                if 'resource_network_bipartite' in st.session_state.plots_post_mining:
                    st.image(st.session_state.plots_post_mining['resource_network_bipartite'])





