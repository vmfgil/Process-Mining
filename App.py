import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io
from streamlit_option_menu import option_menu

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

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILO AVANÇADO ---
st.set_page_config(
    page_title="Process Mining Dashboard",
    page_icon="✨",
    layout="wide"
)

# Estilo CSS para a transformação completa
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    .stApp { background-color: #F0F2F6; }
    .main .block-container { padding: 1rem 2rem 2rem 2rem; }

    /* KPI Cards Personalizados */
    .kpi-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #E2E8F0;
        height: 100%;
    }
    .kpi-card h3 { color: #475569; font-size: 1rem; font-weight: 600; margin: 0; padding: 0; text-transform: uppercase; }
    .kpi-card p { color: #0F172A; font-size: 2.5rem; font-weight: 700; margin: 5px 0 0 0; padding: 0; letter-spacing: -1px; }

    h1 { color: #0F172A; font-weight: 700; letter-spacing: -2px; }
    h2 { color: #1E293B; font-weight: 600; border: none; padding-top: 20px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border: none; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 8px; padding: 10px 20px; color: #475569; border: none; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; color: #3B82F6; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
    
    .streamlit-expanderHeader { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; font-weight: 600; color: #1E293B; }
    .streamlit-expanderContent { background-color: #FFFFFF; border-left: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; border-radius: 0 0 8px 8px; margin-top: -8px; padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}


# --- FUNÇÕES DE ANÁLISE (LÓGICA ORIGINAL E ESTÁVEL DO SEU CÓDIGO) ---
@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    
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

    monthly_throughput = df_projects.groupby(df_projects['end_date'].dt.to_period('M').astype(str)).size().mean()
    kpi_data = {
        'Métrica': ['Total de Projetos (Casos)', 'Total de Tarefas', 'Total de Eventos no Log', 'Total de Recursos Únicos', 'Duração Média dos Projetos (dias)', 'Produtividade Média (Projetos/Mês)'],
        'Valor': [df_projects['project_id'].nunique(), len(df_tasks), len(df_alloc_costs), df_resources['resource_id'].nunique(), df_projects['actual_duration_days'].mean(), monthly_throughput]
    }
    tables['kpi_df'] = pd.DataFrame(kpi_data)

    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(10, 6)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=100, alpha=0.7, palette='bright', ax=ax); ax.axhline(0, color='black', linestyle='--', lw=1); ax.axvline(0, color='black', linestyle='--', lw=1); ax.set_title('Matriz de Performance: Prazo vs. Orçamento')
    plots['performance_matrix'] = fig
    
    fig, ax = plt.subplots(figsize=(10, 5)); sns.boxplot(x=df_projects['actual_duration_days'], color='skyblue', ax=ax); sns.stripplot(x=df_projects['actual_duration_days'], color='blue', size=4, jitter=True, alpha=0.5, ax=ax); ax.set_title('Distribuição da Duração dos Projetos (Lead Time)')
    plots['case_durations_boxplot'] = fig

    log_df_final = df_tasks[['project_id', 'task_id', 'task_name', 'end_date']].copy()
    log_df_final.rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True)
    df_handoff_analysis = log_df_final.copy()
    df_handoff_analysis.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
    df_handoff_analysis['previous_activity_end_time'] = df_handoff_analysis.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff_analysis['handoff_time_days'] = (df_handoff_analysis['time:timestamp'] - df_handoff_analysis['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff_analysis['previous_activity'] = df_handoff_analysis.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff_analysis.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', hue='transition', palette='magma', legend=False, ax=ax); ax.set_title('Top 10 Transições com Maior Tempo de Espera (Handoff)')
    plots['top_handoffs'] = fig

    avg_project_cost_per_day = df_projects['cost_per_day'].mean()
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * avg_project_cost_per_day
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', hue='transition', palette='Reds_r', legend=False, ax=ax); ax.set_title('Top 10 Transições por Custo de Espera Estimado')
    plots['top_handoffs_cost'] = fig
    
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), hue='task_name', palette='viridis', legend=False, ax=ax); ax.set_title('Tempo Médio de Execução por Atividade')
    plots['activity_service_times'] = fig
    
    activity_counts = df_tasks["task_name"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax); ax.set_title('Atividades Mais Frequentes')
    plots['top_activities_plot'] = fig
    
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), hue='resource_name', palette='plasma', legend=False, ax=ax); ax.set_title('Top 10 Recursos por Horas Trabalhadas')
    plots['resource_workload'] = fig
    
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots(figsize=(14, 9)); sns.heatmap(resource_activity_matrix_pivot, cmap='YlGnBu', annot=True, fmt=".0f", ax=ax); ax.set_title('Heatmap de Esforço (Horas) por Recurso e Atividade')
    plots['resource_activity_matrix'] = fig
    
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', hue='resource_type', palette='magma', legend=False, ax=ax); ax.set_title('Custo por Tipo de Recurso')
    plots['cost_by_resource_type'] = fig

    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()
    bins = np.linspace(min_res, max_res, 5, dtype=int) if max_res > min_res else [min_res, max_res]
    df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', hue='team_size_bin_dynamic', palette='flare', legend=False, ax=ax); ax.set_title('Impacto do Tamanho da Equipa no Atraso')
    plots['delay_by_teamsize'] = fig
    
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', hue='team_size_bin_dynamic', palette='crest', legend=False, ax=ax); ax.set_title('Duração Mediana por Tamanho da Equipa')
    plots['median_duration_by_teamsize'] = fig
    
    df_alloc_costs['day_of_week'] = pd.to_datetime(df_alloc_costs['allocation_date']).dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', hue='day_of_week', palette='plasma', legend=False, ax=ax); ax.set_title('Eficiência Semanal (Horas Trabalhadas)'); plt.xticks(rotation=45)
    plots['weekly_efficiency'] = fig
    
    df_tasks_analysis = df_tasks.copy()
    df_tasks_analysis['service_time_days'] = (pd.to_datetime(df_tasks_analysis['end_date']) - pd.to_datetime(df_tasks_analysis['start_date'])).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True)
    df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (pd.to_datetime(df_tasks_analysis['start_date']) - pd.to_datetime(df_tasks_analysis['previous_task_end'])).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    fig, ax = plt.subplots(figsize=(10, 6)); bottleneck_by_activity.plot(kind='bar', stacked=True, color=['royalblue', 'crimson'], ax=ax); ax.set_ylabel('Dias'); ax.set_xlabel('Tipo de Tarefa'); ax.tick_params(axis='x', rotation=45); ax.set_title('Gargalos: Serviço vs. Espera')
    plots['service_vs_wait_stacked'] = fig
    
    df_start_events = df_tasks[['project_id', 'task_id', 'task_name', 'start_date']].copy(); df_start_events.rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True); df_start_events['lifecycle:transition'] = 'start'
    df_complete_events = df_tasks[['project_id', 'task_id', 'task_name', 'end_date']].copy(); df_complete_events.rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}, inplace=True); df_complete_events['lifecycle:transition'] = 'complete'
    log_df_temp = pd.concat([df_start_events, df_complete_events], ignore_index=True)
    resource_mapping = df_resource_allocations.groupby('task_id')['resource_id'].apply(list).reset_index()
    log_df_temp = log_df_temp.merge(resource_mapping, on='task_id', how='left').explode('resource_id')
    log_df_temp = log_df_temp.merge(df_resources[['resource_id', 'resource_name']], on='resource_id', how='left'); log_df_temp.rename(columns={'resource_name': 'org:resource'}, inplace=True)
    log_df_pm4py = log_converter.apply(log_df_temp)
    
    handoff_counts = {}
    for trace in log_df_pm4py:
        resources = [event['org:resource'] for event in trace if 'org:resource' in event and pd.notna(event['org:resource'])]
        for i in range(len(resources) - 1):
            if resources[i] != resources[i+1]:
                pair = (resources[i], resources[i+1]); handoff_counts[pair] = handoff_counts.get(pair, 0) + 1
    df_resource_handoffs = pd.DataFrame([{'De': k[0], 'Para': k[1], 'Contagem': v} for k, v in handoff_counts.items()]).sort_values('Contagem', ascending=False)
    
    df_rh_typed = df_resource_handoffs.merge(df_resources[['resource_name', 'resource_type']], left_on='De', right_on='resource_name').merge(df_resources[['resource_name', 'resource_type']], left_on='Para', right_on='resource_name', suffixes=('_de', '_para'))
    handoff_matrix = df_rh_typed.groupby(['resource_type_de', 'resource_type_para'])['Contagem'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(handoff_matrix, annot=True, fmt=".0f", cmap="BuPu", ax=ax); ax.set_title('Matriz de Handoffs por Tipo de Equipa')
    plots['handoff_matrix_by_type'] = fig

    return plots, tables, df_full_context, log_df_pm4py

@st.cache_data
def run_post_mining_analysis(log_df_pm4py_cached, dfs_cached):
    plots = {}
    metrics = {}
    
    log_df_final = log_df_pm4py_cached
    df_projects = dfs_cached['projects'].copy(); df_tasks_raw = dfs_cached['tasks'].copy(); df_resources = dfs_cached['resources'].copy()

    log_df_final['case:concept:name'] = log_df_final['case:concept:name'].astype(str)
    for df in [df_tasks_raw, df_projects]:
        for col in ['start_date', 'end_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col])
    for col in ['project_id', 'task_id']:
        if col in df_tasks_raw.columns: df_tasks_raw[col] = df_tasks_raw[col].astype(str)
        if col in df_projects.columns: df_projects[col] = df_projects[col].astype(str)
    
    event_log_pm4py = pm4py.convert_to_event_log(log_df_final)

    variants_dict = variants_filter.get_variants(event_log_pm4py)
    top_3_variants = variants_filter.apply(event_log_pm4py, sorted(variants_dict, key=lambda k: len(variants_dict[k]), reverse=True)[:3])

    net_im, im_im, fm_im = pt_converter.apply(inductive_miner.apply(top_3_variants)); gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = gviz_im
    metrics['inductive_miner'] = {"Fitness": replay_fitness_evaluator.apply(top_3_variants, net_im, im_im, fm_im).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(top_3_variants, net_im, im_im, fm_im)}

    net_hm, im_hm, fm_hm = heuristics_miner.apply(top_3_variants); gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = gviz_hm
    metrics['heuristics_miner'] = {"Fitness": replay_fitness_evaluator.apply(top_3_variants, net_hm, im_hm, fm_hm).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(top_3_variants, net_hm, im_hm, fm_hm)}

    dfg_perf, _, _ = pm4py.discover_performance_dfg(event_log_pm4py); gviz_dfg = dfg_visualizer.apply(dfg_perf, log=event_log_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = gviz_dfg
    
    handovers = {pair: handovers.get(pair, 0) + 1 for trace in log_df_final.groupby('case:concept:name') for i in range(len(trace) - 1) if (pair := (trace.iloc[i]['org:resource'], trace.iloc[i+1]['org:resource']))[0] != pair[1]}
    fig_net, ax_net = plt.subplots(figsize=(14, 14)); G = nx.DiGraph();
    for (source, target), weight in handovers.items(): G.add_edge(source, target, weight=weight)
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42); weights = [G[u][v]['weight'] for u,v in G.edges()];
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='gray', width=[w*0.5 for w in weights], ax=ax_net, font_size=10, connectionstyle='arc3,rad=0.1'); nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net); ax_net.set_title('Rede Social de Recursos (Handover Network)')
    img_buf = io.BytesIO(); fig_net.savefig(img_buf, format='png', bbox_inches='tight'); plt.close(fig_net); plots['resource_network_adv'] = img_buf
    
    return plots, metrics

# --- LAYOUT DA APLICAÇÃO ---
st.title("Painel de Análise de Processos")

with st.sidebar:
    st.markdown("<h1 style='color: white; font-size: 24px; letter-spacing: -1px; text-align: center;'>Process Mining</h1>", unsafe_allow_html=True)
    page = option_menu(
        menu_title=None,
        options=["Upload", "Executar Análise", "Resultados"],
        icons=["cloud-upload-fill", "play-circle-fill", "bar-chart-line-fill"],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0F172A"}, "icon": {"color": "#94A3B8", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#1E293B"},
            "nav-link-selected": {"background-color": "#3B82F6"},
        }
    )

if page == "Upload":
    st.header("1. Upload dos Ficheiros de Dados (.csv)")
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    cols = st.columns(3)
    for i, name in enumerate(file_names):
        with cols[i % 3]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                st.success(f"`{name}.csv` carregado.")

    if all(st.session_state.dfs[name] is not None for name in file_names):
        st.subheader("Pré-visualização dos Dados")
        for name, df in st.session_state.dfs.items():
            with st.expander(f"Ver dados de `{name}.csv`"): st.dataframe(df.head())

elif page == "Executar Análise":
    st.header("2. Execução da Análise")
    if not all(st.session_state.dfs[name] is not None for name in file_names):
        st.warning("Por favor, carregue todos os ficheiros na página 'Upload'.")
    else:
        if st.button("🚀 Iniciar Análise Completa", use_container_width=True):
            with st.spinner("A executar análise pré-mineração..."):
                plots_pre, tables_pre, df_full, log_df_pm4py = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre; st.session_state.tables_pre_mining = tables_pre
                st.session_state.log_df_for_cache = pm4py.convert_to_dataframe(log_df_pm4py)
                st.session_state.dfs_for_cache = {'projects': st.session_state.dfs['projects'].copy(), 'tasks': st.session_state.dfs['tasks'].copy(), 'resources': st.session_state.dfs['resources'].copy()}
            with st.spinner("A executar análise de Process Mining..."):
                plots_post, metrics = run_post_mining_analysis(st.session_state.log_df_for_cache, st.session_state.dfs_for_cache)
                st.session_state.plots_post_mining = plots_post; st.session_state.metrics = metrics
            st.session_state.analysis_run = True
            st.success("✅ Análise concluída! Navegue para 'Resultados'."); st.balloons()

elif page == "Resultados":
    st.header("Resultados da Análise")
    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada.")
    else:
        kpi_df = st.session_state.tables_pre_mining['kpi_df']
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='kpi-card'><h3>Total de Projetos</h3><p>{int(kpi_df.loc[kpi_df['Métrica'] == 'Total de Projetos (Casos)', 'Valor'].iloc[0])}</p></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='kpi-card'><h3>Duração Média</h3><p>{kpi_df.loc[kpi_df['Métrica'] == 'Duração Média dos Projetos (dias)', 'Valor'].iloc[0]:.1f}d</p></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='kpi-card'><h3>Total de Recursos</h3><p>{int(kpi_df.loc[kpi_df['Métrica'] == 'Total de Recursos Únicos', 'Valor'].iloc[0])}</p></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Análise Pré-Mineração", "⛏️ Análise de Process Mining"])
        
        with tab1:
            with st.expander("Análise de Casos e Performance 🔎", expanded=True):
                c1, c2 = st.columns(2)
                c1.pyplot(st.session_state.plots_pre_mining['performance_matrix'], use_container_width=True)
                c2.pyplot(st.session_state.plots_pre_mining['case_durations_boxplot'], use_container_width=True)
            with st.expander("Análise de Atividades e Handoffs ⏱️"):
                c1, c2 = st.columns(2)
                c1.pyplot(st.session_state.plots_pre_mining['activity_service_times'], use_container_width=True)
                c2.pyplot(st.session_state.plots_pre_mining['top_activities_plot'], use_container_width=True)
                c1.pyplot(st.session_state.plots_pre_mining['top_handoffs'], use_container_width=True)
                c2.pyplot(st.session_state.plots_pre_mining['top_handoffs_cost'], use_container_width=True)
                st.pyplot(st.session_state.plots_pre_mining['service_vs_wait_stacked'], use_container_width=True)
            with st.expander("Análise Organizacional e de Recursos 👥"):
                 c1, c2 = st.columns(2)
                 c1.pyplot(st.session_state.plots_pre_mining['resource_workload'], use_container_width=True)
                 c2.pyplot(st.session_state.plots_pre_mining['cost_by_resource_type'], use_container_width=True)
                 st.pyplot(st.session_state.plots_pre_mining['resource_activity_matrix'], use_container_width=True)
                 st.pyplot(st.session_state.plots_pre_mining['handoff_matrix_by_type'], use_container_width=True)
            with st.expander("Análise Aprofundada e Benchmarking 📈"):
                c1,c2 = st.columns(2)
                c1.pyplot(st.session_state.plots_pre_mining['delay_by_teamsize'], use_container_width=True)
                c2.pyplot(st.session_state.plots_pre_mining['median_duration_by_teamsize'], use_container_width=True)
                st.pyplot(st.session_state.plots_pre_mining['weekly_efficiency'], use_container_width=True)

        with tab2:
            with st.expander("Descoberta de Modelos de Processo 🗺️", expanded=True):
                c1, c2 = st.columns(2)
                with c1: st.markdown("<h4>Modelo (Inductive Miner)</h4>", unsafe_allow_html=True); st.graphviz_chart(st.session_state.plots_post_mining['model_inductive_petrinet'])
                with c2: st.markdown("<h4>Modelo (Heuristics Miner)</h4>", unsafe_allow_html=True); st.graphviz_chart(st.session_state.plots_post_mining['model_heuristic_petrinet'])
            with st.expander("Análise de Performance e Colaboração 🔥"):
                st.markdown("<h4>Heatmap de Performance</h4>", unsafe_allow_html=True)
                st.graphviz_chart(st.session_state.plots_post_mining['performance_heatmap'])
                st.image(st.session_state.plots_post_mining['resource_network_adv'], caption="Rede Social de Recursos")
