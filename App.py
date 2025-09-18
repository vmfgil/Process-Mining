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

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILO MODERNO ---
st.set_page_config(
    page_title="Painel de Análise de Processos",
    page_icon="🧭",
    layout="wide"
)

# Paleta de Cores Inspirada nas Imagens (Fintech Style)
PRIMARY_COLOR = "#00B68A" # Verde principal
SECONDARY_COLOR = "#2D3748" # Cinza escuro/carvão
TEXT_COLOR = "#1A202C" # Texto principal
SUBTLE_TEXT_COLOR = "#718096" # Texto secundário
POSITIVE_COLOR = "#00B68A" # Verde para ganhos
NEGATIVE_COLOR = "#E53E3E" # Vermelho para perdas
BACKGROUND_COLOR = "#F7FAFC"
CONTENT_BACKGROUND_COLOR = "#FFFFFF"
PALETTE_VIRIDIS = "viridis"
PALETTE_PLASMA = "plasma"
PALETTE_MAGMA = "magma"
PALETTE_ROCKET = "rocket"

# Estilo CSS para replicar a estética moderna
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {BACKGROUND_COLOR}; }}
    [data-testid="stSidebar"] {{ background-color: {SECONDARY_COLOR}; }}
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {{ color: #FFFFFF; }}
    [data-testid="stSidebar"] .st-emotion-cache-1g6k8xl {{ font-size: 1.5rem; }}
    [data-testid="stSidebar"] .st-emotion-cache-vd22tr p{{ color: #A0AEC0; font-size: 1rem; }}
    [data-testid="stSidebar"] .st-emotion-cache-1kyxreq label {{ background-color: rgba(0, 182, 138, 0.1); border-radius: 8px; padding: 10px 0; }}
    [data-testid="stSidebar"] .st-emotion-cache-1kyxreq p {{ color: {PRIMARY_COLOR}; font-weight: 600; }}
    .main .block-container {{ padding: 2rem 3rem; }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT_COLOR}; font-weight: 700; }}
    h1 {{ font-size: 2.25rem; }}
    h2 {{ font-size: 1.875rem; border-bottom: 2px solid {PRIMARY_COLOR}; padding-bottom: 10px; margin-bottom: 25px; }}
    h3 {{ font-size: 1.5rem; }}
    .card {{ background-color: {CONTENT_BACKGROUND_COLOR}; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 12px 0 rgba(0,0,0,0.05); border: 1px solid #E2E8F0; }}
    .kpi-card {{ background-color: {CONTENT_BACKGROUND_COLOR}; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0; text-align: left; }}
    .kpi-title {{ font-size: 0.9rem; color: {SUBTLE_TEXT_COLOR}; margin-bottom: 8px; }}
    .kpi-value {{ font-size: 1.75rem; font-weight: 700; color: {TEXT_COLOR}; margin-bottom: 8px; }}
    .stButton>button {{ background-color: {PRIMARY_COLOR}; color: white; border-radius: 8px; border: none; padding: 12px 24px; width: 100%; font-weight: 600; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; border-bottom: 1px solid #e2e8f0; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; background-color: transparent; padding: 10px 5px; color: {SUBTLE_TEXT_COLOR}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {PRIMARY_COLOR}; border-bottom: 3px solid {PRIMARY_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES ---
def convert_fig_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=150, transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

def style_plot(fig, ax):
    fig.patch.set_alpha(0)
    ax.set_facecolor(CONTENT_BACKGROUND_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_color(SUBTLE_TEXT_COLOR)
    ax.yaxis.label.set_color(SUBTLE_TEXT_COLOR)
    ax.tick_params(colors=SUBTLE_TEXT_COLOR)
    return fig, ax

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'dfs' not in st.session_state: st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}


# --- FUNÇÕES DE ANÁLISE (COMPLETAS E ESTILIZADAS) ---
@st.cache_data
def run_pre_mining_analysis(dfs):
    plots, tables = {}, {}
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    
    # Lógica de processamento de dados
    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['project_id', 'task_id', 'resource_id']:
        for df in [df_projects, df_tasks, df_resources, df_resource_allocations, dfs['dependencies']]:
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
    
    # --- GERAÇÃO DE TODOS OS GRÁFICOS E TABELAS ---

    # 1. KPIs & Outliers
    tables['kpi_df'] = pd.DataFrame({'Métrica': ['Total de Projetos', 'Total de Tarefas', 'Total de Recursos', 'Duração Média (dias)'], 'Valor': [len(df_projects), len(df_tasks), len(df_resources), df_projects['actual_duration_days'].mean()]})
    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.8, ax=ax, palette=PALETTE_VIRIDIS); ax.axhline(0, color=SUBTLE_TEXT_COLOR, ls='--'); ax.axvline(0, color=SUBTLE_TEXT_COLOR, ls='--'); ax.set_title("Matriz de Performance (Prazo vs. Custo)"); fig, ax = style_plot(fig, ax); plots['performance_matrix'] = convert_fig_to_bytes(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=df_projects['actual_duration_days'], ax=ax, color=PRIMARY_COLOR, width=0.4); sns.stripplot(x=df_projects['actual_duration_days'], color=SECONDARY_COLOR, size=4, jitter=True, alpha=0.5, ax=ax); ax.set_title("Distribuição da Duração dos Projetos"); fig, ax = style_plot(fig, ax); plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)

    # 2. Performance Detalhada
    lead_times = log_df_final.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"]).reset_index(); lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
    throughput_per_case = log_df_final.groupby("case:concept:name").apply(lambda g: g.sort_values("time:timestamp")["time:timestamp"].diff().dropna().mean().total_seconds() if len(g) > 1 else 0).reset_index(name="avg_throughput_seconds"); throughput_per_case["avg_throughput_hours"] = throughput_per_case["avg_throughput_seconds"] / 3600
    perf_df = pd.merge(lead_times, throughput_per_case, on="case:concept:name"); tables['perf_stats'] = perf_df[["lead_time_days", "avg_throughput_hours"]].describe()
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["lead_time_days"], bins=20, kde=True, ax=ax, color=PRIMARY_COLOR); ax.set_title("Distribuição do Lead Time (dias)"); fig, ax = style_plot(fig, ax); plots['lead_time_hist'] = convert_fig_to_bytes(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["avg_throughput_hours"], bins=20, kde=True, color=SECONDARY_COLOR, ax=ax); ax.set_title("Distribuição do Throughput (horas)"); fig, ax = style_plot(fig, ax); plots['throughput_hist'] = convert_fig_to_bytes(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=perf_df["avg_throughput_hours"], color=PRIMARY_COLOR, width=0.4, ax=ax); ax.set_title("Boxplot do Throughput (horas)"); fig, ax = style_plot(fig, ax); plots['throughput_boxplot'] = convert_fig_to_bytes(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax, color=SECONDARY_COLOR); ax.set_title("Relação Lead Time vs Throughput"); fig, ax = style_plot(fig, ax); plots['lead_time_vs_throughput'] = convert_fig_to_bytes(fig)

    # 3. Atividades e Handoffs
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index(); service_times['service_time_days'] = service_times['hours_worked'] / 8
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, palette=PALETTE_VIRIDIS); ax.set_title("Tempo Médio de Execução por Atividade"); fig, ax = style_plot(fig, ax); plots['activity_service_times'] = convert_fig_to_bytes(fig)
    
    df_handoff = log_df_final.sort_values(['case:concept:name', 'time:timestamp']); df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1); df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600); df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False); handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', ax=ax, palette=PALETTE_MAGMA); ax.set_title("Top 10 Handoffs por Tempo de Espera"); fig, ax = style_plot(fig, ax); plots['top_handoffs'] = convert_fig_to_bytes(fig)
    
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, palette=PALETTE_MAGMA); ax.set_title("Top 10 Handoffs por Custo de Espera"); fig, ax = style_plot(fig, ax); plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

    # 4. Análise Organizacional
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=df_tasks["task_name"].value_counts().head(10).values, y=df_tasks["task_name"].value_counts().head(10).index, ax=ax, color=PRIMARY_COLOR); ax.set_title("Atividades Mais Frequentes"); fig, ax = style_plot(fig, ax); plots['top_activities_plot'] = convert_fig_to_bytes(fig)
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, palette=PALETTE_PLASMA); ax.set_title("Top 10 Recursos por Horas Trabalhadas"); fig, ax = style_plot(fig, ax); plots['resource_workload'] = convert_fig_to_bytes(fig)
    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index(); resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), ax=ax, palette=PALETTE_PLASMA); ax.set_title("Recursos por Média de Tarefas por Projeto"); fig, ax = style_plot(fig, ax); plots['resource_avg_events'] = convert_fig_to_bytes(fig)
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_activity_matrix_pivot, cmap='Greens', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8}); ax.set_title("Heatmap de Esforço por Recurso e Atividade"); fig, ax = style_plot(fig, ax); plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)
    handoff_counts = Counter((trace[i]['org:resource'], trace[i+1]['org:resource']) for trace in event_log_pm4py for i in range(len(trace) - 1) if 'org:resource' in trace[i] and 'org:resource' in trace[i+1] and trace[i]['org:resource'] != trace[i+1]['org:resource'])
    df_resource_handoffs = pd.DataFrame(handoff_counts.most_common(10), columns=['Handoff', 'Contagem']); df_resource_handoffs['Handoff'] = df_resource_handoffs['Handoff'].apply(lambda x: f"{x[0]} -> {x[1]}")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs, ax=ax, palette=PALETTE_ROCKET); ax.set_title("Top 10 Handoffs entre Recursos"); fig, ax = style_plot(fig, ax); plots['resource_handoffs'] = convert_fig_to_bytes(fig)
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', ax=ax, palette=PALETTE_MAGMA); ax.set_title("Custo por Tipo de Recurso"); fig, ax = style_plot(fig, ax); plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)

    # 5. Variantes e Rework
    variants_df = log_df_final.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='trace'); variants_df['variant_str'] = variants_df['trace'].apply(lambda x: ' -> '.join(x))
    variant_analysis = variants_df['variant_str'].value_counts().reset_index(name='frequency'); variant_analysis['percentage'] = (variant_analysis['frequency'] / variant_analysis['frequency'].sum()) * 100
    tables['variants_table'] = variant_analysis.head(10)
    fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='frequency', y='variant_str', data=variant_analysis.head(10), ax=ax, orient='h', palette=PALETTE_PLASMA); ax.set_title("Top 10 Variantes de Processo por Frequência"); fig, ax = style_plot(fig, ax); plots['variants_frequency'] = convert_fig_to_bytes(fig)
    rework_loops = Counter(f"{trace[i]} -> {trace[i+1]} -> {trace[i]}" for trace in variants_df['trace'] for i in range(len(trace) - 2) if trace[i] == trace[i+2] and trace[i] != trace[i+1])
    tables['rework_loops_table'] = pd.DataFrame(rework_loops.most_common(10), columns=['rework_loop', 'frequency'])
    
    # 6. Análise Aprofundada
    delayed_projects = df_projects[df_projects['days_diff'] > 0]; tables['cost_of_delay_kpis'] = pd.DataFrame({'Métrica': ['Custo Total Projetos Atrasados', 'Atraso Médio (dias)', 'Custo Médio/Dia Atraso'], 'Valor': [delayed_projects['total_actual_cost'].sum(), delayed_projects['days_diff'].mean(), (delayed_projects.get('total_actual_cost', 0) / delayed_projects['days_diff']).mean()]})
    bins = np.linspace(df_projects['num_resources'].min(), df_projects['num_resources'].max(), 5, dtype=int); df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', ax=ax, palette=PALETTE_PLASMA); ax.set_title("Impacto do Tamanho da Equipa no Atraso"); fig, ax = style_plot(fig, ax); plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', ax=ax, palette=PALETTE_VIRIDIS); ax.set_title("Duração Mediana por Tamanho da Equipa"); fig, ax = style_plot(fig, ax); plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)
    df_alloc_costs['day_of_week'] = df_alloc_costs['allocation_date'].dt.day_name(); weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, palette=PALETTE_PLASMA); ax.set_title("Eficiência Semanal (Horas Trabalhadas)"); plt.xticks(rotation=45); fig, ax = style_plot(fig, ax); plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    df_tasks_analysis = df_tasks.copy(); df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60); df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True); df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1); df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60); df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates()
    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, palette=PALETTE_ROCKET); ax.set_title("Top 15 Recursos por Tempo Médio de Espera"); fig, ax = style_plot(fig, ax); plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    fig, ax = plt.subplots(figsize=(8, 5)); bottleneck_by_activity.plot(kind='bar', stacked=True, color=[PRIMARY_COLOR, NEGATIVE_COLOR], ax=ax); ax.set_title("Gargalos: Tempo de Serviço vs. Espera"); fig, ax = style_plot(fig, ax); plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=bottleneck_by_activity.reset_index(), x='service_time_days', y='waiting_time_days', ax=ax, color=SECONDARY_COLOR); ax.set_title("Tempo de Espera vs. Tempo de Execução"); fig, ax = style_plot(fig, ax); plots['wait_vs_service_scatter'] = convert_fig_to_bytes(fig)
    df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id', 'completion_month']], on='project_id'); monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax, color=PRIMARY_COLOR); plt.xticks(rotation=45); ax.set_title("Evolução do Tempo Médio de Espera"); fig, ax = style_plot(fig, ax); plots['wait_time_evolution'] = convert_fig_to_bytes(fig)
    df_perf_full = perf_df.merge(df_projects, left_on='case:concept:name', right_on='project_id')
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_perf_full, x='team_size_bin_dynamic', y='avg_throughput_hours', ax=ax, palette=PALETTE_PLASMA); ax.set_title("Benchmark de Throughput por Tamanho da Equipa"); fig, ax = style_plot(fig, ax); plots['throughput_benchmark_by_teamsize'] = convert_fig_to_bytes(fig)
    df_tasks['phase'] = df_tasks['task_type'].apply(lambda t: 'Desenvolvimento & Design' if t in ['Desenvolvimento', 'Correção', 'Revisão', 'Design'] else 'Teste (QA)' if t == 'Teste' else 'Operações & Deploy' if t in ['Deploy', 'DBA'] else 'Outros')
    phase_times = df_tasks.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index(); phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days; avg_cycle_time_by_phase = phase_times.groupby('phase')['cycle_time_days'].mean()
    fig, ax = plt.subplots(figsize=(8, 4)); avg_cycle_time_by_phase.plot(kind='bar', color=sns.color_palette(PALETTE_VIRIDIS, avg_cycle_time_by_phase.shape[0]), ax=ax); ax.set_title("Duração Média por Fase do Processo"); plt.xticks(rotation=0); fig, ax = style_plot(fig, ax); plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)

    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    plots, metrics = {}, {}
    df_start_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'start_date']].rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}); df_start_events['lifecycle:transition'] = 'start'
    df_complete_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'end_date']].rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'}); df_complete_events['lifecycle:transition'] = 'complete'
    log_df_full_lifecycle = pd.concat([df_start_events, df_complete_events]).sort_values('time:timestamp'); log_full_pm4py = pm4py.convert_to_event_log(log_df_full_lifecycle)

    # 1. Descoberta de Modelos
    variants_dict = variants_filter.get_variants(_event_log_pm4py); top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]; top_variant_names = [v[0] for v in top_variants_list]; log_top_3_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)
    
    pt_inductive = inductive_miner.apply(log_top_3_variants); net_im, im_im, fm_im = pt_converter.apply(pt_inductive); gviz_im = pn_visualizer.apply(net_im, im_im, fm_im); plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    
    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor']); fig, ax = plt.subplots(figsize=(8, 4)); barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, color=PRIMARY_COLOR);
        for p in barplot.patches: ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', color=TEXT_COLOR, weight='bold')
        ax.set_title(title); ax.set_ylim(0, 1.05); fig, ax = style_plot(fig, ax); return fig
        
    metrics_im = {"Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im), "Generalização": generalization_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im), "Simplicidade": simplicity_evaluator.apply(net_im)}
    plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)')); metrics['inductive_miner'] = metrics_im

    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5}); gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm); plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    metrics_hm = {"Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm), "Generalização": generalization_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm), "Simplicidade": simplicity_evaluator.apply(net_hm)}
    plots['metrics_heuristic'] = convert_fig_to_bytes(plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)')); metrics['heuristics_miner'] = metrics_hm

    # 2. Performance e Tempo
    kpi_temporal = _df_projects.groupby('completion_month').agg(avg_lead_time=('actual_duration_days', 'mean'), throughput=('project_id', 'count')).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6)); ax1.plot(kpi_temporal['completion_month'], kpi_temporal['avg_lead_time'], marker='o', color=PRIMARY_COLOR, label='Lead Time'); ax2 = ax1.twinx(); ax2.bar(kpi_temporal['completion_month'], kpi_temporal['throughput'], color=SECONDARY_COLOR, alpha=0.6, label='Throughput'); fig.suptitle('Séries Temporais de KPIs de Performance'); fig.legend(); fig, ax1 = style_plot(fig, ax1); plots['kpi_time_series'] = convert_fig_to_bytes(fig)
    dfg_perf, _, _ = pm4py.discover_performance_dfg(log_full_pm4py); gviz_dfg = dfg_visualizer.apply(dfg_perf, log=log_full_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE); plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)
    log_df_full_lifecycle['weekday'] = log_df_full_lifecycle['time:timestamp'].dt.day_name(); weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]; heatmap_data = log_df_full_lifecycle.groupby('weekday')['case:concept:name'].count().reindex(weekday_order).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(x=heatmap_data.index, y=heatmap_data.values, ax=ax, palette=PALETTE_VIRIDIS); ax.set_title('Ocorrências de Atividades por Dia da Semana'); plt.xticks(rotation=45); fig, ax = style_plot(fig, ax); plots['temporal_heatmap_fixed'] = convert_fig_to_bytes(fig)

    # 3. Análise de Recursos
    log_df_complete = pm4py.convert_to_dataframe(_event_log_pm4py); handovers = Counter((log_df_complete.iloc[i]['org:resource'], log_df_complete.iloc[i+1]['org:resource']) for i in range(len(log_df_complete)-1) if log_df_complete.iloc[i]['case:concept:name'] == log_df_complete.iloc[i+1]['case:concept:name'] and log_df_complete.iloc[i]['org:resource'] != log_df_complete.iloc[i+1]['org:resource'])
    fig_net, ax_net = plt.subplots(figsize=(10, 10)); G = nx.DiGraph();
    for (source, target), weight in handovers.items(): G.add_edge(str(source), str(target), weight=weight)
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42); weights = [G[u][v]['weight'] for u,v in G.edges()]; nx.draw(G, pos, with_labels=True, node_color=PRIMARY_COLOR, node_size=3000, edge_color='#CBD5E0', width=[w*0.5 for w in weights], ax=ax_net, font_size=10, font_color=CONTENT_BACKGROUND_COLOR, font_weight='bold', connectionstyle='arc3,rad=0.1'); nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net, font_color=TEXT_COLOR); ax_net.set_title('Rede Social de Recursos (Handover Network)'); fig_net, ax_net = style_plot(fig_net, ax_net); plots['resource_network_adv'] = convert_fig_to_bytes(fig_net)
    
    # 4. Outras Análises
    variants_df = log_df_full_lifecycle.groupby('case:concept:name').agg(variant=('concept:name', lambda x: tuple(x)), start_timestamp=('time:timestamp', 'min'), end_timestamp=('time:timestamp', 'max')).reset_index(); variants_df['duration_hours'] = (variants_df['end_timestamp'] - variants_df['start_timestamp']).dt.total_seconds() / 3600; variant_durations = variants_df.groupby('variant').agg(count=('case:concept:name', 'count'), avg_duration_hours=('duration_hours', 'mean')).reset_index().sort_values(by='count', ascending=False).head(10); variant_durations['variant_str'] = variant_durations['variant'].apply(lambda x: ' -> '.join([str(i) for i in x][:4]) + '...')
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_duration_hours', y='variant_str', data=variant_durations.astype({'avg_duration_hours':'float'}), ax=ax, palette=PALETTE_PLASMA); ax.set_title('Duração Média das 10 Variantes Mais Comuns'); fig.tight_layout(); fig, ax = style_plot(fig, ax); plots['variant_duration_plot'] = convert_fig_to_bytes(fig)
    
    aligned_traces = alignments_miner.apply(log_full_pm4py, net_im, im_im, fm_im); deviations_list = [{'fitness': trace['fitness'], 'deviations': sum(1 for move in trace['alignment'] if '>>' in move[0] or '>>' in move[1])} for trace in aligned_traces if 'fitness' in trace]; deviations_df = pd.DataFrame(deviations_list)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(x='fitness', y='deviations', data=deviations_df, alpha=0.6, ax=ax, color=PRIMARY_COLOR); ax.set_title('Diagrama de Dispersão (Fitness vs. Desvios)'); fig.tight_layout(); fig, ax = style_plot(fig, ax); plots['deviation_scatter_plot'] = convert_fig_to_bytes(fig)
    
    case_fitness_data = [{'project_id': str(trace.attributes['concept:name']), 'fitness': alignment['fitness']} for trace, alignment in zip(log_full_pm4py, aligned_traces) if 'concept:name' in trace.attributes]; case_fitness_df = pd.DataFrame(case_fitness_data).merge(_df_projects[['project_id', 'end_date']], on='project_id'); case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str); monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax, color=PRIMARY_COLOR); ax.set_title('Score de Conformidade ao Longo do Tempo'); ax.set_ylim(0, 1.05); ax.tick_params(axis='x', rotation=45); fig.tight_layout(); fig, ax = style_plot(fig, ax); plots['conformance_over_time_plot'] = convert_fig_to_bytes(fig)

    return plots, metrics

# --- LAYOUT DA APLICAÇÃO ---
st.sidebar.header("Análise de Processos")
st.sidebar.markdown("Navegue pelas secções da aplicação.")
page = st.sidebar.radio("Menu", ["Upload de Ficheiros", "Executar Análise", "Resultados da Análise"], label_visibility="hidden")
file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']

if page == "Upload de Ficheiros":
    st.title("1. Upload dos Ficheiros de Dados")
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários para a análise. Os dados serão processados para revelar insights sobre os seus processos de TI.")
    
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
             with st.container():
                st.markdown(f"<h5>Ficheiro: <code>{name}.csv</code></h5>", unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)

elif page == "Executar Análise":
    st.title("2. Execução da Análise de Processos")
    if not all(st.session_state.dfs[name] is not None for name in file_names):
        st.warning("Por favor, carregue todos os 5 ficheiros CSV na página de 'Upload' antes de continuar.")
    else:
        st.info("Todos os ficheiros estão carregados. Clique no botão abaixo para iniciar a análise completa dos processos.")
        if st.button("🚀 Iniciar Análise Completa"):
            with st.spinner("A executar a análise pré-mineração (26 gráficos)... Isto pode demorar um momento."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                st.session_state.event_log_for_cache = pm4py.convert_to_dataframe(event_log)
                st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}
            
            with st.spinner("A executar a análise de Process Mining (23 artefactos)... Esta é a parte mais demorada."):
                log_from_df = pm4py.convert_to_event_log(st.session_state.event_log_for_cache)
                dfs_cache = st.session_state.dfs_for_cache
                plots_post, metrics = run_post_mining_analysis(log_from_df, dfs_cache['projects'], dfs_cache['tasks_raw'], dfs_cache['resources'], dfs_cache['full_context'])
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics

            st.session_state.analysis_run = True
            st.success("✅ Análise completa concluída com sucesso! Navegue para 'Resultados da Análise'.")
            st.balloons()

elif page == "Resultados da Análise":
    st.title("Dashboard de Análise de Processos")
    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor, vá à página 'Executar Análise'.")
    else:
        tab1, tab2 = st.tabs(["📊 Análise Geral (Pré-Mineração)", "⛏️ Process Mining (Pós-Mineração)"])
        
        with tab1:
            st.header("Visão Geral da Performance")
            # --- SEÇÃO DE KPIs ---
            kpi_data = st.session_state.tables_pre_mining.get('kpi_df', pd.DataFrame()).set_index('Métrica')['Valor']
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1: st.markdown(f"<div class='kpi-card'><p class='kpi-title'>Total de Projetos</p><p class='kpi-value'>{int(kpi_data.get('Total de Projetos', 0))}</p></div>", unsafe_allow_html=True)
            with kpi2: st.markdown(f"<div class='kpi-card'><p class='kpi-title'>Total de Tarefas</p><p class='kpi-value'>{int(kpi_data.get('Total de Tarefas', 0))}</p></div>", unsafe_allow_html=True)
            with kpi3: st.markdown(f"<div class='kpi-card'><p class='kpi-title'>Total de Recursos</p><p class='kpi-value'>{int(kpi_data.get('Total de Recursos', 0))}</p></div>", unsafe_allow_html=True)
            with kpi4: st.markdown(f"<div class='kpi-card'><p class='kpi-title'>Duração Média</p><p class='kpi-value'>{kpi_data.get('Duração Média (dias)', 0):.1f} dias</p></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- SEÇÕES DE CONTEÚDO ---
            sections = {
                "Análises de Alto Nível e de Casos": ['performance_matrix', 'case_durations_boxplot', 'outlier_cost', 'outlier_duration'],
                "Performance Detalhada": ['lead_time_hist', 'throughput_hist', 'throughput_boxplot', 'lead_time_vs_throughput', 'perf_stats'],
                "Atividades e Handoffs": ['activity_service_times', 'top_handoffs', 'top_handoffs_cost'],
                "Análise Organizacional (Recursos)": ['top_activities_plot', 'resource_workload', 'resource_avg_events', 'resource_handoffs', 'cost_by_resource_type', 'resource_activity_matrix'],
                "Variantes e Rework": ['variants_frequency', 'variants_table', 'rework_loops_table'],
                "Análise Aprofundada e Benchmarking": ['delay_by_teamsize', 'median_duration_by_teamsize', 'weekly_efficiency', 'bottleneck_by_resource', 'service_vs_wait_stacked', 'wait_vs_service_scatter', 'wait_time_evolution', 'throughput_benchmark_by_teamsize', 'cycle_time_breakdown', 'cost_of_delay_kpis']
            }

            for title, items in sections.items():
                st.subheader(title)
                plot_items = [item for item in items if item in st.session_state.plots_pre_mining]
                table_items = [item for item in items if item in st.session_state.tables_pre_mining]
                
                # Layout dinâmico
                all_items = plot_items + table_items
                if not all_items: continue
                
                cols = st.columns(2)
                for i, item_name in enumerate(all_items):
                    col = cols[i % 2]
                    with col.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        if item_name in plot_items:
                            st.image(st.session_state.plots_pre_mining[item_name])
                        else:
                            st.dataframe(st.session_state.tables_pre_mining[item_name], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.header("Descoberta e Análise de Modelos de Processo")
            c1, c2 = st.columns(2)
            with c1.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Modelo (Inductive Miner)"); st.image(st.session_state.plots_post_mining.get('model_inductive_petrinet', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c2.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Métricas (Inductive Miner)"); st.image(st.session_state.plots_post_mining.get('metrics_inductive', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c1.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Modelo (Heuristics Miner)"); st.image(st.session_state.plots_post_mining.get('model_heuristic_petrinet', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c2.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Métricas (Heuristics Miner)"); st.image(st.session_state.plots_post_mining.get('metrics_heuristic', '')); st.markdown('</div>', unsafe_allow_html=True)

            st.header("Análise de Performance e Conformidade")
            c1, c2 = st.columns(2)
            with c1.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Séries Temporais de KPIs"); st.image(st.session_state.plots_post_mining.get('kpi_time_series', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c2.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Atividades por Dia da Semana"); st.image(st.session_state.plots_post_mining.get('temporal_heatmap_fixed', '')); st.markdown('</div>', unsafe_allow_html=True)
            with st.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Heatmap de Performance no Processo"); st.image(st.session_state.plots_post_mining.get('performance_heatmap', '')); st.markdown('</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Rede Social de Recursos"); st.image(st.session_state.plots_post_mining.get('resource_network_adv', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c2.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Duração Média das Variantes"); st.image(st.session_state.plots_post_mining.get('variant_duration_plot', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c1.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Dispersão: Fitness vs Desvios"); st.image(st.session_state.plots_post_mining.get('deviation_scatter_plot', '')); st.markdown('</div>', unsafe_allow_html=True)
            with c2.container(): st.markdown('<div class="card">', unsafe_allow_html=True); st.subheader("Conformidade ao Longo do Tempo"); st.image(st.session_state.plots_post_mining.get('conformance_over_time_plot', '')); st.markdown('</div>', unsafe_allow_html=True)
