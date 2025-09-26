import streamlit as st

import io, zipfile

def export_all_csvs(named_dfs):
  mem = io.BytesIO()
  with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
    for name, df in named_dfs:
      try:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
      except Exception:
        continue
      zf.writestr(str(name) + '.csv', csv_bytes)
  mem.seek(0)
  return mem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io
import uuid

# PM4PY
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

# Página
st.set_page_config(page_title="Painel de Análise de Processos de IT", page_icon="✨", layout="wide")

# CSS
st.markdown("""
<style>
  :root{
    --bg:#0B1220; --panel:#111827; --text:#FFFFFF; --accent:#3B82F6; --accent-2:#60A5FA;
    --border:#1f2a37; --card:#FFFFFF; --card-text:#0F172A; --muted:#93A4BF;
  }
  html, body, .stApp{ background-color:var(--bg)!important; color:var(--text)!important;
    font-family:'Inter', -apple-system, BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,'Helvetica Neue',Arial,sans-serif; }
  .main .block-container{ padding:2rem 2.5rem; }
  h1,h2,h3,h4,h5,h6{ color:var(--text); font-weight:800; }
  h2{ border-bottom:2px solid var(--accent); padding-bottom:8px; margin-bottom:16px; }

  /* Sidebar */
  [data-testid="stSidebar"]{ background:linear-gradient(180deg,#0F172A 0%,#0B1220 100%)!important; border-right:1px solid var(--border); }
  [data-testid="stSidebar"] *{ color:#FFFFFF!important; font-weight:800; }
  .sidebar-header{ font-size:1.2rem; letter-spacing:0.2px; margin-bottom:8px; }
  .sidebar-note{ color:#FFFFFF!important; opacity:1; font-size:0.95rem; font-weight:800; }

  /* Inputs / Uploaders */
  .stTextInput>div>div>input, .stSelectbox, .stTextArea textarea{
    color:#FFFFFF!important; background-color:var(--panel)!important; border:1px solid var(--border)!important; font-weight:800;
  }
  .stFileUploader{
    background-color:var(--panel)!important; border:1px solid var(--border)!important; border-radius:10px; padding:10px;
  }
  .stFileUploader label{ color:#FFFFFF!important; font-weight:800; }
  section[data-testid="stFileUploadDropzone"] div[data-testid="stMarkdownContainer"] p{ color:#FFFFFF!important; font-weight:800; }

  /* Mensagens: evitar warnings visuais no dashboard */
  [data-testid="stWarning"]{ display:none!important; }
  [data-testid="stSuccess"]{ background-color:rgba(22,163,74,0.18)!important; border:1px solid rgba(22,163,74,0.5)!important; }
  [data-testid="stInfo"]{ background-color:rgba(59,130,246,0.18)!important; border:1px solid rgba(59,130,246,0.5)!important; }

  /* Card branco com texto escuro para legibilidade */
  .card{
    background-color:var(--card)!important; color:var(--card-text)!important;
    border:1px solid #e5e7eb; border-radius:14px; padding:18px 18px; box-shadow:0 6px 18px rgba(0,0,0,0.2); margin-bottom:16px;
  }
  .card-header{ display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #e5e7eb;
    padding-bottom:10px; margin-bottom:14px; }
  .card-title{ display:inline-flex; align-items:center; gap:10px; font-weight:900; font-size:1.05rem; color:var(--card-text)!important; }
  .card-actions{ display:inline-flex; align-items:center; gap:8px; }

  /* Dentro do card, forçar textos escuros */
  .card *{ color:var(--card-text)!important; }
  .card .stMetric{ color:var(--card-text)!important; }
  .card .metric-label, .card .metric-value{ color:var(--card-text)!important; }

  /* Botões gerais e de exportação (sempre visíveis) */
  .stButton>button{
    background-color:var(--accent)!important; color:#FFFFFF!important; font-weight:900!important;
    border-radius:10px; border:1px solid rgba(59,130,246,0.5)!important; padding:10px 14px;
  }
  [data-testid="stDownloadButton"] button{
    background-color:var(--accent)!important; color:#FFFFFF!important; font-weight:900!important;
    border-radius:10px; border:1px solid rgba(59,130,246,0.5)!important; padding:8px 12px;
  }

  /* Alinhamento e tamanhos consistentes para cartões no dashboard */
  .dashboard-grid .card{ min-height:380px; }
  .dashboard-grid .card.kpi{ min-height:150px; }
  .dashboard-grid .card.tall{ min-height:520px; }
  .dashboard-grid .card.wide{ min-height:460px; }

  /* Remover “caixas vazias” visuais */
  .element-container:has(.card) { margin-bottom:0!important; }

/* Overrides injected by Julius */


  /* Stronger contrast on dark backgrounds */
  .stApp, body { color: #FFFFFF !important; }
  h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }

  /* Card base and equal height layout */
  .card { background: #FFFFFF; color: #0F172A; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; display: flex; flex-direction: column; }
  .card .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .card .card-title { font-weight: 700; color: #0F172A; }
  .card .card-body { flex: 1; }

  /* KPI cards should be white too */
  .card.kpi { background: #FFFFFF; color: #0F172A; }

  /* Dashboard grid for uniform sizing and alignment */
  .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; align-items: stretch; }

  /* Make Streamlit file uploader readable */
  [data-testid="stFileUploader"] * { color: #0F172A !important; }
  [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; border: 1px dashed #cbd5e1 !important; }
  [data-testid="stFileUploader"] button { background: #111827 !important; color: #FFFFFF !important; border-color: #111827 !important; }

  /* Buttons inside headers */
  .card .stDownloadButton { margin-left: 8px; }

</style>
""", unsafe_allow_html=True)

# Helpers
def convert_fig_to_bytes(fig, format='png'):
  buf = io.BytesIO()
  fig.savefig(buf, format=format, bbox_inches='tight', dpi=150)
  buf.seek(0)
  plt.close(fig)
  return buf

def convert_gviz_to_bytes(gviz, format='png'):
  return io.BytesIO(gviz.pipe(format=format))

def to_csv_bytes(df: pd.DataFrame) -> io.BytesIO:
  buf = io.BytesIO()
  df.to_csv(buf, index=False, encoding='utf-8')
  buf.seek(0)
  return buf

# Estado
if 'dfs' not in st.session_state:
  st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'is_auth' not in st.session_state: st.session_state.is_auth = False
if 'user_name' not in st.session_state: st.session_state.user_name = ""
if 'view' not in st.session_state: st.session_state.view = "config"

# Análise Pré
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

  project_aggregates = df_alloc_costs.groupby('project_id').agg(
    total_actual_cost=('cost_of_work', 'sum'),
    num_resources=('resource_id', 'nunique')
  ).reset_index()
  df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
  df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
  df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)

  df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
  df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
  df_full_context = df_full_context.merge(df_resources, on='resource_id')
  df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

  log_df_final = df_full_context[['project_id', 'task_name', 'allocation_date', 'resource_name']].copy()
  log_df_final.rename(columns={'project_id':'case:concept:name','task_name':'concept:name','allocation_date':'time:timestamp','resource_name':'org:resource'}, inplace=True)
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

  fig, ax = plt.subplots(figsize=(8,5))
  sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.7, ax=ax)
  ax.axhline(0, color='black', ls='--'); ax.axvline(0, color='black', ls='--'); ax.set_title("Matriz de Performance")
  plots['performance_matrix'] = convert_fig_to_bytes(fig)

  fig, ax = plt.subplots(figsize=(8,4))
  sns.boxplot(x=df_projects['actual_duration_days'], ax=ax, color="#60A5FA")
  sns.stripplot(x=df_projects['actual_duration_days'], color="#1f8ef1", size=4, jitter=True, alpha=0.5, ax=ax)
  ax.set_title("Distribuição da Duração dos Projetos")
  plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)

  lead_times = log_df_final.groupby("case:concept:name")["time:timestamp"].agg(["min","max"]).reset_index()
  lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
  def compute_avg_throughput(group):
    group = group.sort_values("time:timestamp"); deltas = group["time:timestamp"].diff().dropna()
    return deltas.mean().total_seconds() if not deltas.empty else 0
  throughput_per_case = log_df_final.groupby("case:concept:name", group_keys=False).apply(compute_avg_throughput).reset_index(name="avg_throughput_seconds")
  throughput_per_case["avg_throughput_hours"] = throughput_per_case["avg_throughput_seconds"] / 3600
  perf_df = pd.merge(lead_times, throughput_per_case, on="case:concept:name")
  tables['perf_stats'] = perf_df[["lead_time_days","avg_throughput_hours"]].describe()

  fig, ax = plt.subplots(figsize=(8,4))
  sns.histplot(perf_df["lead_time_days"], bins=20, kde=True, ax=ax, color="#60A5FA")
  ax.set_title("Distribuição do Lead Time (dias)")
  plots['lead_time_hist'] = convert_fig_to_bytes(fig)

  fig, ax = plt.subplots(figsize=(8,4))
  sns.histplot(perf_df["avg_throughput_hours"], bins=20, kde=True, color='#34D399', ax=ax)
  ax.set_title("Distribuição do Throughput (horas)")
  plots['throughput_hist'] = convert_fig_to_bytes(fig)

  fig, ax = plt.subplots(figsize=(8,4))
  sns.boxplot(x=perf_df["avg_throughput_hours"], color='#10B981', ax=ax)
  ax.set_title("Boxplot do Throughput (horas)")
  plots['throughput_boxplot'] = convert_fig_to_bytes(fig)

  fig, ax = plt.subplots(figsize=(8,5))
  sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax, color='#3B82F6')
  ax.set_title("Relação Lead Time vs Throughput")
  plots['lead_time_vs_throughput'] = convert_fig_to_bytes(fig)

  service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
  service_times['service_time_days'] = service_times['hours_worked'] / 8
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, hue='task_name', legend=False, palette='viridis')
  ax.set_title("Tempo Médio de Execução por Atividade")
  plots['activity_service_times'] = convert_fig_to_bytes(fig)

  df_handoff = log_df_final.sort_values(['case:concept:name','time:timestamp'])
  df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)
  df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)
  df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)
  handoff_stats = df_handoff.groupby(['previous_activity','concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
  handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', ax=ax, hue='transition', legend=False, palette='magma')
  ax.set_title("Top 10 Handoffs por Tempo de Espera")
  plots['top_handoffs'] = convert_fig_to_bytes(fig)

  handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, hue='transition', legend=False, palette='Reds_r')
  ax.set_title("Top 10 Handoffs por Custo de Espera")
  plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

  activity_counts = df_tasks["task_name"].value_counts()
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, color="#3B82F6")
  ax.set_title("Atividades Mais Frequentes")
  plots['top_activities_plot'] = convert_fig_to_bytes(fig)

  resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, hue='resource_name', legend=False, palette='plasma')
  ax.set_title("Top 10 Recursos por Horas Trabalhadas")
  plots['resource_workload'] = convert_fig_to_bytes(fig)

  resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()
  resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), ax=ax, hue='resource_name', legend=False, palette='coolwarm')
  ax.set_title("Recursos por Média de Tarefas por Projeto")
  plots['resource_avg_events'] = convert_fig_to_bytes(fig)

  resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
  fig, ax = plt.subplots(figsize=(12,8))
  sns.heatmap(resource_activity_matrix_pivot, cmap='YlGnBu', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8})
  ax.set_title("Heatmap de Esforço por Recurso e Atividade")
  plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)

  handoff_counts = Counter((trace[i]['org:resource'], trace[i+1]['org:resource'])
                           for trace in event_log_pm4py
                           for i in range(len(trace)-1)
                           if 'org:resource' in trace[i] and 'org:resource' in trace[i+1] and trace[i]['org:resource'] != trace[i+1]['org:resource'])
  df_resource_handoffs = pd.DataFrame(handoff_counts.most_common(10), columns=['Handoff','Contagem'])
  df_resource_handoffs['Handoff'] = df_resource_handoffs['Handoff'].apply(lambda x: f"{x[0]} -> {x[1]}")
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs, ax=ax, hue='Handoff', legend=False, palette='rocket')
  ax.set_title("Top 10 Handoffs entre Recursos")
  plots['resource_handoffs'] = convert_fig_to_bytes(fig)

  cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
  fig, ax = plt.subplots(figsize=(8,4))
  sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', ax=ax, hue='resource_type', legend=False, palette='magma')
  ax.set_title("Custo por Tipo de Recurso")
  plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)

  variants_df = log_df_final.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='trace')
  variants_df['variant_str'] = variants_df['trace'].apply(lambda x: ' -> '.join(x))
  variant_analysis = variants_df['variant_str'].value_counts().reset_index(name='frequency')
  variant_analysis['percentage'] = (variant_analysis['frequency'] / variant_analysis['frequency'].sum()) * 100
  tables['variants_table'] = variant_analysis.head(10)
  fig, ax = plt.subplots(figsize=(12,6))
  sns.barplot(x='frequency', y='variant_str', data=variant_analysis.head(10), ax=ax, orient='h', hue='variant_str', legend=False, palette='coolwarm')
  ax.set_title("Top 10 Variantes de Processo por Frequência")
  plots['variants_frequency'] = convert_fig_to_bytes(fig)

  rework_loops = Counter(f"{trace[i]} -> {trace[i+1]} -> {trace[i]}" for trace in variants_df['trace'] for i in range(len(trace)-2) if trace[i] == trace[i+2] and trace[i] != trace[i+1])
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
  fig, ax = plt.subplots(figsize=(8,5))
  sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='flare')
  ax.set_title("Impacto do Tamanho da Equipa no Atraso")
  plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)

  median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='crest')
  ax.set_title("Duração Mediana por Tamanho da Equipa")
  plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)

  df_alloc_costs['day_of_week'] = df_alloc_costs['allocation_date'].dt.day_name()
  weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, hue='day_of_week', legend=False, palette='plasma')
  plt.xticks(rotation=45)
  ax.set_title("Eficiência Semanal (Horas Trabalhadas)")
  plots['weekly_efficiency'] = convert_fig_to_bytes(fig)

  df_tasks_analysis = df_tasks.copy()
  df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60)
  df_tasks_analysis.sort_values(['project_id','start_date'], inplace=True)
  df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
  df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60)
  df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)

  df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id','resource_name']], on='task_id', how='left').drop_duplicates()
  bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, hue='resource_name', legend=False, palette='rocket')
  ax.set_title("Top 15 Recursos por Tempo Médio de Espera")
  plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)

  bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days','waiting_time_days']].mean()
  fig, ax = plt.subplots(figsize=(8,5))
  bottleneck_by_activity.plot(kind='bar', stacked=True, color=['royalblue','crimson'], ax=ax)
  ax.set_title("Gargalos: Tempo de Serviço vs. Espera")
  plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)

  fig, ax = plt.subplots(figsize=(8,5))
  sns.regplot(data=bottleneck_by_activity.reset_index(), x='service_time_days', y='waiting_time_days', ax=ax, color='#3B82F6')
  ax.set_title("Tempo de Espera vs. Tempo de Execução")
  plots['wait_vs_service_scatter'] = convert_fig_to_bytes(fig)

  df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id','completion_month']], on='project_id')
  monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()
  fig, ax = plt.subplots(figsize=(8,4))
  sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax, color='#3B82F6')
  plt.xticks(rotation=45)
  ax.set_title("Evolução do Tempo Médio de Espera")
  plots['wait_time_evolution'] = convert_fig_to_bytes(fig)

  df_perf_full = perf_df.merge(df_projects, left_on='case:concept:name', right_on='project_id')
  fig, ax = plt.subplots(figsize=(8,5))
  sns.boxplot(data=df_perf_full, x='team_size_bin_dynamic', y='avg_throughput_hours', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='plasma')
  ax.set_title("Benchmark de Throughput por Tamanho da Equipa")
  plots['throughput_benchmark_by_teamsize'] = convert_fig_to_bytes(fig)

  def get_phase(task_type):
    if task_type in ['Desenvolvimento','Correção','Revisão','Design']: return 'Desenvolvimento & Design'
    if task_type == 'Teste': return 'Teste (QA)'
    if task_type in ['Deploy','DBA']: return 'Operações & Deploy'
    return 'Outros'
  df_tasks['phase'] = df_tasks['task_type'].apply(get_phase)
  phase_times = df_tasks.groupby(['project_id','phase']).agg(start=('start_date','min'), end=('end_date','max')).reset_index()
  phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days
  avg_cycle_time_by_phase = phase_times.groupby('phase')['cycle_time_days'].mean()
  fig, ax = plt.subplots(figsize=(8,4))
  avg_cycle_time_by_phase.plot(kind='bar', color=sns.color_palette('muted'), ax=ax)
  ax.set_title("Duração Média por Fase do Processo")
  plt.xticks(rotation=0)
  plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)

  return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

# Análise Pós
@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
  plots = {}
  metrics = {}

  df_start_events = _df_tasks_raw[['project_id','task_id','task_name','start_date']].rename(columns={'start_date':'time:timestamp','task_name':'concept:name','project_id':'case:concept:name'})
  df_start_events['lifecycle:transition'] = 'start'
  df_complete_events = _df_tasks_raw[['project_id','task_id','task_name','end_date']].rename(columns={'end_date':'time:timestamp','task_name':'concept:name','project_id':'case:concept:name'})
  df_complete_events['lifecycle:transition'] = 'complete'
  log_df_full_lifecycle = pd.concat([df_start_events, df_complete_events]).sort_values('time:timestamp')
  log_full_pm4py = pm4py.convert_to_event_log(log_df_full_lifecycle)

  variants_dict = variants_filter.get_variants(_event_log_pm4py)
  top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
  top_variant_names = [v[0] for v in top_variants_list]
  log_top_3_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)

  pt_inductive = inductive_miner.apply(log_top_3_variants)
  net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
  gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
  plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)

  def plot_metrics_chart(metrics_dict, title):
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica','Valor'])
    fig, ax = plt.subplots(figsize=(8,4))
    barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, hue='Métrica', legend=False, palette='viridis')
    for p in barplot.patches:
      ax.annotate(f'{p.get_height():.2f}', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0,9), textcoords='offset points', color='black')
    ax.set_title(title); ax.set_ylim(0,1.05); return fig

  metrics_im = {
    "Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0),
    "Precisão": precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im),
    "Generalização": generalization_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im),
    "Simplicidade": simplicity_evaluator.apply(net_im)
  }
  plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)'))
  metrics['inductive_miner'] = metrics_im

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
  metrics['heuristics_miner'] = metrics_hm

  kpi_temporal = _df_projects.groupby('completion_month').agg(avg_lead_time=('actual_duration_days','mean'), throughput=('project_id','count')).reset_index()
  fig, ax1 = plt.subplots(figsize=(12,6))
  ax1.plot(kpi_temporal['completion_month'], kpi_temporal['avg_lead_time'], marker='o', color='b', label='Lead Time')
  ax2 = ax1.twinx()
  ax2.bar(kpi_temporal['completion_month'], kpi_temporal['throughput'], color='g', alpha=0.6, label='Throughput')
  fig.suptitle('Séries Temporais de KPIs de Performance'); fig.legend()
  plots['kpi_time_series'] = convert_fig_to_bytes(fig)

  fig_gantt, ax_gantt = plt.subplots(figsize=(20, max(10, len(_df_projects)*0.4)))
  all_projects = _df_projects.sort_values('start_date')['project_id'].tolist()
  gantt_data = _df_tasks_raw[_df_tasks_raw['project_id'].isin(all_projects)].sort_values(['project_id','start_date'])
  project_y_map = {proj_id:i for i,proj_id in enumerate(all_projects)}
  color_map = {task_name: plt.get_cmap('viridis', gantt_data['task_name'].nunique())(i) for i, task_name in enumerate(gantt_data['task_name'].unique())}
  for _, task in gantt_data.iterrows():
    ax_gantt.barh(project_y_map[task['project_id']], (task['end_date']-task['start_date']).days+1, left=task['start_date'], height=0.6, color=color_map[task['task_name']], edgecolor='black')
  ax_gantt.set_yticks(list(project_y_map.values()))
  ax_gantt.set_yticklabels([f"Projeto {pid}" for pid in project_y_map.keys()])
  ax_gantt.invert_yaxis()
  ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.xticks(rotation=45)
  handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]
  ax_gantt.legend(handles, color_map.keys(), title='Tipo de Tarefa', bbox_to_anchor=(1.05,1), loc='upper left')
  ax_gantt.set_title('Linha do Tempo de Todos os Projetos (Gantt Chart)')
  fig_gantt.tight_layout()
  plots['gantt_chart_all_projects'] = convert_fig_to_bytes(fig_gantt)

  dfg_perf, _, _ = pm4py.discover_performance_dfg(log_full_pm4py)
  gviz_dfg = dfg_visualizer.apply(dfg_perf, log=log_full_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
  plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)

  fig, ax = plt.subplots(figsize=(8,4))
  log_df_full_lifecycle['weekday'] = log_df_full_lifecycle['time:timestamp'].dt.day_name()
  weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
  heatmap_data = log_df_full_lifecycle.groupby('weekday')['case:concept:name'].count().reindex(weekday_order).fillna(0)
  sns.barplot(x=heatmap_data.index, y=heatmap_data.values, ax=ax, hue=heatmap_data.index, legend=False, palette='viridis')
  ax.set_title('Ocorrências de Atividades por Dia da Semana')
  plt.xticks(rotation=45)
  plots['temporal_heatmap_fixed'] = convert_fig_to_bytes(fig)

  log_df_complete = pm4py.convert_to_dataframe(_event_log_pm4py)
  handovers = Counter((log_df_complete.iloc[i]['org:resource'], log_df_complete.iloc[i+1]['org:resource'])
                      for i in range(len(log_df_complete)-1)
                      if log_df_complete.iloc[i]['case:concept:name'] == log_df_complete.iloc[i+1]['case:concept:name']
                      and log_df_complete.iloc[i]['org:resource'] != log_df_complete.iloc[i+1]['org:resource'])
  fig_net, ax_net = plt.subplots(figsize=(10,10))
  G = nx.DiGraph()
  for (source, target), weight in handovers.items():
    G.add_edge(str(source), str(target), weight=weight)
  pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
  weights = [G[u][v]['weight'] for u,v in G.edges()]
  nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='gray', width=[w*0.5 for w in weights], ax=ax_net, font_size=10, connectionstyle='arc3,rad=0.1')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net)
  ax_net.set_title('Rede Social de Recursos (Handover Network)')
  plots['resource_network_adv'] = convert_fig_to_bytes(fig_net)

  if 'skill_level' in _df_resources.columns:
    perf_recursos = _df_full_context.groupby('resource_id').agg(total_hours=('hours_worked','sum'), total_tasks=('task_id','nunique')).reset_index()
    perf_recursos['avg_hours_per_task'] = perf_recursos['total_hours'] / perf_recursos['total_tasks']
    perf_recursos = perf_recursos.merge(_df_resources[['resource_id','skill_level','resource_name']], on='resource_id')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.regplot(data=perf_recursos, x='skill_level', y='avg_hours_per_task', ax=ax, color='#3B82F6')
    ax.set_title("Relação entre Skill e Performance")
    plots['skill_vs_performance_adv'] = convert_fig_to_bytes(fig)

    resource_role_counts = _df_full_context.groupby(['resource_name','skill_level']).size().reset_index(name='count')
    G_bipartite = nx.Graph()
    resources_nodes = resource_role_counts['resource_name'].unique()
    roles_nodes = resource_role_counts['skill_level'].unique()
    G_bipartite.add_nodes_from(resources_nodes, bipartite=0)
    G_bipartite.add_nodes_from(roles_nodes, bipartite=1)
    for _, row in resource_role_counts.iterrows():
      G_bipartite.add_edge(row['resource_name'], row['skill_level'], weight=row['count'])
    fig, ax = plt.subplots(figsize=(12,10))
    pos = nx.bipartite_layout(G_bipartite, resources_nodes)
    nx.draw(G_bipartite, pos, with_labels=True,
            node_color=['skyblue' if node in resources_nodes else 'lightgreen' for node in G_bipartite.nodes()],
            node_size=2000, ax=ax, font_size=8)
    edge_labels = nx.get_edge_attributes(G_bipartite, 'weight')
    nx.draw_networkx_edge_labels(G_bipartite, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title('Rede de Recursos por Função')
    plots['resource_network_bipartite'] = convert_fig_to_bytes(fig)

  variants_df = log_df_full_lifecycle.groupby('case:concept:name').agg(
    variant=('concept:name', lambda x: tuple(x)),
    start_timestamp=('time:timestamp','min'),
    end_timestamp=('time:timestamp','max')
  ).reset_index()
  variants_df['duration_hours'] = (variants_df['end_timestamp'] - variants_df['start_timestamp']).dt.total_seconds() / 3600
  variant_durations = variants_df.groupby('variant').agg(count=('case:concept:name','count'), avg_duration_hours=('duration_hours','mean')).reset_index().sort_values(by='count', ascending=False).head(10)
  variant_durations['variant_str'] = variant_durations['variant'].apply(lambda x: ' -> '.join([str(i) for i in x][:4]) + '...')
  fig, ax = plt.subplots(figsize=(8,5))
  sns.barplot(x='avg_duration_hours', y='variant_str', data=variant_durations.astype({'avg_duration_hours':'float'}), ax=ax, hue='variant_str', legend=False, palette='plasma')
  ax.set_title('Duração Média das 10 Variantes Mais Comuns'); fig.tight_layout()
  plots['variant_duration_plot'] = convert_fig_to_bytes(fig)

  aligned_traces = alignments_miner.apply(log_full_pm4py, net_im, im_im, fm_im)
  deviations_list = [{'fitness': trace['fitness'], 'deviations': sum(1 for move in trace['alignment'] if '>>' in move[0] or '>>' in move[1])} for trace in aligned_traces if 'fitness' in trace]
  deviations_df = pd.DataFrame(deviations_list)
  fig, ax = plt.subplots(figsize=(8,5))
  sns.scatterplot(x='fitness', y='deviations', data=deviations_df, alpha=0.6, ax=ax, color='#3B82F6')
  ax.set_title('Diagrama de Dispersão (Fitness vs. Desvios)'); fig.tight_layout()
  plots['deviation_scatter_plot'] = convert_fig_to_bytes(fig)

  case_fitness_data = [{'project_id': str(trace.attributes['concept:name']), 'fitness': alignment['fitness']} for trace, alignment in zip(log_full_pm4py, aligned_traces) if 'concept:name' in trace.attributes]
  case_fitness_df = pd.DataFrame(case_fitness_data).merge(_df_projects[['project_id','end_date']], on='project_id')
  case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str)
  monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()
  fig, ax = plt.subplots(figsize=(10,5))
  sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax, color='#3B82F6')
  ax.set_title('Score de Conformidade ao Longo do Tempo'); ax.set_ylim(0,1.05); ax.tick_params(axis='x', rotation=45); fig.tight_layout()
  plots['conformance_over_time_plot'] = convert_fig_to_bytes(fig)

  kpi_daily = _df_projects.groupby(_df_projects['end_date'].dt.date).agg(avg_cost_per_day=('cost_per_day','mean')).reset_index()
  kpi_daily.rename(columns={'end_date':'completion_date'}, inplace=True)
  kpi_daily['completion_date'] = pd.to_datetime(kpi_daily['completion_date'])
  fig, ax = plt.subplots(figsize=(10,5))
  sns.lineplot(data=kpi_daily, x='completion_date', y='avg_cost_per_day', ax=ax, color='#3B82F6')
  ax.set_title('Custo Médio por Dia ao Longo do Tempo'); fig.tight_layout()
  plots['cost_per_day_time_series'] = convert_fig_to_bytes(fig)

  df_projects_sorted = _df_projects.sort_values(by='end_date'); df_projects_sorted['cumulative_throughput'] = range(1, len(df_projects_sorted)+1)
  fig, ax = plt.subplots(figsize=(10,5))
  sns.lineplot(x='end_date', y='cumulative_throughput', data=df_projects_sorted, ax=ax, color='#3B82F6')
  ax.set_title('Gráfico Acumulado de Throughput'); fig.tight_layout()
  plots['cumulative_throughput_plot'] = convert_fig_to_bytes(fig)

  def generate_custom_variants_plot(event_log):
    variants = variants_filter.get_variants(event_log)
    top_variants = sorted(variants.items(), key=lambda item: len(item[1]), reverse=True)[:10]
    variant_sequences = {f"V{i+1} ({len(v)} casos)": [str(a) for a in k] for i, (k, v) in enumerate(top_variants)}
    fig, ax = plt.subplots(figsize=(12,6))
    all_activities = sorted(list(set([act for seq in variant_sequences.values() for act in seq])))
    activity_to_y = {activity: i for i, activity in enumerate(all_activities)}
    for i, (variant_name, sequence) in enumerate(variant_sequences.items()):
      ax.plot(range(len(sequence)), [activity_to_y[activity] for activity in sequence], marker='o', linestyle='-', label=variant_name)
    ax.set_yticks(list(activity_to_y.values()))
    ax.set_yticklabels(list(activity_to_y.keys()))
    ax.set_title('Sequência de Atividades das 10 Variantes Mais Comuns')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig
  plots['custom_variants_sequence_plot'] = convert_fig_to_bytes(generate_custom_variants_plot(log_full_pm4py))

  milestones = ['Analise e Design','Implementacao da Funcionalidade','Execucao de Testes','Deploy da Aplicacao']
  df_milestones = _df_tasks_raw[_df_tasks_raw['task_name'].isin(milestones)].copy()
  milestone_pairs = []
  for project_id, group in df_milestones.groupby('project_id'):
    sorted_tasks = group.sort_values('start_date')
    for i in range(len(sorted_tasks)-1):
      duration = (sorted_tasks.iloc[i+1]['start_date'] - sorted_tasks.iloc[i]['end_date']).total_seconds() / 3600
      if duration >= 0:
        milestone_pairs.append({'transition': f"{sorted_tasks.iloc[i]['task_name']} -> {sorted_tasks.iloc[i+1]['task_name']}", 'duration_hours': duration})
  df_milestone_pairs = pd.DataFrame(milestone_pairs)
  if not df_milestone_pairs.empty:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df_milestone_pairs, x='duration_hours', y='transition', ax=ax, orient='h', hue='transition', legend=False, palette='coolwarm')
    ax.set_title('Análise de Tempo entre Marcos do Processo'); fig.tight_layout()
    plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

  df_tasks_sorted = _df_tasks_raw.sort_values(['project_id','start_date'])
  df_tasks_sorted['previous_end_date'] = df_tasks_sorted.groupby('project_id')['end_date'].shift(1)
  df_tasks_sorted['waiting_time_days'] = (df_tasks_sorted['start_date'] - df_tasks_sorted['previous_end_date']).dt.total_seconds() / (24 * 3600)
  df_tasks_sorted.loc[df_tasks_sorted['waiting_time_days'] < 0, 'waiting_time_days'] = 0
  df_tasks_sorted['previous_task_name'] = df_tasks_sorted.groupby('project_id')['task_name'].shift(1)
  waiting_times_matrix = df_tasks_sorted.pivot_table(index='previous_task_name', columns='task_name', values='waiting_time_days', aggfunc='mean').fillna(0)
  fig, ax = plt.subplots(figsize=(10,8))
  sns.heatmap(waiting_times_matrix * 24, cmap='YlGnBu', annot=True, fmt='.1f', ax=ax, annot_kws={"size": 8})
  ax.set_title('Matriz de Tempo de Espera entre Atividades (horas)'); fig.tight_layout()
  plots['waiting_time_matrix_plot'] = convert_fig_to_bytes(fig)

  resource_efficiency = _df_full_context.groupby('resource_name').agg(total_hours_worked=('hours_worked','sum'), total_tasks_completed=('task_name','count')).reset_index()
  resource_efficiency['avg_hours_per_task'] = resource_efficiency['total_hours_worked'] / resource_efficiency['total_tasks_completed']
  fig, ax = plt.subplots(figsize=(10,6))
  sns.barplot(data=resource_efficiency.sort_values(by='avg_hours_per_task'), x='avg_hours_per_task', y='resource_name', orient='h', ax=ax, hue='resource_name', legend=False, palette='magma')
  ax.set_title('Métricas de Eficiência Individual por Recurso'); fig.tight_layout()
  plots['resource_efficiency_plot'] = convert_fig_to_bytes(fig)

  df_tasks_sorted['sojourn_time_hours'] = df_tasks_sorted['waiting_time_days'] * 24
  waiting_time_by_task = df_tasks_sorted.groupby('task_name')['sojourn_time_hours'].mean().reset_index()
  fig, ax = plt.subplots(figsize=(10,6))
  sns.barplot(data=waiting_time_by_task.sort_values(by='sojourn_time_hours', ascending=False), x='sojourn_time_hours', y='task_name', ax=ax, hue='task_name', legend=False, palette='viridis')
  ax.set_title('Tempo Médio de Espera por Atividade'); fig.tight_layout()
  plots['avg_waiting_time_by_activity_plot'] = convert_fig_to_bytes(fig)

  return plots, metrics

# Cartões
def card_header(title: str, icon: str = "📊", png_bytes: io.BytesIO = None, csv_df: pd.DataFrame = None, png_filename: str = "grafico.png", csv_filename: str = "tabela.csv", card_key: str = None):
  st.markdown(f"""
  <div class="card-header">
    <div class="card-title">{icon} {title}</div>
    <div class="card-actions"></div>
  </div>
  """, unsafe_allow_html=True)
  c_exp = st.columns([1, 1])
  if png_bytes is not None:
    with c_exp[0]:
      st.download_button("📥 PNG", data=png_bytes, file_name=png_filename, mime="image/png", key=f"dl_png_{card_key or title}_{uuid.uuid4()}")
  if csv_df is not None:
    with c_exp[1]:
      st.download_button("📥 CSV", data=to_csv_bytes(csv_df), file_name=csv_filename, mime="text/csv", key=f"dl_csv_{card_key or title}_{uuid.uuid4()}")

def card(title: str, icon: str = "📊", body_fn=None, png_bytes: io.BytesIO = None, csv_df: pd.DataFrame = None, png_filename: str = "grafico.png", csv_filename: str = "tabela.csv", card_key: str = None, extra_class: str = ""):
  st.markdown(f'<div class="card {extra_class}">', unsafe_allow_html=True)
  card_header(title, icon, png_bytes, csv_df, png_filename, csv_filename, card_key=card_key or str(uuid.uuid4()))
  if body_fn is not None:
    body_fn()
  st.markdown('</div>', unsafe_allow_html=True)

# Login
def render_login():
  st.markdown("<h2>🔐 Iniciar Sessão</h2>", unsafe_allow_html=True)
  col_center = st.columns([1, 2, 1])[1]
  with col_center:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><div class="card-title">👋 Bem-vindo</div></div>', unsafe_allow_html=True)
    username = st.text_input("Utilizador", placeholder="ex: vasco", key="login_user")
    password = st.text_input("Palavra-passe", type="password", placeholder="••••••••", key="login_pwd")
    login_btn = st.button("Entrar", key="login_enter")
    if login_btn:
      if username.strip() == "vasco" and password == "1234":
        st.session_state.is_auth = True
        st.session_state.user_name = username.strip().title()
        st.success("Sessão iniciada.")
        st.rerun()
      else:
        st.warning("Credenciais inválidas. Tente novamente.")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
def render_sidebar():
  st.sidebar.markdown('<div class="sidebar-header">Painel de Análise de Processos</div>', unsafe_allow_html=True)
  st.sidebar.markdown('<p class="sidebar-note">Navegue pelas vistas do dashboard.</p>', unsafe_allow_html=True)
  if st.session_state.is_auth:
    st.sidebar.markdown(f'<span class="sidebar-note">👤 {st.session_state.user_name}</span>', unsafe_allow_html=True)

  if st.sidebar.button("🏠 Dashboard Geral — Pré-mineração", key="nav_dashboard_pre"):
    st.session_state.view = "dashboard_pre"; st.rerun()
  if st.sidebar.button("🏠 Dashboard Geral — Pós-mineração", key="nav_dashboard_post"):
    st.session_state.view = "dashboard_post"; st.rerun()
  if st.sidebar.button("⚙️ Configurações — Upload de dados", key="nav_config"):
    st.session_state.view = "config"; st.rerun()
  st.sidebar.divider()
  if st.sidebar.button("🚪 Sair", key="nav_logout"):
    st.session_state.is_auth = False
    st.session_state.user_name = ""
    st.session_state.view = "config"
    st.session_state.analysis_run = False
    st.session_state.plots_pre_mining = {}
    st.session_state.plots_post_mining = {}
    st.session_state.tables_pre_mining = {}
    st.session_state.metrics = {}
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
    st.rerun()

# Configurações / Upload
def render_config():
  st.markdown("<h2>⚙️ Configurações — Upload de dados</h2>", unsafe_allow_html=True)
  file_names = ['projects','tasks','resources','resource_allocations','dependencies']

  # Linha única com 5 uploaders (compacto)
  cols = st.columns(5)
  for i, name in enumerate(file_names):
    with cols[i]:
      def body(name=name):
        uploaded_file = st.file_uploader(f"{name}.csv", type="csv", key=f"upload_{name}")
        if uploaded_file:
          st.session_state.dfs[name] = pd.read_csv(uploaded_file)
          st.success(f"{name}.csv carregado.")
      card(f"📁 Upload — {name}.csv", "📁", body_fn=body, card_key=f"upload_card_{name}", extra_class="kpi")

  # Pré-visualização compacta
  if all(st.session_state.dfs[name] is not None for name in file_names):
    def body_preview():
      cols_prev = st.columns(2)
      items = list(st.session_state.dfs.items())
      half = (len(items)+1)//2
      for i, (name, df) in enumerate(items[:half]):
        with cols_prev[0]:
          st.markdown(f"**{name}.csv — primeiras 5 linhas**")
          st.dataframe(df.head(), use_container_width=True)
          st.download_button("📥 CSV", data=to_csv_bytes(df), file_name=f"{name}_preview.csv", mime="text/csv", key=f"dl_prev_left_{name}")
      for i, (name, df) in enumerate(items[half:]):
        with cols_prev[1]:
          st.markdown(f"**{name}.csv — primeiras 5 linhas**")
          st.dataframe(df.head(), use_container_width=True)
          st.download_button("📥 CSV", data=to_csv_bytes(df), file_name=f"{name}_preview.csv", mime="text/csv", key=f"dl_prev_right_{name}")
    card("🗂️ Pré-visualização dos dados carregados", "🗂️", body_fn=body_preview, card_key="preview_card", extra_class="wide")

    def body_run():
      if st.button("🚀 Iniciar Análise Completa", key="run_analysis_btn"):
        with st.spinner("A analisar dados..."):
          plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
          st.session_state.plots_pre_mining = plots_pre
          st.session_state.tables_pre_mining = tables_pre
          st.session_state.event_log_for_cache = pm4py.convert_to_dataframe(event_log)
          st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}
          log_from_df = pm4py.convert_to_event_log(st.session_state.event_log_for_cache)
          dfs_cache = st.session_state.dfs_for_cache
          plots_post, metrics = run_post_mining_analysis(log_from_df, dfs_cache['projects'], dfs_cache['tasks_raw'], dfs_cache['resources'], dfs_cache['full_context'])
          st.session_state.plots_post_mining = plots_post
          st.session_state.metrics = metrics
        st.session_state.analysis_run = True
        st.success("✅ Análise concluída.")
    card("🚀 Executar análise", "🚀", body_fn=body_run, card_key="run_card", extra_class="kpi")

# Dashboard Pré
def render_dashboard_pre():
  st.markdown("<h2>🏠 Dashboard Geral — Pré-mineração</h2>", unsafe_allow_html=True)
  if not st.session_state.tables_pre_mining:
    return

  kpis = st.session_state.tables_pre_mining['kpi_data']
  st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)

  # KPIs (linha)
  c = st.columns(4)
  def kpi_card(label, value, help=None, icon="📈", key_suffix=""):
    def body(): st.metric(label=label, value=value, help=help)
    with st.container(): card(f"{label}", icon, body_fn=body, card_key=f"kpi_{key_suffix}", extra_class="kpi")
  with c[0]: kpi_card("Total de Projetos", kpis['Total de Projetos'], key_suffix="total_projects")
  with c[1]: kpi_card("Total de Tarefas", kpis['Total de Tarefas'], key_suffix="total_tasks")
  with c[2]: kpi_card("Total de Recursos", kpis['Total de Recursos'], key_suffix="total_resources")
  with c[3]: kpi_card("Duração Média (dias)", kpis['Duração Média (dias)'], help="em dias", key_suffix="avg_duration")

  # Grelha com cartões de tamanho consistente
  row = st.columns(2)
  with row[0]:
    img = st.session_state.plots_pre_mining['performance_matrix']
    def body(): st.image(img, use_container_width=True)
    card("Performance de Custo vs. Prazo", "💹", body_fn=body, png_bytes=img, png_filename="performance_matrix.png")
  with row[1]:
    img = st.session_state.plots_pre_mining['case_durations_boxplot']
    def body(): st.image(img, use_container_width=True)
    card("Distribuição da Duração dos Projetos", "⏱️", body_fn=body, png_bytes=img, png_filename="case_durations_boxplot.png")

  row2 = st.columns(2)
  with row2[0]:
    df = st.session_state.tables_pre_mining['outlier_duration']
    def body(): st.dataframe(df, use_container_width=True)
    card("Top 5 Projetos Mais Longos", "🧭", body_fn=body, csv_df=df, csv_filename="outlier_duration.csv")
  with row2[1]:
    df = st.session_state.tables_pre_mining['outlier_cost']
    def body(): st.dataframe(df, use_container_width=True)
    card("Top 5 Projetos Mais Caros", "💰", body_fn=body, csv_df=df, csv_filename="outlier_cost.csv")

  row3 = st.columns(2)
  with row3[0]:
    img = st.session_state.plots_pre_mining['lead_time_hist']
    def body(): st.image(img, use_container_width=True)
    card("Distribuição do Lead Time", "📐", body_fn=body, png_bytes=img, png_filename="lead_time_hist.png")
  with row3[1]:
    img = st.session_state.plots_pre_mining['throughput_hist']
    def body(): st.image(img, use_container_width=True)
    card("Distribuição do Throughput", "📊", body_fn=body, png_bytes=img, png_filename="throughput_hist.png")

  row4 = st.columns(2)
  with row4[0]:
    img = st.session_state.plots_pre_mining['throughput_boxplot']
    def body(): st.image(img, use_container_width=True)
    card("Boxplot do Throughput", "🗂️", body_fn=body, png_bytes=img, png_filename="throughput_boxplot.png")
  with row4[1]:
    img = st.session_state.plots_pre_mining['lead_time_vs_throughput']
    def body(): st.image(img, use_container_width=True)
    card("Lead Time vs Throughput", "🔗", body_fn=body, png_bytes=img, png_filename="lead_time_vs_throughput.png")

  row5 = st.columns(2)
  with row5[0]:
    img = st.session_state.plots_pre_mining['activity_service_times']
    def body(): st.image(img, use_container_width=True)
    card("Tempo Médio de Execução por Atividade", "🛠️", body_fn=body, png_bytes=img, png_filename="activity_service_times.png")
  with row5[1]:
    img = st.session_state.plots_pre_mining['top_handoffs']
    def body(): st.image(img, use_container_width=True)
    card("Top Handoffs por Tempo de Espera", "🔁", body_fn=body, png_bytes=img, png_filename="top_handoffs.png")

  row6 = st.columns(2)
  with row6[0]:
    img = st.session_state.plots_pre_mining['top_handoffs_cost']
    def body(): st.image(img, use_container_width=True)
    card("Top Handoffs por Custo de Espera", "💸", body_fn=body, png_bytes=img, png_filename="top_handoffs_cost.png")
  with row6[1]:
    img = st.session_state.plots_pre_mining['top_activities_plot']
    def body(): st.image(img, use_container_width=True)
    card("Atividades Mais Frequentes", "📚", body_fn=body, png_bytes=img, png_filename="top_activities_plot.png")

  row7 = st.columns(2)
  with row7[0]:
    img = st.session_state.plots_pre_mining['resource_workload']
    def body(): st.image(img, use_container_width=True)
    card("Top Recursos por Horas Trabalhadas", "👥", body_fn=body, png_bytes=img, png_filename="resource_workload.png")
  with row7[1]:
    img = st.session_state.plots_pre_mining['resource_avg_events']
    def body(): st.image(img, use_container_width=True)
    card("Recursos por Média de Tarefas por Projeto", "🧮", body_fn=body, png_bytes=img, png_filename="resource_avg_events.png")

  row8 = st.columns(2)
  with row8[0]:
    img = st.session_state.plots_pre_mining['resource_handoffs']
    def body(): st.image(img, use_container_width=True)
    card("Top Handoffs entre Recursos", "🔀", body_fn=body, png_bytes=img, png_filename="resource_handoffs.png")
  with row8[1]:
    img = st.session_state.plots_pre_mining['cost_by_resource_type']
    def body(): st.image(img, use_container_width=True)
    card("Custo por Tipo de Recurso", "🏷️", body_fn=body, png_bytes=img, png_filename="cost_by_resource_type.png")

  img = st.session_state.plots_pre_mining['resource_activity_matrix']
  def body_hm(): st.image(img, use_container_width=True)
  card("Heatmap de Esforço por Recurso e Atividade", "🔥", body_fn=body_hm, png_bytes=img, png_filename="resource_activity_matrix.png", extra_class="tall")

  row9 = st.columns(2)
  with row9[0]:
    df = st.session_state.tables_pre_mining['rework_loops_table']
    def body(): st.dataframe(df, use_container_width=True)
    card("Principais Loops de Rework", "♻️", body_fn=body, csv_df=df, csv_filename="rework_loops.csv")
  with row9[1]:
    img = st.session_state.plots_pre_mining['variants_frequency']
    def body(): st.image(img, use_container_width=True)
    card("Frequência das Variantes", "🧬", body_fn=body, png_bytes=img, png_filename="variants_frequency.png")

  delay_kpis = st.session_state.tables_pre_mining['cost_of_delay_kpis']
  c_kpi = st.columns(3)
  with c_kpi[0]:
    def body(): st.metric("Custo Total em Atraso", delay_kpis['Custo Total Projetos Atrasados'])
    card("Custo Total em Atraso", "💶", body_fn=body, extra_class="kpi")
  with c_kpi[1]:
    def body(): st.metric("Atraso Médio (dias)", delay_kpis['Atraso Médio (dias)'], help="em dias")
    card("Atraso Médio (dias)", "⏳", body_fn=body, extra_class="kpi")
  with c_kpi[2]:
    def body(): st.metric("Custo Médio/Dia Atraso", delay_kpis['Custo Médio/Dia Atraso'])
    card("Custo Médio/Dia de Atraso", "📆", body_fn=body, extra_class="kpi")

  row11 = st.columns(2)
  with row11[0]:
    img = st.session_state.plots_pre_mining['delay_by_teamsize']
    def body(): st.image(img, use_container_width=True)
    card("Atraso por Tamanho da Equipa", "👥", body_fn=body, png_bytes=img, png_filename="delay_by_teamsize.png")
  with row11[1]:
    img = st.session_state.plots_pre_mining['median_duration_by_teamsize']
    def body(): st.image(img, use_container_width=True)
    card("Duração Mediana por Tamanho da Equipa", "📏", body_fn=body, png_bytes=img, png_filename="median_duration_by_teamsize.png")

  row12 = st.columns(2)
  with row12[0]:
    img = st.session_state.plots_pre_mining['weekly_efficiency']
    def body(): st.image(img, use_container_width=True)
    card("Eficiência Semanal (Horas Trabalhadas)", "📅", body_fn=body, png_bytes=img, png_filename="weekly_efficiency.png")
  with row12[1]:
    img = st.session_state.plots_pre_mining['bottleneck_by_resource']
    def body(): st.image(img, use_container_width=True)
    card("Recursos com Maior Espera", "🛑", body_fn=body, png_bytes=img, png_filename="bottleneck_by_resource.png")

  row13 = st.columns(2)
  with row13[0]:
    img = st.session_state.plots_pre_mining['service_vs_wait_stacked']
    def body(): st.image(img, use_container_width=True)
    card("Gargalos: Serviço vs Espera", "🧱", body_fn=body, png_bytes=img, png_filename="service_vs_wait_stacked.png")
  with row13[1]:
    img = st.session_state.plots_pre_mining['wait_vs_service_scatter']
    def body(): st.image(img, use_container_width=True)
    card("Tempo de Espera vs Execução", "⚖️", body_fn=body, png_bytes=img, png_filename="wait_vs_service_scatter.png")

  row14 = st.columns(2)
  with row14[0]:
    img = st.session_state.plots_pre_mining['wait_time_evolution']
    def body(): st.image(img, use_container_width=True)
    card("Evolução do Tempo Médio de Espera", "📈", body_fn=body, png_bytes=img, png_filename="wait_time_evolution.png")
  with row14[1]:
    img = st.session_state.plots_pre_mining['throughput_benchmark_by_teamsize']
    def body(): st.image(img, use_container_width=True)
    card("Benchmark de Throughput por Equipa", "🏷️", body_fn=body, png_bytes=img, png_filename="throughput_benchmark_by_teamsize.png")

  img = st.session_state.plots_pre_mining['cycle_time_breakdown']
  def body_cycle(): st.image(img, use_container_width=True)
  card("Duração Média por Fase do Processo", "📊", body_fn=body_cycle, png_bytes=img, png_filename="cycle_time_breakdown.png", extra_class="wide")
  st.markdown('</div>', unsafe_allow_html=True)

# Dashboard Pós
def render_dashboard_post():
  st.markdown("<h2>🏠 Dashboard Geral — Pós-mineração</h2>", unsafe_allow_html=True)
  if not st.session_state.plots_post_mining:
    return

  st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)

  img = st.session_state.plots_post_mining['model_inductive_petrinet']
  def body_ind(): st.image(img, use_container_width=True)
  card("Modelo de Processo (Petri Net) — Inductive Miner", "🧩", body_fn=body_ind, png_bytes=img, png_filename="model_inductive.png", extra_class="wide")

  img = st.session_state.plots_post_mining['model_heuristic_petrinet']
  def body_h(): st.image(img, use_container_width=True)
  card("Modelo de Processo (Petri Net) — Heuristics Miner", "🧩", body_fn=body_h, png_bytes=img, png_filename="model_heuristics.png", extra_class="wide")

  row2 = st.columns(2)
  with row2[0]:
    img = st.session_state.plots_post_mining['metrics_inductive']
    def body(): st.image(img, use_container_width=True)
    card("Métricas de Qualidade — Inductive Miner", "📏", body_fn=body, png_bytes=img, png_filename="metrics_inductive.png")
  with row2[1]:
    img = st.session_state.plots_post_mining['metrics_heuristic']
    def body(): st.image(img, use_container_width=True)
    card("Métricas de Qualidade — Heuristics Miner", "📏", body_fn=body, png_bytes=img, png_filename="metrics_heuristics.png")

  row3 = st.columns(2)
  with row3[0]:
    img = st.session_state.plots_post_mining['kpi_time_series']
    def body(): st.image(img, use_container_width=True)
    card("Séries Temporais de KPIs", "⏱️", body_fn=body, png_bytes=img, png_filename="kpi_time_series.png")
  with row3[1]:
    img = st.session_state.plots_post_mining['temporal_heatmap_fixed']
    def body(): st.image(img, use_container_width=True)
    card("Atividades por Dia da Semana", "📆", body_fn=body, png_bytes=img, png_filename="temporal_heatmap.png")

  img = st.session_state.plots_post_mining['performance_heatmap']
  def body_ph(): st.image(img, use_container_width=True)
  card("Heatmap de Performance no Processo (DFG)", "🔥", body_fn=body_ph, png_bytes=img, png_filename="performance_heatmap.png", extra_class="tall")

  if 'gantt_chart_all_projects' in st.session_state.plots_post_mining:
    img = st.session_state.plots_post_mining['gantt_chart_all_projects']
    def body_g(): st.image(img, use_container_width=True)
    card("Gantt Chart de Todos os Projetos", "📋", body_fn=body_g, png_bytes=img, png_filename="gantt_all_projects.png", extra_class="tall")

  row4 = st.columns(2)
  with row4[0]:
    img = st.session_state.plots_post_mining['resource_network_adv']
    def body(): st.image(img, use_container_width=True)
    card("Rede Social de Recursos (Handover)", "🌐", body_fn=body, png_bytes=img, png_filename="resource_network.png")
  with row4[1]:
    if 'skill_vs_performance_adv' in st.session_state.plots_post_mining:
      img = st.session_state.plots_post_mining['skill_vs_performance_adv']
      def body(): st.image(img, use_container_width=True)
      card("Relação entre Skill e Performance", "🎯", body_fn=body, png_bytes=img, png_filename="skill_vs_performance.png")

  row5 = st.columns(2)
  with row5[0]:
    if 'resource_network_bipartite' in st.session_state.plots_post_mining:
      img = st.session_state.plots_post_mining['resource_network_bipartite']
      def body(): st.image(img, use_container_width=True)
      card("Rede de Recursos por Função (Bipartida)", "🪢", body_fn=body, png_bytes=img, png_filename="resource_bipartite.png")
  with row5[1]:
    if 'resource_efficiency_plot' in st.session_state.plots_post_mining:
      img = st.session_state.plots_post_mining['resource_efficiency_plot']
      def body(): st.image(img, use_container_width=True)
      card("Eficiência Individual por Recurso", "⚙️", body_fn=body, png_bytes=img, png_filename="resource_efficiency.png")

  row6 = st.columns(2)
  with row6[0]:
    img = st.session_state.plots_post_mining['variant_duration_plot']
    def body(): st.image(img, use_container_width=True)
    card("Duração Média das Variantes", "🧬", body_fn=body, png_bytes=img, png_filename="variant_duration.png")
  with row6[1]:
    img = st.session_state.plots_post_mining['deviation_scatter_plot']
    def body(): st.image(img, use_container_width=True)
    card("Fitness vs Desvios", "📎", body_fn=body, png_bytes=img, png_filename="deviation_scatter.png")

  row7 = st.columns(2)
  with row7[0]:
    img = st.session_state.plots_post_mining['conformance_over_time_plot']
    def body(): st.image(img, use_container_width=True)
    card("Conformidade ao Longo do Tempo", "📈", body_fn=body, png_bytes=img, png_filename="conformance_over_time.png")
  with row7[1]:
    img = st.session_state.plots_post_mining['cost_per_day_time_series']
    def body(): st.image(img, use_container_width=True)
    card("Custo por Dia ao Longo do Tempo", "💵", body_fn=body, png_bytes=img, png_filename="cost_per_day.png")

  row8 = st.columns(2)
  with row8[0]:
    img = st.session_state.plots_post_mining['cumulative_throughput_plot']
    def body(): st.image(img, use_container_width=True)
    card("Throughput Acumulado", "🪜", body_fn=body, png_bytes=img, png_filename="cumulative_throughput.png")
  with row8[1]:
    if 'milestone_time_analysis_plot' in st.session_state.plots_post_mining:
      img = st.session_state.plots_post_mining['milestone_time_analysis_plot']
      def body(): st.image(img, use_container_width=True)
      card("Análise de Tempo entre Marcos", "🎯", body_fn=body, png_bytes=img, png_filename="milestone_time.png")

  row9 = st.columns(2)
  with row9[0]:
    if 'waiting_time_matrix_plot' in st.session_state.plots_post_mining:
      img = st.session_state.plots_post_mining['waiting_time_matrix_plot']
      def body(): st.image(img, use_container_width=True)
      card("Matriz de Tempo de Espera (horas)", "🧩", body_fn=body, png_bytes=img, png_filename="waiting_time_matrix.png")
  with row9[1]:
    if 'avg_waiting_time_by_activity_plot' in st.session_state.plots_post_mining:
      img = st.session_state.plots_post_mining['avg_waiting_time_by_activity_plot']
      def body(): st.image(img, use_container_width=True)
      card("Tempo de Espera Médio por Atividade", "⏳", body_fn=body, png_bytes=img, png_filename="avg_waiting_time_by_activity.png")

  if 'custom_variants_sequence_plot' in st.session_state.plots_post_mining:
    img = st.session_state.plots_post_mining['custom_variants_sequence_plot']
    def body(): st.image(img, use_container_width=True)
    card("Sequência de Atividades das Variantes", "🧭", body_fn=body, png_bytes=img, png_filename="variants_sequence.png")

  st.markdown('</div>', unsafe_allow_html=True)

# Fluxo
if not st.session_state.is_auth:
  render_login()
else:
  render_sidebar()
  if st.session_state.view == "config":
    render_config()
  elif st.session_state.view == "dashboard_pre":
    render_dashboard_pre()
  elif st.session_state.view == "dashboard_post":
    render_dashboard_post()
  else:
    render_dashboard_pre()
