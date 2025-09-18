# -*- coding: utf-8 -*-
"""

Aplicação Web Streamlit para Análise de Processos de Gestão de Recursos de TI (Versão Completa e Fiel).



Esta aplicação é uma tradução fiel de um notebook de análise de processos,

incorporando um dashboard completo com todas as 46 visualizações originais, organizadas

de forma intuitiva com um sistema de navegação melhorado para uma experiência de utilizador otimizada.

"""



# --- 1. IMPORTAÇÃO DE BIBLIOTECAS ---

import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx

from io import StringIO, BytesIO

import warnings

from collections import Counter

import base64

import tempfile

import os



# Bibliotecas de Process Mining (PM4PY) e PDF

try:

    import pm4py

    from fpdf import FPDF

    from pm4py.objects.log.util import dataframe_utils

    from pm4py.objects.conversion.log import converter as log_converter

    from pm4py.visualization.dfg import visualizer as dfg_visualizer

    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    from pm4py.algo.discovery.inductive import algorithm as inductive_miner

    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator

    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

    from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator

    from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

except ImportError:

    st.error("Uma ou mais bibliotecas necessárias (pm4py, fpdf) não estão instaladas.")

    st.stop()





# --- 2. CONFIGURAÇÃO DA PÁGINA E ESTADO DA SESSÃO ---

st.set_page_config(

    page_title="Dashboard Completo de Análise de Processos",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded"

)



# Inicialização do estado da sessão

default_files = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']

if 'uploaded_files' not in st.session_state:

    st.session_state.uploaded_files = {k: None for k in default_files}

if 'analysis_complete' not in st.session_state:

    st.session_state.analysis_complete = False

if 'pre_mining_results' not in st.session_state:

    st.session_state.pre_mining_results = {}

if 'post_mining_results' not in st.session_state:

    st.session_state.post_mining_results = {}

if 'dataframes' not in st.session_state:

    st.session_state.dataframes = {}



warnings.filterwarnings("ignore")



# --- 3. ESTÉTICA E CSS PERSONALIZADO ---

st.markdown("""

<style>

    /* Tema Principal */

    .stApp {

        background-color: #f0f2f6;

    }

    /* Estilo dos Títulos */

    h1, h2, h3 {

        color: #1E3A8A; /* Azul Escuro */

    }

    /* Botões */

    .stButton > button {

        border-radius: 20px;

        border: 1px solid #1E3A8A;

        background-color: #3B82F6; /* Azul Primário */

        color: white;

    }

    .stButton > button:hover {

        background-color: #1E3A8A;

        color: white;

        border: 1px solid #3B82F6;

    }

    /* Barra Lateral */

    [data-testid="stSidebar"] {

        background-color: #DBEAFE; /* Azul Claro */

    }

</style>

""", unsafe_allow_html=True)





# --- 4. FUNÇÕES DE ANÁLISE (MODULARIZADAS) ---



@st.cache_data

def load_and_preprocess_data(uploaded_files):

    try:

        dfs = {name: pd.read_csv(StringIO(file.getvalue().decode('utf-8'))) for name, file in uploaded_files.items()}

        

        for name in default_files:

            for col in ['project_id', 'task_id', 'resource_id', 'allocation_id']:

                if col in dfs[name].columns:

                    dfs[name][col] = dfs[name][col].astype(str)



        for df_name in ['projects', 'tasks']:

            for col in ['start_date', 'end_date', 'planned_end_date']:

                if col in dfs[df_name].columns:

                    dfs[df_name][col] = pd.to_datetime(dfs[df_name][col], errors='coerce')

        dfs['resource_allocations']['allocation_date'] = pd.to_datetime(dfs['resource_allocations']['allocation_date'], errors='coerce')



        df_projects = dfs['projects']

        df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days

        df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days

        df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)

        df_projects['completion_quarter'] = df_projects['end_date'].dt.to_period('Q').astype(str)

        

        df_tasks = dfs['tasks']

        df_tasks['task_duration_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.days



        df_alloc_costs = dfs['resource_allocations'].merge(dfs['resources'], on='resource_id')

        df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']

        

        project_aggregates = df_alloc_costs.groupby('project_id').agg(total_actual_cost=('cost_of_work', 'sum'), num_resources=('resource_id', 'nunique')).reset_index()



        dep_counts = dfs['dependencies'].groupby('project_id').size().reset_index(name='dependency_count')

        task_counts = dfs['tasks'].groupby('project_id').size().reset_index(name='task_count')

        project_complexity = pd.merge(dep_counts, task_counts, on='project_id', how='outer').fillna(0)

        project_complexity['complexity_ratio'] = (project_complexity['dependency_count'] / project_complexity['task_count']).fillna(0)

        

        df_projects = df_projects.merge(project_aggregates, on='project_id', how='left').merge(project_complexity, on='project_id', how='left')

        df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']

        df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)

        dfs['projects'] = df_projects



        allocations_to_merge = dfs['resource_allocations'].drop(columns=['project_id'], errors='ignore')

        df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))

        df_full_context = df_full_context.merge(allocations_to_merge, on='task_id').merge(dfs['resources'], on='resource_id')

        df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

        dfs['full_context'] = df_full_context



        log_df = dfs['tasks'].merge(allocations_to_merge, on='task_id').merge(dfs['resources'], on='resource_id')

        log_df.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name', 'end_date': 'time:timestamp', 'resource_name': 'org:resource'}, inplace=True)

        log_df['case:concept:name'] = 'Projeto ' + log_df['case:concept:name']

        log_df.dropna(subset=['time:timestamp'], inplace=True)

        log_df = log_df.sort_values('time:timestamp')

        dfs['log_df'] = log_df

        dfs['event_log'] = log_converter.apply(log_df)



        return dfs

    except Exception as e:

        st.error(f"Erro no pré-processamento: {e}")

        return None



def generate_pre_mining_visuals(dfs):

    results = {}

    df_projects, df_full_context, df_tasks, df_resources, log_df = dfs['projects'], dfs['full_context'], dfs['tasks'], dfs['resources'], dfs['log_df']

    

    results['kpis'] = {'Total de Projetos': df_projects['project_id'].nunique(), 'Total de Tarefas': df_tasks['task_id'].nunique(), 'Total de Eventos': len(log_df), 'Total de Recursos': df_resources['resource_id'].nunique(), 'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.2f}"}

    

    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='path_name', s=80, alpha=0.7, ax=ax); ax.axhline(0, c='k', ls='--'); ax.axvline(0, c='k', ls='--'); ax.set_title('Matriz de Performance: Prazo vs. Orçamento'); results['plot_01'] = fig

    fig, ax = plt.subplots(figsize=(8, 3)); sns.boxplot(x=df_projects['actual_duration_days'], color='skyblue', ax=ax); ax.set_title('Distribuição da Duração dos Projetos'); results['plot_02'] = fig

    

    lead_times = log_df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"]).reset_index()

    lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.days

    fig, ax = plt.subplots(figsize=(8, 3)); sns.histplot(lead_times["lead_time_days"], bins=20, kde=True, ax=ax); ax.set_title('Distribuição do Lead Time por Caso (dias)'); results['plot_03'] = fig

    

    throughput_per_case = log_df.groupby("case:concept:name").apply(lambda g: g['time:timestamp'].diff().mean().total_seconds() / 3600).reset_index(name="avg_throughput_hours")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3)); sns.histplot(throughput_per_case["avg_throughput_hours"], bins=20, kde=True, ax=axes[0], color='green'); axes[0].set_title('Distribuição do Throughput (horas)'); sns.boxplot(x=throughput_per_case["avg_throughput_hours"], ax=axes[1], color='lightgreen'); axes[1].set_title('Boxplot do Throughput'); fig.tight_layout(); results['plot_04_05'] = fig

    

    perf_df = pd.merge(lead_times, throughput_per_case, on="case:concept:name")

    fig, ax = plt.subplots(figsize=(7, 4)); sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax); ax.set_title('Relação entre Lead Time e Throughput'); results['plot_06'] = fig

    

    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='task_name', data=service_times.sort_values('hours_worked', ascending=False).head(10), palette='viridis', ax=ax, hue='task_name', legend=False); ax.set_title('Tempo Médio de Execução por Atividade (Horas)'); results['plot_07'] = fig

    

    df_handoff = log_df[log_df.duplicated(subset=['case:concept:name'], keep=False)].sort_values(['case:concept:name', 'time:timestamp'])

    df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)

    df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)

    df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)

    handoff_stats = df_handoff.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)

    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', palette='magma', ax=ax, hue='transition', legend=False); ax.set_title('Top 10 Transições com Maior Tempo de Espera'); results['plot_08'] = fig

    

    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', palette='Reds_r', ax=ax, hue='transition', legend=False); ax.set_title('Top 10 Transições por Custo de Espera Estimado (€)'); results['plot_09'] = fig

    

    activity_counts = df_tasks["task_name"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='plasma', hue=activity_counts.head(10).index, legend=False); ax.set_title('Atividades Mais Frequentes'); results['plot_10'] = fig

    

    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=resource_workload.head(10).values, y=resource_workload.head(10).index, ax=ax, palette='magma', hue=resource_workload.head(10).index, legend=False); ax.set_title('Top 10 Recursos por Horas Trabalhadas'); results['plot_11'] = fig

    

    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()

    resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), palette='coolwarm', ax=ax, hue='resource_name', legend=False); ax.set_title('Top 10 Recursos por Média de Tarefas por Projeto'); results['plot_12'] = fig

    

    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_activity_matrix_pivot, cmap='YlGnBu', annot=True, fmt=".0f", ax=ax); ax.set_title('Heatmap de Esforço (Horas) por Recurso e Atividade'); results['plot_13'] = fig

    

    handoff_counts = Counter()

    for _, trace in log_df.groupby('case:concept:name'):

        resources = trace['org:resource'].tolist()

        for i in range(len(resources) - 1):

            if resources[i] != resources[i+1]: handoff_counts[(resources[i], resources[i+1])] += 1

    df_resource_handoffs = pd.DataFrame([{'De': k[0], 'Para': k[1], 'Contagem': v} for k,v in handoff_counts.items()]).sort_values('Contagem', ascending=False)

    df_resource_handoffs['Handoff'] = df_resource_handoffs['De'] + ' -> ' + df_resource_handoffs['Para']

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs.head(10), palette='rocket', ax=ax, hue='Handoff', legend=False); ax.set_title('Top 10 Handoffs entre Recursos'); results['plot_14'] = fig

    

    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=cost_by_resource_type.values, y=cost_by_resource_type.index, ax=ax, palette='cividis', hue=cost_by_resource_type.index, legend=False); ax.set_title('Custo Total por Tipo de Recurso'); results['plot_15'] = fig

    

    variants_df = log_df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x)).reset_index(name='variant_str')

    variant_analysis = variants_df['variant_str'].value_counts().reset_index(name='frequency')

    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x='frequency', y='variant_str', data=variant_analysis.head(10), palette='coolwarm', ax=ax, hue='variant_str', legend=False); ax.set_title('Top 10 Variantes de Processo por Frequência'); results['plot_16'] = fig

    

    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()

    bins = np.linspace(min_res, max_res, 4, dtype=int) if max_res > min_res else [min_res, max_res]

    df_projects['team_size_bin'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)

    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_projects, x='team_size_bin', y='days_diff', ax=ax, palette='flare', hue='team_size_bin', legend=False); ax.set_title('Impacto do Tamanho da Equipa no Atraso'); results['plot_17'] = fig

    

    median_duration_by_team_size = df_projects.groupby('team_size_bin')['actual_duration_days'].median().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin', y='actual_duration_days', palette='crest', ax=ax, hue='team_size_bin', legend=False); ax.set_title('Duração Mediana por Tamanho da Equipa'); results['plot_18'] = fig

    

    df_full_context['day_of_week'] = df_full_context['allocation_date'].dt.day_name()

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    weekly_hours = df_full_context.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order)

    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=weekly_hours.index, y=weekly_hours.values, ax=ax, palette='plasma', hue=weekly_hours.index, legend=False); ax.set_title('Horas Trabalhadas por Dia da Semana'); results['plot_19'] = fig

    

    df_tasks_analysis = df_tasks.copy()

    df_tasks_analysis['service_time_days'] = df_tasks_analysis['task_duration_days']

    df_tasks_analysis = df_tasks_analysis.sort_values(['project_id', 'start_date'])

    df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)

    df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*3600)

    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].clip(lower=0)

    

    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates()

    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6)); sns.barplot(y=bottleneck_by_resource.index, x=bottleneck_by_resource.values, palette='rocket', ax=ax, hue=bottleneck_by_resource.index, legend=False); ax.set_title('Recursos por Tempo Médio de Espera (Dias)'); results['plot_20'] = fig

    

    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()

    fig, ax = plt.subplots(figsize=(10, 6)); bottleneck_by_activity.plot(kind='bar', stacked=True, color=['royalblue', 'crimson'], ax=ax); ax.set_title('Gargalos (Tempo de Serviço vs. Espera)'); results['plot_21'] = fig

    fig, ax = plt.subplots(figsize=(7, 4)); sns.regplot(data=bottleneck_by_activity, x='service_time_days', y='waiting_time_days', ax=ax); ax.set_title('Espera vs. Execução'); results['plot_22'] = fig



    df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id', 'completion_month']], on='project_id')

    monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax); ax.set_title("Evolução do Tempo Médio de Espera"); plt.xticks(rotation=45); results['plot_23'] = fig

    

    df_rh_typed = df_resource_handoffs.merge(df_resources[['resource_name', 'resource_type']], left_on='De', right_on='resource_name').merge(df_resources[['resource_name', 'resource_type']], left_on='Para', right_on='resource_name', suffixes=('_de', '_para'))

    handoff_matrix = df_rh_typed.groupby(['resource_type_de', 'resource_type_para'])['Contagem'].sum().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(handoff_matrix, annot=True, fmt=".0f", cmap="BuPu", ax=ax); ax.set_title("Matriz de Handoffs por Tipo de Equipa"); results['plot_24'] = fig

    

    perf_df['project_id'] = perf_df['case:concept:name'].str.replace('Projeto ', '')

    df_perf_full = perf_df.merge(df_projects, on='project_id', how='left')

    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_perf_full, x='team_size_bin', y='avg_throughput_hours', palette='plasma', ax=ax, hue='team_size_bin', legend=False); ax.set_title('Benchmark de Throughput por Tamanho da Equipa'); results['plot_25'] = fig

    

    def get_phase(task_type):

        if task_type in ['Desenvolvimento', 'Correção', 'Revisão', 'Design']: return 'Desenvolvimento & Design'

        if task_type == 'Teste': return 'Teste (QA)'

        if task_type in ['Deploy', 'DBA']: return 'Operações & Deploy'

        return 'Outros'

    df_tasks_phases = df_tasks.copy(); df_tasks_phases['phase'] = df_tasks_phases['task_type'].apply(get_phase)

    phase_times = df_tasks_phases.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index()

    phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days

    avg_cycle_time_by_phase = phase_times.groupby('phase')['cycle_time_days'].mean()

    fig, ax = plt.subplots(figsize=(8, 5)); avg_cycle_time_by_phase.plot(kind='bar', color=sns.color_palette('muted'), ax=ax); ax.set_title('Duração Média por Fase do Processo'); results['plot_26'] = fig

    

    return results



def calculate_model_metrics(log, petri_net, initial_marking, final_marking, title):

    fitness = replay_fitness_evaluator.apply(log, petri_net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)

    precision = precision_evaluator.apply(log, petri_net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)

    generalization = generalization_evaluator.apply(log, petri_net, initial_marking, final_marking)

    simplicity = simplicity_evaluator.apply(petri_net)

    metrics = {"Fitness": fitness.get('average_trace_fitness', 0), "Precisão": precision, "Generalização": generalization, "Simplicidade": simplicity}

    

    df_metrics = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(data=df_metrics, x='Métrica', y='Valor', palette='viridis', ax=ax, hue='Métrica', legend=False)

    ax.set_ylim(0, 1.05); ax.set_ylabel(''); ax.set_xlabel(''); ax.set_title(title)

    for p in ax.patches:

        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    return fig



def generate_post_mining_visuals(dfs):

    results = {}

    event_log, df_tasks, df_projects, log_df = dfs['event_log'], dfs['tasks'], dfs['projects'], dfs['log_df']

    

    process_tree_im = inductive_miner.apply(event_log)

    net_im, im_im, fm_im = pm4py.convert_to_petri_net(process_tree_im)

    results['model_01_inductive'] = pn_visualizer.apply(net_im, im_im, fm_im)

    results['metrics_inductive'] = calculate_model_metrics(event_log, net_im, im_im, fm_im, 'Métricas de Qualidade (Inductive)')



    net_hm, im_hm, fm_hm = heuristics_miner.apply(event_log)

    results['model_02_heuristics'] = pn_visualizer.apply(net_hm, im_hm, fm_hm)

    results['metrics_heuristics'] = calculate_model_metrics(event_log, net_hm, im_hm, fm_hm, 'Métricas de Qualidade (Heuristics)')

    

    dfg_perf, _, _ = pm4py.discover_performance_dfg(event_log)

    results['model_03_performance_dfg'] = dfg_visualizer.apply(dfg_perf, log=event_log, variant=dfg_visualizer.Variants.PERFORMANCE)



    variants = pm4py.get_variants_as_tuples(event_log)

    variants_counts = {str(k): len(v) for k, v in variants.items()}

    variants_df_full = pd.DataFrame(list(variants_counts.items()), columns=['variant', 'count']).sort_values(by='count', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5)); ax.pie(variants_df_full['count'].head(7), labels=[f'Variante {i+1}' for i in range(7)], autopct='%1.1f%%', startangle=90); ax.set_title('Distribuição das 7 Variantes Mais Comuns'); results['chart_04_variants_pie'] = fig

    

    aligned_traces = alignments.apply(event_log, net_im, im_im, fm_im)

    fitness_values = [trace['fitness'] for trace in aligned_traces]

    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(fitness_values, bins=20, kde=True, ax=ax, color='green'); ax.set_title('Distribuição do Fitness de Conformidade'); results['chart_05_conformance_fitness'] = fig

    

    kpi_temporal = df_projects.groupby('completion_month').agg(avg_lead_time=('actual_duration_days', 'mean'), throughput=('project_id', 'count')).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5)); ax1.plot(kpi_temporal['completion_month'], kpi_temporal['avg_lead_time'], marker='o', color='b'); ax1.set_ylabel('Dias', color='b'); ax2 = ax1.twinx(); ax2.bar(kpi_temporal['completion_month'], kpi_temporal['throughput'], color='g', alpha=0.6); ax2.set_ylabel('Nº de Projetos', color='g'); fig.suptitle('Séries Temporais de KPIs de Performance'); results['chart_06_kpi_time_series'] = fig

    

    fig, ax = plt.subplots(figsize=(12, 8)); projects_to_plot = df_projects.sort_values('start_date').head(20); tasks_to_plot = df_tasks[df_tasks['project_id'].isin(projects_to_plot['project_id'])]; project_y_map = {proj_id: i for i, proj_id in enumerate(projects_to_plot['project_id'])}; task_colors = plt.get_cmap('viridis', tasks_to_plot['task_name'].nunique()); color_map = {name: task_colors(i) for i, name in enumerate(tasks_to_plot['task_name'].unique())}; [ax.barh(project_y_map[task['project_id']], (task['end_date'] - task['start_date']).days + 1, left=task['start_date'], color=color_map.get(task['task_name'])) for _, task in tasks_to_plot.iterrows() if task['project_id'] in project_y_map]; ax.set_yticks(list(project_y_map.values())); ax.set_yticklabels([f"Projeto {pid}" for pid in project_y_map.keys()]); ax.invert_yaxis(); ax.set_title('Gráfico de Gantt (20 Primeiros Projetos)'); results['chart_07_gantt_chart'] = fig

    

    variants_df_log = log_df.groupby('case:concept:name').agg(variant=('concept:name', tuple), start=('time:timestamp', 'min'), end=('time:timestamp', 'max')).reset_index()

    variants_df_log['duration_hours'] = (variants_df_log['end'] - variants_df_log['start']).dt.total_seconds() / 3600

    variant_durations = variants_df_log.groupby('variant')['duration_hours'].mean().reset_index().sort_values('duration_hours', ascending=False)

    variant_durations['variant_str'] = variant_durations['variant'].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x='duration_hours', y='variant_str', data=variant_durations.head(10), palette='plasma', ax=ax, hue='variant_str', legend=False); ax.set_title('Duração Média das 10 Variantes Mais Lentas'); results['chart_08_variant_duration'] = fig

    

    deviations_list = [{'fitness': trace['fitness'], 'deviations': sum(1 for move in trace['alignment'] if '>>' in move[0] or '>>' in move[1])} for trace in aligned_traces]

    deviations_df = pd.DataFrame(deviations_list)

    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(x='fitness', y='deviations', data=deviations_df, alpha=0.6, ax=ax); ax.set_title('Diagrama de Dispersão (Fitness vs. Desvios)'); results['chart_09_deviation_scatter'] = fig



    case_fitness_df = pd.DataFrame([{'project_id': trace.attributes['concept:name'].replace('Projeto ', ''), 'fitness': alignment['fitness']} for trace, alignment in zip(event_log, aligned_traces)])

    case_fitness_df = case_fitness_df.merge(df_projects[['project_id', 'end_date']], on='project_id')

    case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str)

    monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax); ax.set_title('Score de Conformidade ao Longo do Tempo'); plt.xticks(rotation=45); results['chart_10_conformance_over_time'] = fig

    

    df_projects_sorted = df_projects.sort_values(by='end_date')

    df_projects_sorted['cumulative_throughput'] = range(1, len(df_projects_sorted) + 1)

    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(x='end_date', y='cumulative_throughput', data=df_projects_sorted, ax=ax); ax.set_title('Gráfico Acumulado de Throughput'); results['chart_11_cumulative_throughput'] = fig

    

    milestones = ['Analise e Design', 'Implementacao da Funcionalidade', 'Execucao de Testes', 'Deploy da Aplicacao']

    df_milestones = df_tasks[df_tasks['task_name'].isin(milestones)].sort_values(['project_id', 'start_date'])

    milestone_pairs = []

    for _, group in df_milestones.groupby('project_id'):

        for i in range(len(group) - 1):

            start_task, end_task = group.iloc[i], group.iloc[i+1]

            duration = (end_task['start_date'] - start_task['end_date']).total_seconds() / 3600

            if duration >= 0: milestone_pairs.append({'transition': f"{start_task['task_name']} -> {end_task['task_name']}", 'duration_hours': duration})

    milestone_df = pd.DataFrame(milestone_pairs)

    if not milestone_df.empty:

        fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=milestone_df, x='duration_hours', y='transition', ax=ax, orient='h', palette='viridis'); ax.set_title('Análise de Tempo entre Marcos do Processo'); results['chart_12_milestone_analysis'] = fig

    else:

        fig, ax = plt.subplots(figsize=(8,4)); ax.text(0.5, 0.5, 'Dados insuficientes para análise de marcos.', ha='center'); results['chart_12_milestone_analysis'] = fig

        

    df_tasks_sorted = df_tasks.sort_values(['project_id', 'start_date'])

    df_tasks_sorted['previous_end_date'] = df_tasks_sorted.groupby('project_id')['end_date'].shift(1)

    df_tasks_sorted['waiting_time_days'] = (df_tasks_sorted['start_date'] - df_tasks_sorted['previous_end_date']).dt.total_seconds() / (24 * 3600)

    df_tasks_sorted.loc[df_tasks_sorted['waiting_time_days'] < 0, 'waiting_time_days'] = 0

    df_tasks_sorted['previous_task_name'] = df_tasks_sorted.groupby('project_id')['task_name'].shift(1)

    waiting_times_matrix = df_tasks_sorted.pivot_table(index='previous_task_name', columns='task_name', values='waiting_time_days', aggfunc='mean').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 10)); sns.heatmap(waiting_times_matrix, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=.5, ax=ax); ax.set_title('Matriz de Tempo de Espera entre Atividades (dias)'); results['chart_13_waiting_time_matrix'] = fig

    

    waiting_time_by_task = df_tasks_sorted.groupby('task_name')['waiting_time_days'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x=waiting_time_by_task.values, y=waiting_time_by_task.index, ax=ax, palette='viridis', hue=waiting_time_by_task.index, legend=False); ax.set_title('Tempo Médio de Espera por Atividade (dias)'); results['chart_14_avg_wait_by_activity'] = fig



    handoff_counts = Counter()

    for _, trace in log_df.groupby('case:concept:name'):

        resources = trace['org:resource'].tolist()

        for i in range(len(resources) - 1):

            if resources[i] != resources[i+1]: handoff_counts[(resources[i], resources[i+1])] += 1

    fig, ax = plt.subplots(figsize=(12, 12)); G = nx.DiGraph();

    for (source, target), weight in handoff_counts.items(): G.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(G, k=1.2, iterations=50, seed=42)

    weights = [G[u][v]['weight'] for u,v in G.edges()]

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, edge_color='gray', width=[w*0.5 for w in weights], ax=ax, font_size=9, connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax); ax.set_title("Rede Social de Colaboração (Handovers)"); results['social_network'] = fig

    

    df_full_context = dfs['full_context']

    resource_role_counts = df_full_context.groupby(['resource_name', 'resource_type']).size().reset_index(name='count')

    G_bipartite = nx.Graph(); resources_nodes = resource_role_counts['resource_name'].unique(); roles_nodes = resource_role_counts['resource_type'].unique()

    G_bipartite.add_nodes_from(resources_nodes, bipartite=0); G_bipartite.add_nodes_from(roles_nodes, bipartite=1)

    for _, row in resource_role_counts.iterrows(): G_bipartite.add_edge(row['resource_name'], row['resource_type'], weight=row['count'])

    pos = nx.bipartite_layout(G_bipartite, resources_nodes, align='vertical')

    fig, ax = plt.subplots(figsize=(12, 10)); nx.draw_networkx_nodes(G_bipartite, pos, nodelist=resources_nodes, node_color='skyblue', node_size=2000, ax=ax); nx.draw_networkx_nodes(G_bipartite, pos, nodelist=roles_nodes, node_color='lightgreen', node_size=4000, ax=ax); nx.draw_networkx_edges(G_bipartite, pos, width=[d['weight']*0.1 for u,v,d in G_bipartite.edges(data=True)], edge_color='gray', ax=ax); nx.draw_networkx_labels(G_bipartite, pos, font_size=9); nx.draw_networkx_edge_labels(G_bipartite, pos, edge_labels={(u,v):d['weight'] for u,v,d in G_bipartite.edges(data=True)}); ax.set_title('Rede de Recursos por Função'); results['bipartite_network'] = fig



    return results



def run_full_analysis():

    with st.spinner('A processar os dados e a gerar as 46 análises... Por favor, aguarde.'):

        st.session_state.dataframes = load_and_preprocess_data(st.session_state.uploaded_files)

        if st.session_state.dataframes:

            st.session_state.pre_mining_results = generate_pre_mining_visuals(st.session_state.dataframes)

            st.session_state.post_mining_results = generate_post_mining_visuals(st.session_state.dataframes)

            st.session_state.analysis_complete = True

            st.success('Análise concluída com sucesso!')

        else:

            st.error("A análise falhou. Verifique os ficheiros e tente novamente.")



def generate_pdf_report(pre_res, post_res):

    pdf = FPDF()

    pdf.set_auto_page_break(auto=True, margin=15)

    

    all_results = {**pre_res, **post_res}

    

    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)

    pdf.cell(0, 10, 'Relatório de Análise de Processos', 0, 1, 'C')

    

    with tempfile.TemporaryDirectory() as temp_dir:

        for name, fig in all_results.items():

            if isinstance(fig, plt.Figure):

                title = name.replace('_', ' ').replace('plot', '').replace('chart', '').strip().title()

                try:

                    path = os.path.join(temp_dir, f"{name}.png")

                    fig.savefig(path, format="png", bbox_inches='tight', dpi=150)

                    

                    if pdf.get_y() > 180:

                        pdf.add_page()

                    

                    pdf.set_font("Arial", 'B', 12)

                    pdf.cell(0, 10, title, 0, 1, 'L')

                    

                    pdf.image(path, x=10, w=190)

                except Exception as e:

                    print(f"Error saving {name}: {e}")

    

    return pdf.output(dest='S').encode('latin-1')

    

# --- 5. LAYOUT DA APLICAÇÃO (UI) ---

st.title("Dashboard Completo de Análise de Processos")

st.sidebar.title("Painel de Controlo")

menu_selection = st.sidebar.radio(

    "Menu", ["1. Carregar Dados", "2. Executar Análise", "3. Visualizar Resultados"],

    captions=["Faça o upload dos 5 ficheiros CSV", "Inicie o processamento dos dados", "Explore o dashboard completo"]

)



if menu_selection == "1. Carregar Dados":

    st.header("1. Upload dos Ficheiros CSV")

    for name in default_files:

        with st.container():

            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")

            if uploaded_file:

                st.session_state.uploaded_files[name] = uploaded_file

                df_preview = pd.read_csv(uploaded_file); uploaded_file.seek(0)

                with st.expander(f"Pré-visualização de `{name}.csv`", expanded=False):

                    st.dataframe(df_preview.head(), height=210)



elif menu_selection == "2. Executar Análise":

    st.header("2. Execução da Análise")

    if all(st.session_state.uploaded_files.values()):

        if st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):

            run_full_analysis()

    else:

        st.error("Por favor, carregue todos os 5 ficheiros na secção '1. Carregar Dados'.")



elif menu_selection == "3. Visualizar Resultados":

    st.header("3. Dashboard de Resultados")

    if not st.session_state.analysis_complete:

        st.warning("A análise ainda não foi executada.")

    else:

        pre_res = st.session_state.pre_mining_results

        post_res = st.session_state.post_mining_results

        

        pdf_buffer = generate_pdf_report(pre_res, post_res)

        st.sidebar.download_button(

            label="📥 Gerar Relatório PDF",

            data=pdf_buffer,

            file_name="relatorio_analise_processos.pdf",

            mime="application/pdf",

        )



        st.sidebar.markdown("---")

        st.sidebar.subheader("Navegação do Dashboard")

        

        main_tab = st.sidebar.radio("Área de Análise", ["Análise Descritiva (Pré-Mineração)", "Análise de Processos (Pós-Mineração)"], label_visibility="collapsed")



        if main_tab == "Análise Descritiva (Pré-Mineração)":

            st.subheader("📊 Análise Descritiva (Pré-Mineração)")

            sections = ["Visão Geral e KPIs", "Performance e Prazos", "Organizacional e Custos", "Gargalos e Handoffs"]

            selected_section = st.sidebar.selectbox("Secção:", sections)



            if selected_section == "Visão Geral e KPIs":

                st.subheader("Visão Geral e KPIs")

                cols = st.columns(len(pre_res.get('kpis', {})))

                for i, (metric, value) in enumerate(pre_res.get('kpis', {}).items()): cols[i].metric(label=metric, value=str(value))

                if 'plot_01' in pre_res: st.pyplot(pre_res['plot_01'])



            if selected_section == "Performance e Prazos":

                st.subheader("Análise de Performance e Prazos")

                if 'plot_02' in pre_res: st.pyplot(pre_res['plot_02'])

                if 'plot_03' in pre_res: st.pyplot(pre_res['plot_03'])

                if 'plot_04_05' in pre_res: st.pyplot(pre_res['plot_04_05'])

                if 'plot_06' in pre_res: st.pyplot(pre_res['plot_06'])

                if 'plot_17' in pre_res: st.pyplot(pre_res['plot_17'])

                if 'plot_18' in pre_res: st.pyplot(pre_res['plot_18'])

                if 'plot_25' in pre_res: st.pyplot(pre_res['plot_25'])

                if 'plot_26' in pre_res: st.pyplot(pre_res['plot_26'])



            if selected_section == "Organizacional e Custos":

                st.subheader("Análise Organizacional, Atividades e Custos")

                if 'plot_07' in pre_res: st.pyplot(pre_res['plot_07'])

                if 'plot_10' in pre_res: st.pyplot(pre_res['plot_10'])

                if 'plot_11' in pre_res: st.pyplot(pre_res['plot_11'])

                if 'plot_12' in pre_res: st.pyplot(pre_res['plot_12'])

                if 'plot_15' in pre_res: st.pyplot(pre_res['plot_15'])

                if 'plot_19' in pre_res: st.pyplot(pre_res['plot_19'])

                if 'plot_13' in pre_res: st.pyplot(pre_res['plot_13'])



            if selected_section == "Gargalos e Handoffs":

                st.subheader("Análise de Gargalos e Handoffs")

                if 'plot_08' in pre_res: st.pyplot(pre_res['plot_08'])

                if 'plot_09' in pre_res: st.pyplot(pre_res['plot_09'])

                if 'plot_14' in pre_res: st.pyplot(pre_res['plot_14'])

                if 'plot_20' in pre_res: st.pyplot(pre_res['plot_20'])

                if 'plot_21' in pre_res: st.pyplot(pre_res['plot_21'])

                if 'plot_22' in pre_res: st.pyplot(pre_res['plot_22'])

                if 'plot_23' in pre_res: st.pyplot(pre_res['plot_23'])

                if 'plot_24' in pre_res: st.pyplot(pre_res['plot_24'])

        

        if main_tab == "Análise de Processos (Pós-Mineração)":

            st.subheader("🗺️ Análise de Processos (Pós-Mineração)")

            sections = ["Descoberta de Modelos", "Variantes e Conformidade", "Análise Temporal", "Tempos de Espera e Recursos"]

            selected_section = st.sidebar.selectbox("Secção:", sections, key='tab2_selectbox')



            if selected_section == "Descoberta de Modelos":

                st.subheader("Descoberta de Modelos e Métricas de Qualidade")

                st.markdown("##### Modelo com Inductive Miner")

                if 'model_01_inductive' in post_res: st.graphviz_chart(post_res['model_01_inductive'])

                st.markdown("###### Métricas de Qualidade (Inductive)")

                if 'metrics_inductive' in post_res: st.pyplot(post_res['metrics_inductive'])

                st.markdown("---")

                st.markdown("##### Modelo com Heuristics Miner")

                if 'model_02_heuristics' in post_res: st.graphviz_chart(post_res['model_02_heuristics'])

                st.markdown("###### Métricas de Qualidade (Heuristics)")

                if 'metrics_heuristics' in post_res: st.pyplot(post_res['metrics_heuristics'])

                st.markdown("---")

                st.subheader("Mapa de Performance do Processo")

                if 'model_03_performance_dfg' in post_res: st.graphviz_chart(post_res['model_03_performance_dfg'])



            if selected_section == "Variantes e Conformidade":

                st.subheader("Análise de Variantes e Conformidade")

                if 'chart_04_variants_pie' in post_res: st.pyplot(post_res['chart_04_variants_pie'])

                if 'chart_08_variant_duration' in post_res: st.pyplot(post_res['chart_08_variant_duration'])

                if 'chart_05_conformance_fitness' in post_res: st.pyplot(post_res['chart_05_conformance_fitness'])

                if 'chart_09_deviation_scatter' in post_res: st.pyplot(post_res['chart_09_deviation_scatter'])

                if 'chart_10_conformance_over_time' in post_res: st.pyplot(post_res['chart_10_conformance_over_time'])



            if selected_section == "Análise Temporal":

                st.subheader("Análise Temporal e de Linha do Tempo")

                if 'chart_06_kpi_time_series' in post_res: st.pyplot(post_res['chart_06_kpi_time_series'])

                if 'chart_11_cumulative_throughput' in post_res: st.pyplot(post_res['chart_11_cumulative_throughput'])

                if 'chart_07_gantt_chart' in post_res: st.pyplot(post_res['chart_07_gantt_chart'])

                if 'chart_12_milestone_analysis' in post_res: st.pyplot(post_res['chart_12_milestone_analysis'])



            if selected_section == "Tempos de Espera e Recursos":

                st.subheader("Análise de Tempos de Espera e Recursos")

                if 'chart_13_waiting_time_matrix' in post_res: st.pyplot(post_res['chart_13_waiting_time_matrix'])

                if 'chart_14_avg_wait_by_activity' in post_res: st.pyplot(post_res['chart_14_avg_wait_by_activity'])

                st.subheader("Redes de Colaboração entre Recursos")

                if 'social_network' in post_res: st.pyplot(post_res['social_network'])

                if 'bipartite_network' in post_res: st.pyplot(post_res['bipartite_network'])
