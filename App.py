# -*- coding: utf-8 -*-
"""
Aplicação Web Streamlit para Análise de Processos de Gestão de Recursos de TI.

Esta aplicação transforma um notebook de análise de processos (usando pm4py)
numa ferramenta web interativa e de fácil utilização, incorporando um dashboard
completo com 46 visualizações organizadas.

Funcionalidades:
1.  **Upload de Dados**: Permite ao utilizador carregar os 5 ficheiros CSV necessários.
2.  **Execução da Análise**: Um botão inicia o processamento completo dos dados.
3.  **Visualização de Resultados**: Apresenta os resultados (KPIs, gráficos e modelos de processo)
    de forma organizada em separadores e subsecções expansíveis.
"""

# --- 1. IMPORTAÇÃO DE BIBLIOTECAS ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from io import StringIO
import warnings

# Bibliotecas de Process Mining (PM4PY)
try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
except ImportError:
    st.error("As bibliotecas de Process Mining (pm4py) não estão instaladas. Por favor, instale-as com 'pip install pm4py'.")
    st.stop()


# --- 2. CONFIGURAÇÃO DA PÁGINA E ESTADO DA SESSÃO ---
st.set_page_config(
    page_title="Dashboard de Análise de Processos de TI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização do estado da sessão
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {k: None for k in ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

# Ignorar warnings para uma UI mais limpa
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 3. FUNÇÕES DE ANÁLISE (MODULARIZADAS) ---

@st.cache_data
def load_and_preprocess_data(uploaded_files):
    """Carrega, pré-processa todos os dados e cria o log de eventos."""
    try:
        # Carregar para DataFrames
        dfs = {name: pd.read_csv(file) for name, file in uploaded_files.items()}

        # Conversões de data
        for col in ['start_date', 'end_date', 'planned_end_date']:
            if col in dfs['projects'].columns: dfs['projects'][col] = pd.to_datetime(dfs['projects'][col], errors='coerce')
            if col in dfs['tasks'].columns: dfs['tasks'][col] = pd.to_datetime(dfs['tasks'][col], errors='coerce')
        if 'allocation_date' in dfs['resource_allocations'].columns:
            dfs['resource_allocations']['allocation_date'] = pd.to_datetime(dfs['resource_allocations']['allocation_date'], errors='coerce')

        # Engenharia de Funcionalidades (Features)
        dfs['projects']['days_diff'] = (dfs['projects']['end_date'] - dfs['projects']['planned_end_date']).dt.days
        dfs['projects']['actual_duration_days'] = (dfs['projects']['end_date'] - dfs['projects']['start_date']).dt.days
        dfs['projects']['completion_month'] = dfs['projects']['end_date'].dt.to_period('M').astype(str)
        
        df_alloc_costs = dfs['resource_allocations'].merge(dfs['resources'], on='resource_id')
        df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']
        
        project_aggregates = df_alloc_costs.groupby('project_id').agg(
            total_actual_cost=('cost_of_work', 'sum'),
            num_resources=('resource_id', 'nunique')
        ).reset_index()

        dfs['projects'] = dfs['projects'].merge(project_aggregates, on='project_id', how='left')
        dfs['projects']['cost_diff'] = dfs['projects']['total_actual_cost'] - dfs['projects']['budget_impact']
        dfs['projects']['cost_per_day'] = dfs['projects']['total_actual_cost'] / dfs['projects']['actual_duration_days'].replace(0, np.nan)

        # DataFrame Unificado (df_full_context)
        allocations_to_merge = dfs['resource_allocations'].drop(columns=['project_id'], errors='ignore')
        df_full_context = dfs['tasks'].merge(dfs['projects'], on='project_id', suffixes=('_task', '_project'))
        df_full_context = df_full_context.merge(allocations_to_merge, on='task_id')
        df_full_context = df_full_context.merge(dfs['resources'], on='resource_id')
        df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
        dfs['full_context'] = df_full_context

        # Log de Eventos para PM4PY
        log_df = dfs['tasks'].merge(allocations_to_merge, on='task_id').merge(dfs['resources'], on='resource_id')
        log_df.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name', 'end_date': 'time:timestamp', 'resource_name': 'org:resource'}, inplace=True)
        log_df['case:concept:name'] = 'Projeto ' + log_df['case:concept:name'].astype(str)
        log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'], errors='coerce')
        log_df.dropna(subset=['time:timestamp'], inplace=True)
        log_df = log_df.sort_values('time:timestamp')
        dfs['log_df'] = log_df
        
        dfs['event_log'] = log_converter.apply(log_df)

        return dfs
    except Exception as e:
        st.error(f"Ocorreu um erro durante o pré-processamento: {e}")
        return None

# --- Funções Geradoras de Gráficos (Pré-Mineração) ---

def generate_pre_mining_visuals(dfs):
    results = {}
    df_projects = dfs['projects']
    df_full_context = dfs['full_context']
    log_df = dfs['log_df']

    # --- Secção 1: Análises de Alto Nível e de Casos ---
    # KPIs
    results['kpis'] = {
        'Total de Projetos': df_projects['project_id'].nunique(),
        'Total de Tarefas': dfs['tasks']['task_id'].nunique(),
        'Total de Recursos': dfs['resources']['resource_id'].nunique(),
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.2f}",
    }
    # plot_01_performance_matrix
    fig, ax = plt.subplots(figsize=(10, 6)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='num_resources', palette='viridis', s=100, alpha=0.7, ax=ax); ax.axhline(0, c='k', ls='--'); ax.axvline(0, c='k', ls='--'); ax.set_title('Matriz de Performance: Prazo vs. Orçamento'); results['plot_01_performance_matrix'] = fig
    # plot_02_case_durations_boxplot
    fig, ax = plt.subplots(figsize=(10, 4)); sns.boxplot(x=df_projects['actual_duration_days'], color='skyblue', ax=ax); ax.set_title('Distribuição da Duração dos Projetos (Lead Time)'); results['plot_02_case_durations_boxplot'] = fig

    # --- Secção 2: Análises de Performance Detalhada ---
    lead_times = log_df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
    # plot_03_lead_time_hist
    fig, ax = plt.subplots(figsize=(10, 4)); sns.histplot(lead_times["lead_time_days"], bins=20, kde=True, ax=ax); ax.set_title('Distribuição do Lead Time por Caso (dias)'); results['plot_03_lead_time_hist'] = fig

    # --- Secção 3: Análise de Atividades e Handoffs ---
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    # plot_07_activity_service_times
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x='hours_worked', y='task_name', data=service_times.sort_values('hours_worked', ascending=False).head(10), palette='viridis', ax=ax, hue='task_name', legend=False); ax.set_title('Tempo Médio de Execução por Atividade (Horas)'); results['plot_07_activity_service_times'] = fig

    # --- Secção 4: Análise Organizacional (Recursos) ---
    # plot_10_top_activities_plot
    activity_counts = dfs['tasks']["task_name"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='plasma', hue=activity_counts.head(10).index, legend=False); ax.set_title('Atividades Mais Frequentes'); results['plot_10_top_activities_plot'] = fig
    # plot_11_resource_workload
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x=resource_workload.head(10).values, y=resource_workload.head(10).index, ax=ax, palette='magma', hue=resource_workload.head(10).index, legend=False); ax.set_title('Top 10 Recursos por Horas Trabalhadas'); results['plot_11_resource_workload'] = fig
    # plot_15_cost_by_resource_type
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x=cost_by_resource_type.values, y=cost_by_resource_type.index, ax=ax, palette='cividis', hue=cost_by_resource_type.index, legend=False); ax.set_title('Custo Total por Tipo de Recurso'); results['plot_15_cost_by_resource_type'] = fig

    # --- Secção 6: Análise Aprofundada ---
    # plot_17_delay_by_teamsize
    bins = np.linspace(df_projects['num_resources'].min(), df_projects['num_resources'].max(), 5, dtype=int)
    df_projects['team_size_bin'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_projects, x='team_size_bin', y='days_diff', ax=ax, palette='flare', hue='team_size_bin', legend=False); ax.set_title('Impacto do Tamanho da Equipa no Atraso'); results['plot_17_delay_by_teamsize'] = fig
    
    # plot_19_weekly_efficiency
    # CORREÇÃO: Usar 'allocation_date' que existe em df_full_context
    df_full_context['day_of_week'] = df_full_context['allocation_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_hours = df_full_context.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x=weekly_hours.index, y=weekly_hours.values, ax=ax, palette='plasma', hue=weekly_hours.index, legend=False); ax.set_title('Total de Horas Trabalhadas por Dia da Semana'); results['plot_19_weekly_efficiency'] = fig

    return results

# --- Funções Geradoras de Gráficos (Pós-Mineração) ---

def generate_post_mining_visuals(dfs):
    results = {}
    event_log = dfs['event_log']
    df_tasks = dfs['tasks']
    df_projects = dfs['projects']
    
    # --- Modelos de Processo ---
    process_tree_im = inductive_miner.apply(event_log)
    net_im, im_im, fm_im = pm4py.convert_to_petri_net(process_tree_im)
    results['inductive_model'] = pn_visualizer.apply(net_im, im_im, fm_im)
    
    net_hm, im_hm, fm_hm = heuristics_miner.apply(event_log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.8})
    results['heuristics_model'] = pn_visualizer.apply(net_hm, im_hm, fm_hm)

    # DFG de Performance
    dfg_perf, sa, ea = pm4py.discover_performance_dfg(event_log)
    results['performance_dfg'] = dfg_visualizer.apply(dfg_perf, log=event_log, variant=dfg_visualizer.Variants.PERFORMANCE)

    # --- Gantt Chart ---
    fig_gantt, ax = plt.subplots(figsize=(20, 10))
    projects_to_plot = df_projects.sort_values('start_date').head(20)
    tasks_to_plot = df_tasks[df_tasks['project_id'].isin(projects_to_plot['project_id'])]
    project_y_map = {proj_id: i for i, proj_id in enumerate(projects_to_plot['project_id'])}
    task_colors = plt.get_cmap('viridis', tasks_to_plot['task_name'].nunique())
    color_map = {name: task_colors(i) for i, name in enumerate(tasks_to_plot['task_name'].unique())}
    for _, task in tasks_to_plot.iterrows():
        if task['project_id'] in project_y_map:
            ax.barh(project_y_map[task['project_id']], (task['end_date'] - task['start_date']).days + 1, left=task['start_date'], color=color_map.get(task['task_name']))
    ax.set_yticks(list(project_y_map.values())); ax.set_yticklabels([f"Projeto {pid}" for pid in project_y_map.keys()]); ax.invert_yaxis()
    ax.set_title('Gráfico de Gantt (20 Primeiros Projetos)'); results['gantt_chart'] = fig_gantt

    # --- Análise de Variantes ---
    variants = pm4py.get_variants_as_tuples(event_log)
    variants_df = pd.DataFrame.from_dict(variants, orient='index', columns=['count']).sort_values(by='count', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6)); ax.pie(variants_df['count'].head(7), labels=[f'Variante {i+1}' for i in range(7)], autopct='%1.1f%%', startangle=90); ax.set_title('Distribuição das 7 Variantes Mais Comuns'); results['variants_pie'] = fig

    # --- Análise de Conformidade ---
    aligned_traces = alignments.apply(event_log, net_im, im_im, fm_im)
    fitness_values = [trace['fitness'] for trace in aligned_traces]
    fig, ax = plt.subplots(figsize=(10, 4)); sns.histplot(fitness_values, bins=20, kde=True, ax=ax, color='green'); ax.set_title('Distribuição do Fitness de Conformidade'); results['conformance_fitness'] = fig
    
    return results

def run_full_analysis():
    """Função principal para orquestrar a análise completa."""
    with st.spinner('A processar os dados e a gerar as análises... Por favor, aguarde.'):
        st.session_state.dataframes = load_and_preprocess_data(st.session_state.uploaded_files)
        
        if st.session_state.dataframes:
            st.session_state.results.update(generate_pre_mining_visuals(st.session_state.dataframes))
            st.session_state.results.update(generate_post_mining_visuals(st.session_state.dataframes))
            
            st.session_state.analysis_complete = True
            st.success('Análise concluída com sucesso! Navegue para "Visualizar Resultados" para ver o dashboard.')
        else:
            st.error("A análise falhou. Verifique os ficheiros e tente novamente.")


# --- 4. LAYOUT DA APLICAÇÃO (UI) ---

st.title("📊 Dashboard de Análise de Processos de TI")
st.markdown("Bem-vindo! Esta ferramenta transforma os seus dados de gestão de projetos num dashboard interativo de Process Mining.")

st.sidebar.title("Painel de Controlo")
menu_selection = st.sidebar.radio(
    "Menu de Navegação",
    ["1. Carregar Dados", "2. Executar Análise", "3. Visualizar Resultados"],
    captions=["Faça o upload dos seus ficheiros CSV", "Inicie o processamento dos dados", "Explore o dashboard interativo"]
)

# --- Secção 1: Upload de Dados ---
if menu_selection == "1. Carregar Dados":
    st.header("1. Upload dos Ficheiros CSV")
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários. Após cada upload, verá uma pré-visualização.")
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    for name in file_names:
        with st.expander(f"Carregar `{name}.csv`", expanded=True):
            uploaded_file = st.file_uploader(f"Selecione `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                st.session_state.uploaded_files[name] = uploaded_file
                df_preview = pd.read_csv(uploaded_file); uploaded_file.seek(0)
                st.dataframe(df_preview.head(), use_container_width=True)
                st.success(f"`{name}.csv` carregado!")

# --- Secção 2: Execução da Análise ---
elif menu_selection == "2. Executar Análise":
    st.header("2. Execução da Análise")
    if all(st.session_state.uploaded_files.values()):
        st.info("Todos os ficheiros foram carregados. Está pronto para iniciar a análise.")
        if st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):
            run_full_analysis()
    else:
        missing = [name for name, f in st.session_state.uploaded_files.items() if f is None]
        st.error(f"Faltam ficheiros: `{', '.join(missing)}`. Por favor, carregue-os na secção '1. Carregar Dados'.")

# --- Secção 3: Visualização dos Resultados ---
elif menu_selection == "3. Visualizar Resultados":
    st.header("3. Dashboard de Resultados")
    if not st.session_state.analysis_complete:
        st.warning("A análise ainda não foi executada. Vá à secção '2. Executar Análise'.")
    else:
        # --- Navegação Principal do Dashboard ---
        tab1, tab2 = st.tabs(["📊 Análise Descritiva (Pré-Mineração)", "🗺️ Análise de Processos (Pós-Mineração)"])

        with tab1:
            st.subheader("Análise Geral do Desempenho e Recursos")
            
            with st.expander(" KPIs de Alto Nível e Matriz de Performance", expanded=True):
                cols = st.columns(4)
                for i, (metric, value) in enumerate(st.session_state.results['kpis'].items()):
                    cols[i].metric(label=metric, value=value)
                st.pyplot(st.session_state.results.get('plot_01_performance_matrix'), use_container_width=True)

            with st.expander("Análise de Duração e Prazos"):
                st.pyplot(st.session_state.results.get('plot_02_case_durations_boxplot'), use_container_width=True)
                st.pyplot(st.session_state.results.get('plot_03_lead_time_hist'), use_container_width=True)
                st.pyplot(st.session_state.results.get('plot_17_delay_by_teamsize'), use_container_width=True)

            with st.expander("Análise Organizacional e de Custos"):
                st.pyplot(st.session_state.results.get('plot_11_resource_workload'), use_container_width=True)
                st.pyplot(st.session_state.results.get('plot_15_cost_by_resource_type'), use_container_width=True)
                st.pyplot(st.session_state.results.get('plot_19_weekly_efficiency'), use_container_width=True)
                st.pyplot(st.session_state.results.get('plot_07_activity_service_times'), use_container_width=True)
                st.pyplot(st.session_state.results.get('plot_10_top_activities_plot'), use_container_width=True)
        
        with tab2:
            st.subheader("Descoberta, Conformidade e Análise de Performance do Processo")

            with st.expander("Descoberta de Modelos de Processo", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Modelo com Inductive Miner**")
                    st.graphviz_chart(st.session_state.results.get('inductive_model'), use_container_width=True)
                with col2:
                    st.markdown("**Modelo com Heuristics Miner**")
                    st.graphviz_chart(st.session_state.results.get('heuristics_model'), use_container_width=True)
            
            with st.expander("Análise de Performance e Gargalos do Processo"):
                st.graphviz_chart(st.session_state.results.get('performance_dfg'), use_container_width=True)
                st.pyplot(st.session_state.results.get('gantt_chart'), use_container_width=True)

            with st.expander("Análise de Variantes e Conformidade"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribuição de Variantes**")
                    st.pyplot(st.session_state.results.get('variants_pie'), use_container_width=True)
                with col2:
                    st.markdown("**Distribuição de Fitness de Conformidade**")
                    st.pyplot(st.session_state.results.get('conformance_fitness'), use_container_width=True)

