# -*- coding: utf-8 -*-
"""
Aplicação Web Streamlit para Análise de Processos de Gestão de Recursos de TI.

Esta aplicação transforma um notebook de análise de processos (usando pm4py)
numa ferramenta web interativa e de fácil utilização.

Funcionalidades:
1.  **Upload de Dados**: Permite ao utilizador carregar os 5 ficheiros CSV necessários.
2.  **Execução da Análise**: Um botão inicia o processamento completo dos dados.
3.  **Visualização de Resultados**: Apresenta os resultados (KPIs, gráficos e modelos de processo)
    de forma organizada em separadores.
"""

# --- 1. IMPORTAÇÃO DE BIBLIOTECAS ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from io import StringIO
import warnings

# Bibliotecas de Process Mining (PM4PY)
# A instalação será gerida pelo ficheiro requirements.txt na plataforma de deploy.
try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
    from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
    from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
except ImportError:
    st.error("As bibliotecas de Process Mining (pm4py) não estão instaladas. Por favor, instale-as com 'pip install pm4py'.")
    st.stop()


# --- 2. CONFIGURAÇÃO DA PÁGINA E ESTADO DA SESSÃO ---

# Configuração inicial da página da aplicação
st.set_page_config(
    page_title="Analisador de Processos de TI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização do estado da sessão para armazenar dados e resultados
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {
        'projects': None,
        'tasks': None,
        'resources': None,
        'resource_allocations': None,
        'dependencies': None
    }
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

# --- 3. FUNÇÕES DE ANÁLISE (Refatoradas do Notebook) ---

# Ignorar warnings específicos para uma apresentação mais limpa
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

@st.cache_data
def load_and_preprocess_data(uploaded_files):
    """
    Carrega os dados dos ficheiros CSV, realiza o pré-processamento e
    cria o log de eventos para a análise de processos.
    """
    try:
        # Carregar para DataFrames
        df_projects = pd.read_csv(uploaded_files['projects'])
        df_tasks = pd.read_csv(uploaded_files['tasks'])
        df_resources = pd.read_csv(uploaded_files['resources'])
        df_resource_allocations = pd.read_csv(uploaded_files['resource_allocations'])
        df_dependencies = pd.read_csv(uploaded_files['dependencies'])

        # Conversões de data
        for df in [df_projects, df_tasks]:
            for col in ['start_date', 'end_date', 'planned_end_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        # Engenharia de funcionalidades
        df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
        df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
        
        df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
        df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']
        
        project_aggregates = df_alloc_costs.groupby('project_id').agg(
            total_actual_cost=('cost_of_work', 'sum'),
            num_resources=('resource_id', 'nunique')
        ).reset_index()

        df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
        df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
        
        # Criação do DataFrame unificado
        # CORREÇÃO DEFINITIVA: Remover 'project_id' de allocations para evitar colisão, como no notebook.
        allocations_to_merge = df_resource_allocations.drop(columns=['project_id'], errors='ignore')
        df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
        df_full_context = df_full_context.merge(allocations_to_merge, on='task_id')
        df_full_context = df_full_context.merge(df_resources, on='resource_id')
        df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
        
        # Criação do Log de Eventos para PM4PY
        # CORREÇÃO DEFINITIVA: Lógica de merge alinhada com o notebook para evitar erro de 'project_id_x'
        log_df = df_tasks.merge(allocations_to_merge, on='task_id').merge(df_resources, on='resource_id')
        log_df.rename(columns={
            'project_id': 'case:concept:name', # CORREÇÃO: Alterado de 'project_id_x' para 'project_id'
            'task_name': 'concept:name',
            'end_date': 'time:timestamp',
            'resource_name': 'org:resource'
        }, inplace=True)
        log_df['case:concept:name'] = 'Projeto ' + log_df['case:concept:name'].astype(str)
        log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'], errors='coerce')
        log_df.dropna(subset=['time:timestamp'], inplace=True)
        log_df = log_df.sort_values('time:timestamp')
        
        event_log = log_converter.apply(log_df)

        return {
            'df_projects': df_projects,
            'df_tasks': df_tasks,
            'df_full_context': df_full_context,
            'event_log': event_log
        }

    except Exception as e:
        st.error(f"Ocorreu um erro durante o pré-processamento: {e}")
        return None

def generate_high_level_visuals(df_projects):
    """Gera os gráficos e KPIs de alto nível."""
    results = {}
    
    # KPIs
    kpis = {
        'Total de Projetos': df_projects['project_id'].nunique(),
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.2f}",
        'Custo Médio (€)': f"{df_projects['total_actual_cost'].mean():,.2f}",
        'Desvio Médio de Prazo (dias)': f"{df_projects['days_diff'].mean():.2f}"
    }
    results['kpis'] = kpis
    
    # Matriz de Performance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='num_resources', size='total_actual_cost',
                    sizes=(50, 500), alpha=0.7, palette='viridis', ax=ax)
    ax.axhline(0, color='grey', linestyle='--', lw=1)
    ax.axvline(0, color='grey', linestyle='--', lw=1)
    ax.set_title('Matriz de Performance: Prazo vs. Orçamento', fontsize=16)
    ax.set_xlabel('Desvio de Prazo (dias)')
    ax.set_ylabel('Desvio de Custo (€)')
    ax.legend(title='Nº de Recursos')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    results['performance_matrix'] = fig
    
    return results

def discover_and_evaluate_models(event_log):
    """Descobre modelos de processo e avalia a sua qualidade."""
    results = {}
    
    # Inductive Miner
    net_im, im_im, fm_im = inductive_miner.apply(event_log)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    results['inductive_model'] = gviz_im
    
    # Heuristics Miner
    net_hm, im_hm, fm_hm = heuristics_miner.apply(event_log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.8})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    results['heuristics_model'] = gviz_hm
    
    return results

def analyze_bottlenecks_and_resources(event_log, df_full_context):
    """Analisa gargalos, handoffs e performance dos recursos."""
    results = {}

    # DFG de Performance
    dfg_perf = pm4py.discover_performance_dfg(event_log, activity_key='concept:name', timestamp_key='time:timestamp', case_id_key='case:concept:name')
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=event_log, variant=dfg_visualizer.Variants.PERFORMANCE)
    results['performance_dfg'] = gviz_dfg
    
    # Carga de trabalho por recurso
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=resource_workload.values, y=resource_workload.index, palette='magma', ax=ax, hue=resource_workload.index, legend=False)
    ax.set_title('Top 15 Recursos por Horas Trabalhadas', fontsize=16)
    ax.set_xlabel('Total de Horas Trabalhadas')
    ax.set_ylabel('Recurso')
    results['resource_workload'] = fig
    
    return results

def run_full_analysis():
    """Função principal para orquestrar a análise completa."""
    with st.spinner('A processar os dados e a gerar as análises... Por favor, aguarde.'):
        # Passo 1: Carregar e pré-processar os dados
        st.session_state.dataframes = load_and_preprocess_data(st.session_state.uploaded_files)
        
        if st.session_state.dataframes:
            # Extrair dataframes e log para fácil acesso
            df_projects = st.session_state.dataframes['df_projects']
            event_log = st.session_state.dataframes['event_log']
            df_full_context = st.session_state.dataframes['df_full_context']

            # Passo 2: Executar os módulos de análise
            st.session_state.results.update(generate_high_level_visuals(df_projects))
            st.session_state.results.update(discover_and_evaluate_models(event_log))
            st.session_state.results.update(analyze_bottlenecks_and_resources(event_log, df_full_context))
            
            # Sinalizar que a análise foi concluída com sucesso
            st.session_state.analysis_complete = True
            st.success('Análise concluída com sucesso! Navegue para "Visualizar Resultados" para ver o dashboard.')
        else:
            st.error("A análise falhou. Verifique os ficheiros e tente novamente.")


# --- 4. LAYOUT DA APLICAÇÃO (UI) ---

# Título Principal da Aplicação
st.title("🚀 Analisador de Processos de Gestão de Recursos de TI")
st.markdown("Bem-vindo! Esta ferramenta transforma os seus dados de gestão de projetos num dashboard interativo de Process Mining.")

# --- Barra Lateral de Navegação ---
st.sidebar.title("Painel de Controlo")
st.sidebar.markdown("Navegue pelas secções da aplicação abaixo.")
menu_selection = st.sidebar.radio(
    "Menu de Navegação",
    ["1. Carregar Dados", "2. Executar Análise", "3. Visualizar Resultados"],
    captions=["Faça o upload dos seus ficheiros CSV", "Inicie o processamento dos dados", "Explore o dashboard interativo"]
)

# --- Secção 1: Upload de Dados ---
if menu_selection == "1. Carregar Dados":
    st.header("1. Upload dos Ficheiros CSV")
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários para a análise. Após cada upload, verá uma pré-visualização das primeiras linhas.")

    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    for name in file_names:
        with st.expander(f"Carregar `{name}.csv`", expanded=True):
            uploaded_file = st.file_uploader(f"Selecione o ficheiro `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file is not None:
                st.session_state.uploaded_files[name] = uploaded_file
                # CORREÇÃO: Revertido para separador vírgula e adicionado .seek(0)
                df_preview = pd.read_csv(uploaded_file)
                uploaded_file.seek(0) # IMPORTANTE: Repõe o ponteiro do ficheiro para o início
                st.dataframe(df_preview.head(), use_container_width=True)
                st.success(f"`{name}.csv` carregado com sucesso!")


# --- Secção 2: Execução da Análise ---
elif menu_selection == "2. Executar Análise":
    st.header("2. Execução da Análise de Processos")
    st.markdown("Quando todos os ficheiros estiverem carregados, clique no botão abaixo para iniciar a análise completa.")

    # Verificar se todos os ficheiros foram carregados
    all_files_uploaded = all(st.session_state.uploaded_files.values())

    if all_files_uploaded:
        st.info("Todos os ficheiros necessários foram carregados. Está pronto para iniciar a análise.")
        if st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):
            run_full_analysis()
    else:
        missing_files = [name for name, f in st.session_state.uploaded_files.items() if f is None]
        st.error(f"Ainda não é possível executar a análise. Por favor, carregue os seguintes ficheiros na secção '1. Carregar Dados': `{', '.join(missing_files)}`")

# --- Secção 3: Visualização dos Resultados ---
elif menu_selection == "3. Visualizar Resultados":
    st.header("3. Resultados da Análise")
    
    if not st.session_state.analysis_complete:
        st.warning("A análise ainda não foi executada. Por favor, vá à secção '2. Executar Análise' e inicie o processo.")
    else:
        st.markdown("Explore os resultados da sua análise de processos nos separadores abaixo.")
        
        # Criação de separadores para organizar os resultados
        tab1, tab2, tab3 = st.tabs(["📊 Análise de Alto Nível", "🗺️ Modelos de Processo", "🔬 Análise de Gargalos e Recursos"])

        with tab1:
            st.subheader("Painel de KPIs de Alto Nível")
            kpis = st.session_state.results.get('kpis', {})
            cols = st.columns(4)
            for i, (metric, value) in enumerate(kpis.items()):
                cols[i].metric(label=metric, value=value)
            
            st.divider()
            
            st.subheader("Matriz de Performance: Prazo vs. Orçamento")
            st.pyplot(st.session_state.results.get('performance_matrix'), use_container_width=True)

        with tab2:
            st.subheader("Modelo de Processo (Inductive Miner)")
            st.markdown("Este modelo é gerado utilizando o Inductive Miner, que garante a produção de um modelo de processo 'sólido' e bem estruturado.")
            st.graphviz_chart(st.session_state.results.get('inductive_model'))
            
            st.divider()
            
            st.subheader("Modelo de Processo (Heuristics Miner)")
            st.markdown("Este modelo é gerado pelo Heuristics Miner, que é mais flexível e foca-se nas relações de dependência mais frequentes, sendo útil para processos menos estruturados.")
            st.graphviz_chart(st.session_state.results.get('heuristics_model'))

        with tab3:
            st.subheader("Mapa de Processo com Performance (DFG)")
            st.markdown("Este diagrama mostra o fluxo do processo. A cor e a espessura das setas indicam a frequência e o tempo médio de transição entre as atividades.")
            st.graphviz_chart(st.session_state.results.get('performance_dfg'))
            
            st.divider()

            st.subheader("Carga de Trabalho por Recurso")
            st.markdown("Visualização das horas totais trabalhadas pelos recursos mais ativos, ajudando a identificar a distribuição do esforço.")
            st.pyplot(st.session_state.results.get('resource_workload'), use_container_width=True)

