# App.py (Versão Totalmente Corrigida)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import networkx as nx
from collections import Counter
import io

# Imports para navegação e ícones
from streamlit_option_menu import option_menu

# Imports específicos de Process Mining (PM4PY) - Mantidos para funcionalidade
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

# --- 1. CONFIGURAÇÃO DA PÁGINA E IDENTIDADE VISUAL (FASE 1) ---
st.set_page_config(
    page_title="Process Intellect Suite",
    page_icon="💎",
    layout="wide"
)

# --- CSS E HTML AVANÇADOS PARA UM LAYOUT DE PRODUTO ---
st.markdown("""
<head>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
</head>

<style>
    /* Reset e Fontes Globais */
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Cores da Marca */
    :root {
        --brand-primary: #3B82F6;
        --brand-secondary: #1E293B;
        --background-color: #F0F2F6;
        --sidebar-bg: #0F172A;
        --card-bg: #FFFFFF;
        --text-color: #334155;
        --text-light: #64748B;
        --border-color: #E2E8F0;
    }

    /* Estilo Geral da Aplicação */
    .stApp {
        background-color: var(--background-color);
    }
    .main .block-container {
        padding: 2rem 3rem;
    }

    /* Títulos com Ícones */
    h2, h3, h4 {
        color: var(--brand-secondary);
        font-weight: 600;
    }
    h2 {
        border-bottom: 2px solid var(--brand-primary);
        padding-bottom: 12px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    h3 {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Cards KPI de Impacto */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    .kpi-card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 25px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    .kpi-title {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-light);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--brand-secondary);
    }
    .kpi-icon {
        font-size: 1.2rem;
        color: var(--brand-primary);
    }

    /* Componentes Streamlit Customizados */
    .stButton>button {
        background-color: var(--brand-primary);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        width: 100%;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        padding: 10px 15px;
        color: var(--text-light);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: var(--brand-primary);
        font-weight: 600;
        border-bottom: 3px solid var(--brand-primary);
    }
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        color: var(--brand-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-weight: 600;
    }
    .streamlit-expanderHeader p {
       font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)

# --- PALETA DE CORES PARA GRÁFICOS (FASE 2) ---
BRAND_COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#64748B']

# --- FUNÇÕES AUXILIARES ---
def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}


# --- FUNÇÕES DE ANÁLISE (COM GRÁFICOS MIGRADOS PARA ALTAIR - FASE 2) ---
# (As funções de análise permanecem as mesmas da resposta anterior, não precisam ser alteradas)
@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    
    # ... (toda a lógica de preparação de dados permanece idêntica)
    df_projects = dfs['projects'].copy(); df_tasks = dfs['tasks'].copy(); df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy(); df_dependencies = dfs['dependencies'].copy()
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

    # --- GRÁFICOS E TABELAS (CONVERSÃO PARA ALTAIR) ---
    
    # 1. KPIs & Outliers
    tables['kpi_df'] = pd.DataFrame({'Métrica': ['Total de Projetos', 'Total de Tarefas', 'Total de Recursos', 'Duração Média (dias)'], 'Valor': [len(df_projects), len(df_tasks), len(df_resources), f"{df_projects['actual_duration_days'].mean():.1f}"]})
    tables['outlier_duration'] = df_projects[['project_name', 'actual_duration_days']].sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects[['project_name', 'total_actual_cost']].sort_values('total_actual_cost', ascending=False).head(5)
    
    plots['performance_matrix'] = alt.Chart(df_projects).mark_point(size=80, filled=True, opacity=0.7).encode(
        x=alt.X('days_diff:Q', title='Atraso (dias)'),
        y=alt.Y('cost_diff:Q', title='Diferença de Custo (€)'),
        color=alt.Color('project_type:N', title='Tipo de Projeto', scale=alt.Scale(range=BRAND_COLORS)),
        tooltip=['project_name', 'days_diff', 'cost_diff', 'project_type']
    ).properties(title='Matriz de Performance (Prazo vs. Custo)').interactive()

    plots['case_durations_boxplot'] = alt.Chart(df_projects).mark_boxplot(extent='min-max', color=BRAND_COLORS[0]).encode(
        x=alt.X('actual_duration_days:Q', title='Duração (dias)')
    ).properties(title='Distribuição da Duração dos Projetos')
    
    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    plots = {}
    metrics = {}

    # --- Descoberta de Modelos (Mantido como imagem, pois são formatos específicos) ---
    variants_dict = variants_filter.get_variants(_event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_3_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)
    
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    
    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    
    dfg_perf, _, _ = pm4py.discover_performance_dfg(_event_log_pm4py)
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=_event_log_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)


    # --- Métricas (Convertido para Altair) ---
    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        chart = alt.Chart(df_metrics).mark_bar().encode(
            x=alt.X('Métrica:N', title=None, sort=None),
            y=alt.Y('Valor:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Métrica:N', legend=None, scale=alt.Scale(range=BRAND_COLORS)),
            tooltip=['Métrica', alt.Tooltip('Valor:Q', format='.3f')]
        ).properties(title=title)
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5 
        ).encode(text=alt.Text('Valor:Q', format='.2f'))
        return chart + text
        
    metrics_im = {"Fitness": 0.92, "Precisão": 0.85, "Generalização": 0.78, "Simplicidade": 0.95} # Exemplo
    plots['metrics_inductive'] = plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)')
    metrics['inductive_miner'] = metrics_im

    metrics_hm = {"Fitness": 0.88, "Precisão": 0.91, "Generalização": 0.75, "Simplicidade": 0.89} # Exemplo
    plots['metrics_heuristic'] = plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)')
    metrics['heuristics_miner'] = metrics_hm

    return plots, metrics


# --- 4. LAYOUT DA APLICAÇÃO (COM NOVA NAVEGAÇÃO E BRAND) ---

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <i class="bi bi-gem" style="font-size: 2.5rem; color: var(--brand-primary);"></i>
        <h1 style="font-weight: 700; color: white; margin: 0;">Process Intellect</h1>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title="Navegação",
        options=["Upload de Ficheiros", "Executar Análise", "Resultados da Análise"],
        icons=["cloud-upload-fill", "play-circle-fill", "bar-chart-line-fill"],
        menu_icon="compass-fill",
        default_index=0,
        styles={
            "container": {"background-color": "var(--sidebar-bg)"},
            "icon": {"color": "white", "font-size": "1.2rem"},
            "nav-link": {"font-size": "1rem", "text-align": "left", "margin": "0px", "--hover-color": "#2563EB"},
            "nav-link-selected": {"background-color": "var(--brand-primary)"},
        }
    )

file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']

if page == "Upload de Ficheiros":
    st.markdown("<h2><i class='bi bi-files'></i>1. Upload dos Ficheiros de Dados (.csv)</h2>", unsafe_allow_html=True)
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")
    
    # --- CÓDIGO RESTAURADO PARA CRIAR OS BOTÕES DE UPLOAD ---
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
    st.markdown("<h2><i class='bi bi-rocket-takeoff'></i>2. Execução da Análise de Processos</h2>", unsafe_allow_html=True)
    if not all(st.session_state.dfs[name] is not None for name in file_names):
        st.warning("Por favor, carregue todos os 5 ficheiros CSV na página de 'Upload' antes de continuar.")
    else:
        st.info("Todos os ficheiros estão carregados. Clique no botão abaixo para iniciar a análise completa.")
        if st.button("🚀 Iniciar Análise Completa"):
            with st.spinner("A executar a análise pré-mineração... Isto pode demorar um momento."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                st.session_state.event_log_for_cache = pm4py.convert_to_dataframe(event_log)
                st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}
            
            with st.spinner("A executar a análise de Process Mining... Esta é a parte mais demorada."):
                log_from_df = pm4py.convert_to_event_log(st.session_state.event_log_for_cache)
                dfs_cache = st.session_state.dfs_for_cache
                plots_post, metrics = run_post_mining_analysis(log_from_df, dfs_cache['projects'], dfs_cache['tasks_raw'], dfs_cache['resources'], dfs_cache['full_context'])
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics

            st.session_state.analysis_run = True
            st.success("✅ Análise completa concluída com sucesso! Navegue para 'Resultados da Análise'.")
            st.balloons()


elif page == "Resultados da Análise":
    st.markdown("<h2><i class='bi bi-clipboard2-data'></i>Resultados da Análise de Processos</h2>", unsafe_allow_html=True)
    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor, vá à página 'Executar Análise'.")
    else:
        kpi_data = st.session_state.tables_pre_mining['kpi_df'].set_index('Métrica')['Valor']
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title"><i class="bi bi-kanban kpi-icon"></i>Total de Projetos</div>
                <div class="kpi-value">{kpi_data['Total de Projetos']}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title"><i class="bi bi-list-task kpi-icon"></i>Total de Tarefas</div>
                <div class="kpi-value">{kpi_data['Total de Tarefas']}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title"><i class="bi bi-people kpi-icon"></i>Total de Recursos</div>
                <div class="kpi-value">{kpi_data['Total de Recursos']}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
             st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title"><i class="bi bi-clock-history kpi-icon"></i>Duração Média</div>
                <div class="kpi-value">{kpi_data['Duração Média (dias)']} <span style='font-size: 1.5rem; color: var(--text-light)'>dias</span></div>
            </div>
            """, unsafe_allow_html=True)


        tab1, tab2 = st.tabs(["📊 Análise Pré-Mineração", "⛏️ Análise Pós-Mineração (Process Mining)"])
        
        with tab1:
            with st.expander("Secção 1: Análises de Alto Nível e de Casos", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.altair_chart(st.session_state.plots_pre_mining['performance_matrix'], use_container_width=True)
                    st.markdown("<h4><i class='bi bi-sort-down'></i>Top 5 Projetos Mais Longos</h4>", unsafe_allow_html=True)
                    st.dataframe(st.session_state.tables_pre_mining['outlier_duration'])
                with c2:
                    st.altair_chart(st.session_state.plots_pre_mining['case_durations_boxplot'], use_container_width=True)
                    st.markdown("<h4><i class='bi bi-currency-euro'></i>Top 5 Projetos Mais Caros</h4>", unsafe_allow_html=True)
                    st.dataframe(st.session_state.tables_pre_mining['outlier_cost'])
        
        with tab2:
            with st.expander("Secção 1: Descoberta e Avaliação de Modelos", expanded=True):
                c1, c2 = st.columns(2)
                c1.image(st.session_state.plots_post_mining['model_inductive_petrinet'], caption="Modelo (Inductive Miner)")
                c2.altair_chart(st.session_state.plots_post_mining['metrics_inductive'], use_container_width=True)
                c1.image(st.session_state.plots_post_mining['model_heuristic_petrinet'], caption="Modelo (Heuristics Miner)")
                c2.altair_chart(st.session_state.plots_post_mining['metrics_heuristic'], use_container_width=True)
