import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io
import altair as alt
from streamlit_option_menu import option_menu

# Imports de Process Mining
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
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
    [data-testid="stSidebar"] { background-color: #0F172A; border-right: none; }
    .kpi-card { background-color: #FFFFFF; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center; border: 1px solid #E2E8F0; }
    .kpi-card h3 { color: #475569; font-size: 1rem; font-weight: 600; margin: 0; padding: 0; text-transform: uppercase; }
    .kpi-card p { color: #0F172A; font-size: 2.5rem; font-weight: 700; margin: 5px 0 0 0; padding: 0; letter-spacing: -1px; }
    h1 { color: #0F172A; font-weight: 700; letter-spacing: -2px; }
    h2 { color: #1E293B; font-weight: 600; border: none; padding: 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border: none; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 8px; padding: 10px 20px; color: #475569; border: none; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; color: #3B82F6; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
    .streamlit-expanderHeader { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; font-weight: 600; color: #1E293B; }
    .streamlit-expanderContent { border-left: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; border-radius: 0 0 8px 8px; margin-top: -8px; }
</style>
""", unsafe_allow_html=True)


# --- FUNÇÕES AUXILIARES ---
def convert_fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=180, transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz):
    return io.BytesIO(gviz.pipe(format='png'))

PALETA_CORES = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#6366F1"]

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'processed_data' not in st.session_state: st.session_state.processed_data = {}

# --- FUNÇÃO DE ANÁLISE (REVISTA E CORRIGIDA) ---
@st.cache_data
def run_full_analysis(dfs):
    processed_data = {}
    
    # --- PRÉ-PROCESSAMENTO ---
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
            
    for col in ['project_id', 'task_id', 'resource_id']:
        for df in [df_projects, df_tasks, df_resources, df_resource_allocations]:
            if col in df.columns: df[col] = df[col].astype(str)

    # --- ENRIQUECIMENTO DOS DADOS (LÓGICA SIMPLIFICADA E CORRIGIDA) ---
    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['project_name'].str.extract(r'Projeto \d+: (.*?) ')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)
    
    df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
    df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']
    
    project_aggregates = df_alloc_costs.groupby('project_id').agg(total_actual_cost=('cost_of_work', 'sum'), num_resources=('resource_id', 'nunique')).reset_index()
    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    
    # CRIAÇÃO DO DATAFRAME UNIFICADO (df_full_context) - VERSÃO CORRIGIDA E ROBUSTA
    df_full_context = df_tasks.merge(df_alloc_costs, on=['project_id', 'task_id'], how='left')
    df_full_context = df_full_context.merge(df_projects.add_suffix('_project'), left_on='project_id', right_on='project_id_project', how='left')
    
    processed_data['df_projects'] = df_projects
    processed_data['df_full_context'] = df_full_context

    # --- LÓGICA DE PROCESS MINING E MODELOS ---
    log_df = df_full_context.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name', 'end_date': 'time:timestamp', 'resource_name': 'org:resource'})
    log_df = log_df[['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource']].dropna()
    log_df['lifecycle:transition'] = 'complete'
    event_log = pm4py.convert_to_event_log(log_df)
    processed_data['event_log_df'] = log_df 

    variants = variants_filter.get_variants(event_log)
    top_3_variants = variants_filter.apply(event_log, sorted(variants, key=lambda k: len(variants[k]), reverse=True)[:3])
    
    net_im, im_im, fm_im = pt_converter.apply(inductive_miner.apply(top_3_variants))
    processed_data['model_inductive_petrinet'] = convert_gviz_to_bytes(pn_visualizer.apply(net_im, im_im, fm_im))
    
    net_hm, im_hm, fm_hm = heuristics_miner.apply(top_3_variants)
    processed_data['model_heuristic_petrinet'] = convert_gviz_to_bytes(pn_visualizer.apply(net_hm, im_hm, fm_hm))

    dfg_perf, _, _ = pm4py.discover_performance_dfg(event_log)
    processed_data['performance_heatmap'] = convert_gviz_to_bytes(dfg_visualizer.apply(dfg_perf, log=event_log, variant=dfg_visualizer.Variants.PERFORMANCE))
    
    handovers = Counter((log_df.iloc[i]['org:resource'], log_df.iloc[i+1]['org:resource']) for i in range(len(log_df)-1) if log_df.iloc[i]['case:concept:name'] == log_df.iloc[i+1]['case:concept:name'] and log_df.iloc[i]['org:resource'] != log_df.iloc[i+1]['org:resource'])
    fig_net, ax_net = plt.subplots(figsize=(10, 10)); G = nx.DiGraph();
    for (source, target), weight in handovers.items(): G.add_edge(str(source), str(target), weight=weight)
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42); weights = [G[u][v]['weight'] for u,v in G.edges()]; nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='#CBD5E1', width=[w*0.5 for w in weights], ax=ax_net, font_size=9, connectionstyle='arc3,rad=0.1'); nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax_net, font_size=8);
    processed_data['resource_network_adv'] = convert_fig_to_bytes(fig_net)

    return processed_data

# --- LAYOUT DA APLICAÇÃO ---
with st.sidebar:
    st.markdown("<h1 style='color: white; font-size: 24px; letter-spacing: -1px; text-align: center;'>Process Mining Dashboard</h1>", unsafe_allow_html=True)
    
    page = option_menu(
        menu_title=None, options=["Upload", "Resultados"], icons=["cloud-upload", "bar-chart-line"],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0F172A"},
            "icon": {"color": "#94A3B8", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#1E293B"},
            "nav-link-selected": {"background-color": "#3B82F6"},
        }
    )
    
    if st.session_state.analysis_run:
        st.markdown("---")
        st.markdown("<h3 style='color: white; font-size: 18px;'>Filtros Dinâmicos</h3>", unsafe_allow_html=True)
        df_projects_filter = st.session_state.processed_data.get('df_projects', pd.DataFrame())
        if not df_projects_filter.empty:
            project_types = ['Todos'] + sorted(df_projects_filter['project_type'].unique().tolist())
            st.session_state.selected_type = st.selectbox("Tipo de Projeto", project_types)
            min_date = df_projects_filter['start_date'].min().date()
            max_date = df_projects_filter['end_date'].max().date()
            st.session_state.date_range = st.date_input("Intervalo de Datas (Conclusão)", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if page == "Upload":
    st.header("1. Upload dos Ficheiros de Dados (.csv)")
    st.markdown("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    cols = st.columns(3)
    for i, name in enumerate(file_names):
        with cols[i % 3]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                st.session_state.dfs[name] = pd.read_csv(uploaded_file)
                st.success(f"`{name}.csv` carregado.")
    
    if all(st.session_state.dfs[name] is not None for name in file_names):
        st.info("Todos os ficheiros estão carregados. Pode iniciar a análise.")
        if st.button("🚀 Iniciar Análise Completa", use_container_width=True):
            with st.spinner("A processar dados e a descobrir os modelos... Isto pode demorar um momento."):
                st.session_state.processed_data = run_full_analysis(st.session_state.dfs)
                st.session_state.analysis_run = True
            st.success("✅ Análise concluída! Navegue para a página de 'Resultados'.")
            st.balloons()

elif page == "Resultados":
    st.header("Resultados da Análise de Processos")
    
    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Por favor, volte à página de 'Upload' e inicie o processo.")
    else:
        # --- LÓGICA DE FILTRAGEM ---
        df_projects_f = st.session_state.processed_data['df_projects'].copy()
        df_full_context_f = st.session_state.processed_data['df_full_context'].copy()

        if st.session_state.selected_type != 'Todos':
            df_projects_f = df_projects_f[df_projects_f['project_type'] == st.session_state.selected_type]
            df_full_context_f = df_full_context_f[df_full_context_f['project_type'] == st.session_state.selected_type]

        if len(st.session_state.date_range) == 2:
            start_date, end_date = pd.to_datetime(st.session_state.date_range[0]), pd.to_datetime(st.session_state.date_range[1])
            # CORREÇÃO FINAL AQUI:
            df_projects_f = df_projects_f[(df_projects_f['end_date'] >= start_date) & (df_projects_f['end_date'] <= end_date)]
            df_full_context_f = df_full_context_f[(df_full_context_f['end_date_project'] >= start_date) & (df_full_context_f['end_date_project'] <= end_date)]
        
        # --- KPIs DINÂMICOS ---
        total_projetos = len(df_projects_f) if not df_projects_f.empty else 0
        duracao_media = df_projects_f['actual_duration_days'].mean() if not df_projects_f.empty else 0
        custo_total = df_projects_f['total_actual_cost'].sum() if not df_projects_f.empty else 0
        desvio_prazo_medio = df_projects_f['days_diff'].mean() if not df_projects_f.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="kpi-card"><h3>Total de Projetos</h3><p>{total_projetos}</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="kpi-card"><h3>Duração Média</h3><p>{duracao_media:.1f}d</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="kpi-card"><h3>Custo Total</h3><p>€{custo_total:,.0f}</p></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="kpi-card"><h3>Desvio de Prazo</h3><p>{desvio_prazo_medio:.1f}d</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # --- ABAS E GRÁFICOS ---
        tab1, tab2 = st.tabs(["📊 Visão Geral e Performance", "⛏️ Descoberta de Processos"])

        with tab1:
            with st.expander("Análise de Performance e Casos", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    chart = alt.Chart(df_projects_f).mark_circle(size=100, opacity=0.7).encode(
                        x=alt.X('days_diff:Q', title='Desvio de Prazo (dias)'),
                        y=alt.Y('cost_diff:Q', title='Desvio de Custo (€)'),
                        color=alt.Color('project_type:N', title='Tipo de Projeto', scale=alt.Scale(range=PALETA_CORES)),
                        tooltip=['project_name', 'actual_duration_days', 'total_actual_cost']
                    ).properties(title="Matriz de Performance: Prazo vs. Custo").interactive()
                    st.altair_chart(chart, use_container_width=True)
                with c2:
                    chart = alt.Chart(df_projects_f).mark_boxplot(extent='min-max').encode(
                        x=alt.X('actual_duration_days:Q', title='Duração dos Projetos (dias)'),
                        color=alt.value(PALETA_CORES[0])
                    ).properties(title="Distribuição da Duração dos Projetos")
                    st.altair_chart(chart, use_container_width=True)

            with st.expander("Análise de Atividades e Gargalos"):
                c1, c2 = st.columns(2)
                with c1:
                    service_times = df_full_context_f.groupby('task_name')['hours_worked'].mean().reset_index().nlargest(10, 'hours_worked')
                    chart = alt.Chart(service_times).mark_bar().encode(
                        x=alt.X('hours_worked:Q', title='Horas Médias'),
                        y=alt.Y('task_name:N', title='Atividade', sort='-x'),
                        color=alt.value(PALETA_CORES[1]),
                        tooltip=['task_name', 'hours_worked']
                    ).properties(title='Atividades com Maior Tempo de Execução')
                    st.altair_chart(chart, use_container_width=True)
                with c2:
                    df_handoff = df_full_context_f.sort_values(['project_id_task', 'end_date_task'])
                    df_handoff['previous_end_date'] = df_handoff.groupby('project_id_task')['end_date_task'].shift(1)
                    df_handoff['handoff_hours'] = (df_handoff['start_date_task'] - df_handoff['previous_end_date']).dt.total_seconds() / 3600
                    df_handoff.loc[df_handoff['handoff_hours'] < 0, 'handoff_hours'] = 0
                    df_handoff['previous_task'] = df_handoff.groupby('project_id_task')['task_name'].shift(1)
                    handoff_stats = df_handoff.dropna(subset=['previous_task']).groupby(['previous_task', 'task_name'])['handoff_hours'].mean().reset_index().nlargest(10, 'handoff_hours')
                    
                    chart = alt.Chart(handoff_stats).mark_bar().encode(
                        x=alt.X('handoff_hours:Q', title='Horas de Espera'),
                        y=alt.Y('previous_task:N', title=None, sort='-x'),
                        color=alt.value(PALETA_CORES[3]),
                        tooltip=['previous_task', 'task_name', 'handoff_hours']
                    ).properties(title="Maiores Tempos de Espera (Handoffs)")
                    st.altair_chart(chart, use_container_width=True)
            
        with tab2:
            with st.expander("Descoberta de Modelos de Processo (Estático)", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<h4>Modelo (Inductive Miner)</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.processed_data['model_inductive_petrinet'], use_container_width=True)
                with c2:
                    st.markdown("<h4>Modelo (Heuristics Miner)</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.processed_data['model_heuristic_petrinet'], use_container_width=True)
            
            with st.expander("Análise de Performance e Recursos (Estático)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<h4>Heatmap de Performance</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.processed_data['performance_heatmap'], use_container_width=True)
                with c2:
                    st.markdown("<h4>Rede Social de Recursos</h4>", unsafe_allow_html=True)
                    st.image(st.session_state.processed_data['resource_network_adv'], use_container_width=True)

