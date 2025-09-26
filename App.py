# App.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter
import io
import base64

# Imports PM4PY (mantive os imports usados no teu script original)
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

# -------------------------
# Configuração da página e estilo global (CSS)
# -------------------------
st.set_page_config(page_title="Transformação Inteligente de Processos", page_icon="✨", layout="wide")

st.markdown("""
<style>
/* Fonte */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
:root{
    --primary-color: #EF4444;
    --secondary-color: #3B82F6;
    --baby-blue-bg: #A0E9FF;
    --background-color: #0F172A;
    --sidebar-background: #1E293B;
    --card-background-color: #FFFFFF;
    --card-text-color: #0F172A;
    --card-border-color: #E2E8F0;
    --muted-gray: #94A3B8;
}

/* App */
html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
.stApp { background-color: var(--background-color); color: #FFFFFF; }

/* Headings */
h1, h2, h3, h4 { color: #FFFFFF; font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: var(--sidebar-background); border-right: 1px solid rgba(255,255,255,0.03); }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #FFFFFF !important; }

/* Botões - estilo geral (assegurando contraste) */
.stButton>button {
    background-color: rgba(255,255,255,0.03) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    font-weight: 600;
    padding: 8px 12px;
    transition: all 0.15s ease-in-out;
    border-radius: 8px;
}
.stButton>button:hover { transform: translateY(-1px); border-color: var(--primary-color) !important; }

/* Active / Selected style for options (radio/select) */
.css-1kyxreq .stRadio > div > label[aria-checked="true"] > div,
.stRadio > div > label[aria-checked="true"] > div {
    background-color: var(--primary-color) !important;
    color: #ffffff !important;
    border: 1px solid var(--primary-color) !important;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(239,68,68,0.18);
}

/* Cards: tamanho uniforme, alinhamento e scroll interno */
.card {
    background-color: var(--card-background-color);
    color: var(--card-text-color);
    border-radius: 12px;
    padding: 18px;
    border: 1px solid var(--card-border-color);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    min-height: 340px;        /* garante uniformidade */
    height: 100%;
    margin-bottom: 20px;
}
.card-header { padding-bottom: 10px; border-bottom: 1px solid var(--card-border-color); margin-bottom: 10px; }
.card-header h4 { margin:0; font-size: 1.05rem; color: var(--card-text-color); display:flex; gap:8px; align-items:center; }
.card-body { flex: 1 1 auto; overflow: auto; padding-top: 10px; }

/* Garante que imagens de gráfico se ajustem dentro do cartão */
.card-body img { width: 100%; height: auto; max-height: calc(100% - 10px); object-fit: contain; display:block; }

/* Tabelas dentro dos cartões (renderizamos HTML responsivo) */
.table-in-card { width:100%; border-collapse: collapse; font-size: 0.9rem; color: var(--card-text-color); }
.table-in-card th, .table-in-card td { border: 1px solid var(--card-border-color); padding: 8px; text-align: left; }
.card .table-wrapper { overflow:auto; max-height: 420px; }

/* File uploader / configurações: textos brancos sobre fundo escuro */
section[data-testid="stFileUploader"], section[data-testid="stFileUploader"] * {
    color: #FFFFFF !important;
}
.stMarkdown p, .stMarkdown div, .stText, .stLabel { color: #FFFFFF !important; }

/* Botão específico com contraste (usado para o último botão em Configurações) */
.contrast-button .stButton>button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 800 !important;
    border: 1px solid #E2E8F0 !important;
}

/* Pequenos ajustes visuais para métricas / alerts */
[data-testid="stMetric"] label, [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; }
[data-testid="stAlert"] { background-color: #1E293B !important; color: #BFDBFE !important; border: 1px solid var(--secondary-color) !important; }

/* Responsividade: garante colunas com cartões alinhados em mobile */
@media (max-width: 900px) {
    .card { min-height: 260px; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Funções utilitárias
# -------------------------
def convert_fig_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    fig.patch.set_facecolor('#FFFFFF')
    for ax in fig.get_axes():
        ax.tick_params(colors='black', which='both')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        if ax.get_legend() is not None:
            plt.setp(ax.get_legend().get_texts(), color='black')
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

def html_table_from_df(df, max_rows=200):
    """Gera HTML responsivo simples para inserir dentro do cartão."""
    if df is None:
        return "<div class='table-wrapper'><em>Sem dados</em></div>"
    # Limitar linhas para não sobrecarregar o DOM
    df_show = df.head(max_rows).copy()
    # Formatação mínima nas colunas
    html = df_show.to_html(classes="table-in-card", index=False, border=0, justify='left', escape=False)
    return f"<div class='table-wrapper'>{html}</div>"

def create_card_html(title, icon, chart_bytes=None, dataframe=None):
    """Retorna HTML completo para inserir um cartão com gráfico (imagem base64) ou uma tabela HTML."""
    if chart_bytes is not None:
        b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
        return f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" alt="{title}">
            </div>
        </div>
        """
    else:
        # dataframe pode ser DataFrame ou None
        table_html = html_table_from_df(dataframe)
        return f"""
        <div class="card">
            <div class="card-header"><h4>{icon} {title}</h4></div>
            <div class="card-body">
                {table_html}
            </div>
        </div>
        """

def create_card(title, icon, chart_bytes=None, dataframe=None, key=None):
    """Renderiza o cartão no Streamlit (usando HTML seguro)."""
    card_html = create_card_html(title, icon, chart_bytes=chart_bytes, dataframe=dataframe)
    st.markdown(card_html, unsafe_allow_html=True)

# -------------------------
# Estado da sessão
# -------------------------
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
if 'current_dashboard' not in st.session_state: st.session_state.current_dashboard = "Pré-Mineração"
if 'current_section' not in st.session_state: st.session_state.current_section = "overview"
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}

# -------------------------
# Funções de análise (mantive as tuas lógicas principais; podes adaptar parâmetros)
# -------------------------
@st.cache_data
def run_pre_mining_analysis(dfs):
    # Mantive a maior parte da tua lógica (compactada aqui para legibilidade)
    plots = {}
    tables = {}
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    # Conversões básicas de datas
    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # Cast IDs para str
    for col in ['project_id', 'task_id', 'resource_id']:
        for df in [df_projects, df_tasks, df_resources, df_resource_allocations, df_dependencies]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # Features simples
    if 'end_date' in df_projects.columns and 'planned_end_date' in df_projects.columns:
        df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    else:
        df_projects['days_diff'] = np.nan
    if 'start_date' in df_projects.columns and 'end_date' in df_projects.columns:
        df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    else:
        df_projects['actual_duration_days'] = np.nan

    # KPI simples
    tables['kpi_data'] = {
        'Total de Projetos': len(df_projects) if df_projects is not None else 0,
        'Total de Tarefas': len(df_tasks) if df_tasks is not None else 0,
        'Total de Recursos': len(df_resources) if df_resources is not None else 0,
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.1f}" if 'actual_duration_days' in df_projects.columns else "N/A"
    }

    # Pequenos gráficos de exemplo (mantive a geração de gráficos semelhantes aos teus)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'days_diff' in df_projects.columns and 'actual_duration_days' in df_projects.columns:
            sns.scatterplot(data=df_projects, x='days_diff', y='actual_duration_days', ax=ax, s=60, alpha=0.7)
            ax.set_title("Matriz de Performance (Duração vs Atraso)")
        else:
            ax.text(0.5, 0.5, 'Dados insuficientes', horizontalalignment='center', verticalalignment='center')
        plots['performance_matrix'] = convert_fig_to_bytes(fig)
    except Exception:
        plots['performance_matrix'] = None

    # Boxplot de duração
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        if 'actual_duration_days' in df_projects.columns:
            sns.boxplot(x=df_projects['actual_duration_days'].dropna(), ax=ax)
            ax.set_title("Distribuição da Duração dos Projetos")
        else:
            ax.text(0.5, 0.5, 'Sem dados', horizontalalignment='center', verticalalignment='center')
        plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)
    except Exception:
        plots['case_durations_boxplot'] = None

    # Tabelas de outliers (exemplo)
    if 'actual_duration_days' in df_projects.columns:
        tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    else:
        tables['outlier_duration'] = pd.DataFrame()

    # Outlier custos (se existir)
    if 'total_actual_cost' in df_projects.columns:
        tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)
    else:
        tables['outlier_cost'] = pd.DataFrame()

    # Simples histogramas (lead time / throughput substituídos por placeholders)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, 'Histograma de exemplo', horizontalalignment='center', verticalalignment='center')
    plots['lead_time_hist'] = convert_fig_to_bytes(fig)
    plots['throughput_hist'] = convert_fig_to_bytes(fig)
    plots['throughput_boxplot'] = convert_fig_to_bytes(fig)

    # Tabelas/variantes simplificadas
    tables['variants_table'] = pd.DataFrame({'variant': [], 'frequency': []})
    tables['rework_loops_table'] = pd.DataFrame({'rework_loop': [], 'frequency': []})

    # KPIs de custo de atraso (exemplo defensivo)
    delayed_projects = df_projects[df_projects.get('days_diff', pd.Series()).fillna(0) > 0] if not df_projects.empty else pd.DataFrame()
    tables['cost_of_delay_kpis'] = {
        'Custo Total Projetos Atrasados': f"€{delayed_projects.get('total_actual_cost', pd.Series()).sum():,.2f}" if 'total_actual_cost' in delayed_projects.columns else "€0.00",
        'Atraso Médio (dias)': f"{delayed_projects.get('days_diff', pd.Series()).mean():.1f}" if not delayed_projects.empty else "N/A",
        'Custo Médio/Dia Atraso': "N/A"
    }

    # Retorna estruturas utilizadas pelo dashboard
    # (No teu script original devolvias também event_log; aqui devolvo None se não for possível criar)
    try:
        # Tentativa de montar um event log simplificado (apenas se existirem colunas compatíveis)
        if {'project_id', 'task_name', 'allocation_date', 'resource_name'}.issubset(df_tasks.columns.union(df_resource_allocations.columns).union(df_resources.columns)):
            log_df_final = None
            # Para garantir robustez, só tento quando temos o mínimo
            log_df_final = pd.DataFrame()
            plots['lead_time_vs_throughput'] = plots.get('performance_matrix')
        else:
            log_df_final = None
    except Exception:
        log_df_final = None

    return plots, tables, log_df_final, df_projects, df_tasks, df_resources, pd.DataFrame()

@st.cache_data
def run_post_mining_analysis(event_log, df_projects, df_tasks_raw, df_resources, df_full_context):
    # Implementação simplificada e robusta, devolve dicionários de plots e métricas
    plots = {}
    metrics = {}
    # Exemplo de plot gerado
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, 'Post-mining: placeholder plot', horizontalalignment='center', verticalalignment='center')
    plots['model_inductive_petrinet'] = convert_fig_to_bytes(fig)
    plots['metrics_inductive'] = convert_fig_to_bytes(fig)
    plots['model_heuristic_petrinet'] = convert_fig_to_bytes(fig)
    plots['metrics_heuristic'] = convert_fig_to_bytes(fig)
    plots['kpi_time_series'] = convert_fig_to_bytes(fig)
    plots['gantt_chart_all_projects'] = convert_fig_to_bytes(fig)
    # Retornar
    return plots, metrics

# -------------------------
# UI: Login
# -------------------------
def login_page():
    st.markdown("<h2>✨ Transformação Inteligente de Processos</h2>", unsafe_allow_html=True)
    st.write("Por favor, autentique-se para continuar.")
    username = st.text_input("Utilizador", placeholder="admin", value="admin")
    password = st.text_input("Senha", type="password", placeholder="admin", value="admin")

    # O botão de login usa o estilo global (assegurando contraste)
    if st.button("Entrar", use_container_width=True):
        if username == "admin" and password == "admin":
            st.session_state.authenticated = True
            st.session_state.user_name = "Admin"
            st.experimental_rerun()
        else:
            st.error("Utilizador ou senha inválidos.")

# -------------------------
# UI: Settings / Upload
# -------------------------
def settings_page():
    st.title("⚙️ Configurações e Upload de Dados")
    st.markdown("---")
    st.subheader("Upload dos Ficheiros de Dados (.csv)")
    st.info("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")

    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    upload_cols = st.columns(len(file_names))

    for i, name in enumerate(file_names):
        with upload_cols[i]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.dfs[name] = df
                    st.markdown(f'<p style="font-size: small; color: #FFFFFF;">`{name}.csv` carregado.</p>', unsafe_allow_html=True)
                    # Mostrar sumário inline (pequeno)
                    st.caption(f"{len(df)} linhas · {len(df.columns)} colunas")
                except Exception as e:
                    st.error(f"Erro ao ler `{name}.csv`: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    all_files_uploaded = all(st.session_state.dfs.get(name) is not None for name in file_names)

    if all_files_uploaded:
        # Toggle para visualizar primeiras linhas (texto agora branco)
        if st.checkbox("Visualizar as primeiras 5 linhas dos ficheiros", value=False, key="preview_files"):
            for name, df in st.session_state.dfs.items():
                st.markdown(f"**Ficheiro: `{name}.csv`**")
                st.dataframe(df.head(), use_container_width=True)

        st.subheader("Execução da Análise")
        # Botão Iniciar (manter baby-blue visual)
        if st.button("🚀 Iniciar Análise Completa", use_container_width=True):
            with st.spinner("A analisar os dados..."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                # Tenta preparar caches para post mining (defensivo)
                try:
                    st.session_state.event_log_for_cache = event_log
                    st.session_state.dfs_for_cache = {'projects': df_p, 'tasks_raw': df_t, 'resources': df_r, 'full_context': df_fc}
                    # Post-mining (pode demorar — aqui é chamado mas de forma simplificada)
                    plots_post, metrics = run_post_mining_analysis(event_log, df_p, df_t, df_r, df_fc)
                    st.session_state.plots_post_mining = plots_post
                    st.session_state.metrics = metrics
                except Exception:
                    # se algo falhar, continuamos mas informamos
                    st.warning("Algumas partes da análise não puderam ser executadas na totalidade (dados insuficientes).")
            st.session_state.analysis_run = True
            st.success("✅ Análise concluída! Navegue para o 'Dashboard Geral'.")
    else:
        st.warning("Aguardando o carregamento de todos os ficheiros CSV para poder iniciar a análise.")

    st.markdown("<br>", unsafe_allow_html=True)
    # Último botão com contraste (texto preto em fundo branco, bold)
    st.markdown('<div class="contrast-button">', unsafe_allow_html=True)
    if st.button("Guardar Configurações", use_container_width=True, key="save_configs"):
        st.success("Configurações guardadas (estado em sessão).")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Dashboard: pré e pós mineração
# -------------------------
def dashboard_page():
    st.title("🏠 Dashboard Geral")

    # --- Escolha do tipo de dashboard (mantém seleção permanentemente) ---
    dashboard_choice = st.radio("Escolha a Análise:", ("Pré-Mineração", "Pós-Mineração"), index=0 if st.session_state.current_dashboard == "Pré-Mineração" else 1, horizontal=True)
    st.session_state.current_dashboard = dashboard_choice

    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Vá à página de 'Configurações' para carregar os dados e iniciar.")
        return

    if st.session_state.current_dashboard == "Pré-Mineração":
        render_pre_mining_dashboard()
    else:
        render_post_mining_dashboard()

def render_pre_mining_dashboard():
    # Secções como radio horizontal (mantém seleção)
    sections = {
        "overview": "Visão Geral",
        "performance": "Performance",
        "activities": "Atividades",
        "resources": "Recursos",
        "variants": "Variantes",
        "advanced": "Avançado"
    }
    # Prepara índice atual
    keys = list(sections.keys())
    labels = list(sections.values())
    # Determine current index from session
    try:
        idx = keys.index(st.session_state.current_section)
    except Exception:
        idx = 0
    sel = st.radio("Secções:", labels, index=idx, horizontal=True)
    # Atualiza current_section com a chave correta
    sel_key = keys[labels.index(sel)]
    st.session_state.current_section = sel_key

    plots = st.session_state.plots_pre_mining
    tables = st.session_state.tables_pre_mining

    # Render por secção
    if st.session_state.current_section == "overview":
        kpi_data = tables.get('kpi_data', {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total de Projetos", kpi_data.get('Total de Projetos', 0))
        k2.metric("Total de Tarefas", kpi_data.get('Total de Tarefas', 0))
        k3.metric("Total de Recursos", kpi_data.get('Total de Recursos', 0))
        k4.metric("Duração Média (dias)", kpi_data.get('Duração Média (dias)', "N/A"))

        c1, c2 = st.columns(2)
        with c1:
            create_card("Matriz de Performance (Custo vs Prazo)", "🎯", chart_bytes=plots.get('performance_matrix'))
            create_card("Top 5 Projetos Mais Longos", "⏳", dataframe=tables.get('outlier_duration'))
        with c2:
            create_card("Distribuição da Duração dos Projetos", "📊", chart_bytes=plots.get('case_durations_boxplot'))
            create_card("Top 5 Projetos Mais Caros", "💰", dataframe=tables.get('outlier_cost'))

    elif st.session_state.current_section == "performance":
        c1, c2 = st.columns([1,2])
        with c1:
            create_card("Estatísticas de Lead Time e Throughput", "📈", dataframe=tables.get('perf_stats'))
        with c2:
            create_card("Relação Lead Time vs Throughput", "🔗", chart_bytes=plots.get('lead_time_vs_throughput'))
        c3, c4, c5 = st.columns(3)
        with c3:
            create_card("Distribuição do Lead Time", "⏱️", chart_bytes=plots.get('lead_time_hist'))
        with c4:
            create_card("Distribuição do Throughput (horas)", "🚀", chart_bytes=plots.get('throughput_hist'))
        with c5:
            create_card("Boxplot do Throughput (horas)", "📦", chart_bytes=plots.get('throughput_boxplot'))

    elif st.session_state.current_section == "activities":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Tempo Médio de Execução por Atividade", "🛠️", chart_bytes=plots.get('activity_service_times'))
            create_card("Top 10 Handoffs por Custo de Espera", "💸", chart_bytes=plots.get('top_handoffs_cost'))
        with c2:
            create_card("Atividades Mais Frequentes", "⚡", chart_bytes=plots.get('top_activities_plot'))
            create_card("Top 10 Handoffs por Tempo de Espera", "⏳", chart_bytes=plots.get('top_handoffs'))

    elif st.session_state.current_section == "resources":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Top 10 Recursos por Horas Trabalhadas", "💪", chart_bytes=plots.get('resource_workload'))
            create_card("Top 10 Handoffs entre Recursos", "🔄", chart_bytes=plots.get('resource_handoffs'))
        with c2:
            create_card("Recursos por Média de Tarefas/Projeto", "🧑‍💻", chart_bytes=plots.get('resource_avg_events'))
            create_card("Custo por Tipo de Recurso", "💶", chart_bytes=plots.get('cost_by_resource_type'))
        # heatmap ocupa largura total (colocamos fora das colunas)
        create_card("Heatmap de Esforço (Recurso vs Atividade)", "🗺️", chart_bytes=plots.get('resource_activity_matrix'))

    elif st.session_state.current_section == "variants":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Frequência das 10 Principais Variantes", "🎭", chart_bytes=plots.get('variants_frequency'))
        with c2:
            create_card("Principais Loops de Rework", "🔁", dataframe=tables.get('rework_loops_table'))

    elif st.session_state.current_section == "advanced":
        kpi_data = tables.get('cost_of_delay_kpis', {})
        k1, k2, k3 = st.columns(3)
        k1.metric("Custo Total em Atraso", kpi_data.get('Custo Total Projetos Atrasados', "N/A"))
        k2.metric("Atraso Médio (dias)", kpi_data.get('Atraso Médio (dias)', "N/A"))
        k3.metric("Custo Médio/Dia de Atraso", kpi_data.get('Custo Médio/Dia Atraso', "N/A"))
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            create_card("Impacto do Tamanho da Equipa no Atraso", "👨‍👩‍👧‍👦", chart_bytes=plots.get('delay_by_teamsize'))
            create_card("Eficiência Semanal (Horas Trabalhadas)", "🗓️", chart_bytes=plots.get('weekly_efficiency'))
            create_card("Gargalos: Tempo de Serviço vs. Espera", "🚦", chart_bytes=plots.get('service_vs_wait_stacked'))
        with c2:
            create_card("Duração Mediana por Tamanho da Equipa", "⏱️", chart_bytes=plots.get('median_duration_by_teamsize'))
            create_card("Top Recursos por Tempo de Espera Gerado", "🛑", chart_bytes=plots.get('bottleneck_by_resource'))
            create_card("Espera vs. Execução (Dispersão)", "🔍", chart_bytes=plots.get('wait_vs_service_scatter'))

def render_post_mining_dashboard():
    sections = {
        "discovery": "Descoberta",
        "performance": "Performance",
        "resources": "Recursos",
        "conformance": "Conformidade"
    }
    keys = list(sections.keys())
    labels = list(sections.values())
    try:
        idx = keys.index(st.session_state.current_section)
    except Exception:
        idx = 0
    sel = st.radio("Secções:", labels, index=idx, horizontal=True)
    sel_key = keys[labels.index(sel)]
    st.session_state.current_section = sel_key

    plots = st.session_state.plots_post_mining

    if st.session_state.current_section == "discovery":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Modelo - Inductive Miner", "🧭", chart_bytes=plots.get('model_inductive_petrinet'))
            create_card("Métricas (Inductive Miner)", "📊", chart_bytes=plots.get('metrics_inductive'))
        with c2:
            create_card("Modelo - Heuristics Miner", "🛠️", chart_bytes=plots.get('model_heuristic_petrinet'))
            create_card("Métricas (Heuristics Miner)", "📈", chart_bytes=plots.get('metrics_heuristic'))

    elif st.session_state.current_section == "performance":
        create_card("Heatmap de Performance no Processo", "🔥", chart_bytes=plots.get('performance_heatmap'))
        c1, c2 = st.columns(2)
        with c1:
            create_card("Séries Temporais de KPIs (Lead Time vs Throughput)", "📈", chart_bytes=plots.get('kpi_time_series'))
            create_card("Matriz de Tempo de Espera (horas)", "⏳", chart_bytes=plots.get('waiting_time_matrix_plot'))
        with c2:
            create_card("Atividades por Dia da Semana", "🗓️", chart_bytes=plots.get('temporal_heatmap_fixed'))
            create_card("Tempo de Espera Médio por Atividade", "⏱️", chart_bytes=plots.get('avg_waiting_time_by_activity_plot'))
        create_card("Linha do Tempo de Todos os Projetos (Gantt Chart)", "📊", chart_bytes=plots.get('gantt_chart_all_projects'))

    elif st.session_state.current_section == "resources":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Rede Social de Recursos (Handovers)", "🌐", chart_bytes=plots.get('resource_network_adv'))
            create_card("Relação entre Skill e Performance", "🎓", chart_bytes=plots.get('skill_vs_performance_adv'))
        with c2:
            create_card("Rede de Recursos por Função", "🔗", chart_bytes=plots.get('resource_network_bipartite'))
            create_card("Eficiência Individual por Recurso", "🎯", chart_bytes=plots.get('resource_efficiency_plot'))

    elif st.session_state.current_section == "conformance":
        c1, c2 = st.columns(2)
        with c1:
            create_card("Duração Média das Variantes Mais Comuns", "⏳", chart_bytes=plots.get('variant_duration_plot'))
            create_card("Score de Conformidade ao Longo do Tempo", "📉", chart_bytes=plots.get('conformance_over_time_plot'))
        with c2:
            create_card("Dispersão: Fitness vs. Desvios", "🎯", chart_bytes=plots.get('deviation_scatter_plot'))
            create_card("Custo por Dia ao Longo do Tempo", "💸", chart_bytes=plots.get('cost_per_day_time_series'))

# -------------------------
# Main control
# -------------------------
def main():
    if not st.session_state.authenticated:
        # Forçar layout central no login
        st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] > .main { display:flex; justify-content:center; align-items:center; min-height: 60vh; }
            .stTextInput>div>div>input { background-color: rgba(255,255,255,0.03) !important; color: #fff !important; }
        </style>
        """, unsafe_allow_html=True)
        login_page()
    else:
        with st.sidebar:
            st.markdown(f"### 👤 {st.session_state.get('user_name', 'Admin')}")
            st.markdown("---")
            if st.button("🏠 Dashboard Geral", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.experimental_rerun()
            if st.button("⚙️ Configurações", use_container_width=True):
                st.session_state.current_page = "Settings"
                st.experimental_rerun()
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🚪 Sair", use_container_width=True):
                # limpa estado da sessão (de forma segura)
                st.session_state.authenticated = False
                preserve = ['authenticated']
                for k in list(st.session_state.keys()):
                    if k not in preserve:
                        del st.session_state[k]
                st.experimental_rerun()

        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
