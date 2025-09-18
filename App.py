import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import nbformat
from nbconvert import PythonExporter

# 1) Aponte para o teu .ipynb
notebook_path = "PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb"

# 2) Leia o notebook
nb_node = nbformat.read(notebook_path, as_version=4)

# 3) Converta-o para código Python
py_exporter = PythonExporter()
source, _ = py_exporter.from_notebook_node(nb_node)

# 4) Execute o código num namespace próprio
analysis_ns = {}
exec(source, analysis_ns)

# 5) Extraia as funções definidas no notebook
run_pre_mining  = analysis_ns["run_pre_mining"]
run_post_mining = analysis_ns["run_post_mining"]


#───────────────────────────────
# 1. CONFIGURAÇÃO & ESTILO
#───────────────────────────────
st.set_page_config(
    page_title="IT Resource Mgmt Dashboard",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for brand-like look (gradient header, modern font)
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, #root, .viewerBadge_container {
        font-family: 'Inter', sans-serif;
      }
      .css-18e3th9 {
        background: linear-gradient(90deg, #0D47A1, #1976D2);
      }
      .css-1v0mbdj e1fqkh3o4 {
        color: white;
      }
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0D47A1;
      }
      .sidebar .css-1d391kg {
        font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navegação")
page = st.sidebar.radio("", [
    "1. Carregar Dados",
    "2. Executar Análise",
    "3. Resultados"
])

#───────────────────────────────
# 2. SESSION STATE STORAGE
#───────────────────────────────
if "dfs" not in st.session_state:
    st.session_state.dfs = {}

#───────────────────────────────
# 3. PAGE: CARREGAR DADOS
#───────────────────────────────
if page == "1. Carregar Dados":
    st.header("📂 Carregar e Pré-visualizar Dados")
    uploaded = st.file_uploader(
        "Arraste os 5 CSVs ou selecione aqui:",
        type="csv",
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded:
        # map filenames to dfs
        required = {
            "projects.csv": "projects",
            "tasks.csv": "tasks",
            "resources.csv": "resources",
            "resource_allocations.csv": "allocs",
            "dependencies.csv": "deps"
        }
        missing = set(required) - {f.name for f in uploaded}
        if missing:
            st.error(f"Faltam estes ficheiros: {', '.join(missing)}")
        else:
            # read into session_state.dfs
            for f in uploaded:
                key = required[f.name]
                st.session_state.dfs[key] = pd.read_csv(f)
            st.success("📥 Ficheiros carregados com sucesso!")
            # preview heads
            for name, df in st.session_state.dfs.items():
                st.subheader(f"Preview: {name}")
                st.dataframe(df.head(), height=200)

# 4. PAGE: EXECUTAR ANÁLISE
if page == "2. Executar Análise":
    st.header("⚙️ Executar Pipeline de Análise")
    if len(st.session_state.dfs) < 5:
        st.warning("…")
    else:
        if st.button("▶️ Executar Análise Completa"):
            with st.spinner("🔄 A correr análises pré-mineração…"):
                run_pre_mining(
                    st.session_state.dfs["projects"],
                    st.session_state.dfs["tasks"],
                    st.session_state.dfs["resources"],
                    st.session_state.dfs["allocs"],
                    st.session_state.dfs["deps"],
                )
            with st.spinner("🔄 A correr análises pós-mineração…"):
                run_post_mining(
                    st.session_state.dfs["projects"],
                    st.session_state.dfs["tasks"],
                    st.session_state.dfs["resources"],
                    st.session_state.dfs["allocs"],
                    st.session_state.dfs["deps"],
                )
            st.success("✅ Análise concluída! Veja “Resultados”.")
            st.balloons()

#────────────────────────────────────
# 5. PAGE: RESULTADOS
#────────────────────────────────────
import os
from glob import glob

if page == "3. Resultados":
    st.header("🔎 Resultados")

    # define os diretórios onde seus scripts salvam os plots
    pre_dir  = "Process_Analysis_Dashboard/plots"
    post_dir = "Relatorio_Unificado_Analise_Processos/plots"

    def show_imgs(folder, pattern):
        """Encontra e exibe, em ordem alfabética, todos os PNGs que batem com o pattern."""
        for img in sorted(glob(os.path.join(folder, pattern))):
            st.image(img, use_column_width=True)

    # criamos duas tabs para Pré e Pós-Mineração
    pre_tab, post_tab = st.tabs(["📊 Pré-Mineração", "🧩 Pós-Mineração"])

    # ─────────────────────────────────
    # Aba 1: PRÉ-MINERAÇÃO
    # ─────────────────────────────────
    with pre_tab:
        # Secção 1
        with st.expander("Secção 1: Análises de Alto Nível e de Casos", expanded=True):
            show_imgs(pre_dir, "plot_01_*.png")
            show_imgs(pre_dir, "plot_02_*.png")

        # Secção 2
        with st.expander("Secção 2: Análises de Performance Detalhada"):
            for i in range(3, 7):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

        # Secção 3
        with st.expander("Secção 3: Análise de Atividades e Handoffs"):
            for i in range(7, 10):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")
            # o gráfico 10 pertence à secção 4

        # Secção 4
        with st.expander("Secção 4: Análise Organizacional (Recursos)"):
            for i in range(10, 15):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

        # Secção 5
        with st.expander("Secção 5: Análise de Variantes e Rework"):
            show_imgs(pre_dir, "plot_16_*.png")

        # Secção 6
        with st.expander("Secção 6: Análise Aprofundada (Causa-Raiz, Financeira e Benchmarking)"):
            for i in range(17, 27):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

    # ─────────────────────────────────
    # Aba 2: PÓS-MINERAÇÃO
    # ─────────────────────────────────
    with post_tab:
        # Secção 1
        with st.expander("Secção 1: Painel de KPIs e Análise de Alto Nível", expanded=True):
            show_imgs(post_dir, "01_performance_matrix.png")

        # Secção 2
        with st.expander("Secção 2: Descoberta e Avaliação de Modelos de Processo"):
            show_imgs(post_dir, "02_model_inductive_petrinet.png")
            show_imgs(post_dir, "03_metrics_inductive.png")
            show_imgs(post_dir, "04_model_heuristic_petrinet.png")
            show_imgs(post_dir, "05_metrics_heuristic.png")

        # Secção 3
        with st.expander("Secção 3: Análise de Performance e Tempo de Ciclo (Avançada)"):
            show_imgs(post_dir, "06_kpi_time_series.png")
            show_imgs(post_dir, "07_gantt_chart_all_projects.png")

        # Secção 4
        with st.expander("Secção 4: Análise de Gargalos"):
            show_imgs(post_dir, "08_bottleneck_ranking_adv.png")
            show_imgs(post_dir, "09_performance_heatmap.png")
            show_imgs(post_dir, "10_temporal_heatmap_fixed.png")

        # Secção 5
        with st.expander("Secção 5: Análise de Recursos"):
            show_imgs(post_dir, "11_resource_network_adv.png")
            show_imgs(post_dir, "12_skill_vs_performance_adv.png")

        # Secção 6
        with st.expander("Secção 6: Novas Análises e Visualizações"):
            show_imgs(post_dir, "13_variant_duration_plot.png")
            show_imgs(post_dir, "14_deviation_scatter_plot.png")
            show_imgs(post_dir, "15_conformance_over_time_plot.png")
            show_imgs(post_dir, "16_cost_per_day_time_series.png")
            show_imgs(post_dir, "17_cumulative_throughput_plot.png")
            show_imgs(post_dir, "18_custom_variants_sequence_plot.png")
            show_imgs(post_dir, "19_milestone_time_analysis_plot.png")
            show_imgs(post_dir, "20_waiting_time_matrix_plot.png")
            show_imgs(post_dir, "21_resource_efficiency_plot.png")
            show_imgs(post_dir, "22_avg_waiting_time_by_activity_plot.png")
            show_imgs(post_dir, "23_resource_network_bipartite.png")



