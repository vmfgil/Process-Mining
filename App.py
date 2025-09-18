import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import nbformat
from nbconvert import PythonExporter

# ──────────────────────────────────────────────────────────────────────────
# 0. CARREGAR E LIMPAR O NOTEBOOK COMO CÓDIGO PYTHON
# ──────────────────────────────────────────────────────────────────────────
notebook_path = "PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb"

# 1) Leia o .ipynb
nb_node = nbformat.read(notebook_path, as_version=4)

# 2) Converta para fonte Python
py_exporter = PythonExporter()
source, _ = py_exporter.from_notebook_node(nb_node)

# 3) Limpe linhas de shell/magics e imports de Colab
clean_lines = []
for ln in source.splitlines():
    stripped = ln.lstrip()
    if stripped.startswith("!") or stripped.startswith("%") or "google.colab" in ln:
        continue
    clean_lines.append(ln)
clean_source = "\n".join(clean_lines)

# 4) Execute o código limpo num namespace próprio
analysis_ns = {}
exec(clean_source, analysis_ns)

# 5) Extraia as funções definidas no notebook
run_pre_mining  = analysis_ns["run_pre_mining"]
run_post_mining = analysis_ns["run_post_mining"]


# ──────────────────────────────────────────────────────────────────────────
# 1. CONFIGURAÇÃO & ESTILO STREAMLIT
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Resource Mgmt Dashboard",
    layout="wide",
    page_icon="📊"
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, #root, .viewerBadge_container { font-family: 'Inter', sans-serif; }
      .css-18e3th9 { background: linear-gradient(90deg, #0D47A1, #1976D2); }
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #0D47A1; }
      .sidebar .css-1d391kg { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────
# 2. SIDEBAR E ESTADO
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("Navegação")
page = st.sidebar.radio("", [
    "1. Carregar Dados",
    "2. Executar Análise",
    "3. Resultados"
])

if "dfs" not in st.session_state:
    st.session_state.dfs = {}


# ──────────────────────────────────────────────────────────────────────────
# 3. PAGE: CARREGAR DADOS
# ──────────────────────────────────────────────────────────────────────────
if page == "1. Carregar Dados":
    st.header("📂 Carregar e Pré-visualizar Dados")
    uploaded = st.file_uploader(
        "Arraste os 5 CSVs ou selecione aqui:",
        type="csv",
        accept_multiple_files=True
    )
    if uploaded:
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
            for f in uploaded:
                key = required[f.name]
                st.session_state.dfs[key] = pd.read_csv(f)
            st.success("📥 Ficheiros carregados com sucesso!")
            for name, df in st.session_state.dfs.items():
                st.subheader(f"Preview: {name}")
                st.dataframe(df.head(), height=200)


# ──────────────────────────────────────────────────────────────────────────
# 4. PAGE: EXECUTAR ANÁLISE
# ──────────────────────────────────────────────────────────────────────────
if page == "2. Executar Análise":
    st.header("⚙️ Executar Pipeline de Análise")
    if len(st.session_state.dfs) < 5:
        st.warning("Antes de executar, carregue todos os 5 ficheiros em “Carregar Dados”.")
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
            st.success("✅ Análise concluída! Vá a “Resultados”.")
            st.balloons()


# ──────────────────────────────────────────────────────────────────────────
# 5. PAGE: RESULTADOS
# ──────────────────────────────────────────────────────────────────────────
if page == "3. Resultados":
    st.header("🔎 Resultados")

    pre_dir  = "Process_Analysis_Dashboard/plots"
    post_dir = "Relatorio_Unificado_Analise_Processos/plots"

    def show_imgs(folder, pattern):
        for img in sorted(glob.glob(os.path.join(folder, pattern))):
            st.image(img, use_column_width=True)

    pre_tab, post_tab = st.tabs(["📊 Pré-Mineração", "🧩 Pós-Mineração"])

    # ── Aba 1: Pré-Mineração ─────────────────────────
    with pre_tab:
        with st.expander("Secção 1: Análises de Alto Nível e de Casos", expanded=True):
            show_imgs(pre_dir, "plot_01_*.png")
            show_imgs(pre_dir, "plot_02_*.png")

        with st.expander("Secção 2: Análises de Performance Detalhada"):
            for i in range(3, 7):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

        with st.expander("Secção 3: Análise de Atividades e Handoffs"):
            for i in range(7, 10):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

        with st.expander("Secção 4: Análise Organizacional (Recursos)"):
            for i in range(10, 15):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

        with st.expander("Secção 5: Análise de Variantes e Rework"):
            show_imgs(pre_dir, "plot_16_*.png")

        with st.expander("Secção 6: Análise Aprofundada"):
            for i in range(17, 27):
                show_imgs(pre_dir, f"plot_{i:02d}_*.png")

    # ── Aba 2: Pós-Mineração ─────────────────────────
    with post_tab:
        with st.expander("Secção 1: KPIs e Análise de Alto Nível", expanded=True):
            show_imgs(post_dir, "01_performance_matrix.png")

        with st.expander("Secção 2: Descoberta e Avaliação de Modelos"):
            for fn in [
                "02_model_inductive_petrinet.png",
                "03_metrics_inductive.png",
                "04_model_heuristic_petrinet.png",
                "05_metrics_heuristic.png"
            ]:
                show_imgs(post_dir, fn)

        with st.expander("Secção 3: Tempo de Ciclo Avançado"):
            show_imgs(post_dir, "06_kpi_time_series.png")
            show_imgs(post_dir, "07_gantt_chart_all_projects.png")

        with st.expander("Secção 4: Análise de Gargalos"):
            for fn in [
                "08_bottleneck_ranking_adv.png",
                "09_performance_heatmap.png",
                "10_temporal_heatmap_fixed.png"
            ]:
                show_imgs(post_dir, fn)

        with st.expander("Secção 5: Análise de Recursos"):
            show_imgs(post_dir, "11_resource_network_adv.png")
            show_imgs(post_dir, "12_skill_vs_performance_adv.png")

        with st.expander("Secção 6: Novas Visualizações"):
            for fn in sorted(os.listdir(post_dir)):
                if fn.startswith(("13_variant", "14_deviation", "15_conformance",
                                  "16_cost_per_day", "17_cumulative_throughput",
                                  "18_custom_variants", "19_milestone_time",
                                  "20_waiting_time_matrix", "21_resource_efficiency",
                                  "22_avg_waiting_time_by_activity",
                                  "23_resource_network_bipartite")) and fn.endswith(".png"):
                    show_imgs(post_dir, fn)
