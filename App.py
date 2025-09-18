# streamlit_app.py
# Streamlit app para executar o notebook e mostrar os artefactos gerados.
# Coloca este ficheiro na raíz do repositório (junto do teu notebook).
# Nome do notebook esperado (ajusta se necessário):
NOTEBOOK_FILENAME = "PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb"

import streamlit as st
import tempfile, os, uuid, json, shutil, nbformat, re
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

st.set_page_config(page_title="Analytica — App", layout="wide")

# Helper UI
st.sidebar.markdown("## Analytica — App")
st.sidebar.write("Carrega os ficheiros com os nomes exatos que o notebook espera:")
st.sidebar.markdown("""
- **projects.csv**  
- **tasks.csv**  
- **resources.csv**  
- **resource_allocations.csv**  
- **dependencies.csv**
""")
st.sidebar.caption("Todos os dados processados no servidor Streamlit (ou local).")

st.title("Analytica — Upload • Execução • Resultados")
st.write("Carrega os 5 ficheiros, clica em **Executar Análise** e espera a confirmação. Quando terminar encontrarás os artefactos gerados pelo notebook.")

# Upload widgets
col1, col2 = st.columns(2)
with col1:
    f_projects = st.file_uploader("projects.csv", type=["csv"], key="projects")
    f_tasks = st.file_uploader("tasks.csv", type=["csv"], key="tasks")
with col2:
    f_resources = st.file_uploader("resources.csv", type=["csv"], key="resources")
    f_alloc = st.file_uploader("resource_allocations.csv", type=["csv"], key="resource_allocations")
    f_deps = st.file_uploader("dependencies.csv", type=["csv"], key="dependencies")

# Small preview function
def preview_csv_bytes(b):
    s = b.decode("utf-8", errors="ignore")
    lines = s.splitlines()[:6]
    return "<br>".join([st.markdown(l) for l in lines])

st.markdown("---")
run_col, info_col = st.columns([1,3])
with run_col:
    run_button = st.button("▶️ Executar Análise")
with info_col:
    st.info("A execução pode demorar. Vais receber notificação quando o processo terminar.")

# Inject code that will be placed at top of notebook to capture artefacts
INJECT_CODE = r'''
import os, json, matplotlib
report_dir = os.path.abspath('report_output')
plots_dir = os.path.join(report_dir, 'plots')
models_dir = os.path.join(report_dir, 'models')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
manifest = []
manifest_path = os.path.join(report_dir, 'manifest.json')
plot_counter = 0

def _sanitize_filename(s):
    import re
    s = re.sub(r'[^0-9A-Za-z_\\-\\s]', '_', str(s))
    s = s.strip().replace(' ', '_')
    return s[:200]

def save_artefact_and_add_to_html_report(artefact, title, filename_base, artefact_type='plot', content_text=''):
    global plot_counter, manifest
    plot_counter += 1
    filename_base_counted = f"{plot_counter:02d}_" + _sanitize_filename(filename_base)
    directory = models_dir if artefact_type == 'model' else plots_dir
    os.makedirs(directory, exist_ok=True)
    png_path = os.path.join(directory, f"{filename_base_counted}.png")
    try:
        if hasattr(artefact, 'savefig'):
            artefact.savefig(png_path, bbox_inches='tight')
        elif isinstance(artefact, matplotlib.figure.Figure):
            artefact.savefig(png_path, bbox_inches='tight')
        else:
            import matplotlib.pyplot as plt
            plt.savefig(png_path, bbox_inches='tight')
        saved = png_path
    except Exception as e:
        txt_path = os.path.join(directory, f"{filename_base_counted}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(str(content_text) or str(title) or str(e))
        saved = txt_path
    rel = os.path.relpath(saved, report_dir)
    manifest.append({'name': title, 'file': rel, 'type': artefact_type})
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False)

def save_plot_and_add_to_html_report(plot_obj, title, filename_base, artefact_type='plot', content_text=''):
    global plot_counter, manifest
    plot_counter += 1
    filename_base_counted = f"{plot_counter:02d}_" + _sanitize_filename(filename_base)
    directory = plots_dir
    os.makedirs(directory, exist_ok=True)
    png_path = os.path.join(directory, f"{filename_base_counted}.png")
    try:
        if hasattr(plot_obj, 'figure'):
            fig = plot_obj.figure
            fig.savefig(png_path, bbox_inches='tight')
        elif hasattr(plot_obj, 'savefig'):
            plot_obj.savefig(png_path, bbox_inches='tight')
        else:
            import matplotlib.pyplot as plt
            plt.savefig(png_path, bbox_inches='tight')
        saved = png_path
    except Exception as e:
        txt_path = os.path.join(directory, f"{filename_base_counted}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(str(content_text) or str(title) or str(e))
        saved = txt_path
    rel = os.path.relpath(saved, report_dir)
    manifest.append({'name': title, 'file': rel, 'type': artefact_type})
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False)
'''

def save_uploaded_files_to_dir(run_dir):
    # expects the five file objects; returns dict of saved names
    mapping = {}
    mapping['projects'] = None
    mapping['tasks'] = None
    mapping['resources'] = None
    mapping['resource_allocations'] = None
    mapping['dependencies'] = None
    files = {
        'projects': f_projects,
        'tasks': f_tasks,
        'resources': f_resources,
        'resource_allocations': f_alloc,
        'dependencies': f_deps
    }
    for key, f in files.items():
        if f is None:
            continue
        saved_path = os.path.join(run_dir, f.name)
        with open(saved_path, "wb") as out:
            out.write(f.getbuffer())
        mapping[key] = saved_path
    return mapping

# Run logic
if run_button:
    # Validate uploads
    if not (f_projects and f_tasks and f_resources and f_alloc and f_deps):
        st.error("Por favor carrega os 5 ficheiros obrigatórios antes de executar.")
    else:
        run_id = uuid.uuid4().hex
        st.info(f"Inicio da análise — run id: {run_id}")
        tmp_root = tempfile.mkdtemp(prefix=f"run_{run_id}_")
        try:
            # 1) salvar ficheiros
            saved_map = save_uploaded_files_to_dir(tmp_root)

            # 2) copy notebook into run dir
            nb_src = Path(NOTEBOOK_FILENAME)
            if not nb_src.exists():
                st.error(f"Notebook não encontrado no servidor: {NOTEBOOK_FILENAME}")
            else:
                notebook_copy = os.path.join(tmp_root, nb_src.name)
                shutil.copy2(str(nb_src), notebook_copy)

                # 3) read notebook and inject code
                nb = nbformat.read(notebook_copy, as_version=4)
                injected = nbformat.v4.new_code_cell(INJECT_CODE)
                nb.cells.insert(0, injected)
                # write modified notebook
                exec_nb_path = os.path.join(tmp_root, "executed_notebook.ipynb")
                nbformat.write(nb, exec_nb_path)

                # 4) execute notebook
                with st.spinner("A executar o notebook (pode demorar alguns minutos)..."):
                    ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
                    # run in tmp_root so notebook encontra ficheiros
                    cur = os.getcwd()
                    os.chdir(tmp_root)
                    try:
                        ep.preprocess(nb, {'metadata': {'path': tmp_root}})
                    finally:
                        os.chdir(cur)

                st.success("✅ Análise concluída. A carregar artefactos...")

                # 5) load manifest and show artifacts
                report_output_dir = os.path.join(tmp_root, "report_output")
                manifest_path = os.path.join(report_output_dir, "manifest.json")
                artifacts = []
                if os.path.exists(manifest_path):
                    with open(manifest_path, encoding="utf-8") as mf:
                        artifacts = json.load(mf)
                else:
                    # fallback: list plots folder
                    plots_dir = os.path.join(report_output_dir, "plots")
                    if os.path.isdir(plots_dir):
                        for fn in sorted(os.listdir(plots_dir)):
                            artifacts.append({"name": fn, "file": f"plots/{fn}", "type": "plot"})

                # 6) display artifacts in structured UI
                if not artifacts:
                    st.warning("Nenhum artefacto encontrado (verifica se o notebook chama as funções de 'save_artefact...' corretamente).")
                else:
                    st.markdown("### Artefactos gerados")
                    # create categories by keyword
                    categories = {
                        "Performance & Métricas": [],
                        "Processos & Modelos": [],
                        "Gantt / Throughput": [],
                        "Gargalos & Esperas": [],
                        "Frequência & Ocorrências": [],
                        "Outros": []
                    }
                    for a in artifacts:
                        n = a["name"].lower()
                        if re.search(r"matriz|métric|kpi|séries temporais|throughput", n):
                            categories["Performance & Métricas"].append(a)
                        elif re.search(r"processo|modelo|inductive|heuristics|variante|redis|rede social|dfg", n):
                            categories["Processos & Modelos"].append(a)
                        elif re.search(r"gantt|linha do tempo|throughput", n):
                            categories["Gantt / Throughput"].append(a)
                        elif re.search(r"gargalo|espera|handoff|ranking", n):
                            categories["Gargalos & Esperas"].append(a)
                        elif re.search(r"ocorr|frequência|dia da semana|tarefas que ocorrem", n):
                            categories["Frequência & Ocorrências"].append(a)
                        else:
                            categories["Outros"].append(a)

                    # show category tabs
                    tabs = st.tabs(list(categories.keys()))
                    for i, (cat, items) in enumerate(categories.items()):
                        with tabs[i]:
                            if not items:
                                st.write("_(nenhum artefacto nesta categoria)_")
                            else:
                                for art in items:
                                    st.markdown(f"**{art['name']}**")
                                    target = os.path.join(report_output_dir, art['file'])
                                    if os.path.exists(target) and target.lower().endswith((".png",".jpg",".jpeg",".gif")):
                                        st.image(target, use_column_width=True)
                                        st.download_button("Download imagem", target, file_name=os.path.basename(target))
                                    else:
                                        # show as link or text
                                        if os.path.exists(target):
                                            st.download_button("Download ficheiro", target, file_name=os.path.basename(target))
                                        else:
                                            st.write(f"Ficheiro referenciado não encontrado: {art['file']}")

        except Exception as e:
            st.error(f"Erro durante a execução: {e}")
        finally:
            st.caption(f"Run dir: {tmp_root} (mantido para debug).")
            # opcional: eliminar tmp_root se quiseres
            # shutil.rmtree(tmp_root, ignore_errors=True)
