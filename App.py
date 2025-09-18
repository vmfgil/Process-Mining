import streamlit as st
import os
import nbformat
from nbconvert import PythonExporter
import subprocess
import glob
import pandas as pd

# Caminhos principais
NOTEBOOK_PATH = "PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb"
UPLOAD_DIR = "uploaded_data"

# Configuração da página
st.set_page_config(page_title="Process Mining App", layout="wide")

# Navegação lateral
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Upload & Pré-visualização", "Executar Análises", "Resultados"])

# -------------------------------------------------------------------
# 1. Upload & Pré-visualização
# -------------------------------------------------------------------
if page == "Upload & Pré-visualização":
    st.title("Upload dos Ficheiros CSV")
    uploaded_files = st.file_uploader(
        "Carregue os 5 ficheiros CSV (projects, tasks, resources, resource_allocations, dependencies)",
        accept_multiple_files=True,
        type="csv"
    )

    if uploaded_files:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for f in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
            st.success(f"{f.name} guardado com sucesso.")
            try:
                df = pd.read_csv(save_path)
                st.write(f"Pré-visualização de **{f.name}**:")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Erro ao ler {f.name}: {e}")

# -------------------------------------------------------------------
# 2. Executar análises (notebook -> script -> execução)
# -------------------------------------------------------------------
elif page == "Executar Análises":
    st.title("Executar Notebook")

    if st.button("Executar"):
        if not os.path.exists(NOTEBOOK_PATH):
            st.error("Notebook não encontrado no repositório.")
        else:
            st.info("A converter notebook em script Python...")
            with open(NOTEBOOK_PATH) as f:
                nb = nbformat.read(f, as_version=4)

            exporter = PythonExporter()
            source, _ = exporter.from_notebook_node(nb)

            # --- Substituição do upload Colab pelos CSV da app ---
            csv_loader = """import pandas as pd
projects = pd.read_csv("uploaded_data/projects.csv")
tasks = pd.read_csv("uploaded_data/tasks.csv")
resources = pd.read_csv("uploaded_data/resources.csv")
resource_allocations = pd.read_csv("uploaded_data/resource_allocations.csv")
dependencies = pd.read_csv("uploaded_data/dependencies.csv")"""

            source = source.replace("files.upload()", "# Substituído pelo Streamlit")
            source = source.replace("uploaded", "# Substituído pelo Streamlit")
            source = csv_loader + "\n\n" + source

            # Guardar script convertido
            script_path = "notebook_script.py"
            with open(script_path, "w") as f:
                f.write(source)

            # Executar script
            st.info("A executar o script gerado a partir do notebook...")
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.success("Notebook executado com sucesso.")
                if result.stdout:
                    st.code(result.stdout)
            except subprocess.CalledProcessError as e:
                st.error("Erro ao executar o notebook.")
                st.code(e.stdout + "\n" + e.stderr)

# -------------------------------------------------------------------
# 3. Resultados
# -------------------------------------------------------------------
elif page == "Resultados":
    st.title("Resultados da Análise")

    pre_dir = "Process_Analysis_Dashboard"
    post_dir = "Relatorio_Unificado_Analise_Processos"

    def show_images_from_dir(directory, titulo):
        if os.path.exists(directory):
            st.header(titulo)
            imgs = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
            if not imgs:
                st.warning(f"Sem imagens encontradas em {directory}.")
            for img in sorted(imgs):
                st.image(img, use_column_width=True, caption=os.path.basename(img))
        else:
            st.warning(f"Diretório {directory} não encontrado.")

    # Mostrar secções de resultados
    show_images_from_dir(pre_dir, "Pré-Mineracão")
    show_images_from_dir(post_dir, "Pós-Mineracão")
