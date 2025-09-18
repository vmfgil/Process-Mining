# streamlit_app.py
# Streamlit app to reproduce your notebook outputs (uploads -> run analyses -> results)
# IMPORTANT: this app executes the notebook present at /mnt/data/PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb
# It saves uploaded files to the working directory, runs the notebook as a script, and then
# collects the generated plots from the two report folders the notebook creates:
#  - Process_Analysis_Dashboard (pre-mining / penultimate cell)
#  - Relatorio_Unificado_Analise_Processos (post-mining / last cell)

import streamlit as st
import os, sys, io, shutil, tempfile, subprocess
from pathlib import Path
import nbformat
import pandas as pd
from typing import List, Dict

# App config
st.set_page_config(page_title='Process Mining Dashboard', layout='wide')

# Minimal CSS (uses single quotes to avoid JSON escape issues)
st.markdown(
    "<style>\n    .main > div {padding: 1rem 2rem;}\n    .card {background: linear-gradient(180deg, #ffffff, #f7f9fc); border-radius:12px; padding:1rem; box-shadow: 0 6px 18px rgba(20,30,60,0.08);}\n    .brand {font-weight:700; font-size:20px;}\n    .muted {color:#6b7280;}\n    </style>", unsafe_allow_html=True)

REQUIRED_FILENAMES = ['projects.csv', 'tasks.csv', 'resources.csv', 'resource_allocations.csv', 'dependencies.csv']
NOTEBOOK_PATH = 'PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb'
# fallback path (original upload)
if not os.path.exists(NOTEBOOK_PATH):
    NOTEBOOK_PATH = '/mnt/data/PM_na_Gestão_de_recursos_de_IT_v5.0.ipynb'

PREMINING_DIR = 'Process_Analysis_Dashboard'
POSTMINING_DIR = 'Relatorio_Unificado_Analise_Processos'

def save_uploaded_files(uploaded_files: Dict[str, io.BytesIO], dest_dir: str = '.'):
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for name, up in uploaded_files.items():
        path = os.path.join(dest_dir, name)
        with open(path, 'wb') as f:
            f.write(up.getbuffer())
        saved.append(path)
    return saved

def preview_uploaded(uploaded_files: Dict[str, io.BytesIO], nrows=5):
    previews = {}
    for name, up in uploaded_files.items():
        try:
            up.seek(0)
            df = pd.read_csv(up)
        except Exception:
            try:
                up.seek(0)
                df = pd.read_excel(up)
            except Exception:
                df = None
        previews[name] = df.head(nrows) if isinstance(df, pd.DataFrame) else None
    return previews

def sanitize_and_write_script(nb_path: str, out_path: str):
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c.source for c in nb.cells if c.cell_type == 'code']
    combined = '\n\n# --- Cell boundary ---\n\n'.join(code_cells)
    # Neutralize common interactive or install lines
    combined = combined.replace('files.upload()', 'None  # neutralized')
    combined = combined.replace("from google.colab import files", '# removed')
    import re
    combined = re.sub(r"(^|\n)\s*!pip\s+install[^\n]*", '\n# pip removed', combined)
    combined = re.sub(r"(^|\n)\s*get_ipython\(\)\.system\([^\)]*\)", '\n# system call removed', combined)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Auto-generated script from notebook for Streamlit execution\n')
        f.write('import os\n')
        f.write('\n')
        f.write(combined)
    return out_path

def run_script_and_wait(script_path: str, workdir: str = '.'):
    try:
        proc = subprocess.run([sys.executable, script_path], cwd=workdir, capture_output=True, text=True, timeout=1800)
        output = proc.stdout + '\n' + proc.stderr
        success = proc.returncode == 0
        return success, output
    except subprocess.TimeoutExpired as e:
        return False, f'Script timed out after {e.timeout} seconds'
    except Exception as e:
        return False, str(e)

def collect_images_from_dir(base_dir: str):
    imgs = []
    base = Path(base_dir)
    if not base.exists():
        return imgs
    for p in base.rglob('*'):
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.gif']:
            imgs.append(str(p))
    imgs.sort()
    return imgs

def build_subsections_from_headings(nb_path: str):
    nb = nbformat.read(nb_path, as_version=4)
    headings = []
    for c in nb.cells:
        if c.cell_type == 'markdown':
            for line in c.source.splitlines():
                line = line.strip()
                if line and (line.startswith('#') or (line.isupper() and len(line) < 80)):
                    h = line.lstrip('#').strip()
                    if h and h not in headings:
                        headings.append(h)
    if not headings:
        headings = ['Secção 1: Análises de Alto Nível e de Casos', 'Secção 2: Descida e Descoberta de Modelos']
    return headings

def match_images_to_sections(images: List[str], sections: List[str]):
    mapping = {s: [] for s in sections}
    others = []
    for img in images:
        name = os.path.basename(img).lower()
        matched = False
        for s in sections:
            key = ''.join(ch.lower() for ch in s if ch.isalnum() or ch.isspace())
            tokens = key.split()[:5]
            if all(t in name for t in tokens if len(t) > 2):
                mapping[s].append(img)
                matched = True
                break
        if not matched:
            others.append(img)
    if others:
        mapping['Outros'] = others
    return mapping

# UI
st.sidebar.title('Navega\u00e7\u00e3o')
page = st.sidebar.radio('', ['Upload & Pr\u00e9-visualiza\u00e7\u00e3o', 'Executar An\u00e1lises', 'Resultados'])

st.sidebar.markdown('---')
st.sidebar.markdown('Brand: Process Mining Studio')

st.markdown('<div class="card"><div class="brand">Process Mining Dashboard</div><div class="muted">Upload, executar e explorar resultados</div></div>', unsafe_allow_html=True)

if page == 'Upload & Pr\u00e9-visualiza\u00e7\u00e3o':
    st.header('1) Upload dos 5 ficheiros (obrigat\u00f3rios)')
    st.write('Os ficheiros que o notebook espera s\u00e3o:')
    st.write(', '.join(REQUIRED_FILENAMES))
    uploaded = {}
    cols = st.columns(2)
    i = 0
    for fn in REQUIRED_FILENAMES:
        c = cols[i % 2]
        with c:
            up = st.file_uploader(fn, type=['csv','xlsx','xls'], key=f'up_{fn}')
            if up is not None:
                uploaded[fn] = up
        i += 1
    if st.button('Guardar ficheiros no servidor'):
        missing = [f for f in REQUIRED_FILENAMES if f not in uploaded]
        if missing:
            st.error(f'Faltam ficheiros: {missing} \u2014 carregue todos os 5 ficheiros antes de guardar.')
        else:
            saved = save_uploaded_files(uploaded, dest_dir='uploaded_data')
            st.success('Ficheiros guardados: ' + ', '.join(os.path.basename(s) for s in saved))
            st.info('Pode agora ir a "Executar An\u00e1lises" para correr o notebook e gerar os gr\u00e1ficos.')
    st.markdown('---')
    st.header('Pr\u00e9-visualiza\u00e7\u00e3o (primeiras linhas)')
    previews = preview_uploaded(uploaded, nrows=5)
    for name, df in previews.items():
        st.subheader(name)
        if df is None:
            st.write('Formato n\u00e3o reconhecido.')
        else:
            st.dataframe(df)

elif page == 'Executar An\u00e1lises':
    st.header('2) Executar todas as an\u00e1lises (vai correr o notebook)')
    st.write('Certifique-se que j\u00e1 guardou os ficheiros na p\u00e1gina de upload.')
    if not os.path.exists('uploaded_data'):
        st.warning('Pasta uploaded_data n\u00e3o encontrada. Faça o upload e guarde os ficheiros primeiro.')
    else:
        st.write('Ficheiros guardados detectados:')
        st.write(os.listdir('uploaded_data'))
    col1, col2 = st.columns([1,3])
    with col1:
        run_btn = st.button('Executar Notebook')
    with col2:
        st.info('A execução pode demorar. Sa\u00edda e logs mostrados abaixo.')
    if run_btn:
        for fn in REQUIRED_FILENAMES:
            src = os.path.join('uploaded_data', fn)
            if os.path.exists(src):
                shutil.copy(src, fn)
        tmpdir = tempfile.mkdtemp(prefix='nb_exec_')
        script_path = os.path.join(tmpdir, 'notebook_script.py')
        try:
            sanitize_and_write_script(NOTEBOOK_PATH, script_path)
        except Exception as e:
            st.error('N\u00e3o foi poss\u00edvel preparar o script: ' + str(e))
        st.info('A executar o script gerado a partir do notebook...')
        with st.spinner('Executando...'):
            success, output = run_script_and_wait(script_path, workdir='.')
        if success:
            st.success('Notebook executado com sucesso.')
        else:
            st.error('A execu\u00e7\u00e3o falhou. Ver logs.')
        st.code(output[:20000])

elif page == 'Resultados':
    st.header('3) Resultados')
    st.write('Duas sec\u00e7\u00f5es: Pr\u00e9-minera\u00e7\u00e3o e P\u00f3s-minera\u00e7\u00e3o')
    pre_images = collect_images_from_dir(PREMINING_DIR)
    post_images = collect_images_from_dir(POSTMINING_DIR)
    st.markdown('---')
    st.subheader('Pr\u00e9-minera\u00e7\u00e3o — Process_Analysis_Dashboard')
    if not pre_images:
        st.warning('Nenhuma imagem encontrada em "Process_Analysis_Dashboard". Execute o notebook primeiro.')
    else:
        headings = build_subsections_from_headings(NOTEBOOK_PATH)
        mapping = match_images_to_sections(pre_images, headings)
        for sec, imgs in mapping.items():
            with st.expander(sec, expanded=False):
                if not imgs:
                    st.write('Sem imagens para esta subse\u00e7\u00e3o.')
                else:
                    cols = st.columns(3)
                    for i, im in enumerate(imgs):
                        with cols[i % 3]:
                            st.image(im, use_column_width=True, caption=os.path.basename(im))
    st.markdown('---')
    st.subheader('P\u00f3s-minera\u00e7\u00e3o — Relatorio_Unificado_Analise_Processos')
    if not post_images:
        st.warning('Nenhuma imagem encontrada em "Relatorio_Unificado_Analise_Processos". Execute o notebook primeiro.')
    else:
        headings_post = build_subsections_from_headings(NOTEBOOK_PATH)
        mapping_post = match_images_to_sections(post_images, headings_post)
        for sec, imgs in mapping_post.items():
            with st.expander(sec, expanded=False):
                if not imgs:
                    st.write('Sem imagens para esta subse\u00e7\u00e3o.')
                else:
                    cols = st.columns(3)
                    for i, im in enumerate(imgs):
                        with cols[i % 3]:
                            st.image(im, use_column_width=True, caption=os.path.basename(im))

    st.markdown('\n---\n')
    st.write('Se quer que reorganize as subse\u00e7\u00f5es ou filtre imagens, diga-me como.')

