
# -*- coding: utf-8 -*-
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import streamlit as st
    STREAMLIT_OK = True
except Exception:
    STREAMLIT_OK = False

if STREAMLIT_OK:
    st.set_page_config(page_title="Process Mining App", page_icon="📊", layout="wide")

def autodetect_columns(cols):
    low = [c.lower() for c in cols]
    case_cands = ["case", "case_id", "caseid", "case:concept:name", "trace"]
    act_cands = ["activity", "concept:name", "task", "act"]
    ts_cands = ["time:timestamp", "timestamp", "time", "event_time", "date"]
    case_col = next((cols[i] for i,c in enumerate(low) if c in case_cands), None)
    act_col = next((cols[i] for i,c in enumerate(low) if c in act_cands), None)
    ts_col = next((cols[i] for i,c in enumerate(low) if c in ts_cands), None)
    return case_col, act_col, ts_col

import io
from datetime import datetime

try:
    import pm4py
    PM4PY_OK = True
except Exception:
    PM4PY_OK = False


def compute_kpis(df, case_col, act_col, ts_col):
    out = {"eventos": 0, "atividades": 0, "casos": 0, "throughput_h_med": None}
    if df is None or len(df) == 0:
        return out
    d = df.copy()
    if ts_col and ts_col in d.columns:
        d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    out["eventos"] = int(len(d))
    if act_col and act_col in d.columns:
        out["atividades"] = int(d[act_col].nunique())
    if case_col and case_col in d.columns:
        out["casos"] = int(d[case_col].nunique())
    if case_col and ts_col and case_col in d.columns and ts_col in d.columns:
        agg = d.groupby(case_col)[ts_col].agg(["min", "max"])  # type: ignore
        med_h = ((agg["max"] - agg["min"]).dt.total_seconds() / 3600.0).median()
        out["throughput_h_med"] = float(med_h) if pd.notnull(med_h) else None
    return out


def build_dfg(df, case_col, act_col, ts_col):
    if df is None or not case_col or not act_col or not ts_col:
        return pd.DataFrame(columns=["source", "target", "count"])
    if not set([case_col, act_col, ts_col]).issubset(df.columns):
        return pd.DataFrame(columns=["source", "target", "count"])
    d = df[[case_col, act_col, ts_col]].dropna().copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.sort_values([case_col, ts_col])
    edges = {}
    for cid, g in d.groupby(case_col):
        acts = g[act_col].astype(str).tolist()
        for i in range(len(acts) - 1):
            k = (acts[i], acts[i + 1])
            edges[k] = edges.get(k, 0) + 1
    rows = []
    for k, v in edges.items():
        rows.append({"source": k[0], "target": k[1], "count": v})
    return pd.DataFrame(rows)


def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ---- Session and Auth ----
if STREAMLIT_OK:
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "df" not in st.session_state:
        st.session_state.df = None
    if "map_case" not in st.session_state:
        st.session_state.map_case = None
    if "map_act" not in st.session_state:
        st.session_state.map_act = None
    if "map_ts" not in st.session_state:
        st.session_state.map_ts = None

VALID_USERNAME = "admin"
VALID_PASSWORD = "admin123"

# ---- Sidebar ----
if STREAMLIT_OK:
    with st.sidebar:
        st.title("🧭 Navegação")
        if st.session_state.auth:
            st.session_state.page = st.radio("Ir para", ["Dashboard", "Descoberta", "Análise", "Exportar"],
                                             index=["Dashboard", "Descoberta", "Análise", "Exportar"].index(st.session_state.page))
            st.caption("Ligado como admin")
            if st.button("Sair"):
                st.session_state.auth = False
                st.session_state.page = "Dashboard"
                st.rerun()
        else:
            st.info("Faça login para aceder")

# ---- Login Page ----
if STREAMLIT_OK and not st.session_state.auth:
    st.title("📊 Process Mining App (moderno)")
    with st.form("login"):
        u = st.text_input("Utilizador")
        p = st.text_input("Palavra-passe", type="password")
        ok = st.form_submit_button("Entrar")
    if ok:
        if u == VALID_USERNAME and p == VALID_PASSWORD:
            st.session_state.auth = True
            st.success("Autenticado")
            st.rerun()
        else:
            st.error("Credenciais inválidas")
    st.stop()
