import streamlit as st
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date, time
import uuid
from collections import Counter

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vendor Selection & Project Control",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stApp { background-color: #f7f8fa; color: #1f2937; }
    h1 { color: #111827; font-size: 1.75rem !important; font-weight: 600 !important; }
    h2 { color: #111827; font-size: 1.35rem !important; font-weight: 600 !important;
         border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }
    h3 { color: #1f2937; font-size: 1.05rem !important; font-weight: 600 !important; }
    h4 { color: #374151; font-size: 0.95rem !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.5rem !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.8rem !important; font-weight: 500; }
    .stButton > button {
        background-color: #1e40af; color: white; border: none;
        border-radius: 6px; padding: 8px 20px; font-weight: 500;
        font-size: 0.9rem; width: 100%;
    }
    .stButton > button:hover { background-color: #1e3a8a; }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #047857 !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #065f46 !important;
    }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    .stDataFrame { border: 1px solid #e5e7eb; border-radius: 6px; }
    .pill {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.72rem; font-weight: 600;
    }
    .pill-ok { background: #d1fae5; color: #065f46; }
    .pill-warn { background: #fef3c7; color: #92400e; }
    .pill-fail { background: #fee2e2; color: #991b1b; }
    .pill-info { background: #dbeafe; color: #1e40af; }
    [data-testid="stMetricDelta"] svg { display: none; }
    .step-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 0; font-size: 0.85rem;
    }
    .step-dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block;
    }
    .dot-ok { background: #047857; }
    .dot-idle { background: #d1d5db; }
    .stepper {
        display: flex; align-items: center; gap: 4px;
        margin-bottom: 16px; flex-wrap: wrap;
    }
    .step-chip {
        display: flex; align-items: center; gap: 6px;
        padding: 8px 14px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 500;
        border: 1px solid #e5e7eb;
        background: #ffffff; color: #6b7280;
    }
    .step-chip.active {
        background: #1e40af; color: #ffffff;
        border-color: #1e40af; font-weight: 600;
    }
    .step-chip.done {
        background: #d1fae5; color: #065f46;
        border-color: #a7f3d0;
    }
    .step-chip .num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 20px; height: 20px; border-radius: 50%;
        background: rgba(255,255,255,0.2); font-size: 0.75rem; font-weight: 700;
    }
    .step-chip:not(.active):not(.done) .num { background: #f3f4f6; color: #6b7280; }
    .step-chip.active .num { background: rgba(255,255,255,0.25); color: #ffffff; }
    .step-chip.done .num { background: #047857; color: #ffffff; }
    .hint-box {
        background: #eff6ff; border-left: 3px solid #1e40af;
        padding: 8px 12px; border-radius: 4px; margin: 6px 0;
        font-size: 0.85rem; color: #1e40af;
    }
    .flow-hint {
        background: #f9fafb; border: 1px dashed #d1d5db;
        padding: 10px 14px; border-radius: 6px;
        font-size: 0.82rem; color: #4b5563;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER
# =========================================================
def format_rupiah(juta):
    try:
        juta = float(juta)
    except (TypeError, ValueError):
        return "Rp 0 juta"
    if abs(juta) >= 1000:
        return f"Rp {juta/1000:,.2f} miliar"
    return f"Rp {juta:,.0f} juta"


def sigmoid_kernel(t, k=4.0):
    """
    Kernel S untuk distribusi progress dalam satu item.
    t = 0..1 (progress waktu dari start ke end item)
    Return: 0..1, dengan bentuk S (landai → curam → landai)
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 0.5 * (1 + np.tanh(k * (t - 0.5)))


def _create_binary_vars(prob, var_name, index_list):
    try:
        return prob.add_variable_dicts(var_name, index_list, lowBound=0, upBound=1, cat="Binary")
    except AttributeError:
        pass
    try:
        return pulp.LpVariable.dicts(var_name, index_list, lowBound=0, upBound=1, cat="Binary")
    except AttributeError:
        pass
    x = {}
    for i in index_list:
        var = pulp.LpVariable(f"{var_name}_{i}")
        var.lowBound = 0
        var.upBound = 1
        var.cat = pulp.LpInteger
        x[i] = var
    return x


def solve_vendor_lp(vendor_names, costs, budget, scores=None,
                    weight_price=0.5, weight_perf=0.5):
    try:
        prob = pulp.LpProblem("Vendor_Selection", pulp.LpMaximize)
        x = _create_binary_vars(prob, "select", vendor_names)

        if scores is None:
            scores = {n: 100 for n in vendor_names}

        min_cost = min(costs.values())
        max_cost = max(costs.values())
        cost_range = max_cost - min_cost if max_cost > min_cost else 1

        # Normalisasi min-max: harga termurah = 100, termahal = 0
        cost_scores = {
            n: 100 * (max_cost - costs[n]) / cost_range
            for n in vendor_names
        }

        prob += pulp.lpSum(
            (weight_price * cost_scores[n] / 100
             + weight_perf * scores[n] / 100) * x[n]
            for n in vendor_names
        )
        prob += pulp.lpSum(x[n] for n in vendor_names) == 1
        prob += pulp.lpSum(costs[n] * x[n] for n in vendor_names) <= budget

        prob.solve()
        status = pulp.LpStatus.get(prob.status, "Unknown")
        if status != "Optimal":
            return None, status
        for n in vendor_names:
            val = pulp.value(x[n])
            if val is not None and val > 0.5:
                return n, status
        return None, "No vendor selected"
    except Exception as e:
        return None, f"Error: {e}"


# =========================================================
# SESSION STATE
# =========================================================
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "active_project" not in st.session_state:
    st.session_state.active_project = None
if "step" not in st.session_state:
    st.session_state.step = 0


STEPS = [
    {"key": "project_list", "label": "Daftar Project"},
    {"key": "tender", "label": "Tender & BOQ"},
    {"key": "vendor_selection", "label": "Vendor Qualification"},
    {"key": "site_execution", "label": "Site Execution"},
    {"key": "quality_control", "label": "Quality Control"},
    {"key": "done", "label": "Selesai"},
]


def goto_step(n):
    st.session_state.step = max(0, min(len(STEPS) - 1, n))


def next_step():
    goto_step(st.session_state.step + 1)


def prev_step():
    goto_step(st.session_state.step - 1)


def create_project(name, client, location, budget, deadline):
    pid = str(uuid.uuid4())[:8]
    st.session_state.projects[pid] = {
        "meta": {
            "id": pid, "name": name, "client": client, "location": location,
            "budget": float(budget), "deadline": int(deadline),
            "rab_awal": float(budget),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "live_data": {
            "tender": {}, "vendor_selection": {},
            "site_execution": {}, "quality_control": {},
        },
        "wbs_data": None,
        "items_data": None,
        "vendor_data": {},
        "appointments": [],
    }
    return pid


def get_active():
    if st.session_state.active_project is None:
        return None
    return st.session_state.projects.get(st.session_state.active_project)


def label(tahap):
    return {
        "tender": "Tender & BOQ",
        "vendor_selection": "Vendor Qualification",
        "site_execution": "Site Execution",
        "quality_control": "Quality Control",
    }.get(tahap, tahap)


def stage_status(project, tahap):
    return bool(project["live_data"].get(tahap))


def project_progress(project):
    stages = ["tender", "vendor_selection", "site_execution", "quality_control"]
    return sum(1 for t in stages if stage_status(project, t)), len(stages)


def default_items():
    return pd.DataFrame([
        {"No": 1, "Item Pekerjaan": "Fabrikasi Pipa Carbon Steel", "Volume": "500", "Satuan": "m",
         "Spesifikasi": "ASTM A106 Gr.B", "Kompleksitas": "Sedang"},
        {"No": 2, "Item Pekerjaan": "Welding SMAW + GTAW", "Volume": "300", "Satuan": "joint",
         "Spesifikasi": "WPS sesuai ASME IX", "Kompleksitas": "Tinggi"},
        {"No": 3, "Item Pekerjaan": "NDT Radiografi", "Volume": "150", "Satuan": "titik",
         "Spesifikasi": "ASME V Article 2", "Kompleksitas": "Tinggi"},
        {"No": 4, "Item Pekerjaan": "Hydrotest", "Volume": "3", "Satuan": "line",
         "Spesifikasi": "ASME B31.3", "Kompleksitas": "Sedang"},
        {"No": 5, "Item Pekerjaan": "Support & Hanger Installation", "Volume": "80", "Satuan": "unit",
         "Spesifikasi": "MSS SP-58", "Kompleksitas": "Rendah"},
        {"No": 6, "Item Pekerjaan": "Painting & Coating", "Volume": "800", "Satuan": "m²",
         "Spesifikasi": "SSPC-SP10", "Kompleksitas": "Rendah"},
    ])


def default_wbs():
    return pd.DataFrame([
        {"No": 1, "Item Pekerjaan": "Persiapan & Mobilisasi", "Bobot (%)": 5.0,
         "Mulai (minggu)": 1, "Selesai (minggu)": 3, "Realisasi (%)": 100.0},
        {"No": 2, "Item Pekerjaan": "Pekerjaan Tanah & Pondasi", "Bobot (%)": 15.0,
         "Mulai (minggu)": 2, "Selesai (minggu)": 6, "Realisasi (%)": 100.0},
        {"No": 3, "Item Pekerjaan": "Fabrikasi Pipa", "Bobot (%)": 25.0,
         "Mulai (minggu)": 4, "Selesai (minggu)": 10, "Realisasi (%)": 80.0},
        {"No": 4, "Item Pekerjaan": "Instalasi Pipa & Support", "Bobot (%)": 30.0,
         "Mulai (minggu)": 6, "Selesai (minggu)": 12, "Realisasi (%)": 45.0},
        {"No": 5, "Item Pekerjaan": "Welding & NDT", "Bobot (%)": 10.0,
         "Mulai (minggu)": 8, "Selesai (minggu)": 13, "Realisasi (%)": 20.0},
        {"No": 6, "Item Pekerjaan": "Painting & Insulation", "Bobot (%)": 8.0,
         "Mulai (minggu)": 10, "Selesai (minggu)": 14, "Realisasi (%)": 0.0},
        {"No": 7, "Item Pekerjaan": "Testing, Commissioning & Handover", "Bobot (%)": 7.0,
         "Mulai (minggu)": 12, "Selesai (minggu)": 15, "Realisasi (%)": 0.0},
    ])


INVITED_VENDORS = [
    "PT Baja Nusantara Konstruksi",
    "CV Prima Pipa Sejahtera",
    "PT Karya Logam Presisi",
    "PT Sinergi Teknik Mandiri",
    "PT Anugerah Pipa Internasional",
]


PERFORMANCE_ASPECTS = [
    "Kualitas Pekerjaan",
    "Ketepatan Waktu",
    "K3 & Lingkungan",
    "Koordinasi Tim",
    "Responsif terhadap Masalah",
    "Referensi Klien Sebelumnya",
]


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Project Manager")

    if st.session_state.projects:
        options = list(st.session_state.projects.keys())
        current_idx = 0
        if st.session_state.active_project in options:
            current_idx = options.index(st.session_state.active_project)

        selected = st.selectbox(
            "Project aktif",
            options=options,
            index=current_idx,
            format_func=lambda x: st.session_state.projects[x]["meta"]["name"],
        )
        st.session_state.active_project = selected
    else:
        st.info("Belum ada project. Tambahkan di bawah.")

    st.markdown("---")
    st.markdown("**Tambah Project Baru**")

    new_name = st.text_input("Nama project", placeholder="Contoh: Piping Area 5B", key="new_name")
    new_client = st.text_input("Klien", placeholder="Contoh: PT Pertamina", key="new_client")
    new_location = st.text_input("Lokasi", placeholder="Contoh: Balikpapan", key="new_location")
    new_budget = st.number_input(
        "Budget project (juta Rp)",
        min_value=10.0, value=300.0, step=10.0, key="new_budget",
    )
    new_deadline = st.number_input(
        "Deadline project (hari)",
        min_value=1, value=90, step=5, key="new_deadline"
    )
    if st.button("Tambah Project", key="btn_add_project"):
        if new_name.strip():
            pid = create_project(
                new_name.strip(),
                new_client.strip() or "-",
                new_location.strip() or "-",
                new_budget,
                new_deadline,
            )
            st.session_state.active_project = pid
            st.session_state.step = 1
            st.rerun()
        else:
            st.error("Nama project wajib diisi.")

    st.markdown("---")
    active = get_active()
    if active:
        st.markdown("**Status Tahapan**")
        for i, t in enumerate(STEPS[1:5], start=1):
            ok = stage_status(active, t["key"])
            dot = "dot-ok" if ok else "dot-idle"
            text_color = "#047857" if ok else "#9ca3af"
            st.markdown(
                f"<div class='step-row'>"
                f"<span class='step-dot {dot}'></span>"
                f"<span style='color:{text_color};'>{i}. {t['label']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        done, total = project_progress(active)
        st.caption(f"{done} dari {total} tahap selesai")

        st.markdown("---")
        st.markdown("**Parameter Kriteria Vendor**")
        min_quality = st.slider("Kualitas minimum", 0, 100, 82, key="min_quality")
        min_hse = st.slider("Skor K3 minimum", 0, 100, 85, key="min_hse")
        min_performance = st.slider("Skor performa minimum", 0, 100, 70, key="min_perf")
        budget_weight = st.slider(
            "Bobot harga vs performa (%)",
            0, 100, 50, key="weight_price",
            help="50% = harga & performa seimbang."
        )
        weight_price = budget_weight / 100
        weight_perf = 1 - weight_price

        st.markdown("---")
        st.markdown("**Budget & Deadline Project**")
        new_budget_edit = st.number_input(
            "Budget project (juta Rp)",
            min_value=10.0,
            value=float(active["meta"]["budget"]),
            step=10.0,
            key=f"edit_budget_{active['meta']['id']}",
        )
        new_deadline_edit = st.number_input(
            "Deadline project (hari)",
            min_value=1,
            value=int(active["meta"]["deadline"]),
            step=5,
            key=f"edit_deadline_{active['meta']['id']}"
        )
        if (new_budget_edit != active["meta"]["budget"]
                or new_deadline_edit != active["meta"]["deadline"]):
            active["meta"]["budget"] = float(new_budget_edit)
            active["meta"]["deadline"] = int(new_deadline_edit)
            st.rerun()

        st.markdown("---")
        if st.button("Hapus Project Ini", key="btn_delete_project"):
            del st.session_state.projects[st.session_state.active_project]
            st.session_state.active_project = (
                list(st.session_state.projects.keys())[0]
                if st.session_state.projects else None
            )
            st.session_state.step = 0
            st.rerun()
    else:
        min_quality, min_hse, min_performance = 82, 85, 70
        weight_price, weight_perf = 0.5, 0.5

# =========================================================
# HEADER
# =========================================================
active = get_active()

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("# Vendor Selection & Project Control")
    if active:
        meta = active["meta"]
        st.caption(
            f"{meta['name']} · {meta['client']} · {meta['location']} · "
            f"Budget: {format_rupiah(meta['budget'])} · Deadline: {meta['deadline']} hari"
        )
    else:
        st.caption("Belum ada project aktif")

with col_status:
    total_projects = len(st.session_state.projects)
    st.markdown(
        f"<div style='text-align:right; padding-top:12px;'>"
        f"<span class='pill pill-info'>{total_projects} project</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

step = st.session_state.step

# =========================================================
# STEP 0 — DAFTAR PROJECT
# =========================================================
if step == 0:
    st.markdown("## Daftar Project")

    if not st.session_state.projects:
        st.info("Belum ada project. Tambahkan project baru dari sidebar.")
        st.stop()

    summary_rows = []
    for pid_, p in st.session_state.projects.items():
        m = p["meta"]
        live = p["live_data"]
        vs = live.get("vendor_selection", {})
        qc = live.get("quality_control", {})
        se = live.get("site_execution", {})

        if qc and qc.get("final_status") == "Ya":
            status = "Selesai"
        elif se:
            status = "Eksekusi"
        elif vs:
            status = "Vendor dipilih"
        elif live.get("tender"):
            status = "Tender"
        else:
            status = "Draft"

        done, total = project_progress(p)
        summary_rows.append({
            "Project": m["name"],
            "Klien": m["client"],
            "Lokasi": m["location"],
            "Budget": format_rupiah(m["budget"]),
            "Tahapan": f"{done}/{total}",
            "Vendor Terpilih": vs.get("selected_vendor", "-"),
            "Nilai Kontrak": format_rupiah(vs.get("selected_cost", 0))
            if vs.get("selected_cost") else "-",
            "Status": status,
        })

    df_summary = pd.DataFrame(summary_rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total project", len(df_summary))
    total_budget = sum(p["meta"]["budget"] for p in st.session_state.projects.values())
    c2.metric("Total budget", format_rupiah(total_budget))
    c3.metric("Project selesai", df_summary[df_summary["Status"] == "Selesai"].shape[0])

    st.markdown("---")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Buka Project")
    project_options = list(st.session_state.projects.keys())
    selected_pid = st.selectbox(
        "Pilih project untuk dibuka",
        options=project_options,
        format_func=lambda x: st.session_state.projects[x]["meta"]["name"],
        key="open_project_select"
    )
    if st.button("Buka Project", key="btn_open"):
        st.session_state.active_project = selected_pid
        st.session_state.step = 1
        st.rerun()

    st.stop()


if active is None:
    st.info("Belum ada project aktif. Tambahkan project baru dari sidebar.")
    st.stop()

live_data = active["live_data"]
meta = active["meta"]
budget = float(meta["budget"])
deadline = int(meta["deadline"])
pid = st.session_state.active_project

# =========================================================
# STEPPER DI ATAS
# =========================================================
stepper_html = "<div class='stepper'>"
for i, s in enumerate(STEPS):
    num = i
    if i == step:
        cls = "active"
    elif i > 0 and i < step:
        cls = "done"
    else:
        cls = ""
    stepper_html += (
        f"<div class='step-chip {cls}'>"
        f"<span class='num'>{num}</span>"
        f"<span>{s['label']}</span>"
        f"</div>"
    )
stepper_html += "</div>"
st.markdown(stepper_html, unsafe_allow_html=True)


def render_nav(show_next=True, next_label="Lanjut"):
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if step > 1:
            if st.button("← Sebelumnya", key=f"prev_{step}"):
                prev_step()
                st.rerun()
    with c3:
        if show_next and step < len(STEPS) - 1:
            if st.button(next_label + " →", key=f"next_{step}"):
                next_step()
                st.rerun()


# =========================================================
# STEP 1 — TENDER & BOQ
# =========================================================
if step == 1:
    st.markdown("## Tahap 1 — Tender & BOQ")

    st.markdown(
        "<div class='flow-hint'>"
        "Alur: Klien membuka tender → kita siapkan BOQ (Bill of Quantities) → "
        "vendor diundang untuk mengisi RAB & performa di tahap berikutnya."
        "</div>",
        unsafe_allow_html=True,
    )

    sub1, sub2 = st.tabs(["1. Dokumen & Timeline Tender", "2. BOQ — Bill of Quantities"])

    with sub1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Dokumen Tender")
            doc_drawing = st.checkbox("Gambar kerja", value=True, key="td_drawing")
            doc_rks = st.checkbox("RKS — Rencana Kerja & Syarat", value=True, key="td_rks")
            doc_spek = st.checkbox("Spesifikasi teknis", value=True, key="td_spek")
            doc_boq = st.checkbox("Bill of Quantities", value=True, key="td_boq_check")
            doc_k3 = st.checkbox("Dokumen K3 / HSE Plan", value=True, key="td_k3")

            st.markdown("#### Catatan Aanwijzing")
            catatan = st.text_area(
                "Ringkasan hasil aanwijzing",
                placeholder="Tulis ringkasan minimal 10 karakter...",
                height=120,
                key="td_catatan",
            )

        with col2:
            st.markdown("#### RAB Awal (Estimasi Kita)")
            st.markdown(
                f"<div class='hint-box'>"
                f"Budget project: <b>{format_rupiah(meta['budget'])}</b>."
                f"</div>",
                unsafe_allow_html=True,
            )
            rab_awal_input = st.number_input(
                "RAB Estimasi (juta Rp)",
                min_value=10.0,
                value=float(meta.get("rab_awal", meta.get("budget", 300.0))),
                step=10.0,
                key=f"rab_awal_{meta['id']}",
            )
            budget_input = st.number_input(
                "Budget approved (juta Rp)",
                min_value=10.0,
                value=float(meta.get("budget", 300.0)),
                step=10.0,
                key=f"budget_tender_{meta['id']}",
            )
            st.metric("Selisih", format_rupiah(budget_input - rab_awal_input))

            st.markdown("#### Timeline")
            tgl_aanwijzing = st.date_input("Aanwijzing", datetime(2024, 9, 7), key="td_tgl_aan")
            tgl_penawaran = st.date_input("Batas penawaran", datetime(2024, 9, 15), key="td_tgl_pen")
            tgl_kontrak = st.date_input("Target kontrak", datetime(2024, 9, 30), key="td_tgl_kon")

    with sub2:
        st.markdown("### BOQ — Bill of Quantities")
        st.caption("Daftar item pekerjaan yang akan dilelang ke vendor.")

        if active.get("items_data") is None:
            active["items_data"] = default_items().to_dict("records")

        items_df = pd.DataFrame(active["items_data"])
        items_edited = st.data_editor(
            items_df, use_container_width=True, hide_index=True,
            num_rows="dynamic", key=f"items_editor_{pid}",
            column_config={
                "No": st.column_config.NumberColumn("No", min_value=1, step=1, width="small"),
                "Item Pekerjaan": st.column_config.TextColumn("Item Pekerjaan", required=True, width="large"),
                "Volume": st.column_config.TextColumn("Volume", width="small"),
                "Satuan": st.column_config.TextColumn("Satuan", width="small"),
                "Spesifikasi": st.column_config.TextColumn("Spesifikasi"),
                "Kompleksitas": st.column_config.SelectboxColumn(
                    "Kompleksitas", options=["Rendah", "Sedang", "Tinggi"]
                ),
            },
        )
        active["items_data"] = items_edited.to_dict("records")

    dokumen_lengkap = all([doc_drawing, doc_rks, doc_spek, doc_boq, doc_k3])
    catatan_lengkap = len(catatan.strip()) >= 10
    timeline_valid = tgl_aanwijzing < tgl_penawaran < tgl_kontrak
    items_ok = active.get("items_data") and len(active["items_data"]) > 0
    form_valid = dokumen_lengkap and catatan_lengkap and timeline_valid and items_ok

    if not form_valid:
        missing = []
        if not dokumen_lengkap:
            missing.append("semua dokumen wajib dicentang")
        if not catatan_lengkap:
            missing.append("catatan minimal 10 karakter")
        if not timeline_valid:
            missing.append("urutan tanggal tidak valid")
        if not items_ok:
            missing.append("BOQ minimal 1 item")
        st.warning("Belum lengkap — " + "; ".join(missing))

    if st.button("Simpan Data Tender & BOQ", type="primary", key="btn_save_tender"):
        if not form_valid:
            st.error("Data belum lengkap.")
        else:
            meta["budget"] = float(budget_input)
            meta["rab_awal"] = float(rab_awal_input)
            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rab_awal": float(rab_awal_input),
                "budget": float(budget_input),
                "variance": float(budget_input - rab_awal_input),
                "dokumen": [d for d, v in [
                    ("Drawing", doc_drawing), ("RKS", doc_rks),
                    ("Spesifikasi", doc_spek), ("BoQ", doc_boq), ("K3", doc_k3),
                ] if v],
                "timeline": {
                    "aanwijzing": str(tgl_aanwijzing),
                    "penawaran": str(tgl_penawaran),
                    "kontrak": str(tgl_kontrak),
                },
                "catatan": catatan,
                "boq_count": len(active["items_data"]),
            }
            live_data["tender"] = payload
            st.success("Tersimpan. Lanjut ke Vendor Qualification...")
            next_step()
            st.rerun()

    render_nav(show_next=stage_status(active, "tender"),
               next_label="Lanjut ke Vendor Qualification")


# =========================================================
# STEP 2 — VENDOR QUALIFICATION
# =========================================================
elif step == 2:
    st.markdown("## Tahap 2 — Vendor Qualification & Selection")

    st.markdown(
        "<div class='flow-hint'>"
        "Alur: Vendor diundang → mereka isi RAB + kemampuan + performa → "
        "klarifikasi teknis → optimasi → deal harga & SPK."
        "</div>",
        unsafe_allow_html=True,
    )

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "1. BOQ & Vendor Diundang",
        "2. Input RAB + Kemampuan + Performa",
        "3. Appointment Klarifikasi Teknis",
        "4. Optimasi Setelah Klarifikasi",
        "5. Deal Harga & SPK",
    ])

    # SUB 1
    with sub1:
        st.markdown("### Item Pekerjaan (BOQ)")
        items_data = active.get("items_data", [])
        if not items_data:
            st.warning("BOQ belum diisi. Kembali ke Tahap 1.")
        else:
            st.dataframe(pd.DataFrame(items_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Vendor yang Diundang")
        if "invited_vendors" not in active or not active["invited_vendors"]:
            active["invited_vendors"] = INVITED_VENDORS.copy()

        invited_df = pd.DataFrame({"Vendor": active["invited_vendors"]})
        invited_edited = st.data_editor(
            invited_df, use_container_width=True, hide_index=True,
            num_rows="dynamic", key=f"invited_{pid}",
            column_config={"Vendor": st.column_config.TextColumn("Vendor", required=True)},
        )
        active["invited_vendors"] = invited_edited["Vendor"].dropna().tolist()

    # SUB 2
    with sub2:
        st.markdown("### Input Data Vendor")
        invited = active.get("invited_vendors", [])
        if not invited:
            st.warning("Belum ada vendor diundang.")
        else:
            if "vendor_data" not in active or not active["vendor_data"]:
                active["vendor_data"] = {}

            for v in invited:
                if v not in active["vendor_data"]:
                    active["vendor_data"][v] = {
                        "harga_rab": 0.0, "durasi": 0,
                        "kemampuan": {},
                        "performa": {aspek: 75 for aspek in PERFORMANCE_ASPECTS},
                        "catatan_vendor": "",
                    }

            vendor_pick = st.selectbox("Pilih vendor", options=invited, key=f"vp_{pid}")
            v_data = active["vendor_data"][vendor_pick]

            st.markdown(f"#### A. RAB & Durasi — {vendor_pick}")
            c1, c2 = st.columns(2)
            with c1:
                v_data["harga_rab"] = st.number_input(
                    "Harga RAB (juta Rp)", min_value=0.0,
                    value=float(v_data["harga_rab"]), step=5.0,
                    key=f"rab_{vendor_pick}",
                )
            with c2:
                v_data["durasi"] = st.number_input(
                    "Durasi (hari)", min_value=1,
                    value=int(v_data["durasi"]) if v_data["durasi"] > 0 else 60,
                    step=1, key=f"dur_{vendor_pick}",
                )

            st.markdown(f"#### B. Kemampuan Teknis per Item — {vendor_pick}")
            items_data = active.get("items_data", [])
            for item in items_data:
                item_no = str(item["No"])
                if item_no not in v_data["kemampuan"]:
                    v_data["kemampuan"][item_no] = {"status": "Sanggup", "catatan": ""}

                c1, c2, c3 = st.columns([2, 1, 2])
                with c1:
                    st.markdown(f"**{item['No']}. {item['Item Pekerjaan']}**")
                    st.caption(f"{item['Spesifikasi']} · {item['Volume']} {item['Satuan']}")
                with c2:
                    status_val = st.selectbox(
                        f"Status {item_no}",
                        ["Sanggup", "Sanggup dengan Catatan", "Tidak Sanggup"],
                        index=["Sanggup", "Sanggup dengan Catatan", "Tidak Sanggup"].index(
                            v_data["kemampuan"][item_no]["status"]
                        ),
                        key=f"st_{vendor_pick}_{item_no}",
                        label_visibility="collapsed",
                    )
                with c3:
                    catatan_val = st.text_input(
                        f"Catatan {item_no}",
                        value=v_data["kemampuan"][item_no]["catatan"],
                        key=f"ct_{vendor_pick}_{item_no}",
                        placeholder="Catatan",
                        label_visibility="collapsed",
                    )
                v_data["kemampuan"][item_no] = {"status": status_val, "catatan": catatan_val}

            st.markdown(f"#### C. Penilaian Performa — {vendor_pick}")
            c1, c2 = st.columns(2)
            for i, aspek in enumerate(PERFORMANCE_ASPECTS):
                target = c1 if i % 2 == 0 else c2
                with target:
                    v_data["performa"][aspek] = st.slider(
                        aspek, 0, 100, v_data["performa"][aspek],
                        key=f"pf_{vendor_pick}_{aspek}"
                    )

            st.markdown(f"#### D. Catatan — {vendor_pick}")
            v_data["catatan_vendor"] = st.text_area(
                "Catatan", value=v_data["catatan_vendor"], height=100,
                key=f"nt_{vendor_pick}",
            )

            active["vendor_data"][vendor_pick] = v_data

            st.markdown("---")
            st.markdown("#### Ringkasan")
            rows = []
            for v in invited:
                vd = active["vendor_data"][v]
                avg_perf = sum(vd["performa"].values()) / len(vd["performa"]) if vd["performa"] else 0
                sanggup = sum(1 for k, val in vd["kemampuan"].items() if val["status"] == "Sanggup")
                total = len(vd["kemampuan"]) if vd["kemampuan"] else 1
                rows.append({
                    "Vendor": v,
                    "RAB (jt)": vd["harga_rab"],
                    "Durasi": vd["durasi"],
                    "Kemampuan": f"{sanggup}/{total}",
                    "Skor Performa": round(avg_perf, 1),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # SUB 3
    with sub3:
        st.markdown("### Appointment Klarifikasi Teknis")
        invited = active.get("invited_vendors", [])
        if not invited:
            st.warning("Belum ada vendor.")
        else:
            with st.expander("➕ Buat Appointment", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    appt_vendor = st.selectbox("Vendor", options=invited, key=f"apv_{pid}")
                    appt_mode = st.radio("Mode", ["Tatap muka (Kantor)", "Tatap muka (On-site)", "Zoom Meeting"], key=f"apm_{pid}")
                    appt_date = st.date_input("Tanggal", date.today(), key=f"apd_{pid}")
                    appt_time = st.time_input("Jam", time(9, 0), key=f"apt_{pid}")
                    appt_duration = st.number_input("Durasi (menit)", 30, 480, 90, 15, key=f"apdur_{pid}")
                with c2:
                    appt_location = st.text_input("Lokasi / Link Zoom", placeholder="Meeting Room A", key=f"apl_{pid}")
                    appt_participants = st.text_area("Peserta kami", height=80, key=f"app_{pid}")
                    appt_vendor_participants = st.text_area("Peserta vendor", height=80, key=f"apvp_{pid}")
                    appt_agenda = st.text_area("Agenda", placeholder="Klarifikasi WPS, NDT, hydrotest...", height=100, key=f"apa_{pid}")

                if st.button("Simpan Appointment", type="primary", key=f"btna_{pid}"):
                    if not appt_location.strip():
                        st.error("Lokasi wajib diisi.")
                    elif len(appt_agenda.strip()) < 10:
                        st.error("Agenda minimal 10 karakter.")
                    else:
                        active["appointments"].append({
                            "id": str(uuid.uuid4())[:6],
                            "vendor": appt_vendor, "mode": appt_mode,
                            "date": str(appt_date), "time": str(appt_time),
                            "duration": int(appt_duration),
                            "location": appt_location,
                            "participants": appt_participants,
                            "vendor_participants": appt_vendor_participants,
                            "agenda": appt_agenda, "status": "Terjadwal",
                            "hasil": "",
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        st.success("Appointment dibuat.")
                        st.rerun()

            st.markdown("---")
            if not active["appointments"]:
                st.info("Belum ada appointment.")
            else:
                for i, appt in enumerate(active["appointments"]):
                    with st.expander(
                        f"[{appt['status']}] {appt['vendor']} · {appt['mode']} · {appt['date']} {appt['time']}",
                        expanded=(i == 0),
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Vendor:** {appt['vendor']}")
                            st.markdown(f"**Mode:** {appt['mode']}")
                            st.markdown(f"**Waktu:** {appt['date']} · {appt['time']} ({appt['duration']} menit)")
                            st.markdown(f"**Lokasi:** {appt['location']}")
                        with c2:
                            st.markdown(f"**Peserta kami:** {appt['participants'] or '-'}")
                            st.markdown(f"**Peserta vendor:** {appt['vendor_participants'] or '-'}")
                            st.markdown(f"**Agenda:** {appt['agenda']}")
                        st.markdown("---")
                        new_status = st.selectbox(
                            "Status",
                            ["Terjadwal", "Berlangsung", "Selesai", "Dibatalkan"],
                            index=["Terjadwal", "Berlangsung", "Selesai", "Dibatalkan"].index(appt["status"]),
                            key=f"as_{appt['id']}",
                        )
                        hasil = st.text_area("Hasil klarifikasi", value=appt.get("hasil", ""),
                                             height=120, key=f"ah_{appt['id']}")
                        cs, cd = st.columns([3, 1])
                        with cs:
                            if st.button("Simpan", key=f"sv_{appt['id']}"):
                                active["appointments"][i]["status"] = new_status
                                active["appointments"][i]["hasil"] = hasil
                                st.success("Tersimpan.")
                                st.rerun()
                        with cd:
                            if st.button("Hapus", key=f"dl_{appt['id']}"):
                                active["appointments"].pop(i)
                                st.rerun()

    # SUB 4
    with sub4:
        st.markdown("### Optimasi Setelah Klarifikasi")
        invited = active.get("invited_vendors", [])
        vendor_data = active.get("vendor_data", {})

        prasyarat = []
        if not invited:
            prasyarat.append("vendor diundang")
        if not vendor_data or all(v["harga_rab"] == 0 for v in vendor_data.values()):
            prasyarat.append("RAB vendor")
        if not active["appointments"]:
            prasyarat.append("appointment klarifikasi")

        if prasyarat:
            st.warning("Belum bisa optimasi. Selesaikan: " + ", ".join(prasyarat))
        else:
            st.caption(
                "Optimasi pakai **normalisasi min-max** untuk harga + skor performa. "
                "Bobot bisa diubah di sidebar."
            )

            scores = {}
            harga = {}
            for v in invited:
                vd = vendor_data.get(v, {})
                perf_values = list(vd.get("performa", {}).values())
                avg_perf = sum(perf_values) / len(perf_values) if perf_values else 0
                scores[v] = avg_perf
                harga[v] = vd.get("harga_rab", 0)

            st.markdown("#### Data untuk Optimasi")
            rows = []
            for v in invited:
                rows.append({
                    "Vendor": v,
                    "RAB (juta Rp)": format_rupiah(harga[v]),
                    "Skor Performa": round(scores[v], 1),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            eligible = [
                v for v in invited
                if harga[v] > 0 and harga[v] <= budget and scores[v] >= min_performance
            ]

            if not eligible:
                st.error("Tidak ada vendor yang lolos kriteria.")
            else:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Budget", format_rupiah(budget))
                cc2.metric("Min performa", f"{min_performance}")
                cc3.metric("Bobot harga", f"{int(weight_price*100)}%")
                cc4.metric("Bobot performa", f"{int(weight_perf*100)}%")

                st.markdown("---")
                selected_name, solver_status = solve_vendor_lp(
                    eligible, harga, budget, scores, weight_price, weight_perf
                )

                if selected_name is None:
                    st.error(f"Optimasi gagal: {solver_status}")
                else:
                    selected_cost = harga[selected_name]
                    selected_score = scores[selected_name]

                    st.success(f"**Vendor terbaik: {selected_name}**")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Vendor", selected_name)
                    r2.metric("Harga RAB", format_rupiah(selected_cost))
                    r3.metric("Skor Performa", f"{selected_score:.1f}")
                    r4.metric("Sisa budget", format_rupiah(budget - selected_cost))

                    chart_df = pd.DataFrame({
                        "Vendor": eligible,
                        "Harga": [harga[v] for v in eligible],
                        "Performa": [scores[v] for v in eligible],
                    }).sort_values("Harga")

                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    x_pos = np.arange(len(chart_df))
                    colors = ["#047857" if v == selected_name else "#1e40af" for v in chart_df["Vendor"]]
                    bars = ax1.bar(x_pos, chart_df["Harga"], color=colors, width=0.55, label="Harga RAB")
                    for bar, val in zip(bars, chart_df["Harga"]):
                        ax1.text(bar.get_x() + bar.get_width()/2, val + 5,
                                 f"Rp {val:,.0f} jt", ha="center", fontsize=8.5, color="#374151")
                    ax1.axhline(budget, linestyle="--", color="#dc2626", linewidth=1.5,
                                label=f"Budget: {format_rupiah(budget)}")
                    ax1.set_xticks(x_pos)
                    ax1.set_xticklabels(chart_df["Vendor"], rotation=15, ha="right", fontsize=9)
                    ax1.set_ylabel("Harga RAB (juta Rp)", fontsize=9)
                    ax1.tick_params(labelsize=9)
                    ax1.legend(fontsize=9, loc="upper left")
                    ax1.grid(axis="y", alpha=0.25)
                    ax1.spines["top"].set_visible(False)
                    ax1.spines["right"].set_visible(False)

                    ax2 = ax1.twinx()
                    ax2.plot(x_pos, chart_df["Performa"], color="#f59e0b",
                             linewidth=2, marker="o", markersize=8, label="Skor Performa")
                    ax2.set_ylabel("Skor Performa", fontsize=9, color="#f59e0b")
                    ax2.tick_params(axis="y", labelcolor="#f59e0b", labelsize=9)
                    ax2.set_ylim(0, 105)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.session_state[f"optimized_vendor_{pid}"] = selected_name
                    st.session_state[f"optimized_cost_{pid}"] = selected_cost
                    st.session_state[f"optimized_score_{pid}"] = selected_score

    # SUB 5
    with sub5:
        st.markdown("### Deal Harga & SPK")
        optimized_vendor = st.session_state.get(f"optimized_vendor_{pid}")
        optimized_cost = st.session_state.get(f"optimized_cost_{pid}", 0)

        if not optimized_vendor:
            st.warning("Selesaikan optimasi dulu (sub-tab 4).")
        else:
            st.markdown(
                f"<div class='hint-box'>"
                f"<b>Vendor:</b> {optimized_vendor}<br>"
                f"<b>Harga RAB awal:</b> {format_rupiah(optimized_cost)}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                final_price = st.number_input(
                    "Harga kesepakatan (juta Rp)",
                    min_value=1.0, value=float(optimized_cost), step=5.0,
                    key=f"fp_{pid}",
                )
            with c2:
                saving = optimized_cost - final_price
                st.metric("Penghematan", format_rupiah(saving))

            st.markdown("#### SPK")
            c1, c2, c3 = st.columns(3)
            with c1:
                spk_number = st.text_input("Nomor SPK", placeholder="SPK/KPOP/2024/001", key=f"spkn_{pid}")
            with c2:
                spk_date = st.date_input("Tanggal SPK", date.today(), key=f"spkd_{pid}")
            with c3:
                spk_signer = st.text_input("Ditandatangani oleh", placeholder="Nama PM", key=f"spks_{pid}")

            approval_drawing = st.radio(
                "Status Approval Drawing Vendor",
                ["Belum", "Diajukan", "Disetujui", "Revisi"],
                index=0, key=f"vsa_{pid}",
            )

            negosiasi_notes = st.text_area(
                "Catatan Negosiasi", height=100, key=f"ngn_{pid}",
            )

            valid = (
                spk_number and len(spk_number.strip()) >= 5
                and spk_signer and len(spk_signer.strip()) >= 3
                and approval_drawing == "Disetujui"
                and final_price > 0
            )

            if not valid:
                missing = []
                if not (spk_number and len(spk_number.strip()) >= 5):
                    missing.append("nomor SPK min 5 karakter")
                if not (spk_signer and len(spk_signer.strip()) >= 3):
                    missing.append("penandatangan SPK")
                if approval_drawing != "Disetujui":
                    missing.append("approval drawing 'Disetujui'")
                st.warning("Belum lengkap — " + "; ".join(missing))

            if st.button("Simpan Keputusan & SPK", type="primary", key="btn_vs"):
                if not valid:
                    st.error("Data belum lengkap.")
                else:
                    vd = vendor_data.get(optimized_vendor, {})
                    payload = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "selected_vendor": optimized_vendor,
                        "rab_awal_vendor": float(optimized_cost),
                        "selected_cost": float(final_price),
                        "saving": float(saving),
                        "selected_duration": int(vd.get("durasi", 0)),
                        "selected_performance": float(st.session_state.get(f"optimized_score_{pid}", 0)),
                        "spk_number": spk_number,
                        "spk_date": str(spk_date),
                        "spk_signer": spk_signer,
                        "approval_drawing": approval_drawing,
                        "negosiasi_notes": negosiasi_notes,
                        "appointment_count": len(active["appointments"]),
                        "invited_vendors": len(active.get("invited_vendors", [])),
                    }
                    live_data["vendor_selection"] = payload
                    st.success("Keputusan & SPK tersimpan. Lanjut ke Site Execution...")
                    next_step()
                    st.rerun()

    render_nav(show_next=stage_status(active, "vendor_selection"),
               next_label="Lanjut ke Site Execution")


# =========================================================
# STEP 3 — SITE EXECUTION (dengan S-CURVE yang benar)
# =========================================================
elif step == 3:
    st.markdown("## Tahap 3 — Site Execution")

    st.markdown(
        "<div class='flow-hint'>"
        "Alur: PCM → Mobilisasi → Konstruksi → Monitoring S-Curve → "
        "Analisa kendala → Evaluasi KPI kontraktor."
        "</div>",
        unsafe_allow_html=True,
    )

    vs = live_data.get("vendor_selection", {})
    selected_vendor = vs.get("selected_vendor", "-")
    selected_duration = vs.get("selected_duration", 78)
    final_cost = vs.get("selected_cost", 0)

    st.caption(
        f"Kontraktor: {selected_vendor} · Nilai kontrak: {format_rupiah(final_cost)} · "
        f"Durasi: {selected_duration} hari"
    )

    st.markdown("### Template WBS")
    st.caption(
        "Isi tabel item pekerjaan. **S-Curve akan dihitung pakai kernel S (sigmoid)** "
        "per item — bukan linear. Hasilnya kurva S asli: landai → curam → landai."
    )

    if active.get("wbs_data") is None:
        active["wbs_data"] = default_wbs().to_dict("records")

    wbs_df = pd.DataFrame(active["wbs_data"])
    wbs_edited = st.data_editor(
        wbs_df, use_container_width=True, hide_index=True,
        num_rows="dynamic", key=f"wbs_editor_{pid}",
        column_config={
            "No": st.column_config.NumberColumn("No", min_value=1, step=1, width="small"),
            "Item Pekerjaan": st.column_config.TextColumn("Item Pekerjaan", required=True, width="large"),
            "Bobot (%)": st.column_config.NumberColumn("Bobot (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
            "Mulai (minggu)": st.column_config.NumberColumn("Mulai (minggu)", min_value=1, step=1, width="small"),
            "Selesai (minggu)": st.column_config.NumberColumn("Selesai (minggu)", min_value=1, step=1, width="small"),
            "Realisasi (%)": st.column_config.NumberColumn("Realisasi (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        },
    )
    active["wbs_data"] = wbs_edited.to_dict("records")

    if not wbs_edited.empty:
        total_bobot = float(wbs_edited["Bobot (%)"].sum())
        max_week = int(wbs_edited["Selesai (minggu)"].max())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Bobot", f"{total_bobot:.1f}%",
                  delta=f"{total_bobot - 100:+.1f}%" if abs(total_bobot - 100) > 0.1 else "OK")
        c2.metric("Jumlah Item", len(wbs_edited))
        c3.metric("Durasi", f"{max_week} minggu")

        if abs(total_bobot - 100) > 0.1:
            st.warning(f"Total bobot = {total_bobot:.1f}%. Sebaiknya tepat 100%.")

        invalid_rows = wbs_edited[wbs_edited["Mulai (minggu)"] > wbs_edited["Selesai (minggu)"]]
        if invalid_rows.empty:
            current_week = st.number_input(
                "Minggu saat ini",
                min_value=1, max_value=max_week,
                value=min(max_week, max(1, int(max_week * 0.65))),
                step=1, key=f"current_week_{pid}",
            )

            weeks = np.arange(1, max_week + 1)

            # =========================================================
            # PERHITUNGAN S-CURVE (pakai kernel S / sigmoid per item)
            # =========================================================
            plan_weekly = np.zeros(max_week)

            for _, row in wbs_edited.iterrows():
                w_start = int(row["Mulai (minggu)"])
                w_end = int(row["Selesai (minggu)"])
                bobot = float(row["Bobot (%)"])
                span = w_end - w_start + 1
                if span <= 0 or bobot <= 0:
                    continue

                # Bangun kurva S untuk item ini: dari S(t=0)=0 sampai S(t=1)=bobot
                # Nilai S per minggu dihitung dari selisih sigmoid antar minggu
                ts = np.linspace(0, 1, span + 1)
                s_values = np.array([sigmoid_kernel(t, k=4.0) for t in ts])
                # Bobot per minggu = selisih kumulatif (deferensial sigmoid)
                weekly_weights = np.diff(s_values) * bobot
                # Normalisasi agar total = bobot (untuk amannya)
                if weekly_weights.sum() > 0:
                    weekly_weights = weekly_weights / weekly_weights.sum() * bobot

                plan_weekly[w_start - 1:w_end] += weekly_weights

            plan_cum = np.cumsum(plan_weekly)

            # Realisasi
            actual_weekly = np.zeros(max_week)
            for _, row in wbs_edited.iterrows():
                w_start = int(row["Mulai (minggu)"])
                w_end = int(row["Selesai (minggu)"])
                bobot = float(row["Bobot (%)"])
                realisasi_pct = float(row["Realisasi (%)"])
                if realisasi_pct <= 0:
                    continue

                target = bobot * (realisasi_pct / 100.0)

                # Batasi sampai current_week
                w_actual_end = min(current_week, w_end)
                if w_actual_end < w_start:
                    continue

                # Distribusi aktual dengan kernel S yang sama
                ts = np.linspace(0, 1, (w_actual_end - w_start + 1) + 1)
                s_values = np.array([sigmoid_kernel(t, k=4.0) for t in ts])
                weekly_weights = np.diff(s_values) * target
                if weekly_weights.sum() > 0:
                    weekly_weights = weekly_weights / weekly_weights.sum() * target

                actual_weekly[w_start - 1:w_actual_end] += weekly_weights

            actual_cum = np.cumsum(actual_weekly)

            # =========================================================
            # PLOT S-CURVE
            # =========================================================
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.fill_between(weeks, plan_cum, actual_cum,
                            where=(actual_cum < plan_cum),
                            color="#dc2626", alpha=0.12, label="Di bawah rencana")
            ax.fill_between(weeks, plan_cum, actual_cum,
                            where=(actual_cum >= plan_cum),
                            color="#047857", alpha=0.12, label="Di atas rencana")

            ax.plot(weeks, plan_cum, color="#1e40af", linewidth=2.5,
                    marker="o", markersize=5, label="Rencana (S-Curve)")
            ax.plot(weeks, actual_cum, color="#dc2626", linewidth=2.5,
                    marker="s", markersize=5, label="Realisasi")

            ax.axvline(current_week, linestyle=":", color="#6b7280",
                       linewidth=1.5, label=f"Minggu ini (W{current_week})")

            plan_now = plan_cum[current_week - 1]
            actual_now = actual_cum[current_week - 1]

            ax.scatter([current_week], [plan_now], color="#1e40af", s=80,
                       zorder=5, edgecolor="white", linewidth=1.5)
            ax.scatter([current_week], [actual_now], color="#dc2626", s=80,
                       marker="D", zorder=5, edgecolor="white", linewidth=1.5)

            ax.annotate(f"Rencana: {plan_now:.1f}%",
                        xy=(current_week, plan_now), xytext=(10, 10),
                        textcoords="offset points", color="#1e40af",
                        fontsize=9, fontweight="600")
            ax.annotate(f"Realisasi: {actual_now:.1f}%",
                        xy=(current_week, actual_now), xytext=(10, -20),
                        textcoords="offset points", color="#dc2626",
                        fontsize=9, fontweight="600")

            ax.set_xlabel("Minggu ke-", fontsize=10)
            ax.set_ylabel("Progress Kumulatif (%)", fontsize=10)
            ax.set_title(f"S-Curve — {selected_vendor}", fontsize=12, fontweight="600")
            ax.set_ylim(0, 105)
            ax.set_xlim(1, max_week)
            ax.set_xticks(weeks)
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

            deviasi = actual_now - plan_now
            spi = actual_now / plan_now if plan_now > 0 else 0

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric(f"Rencana (W{current_week})", f"{plan_now:.1f}%")
            sc2.metric(f"Realisasi (W{current_week})", f"{actual_now:.1f}%")
            sc3.metric("Deviasi", f"{deviasi:+.1f}%")
            sc4.metric("SPI", f"{spi:.3f}")

            if deviasi < -5:
                st.error(f"Deviasi {deviasi:.1f}% — proyek berpotensi terlambat.")
            elif deviasi < 0:
                st.warning(f"Deviasi {deviasi:.1f}% — perlu perhatian.")
            else:
                st.success(f"Progress sesuai atau lebih cepat {deviasi:+.1f}%.")

            with st.expander("Lihat tabel S-Curve"):
                df_sc = pd.DataFrame({
                    "Minggu": weeks,
                    "Rencana (%)": np.round(plan_cum, 2),
                    "Realisasi (%)": np.round(actual_cum, 2),
                    "Deviasi (%)": np.round(actual_cum - plan_cum, 2),
                })
                st.dataframe(df_sc, use_container_width=True, hide_index=True)
                st.download_button(
                    "Unduh S-Curve (CSV)",
                    data=df_sc.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"s_curve_{meta['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Pre Construction Meeting")
                pcm_date = st.date_input("Tanggal PCM", datetime(2024, 10, 1), key="se_pcm_date")
                pcm_notes = st.text_area("Catatan PCM", placeholder="Minimal 10 karakter...", height=100, key="se_pcm_notes")
            with col2:
                st.markdown("#### Analisa Kendala")
                kendala = st.selectbox(
                    "Kendala utama",
                    ["Pilih kendala...", "Tidak ada", "Perubahan desain",
                     "Keterlambatan material", "Cuaca ekstrem", "Akses terbatas"],
                    key="se_kendala",
                )
                if kendala == "Tidak ada":
                    st.info("Tidak ada kendala.")
                    aksi = "Tidak diperlukan"
                    aksi_valid = True
                elif kendala == "Pilih kendala...":
                    aksi = None
                    aksi_valid = False
                else:
                    aksi = st.radio("Langkah penanganan", [
                        "Pilih aksi...", "Evaluasi dengan klien", "Revisi gambar kerja",
                        "Pengukuran ulang aktual", "Konsultasi GSB & GSJ",
                    ], key="se_aksi")
                    aksi_valid = aksi != "Pilih aksi..."
                kpi = st.slider("KPI kontraktor (%)", 0, 100, 0, key="se_kpi")

            pcm_valid = len(pcm_notes.strip()) >= 10
            kendala_valid = kendala != "Pilih kendala..."
            kpi_valid = kpi > 0
            form_valid = pcm_valid and kendala_valid and aksi_valid and kpi_valid

            if not form_valid:
                missing = []
                if not pcm_valid:
                    missing.append("catatan PCM min 10 karakter")
                if not kendala_valid:
                    missing.append("kendala belum dipilih")
                if not aksi_valid:
                    missing.append("aksi belum dipilih")
                if not kpi_valid:
                    missing.append("KPI harus > 0%")
                st.warning("Belum lengkap — " + "; ".join(missing))

            if st.button("Simpan Data Site Execution", type="primary", key="btn_se"):
                if not form_valid:
                    st.error("Data belum lengkap.")
                else:
                    payload = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "vendor": selected_vendor,
                        "current_week": current_week,
                        "plan_pct": float(plan_now),
                        "actual_pct": float(actual_now),
                        "deviation_pct": float(deviasi),
                        "spi": float(spi),
                        "pcm_date": str(pcm_date),
                        "pcm_notes": pcm_notes,
                        "kendala": kendala,
                        "aksi": aksi,
                        "kpi": kpi,
                        "wbs_count": len(wbs_edited),
                    }
                    live_data["site_execution"] = payload
                    st.success("Site Execution tersimpan. Lanjut ke Quality Control...")
                    next_step()
                    st.rerun()

    render_nav(show_next=stage_status(active, "site_execution"),
               next_label="Lanjut ke Quality Control")


# =========================================================
# STEP 4 — QUALITY CONTROL
# =========================================================
elif step == 4:
    st.markdown("## Tahap 4 — Quality Control & Handover")

    st.markdown(
        "<div class='flow-hint'>"
        "Alur: Pemberitahuan selesai → Inspeksi → Commissioning → BAST → Selesai."
        "</div>",
        unsafe_allow_html=True,
    )

    live = live_data.get("quality_control", {})
    if live:
        c1, c2 = st.columns(2)
        c1.metric("Status akhir", live.get("final_status", "-"))
        c2.metric("BAST", live.get("bast_number", "-"))
        st.caption(f"Terakhir disimpan: {live.get('timestamp', '-')}")
        st.markdown("---")

    st.markdown("#### Inspeksi Mutu")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mat_insp = st.selectbox("Material", ["Pilih...", "PASS", "FAIL"], key="qc_mat")
    with c2:
        ndt = st.selectbox("NDT / Weld", ["Pilih...", "PASS", "FAIL"], key="qc_ndt")
    with c3:
        hydro = st.selectbox("Hydrotest", ["Pilih...", "PASS", "FAIL"], key="qc_hydro")
    with c4:
        visual = st.selectbox("Visual", ["Pilih...", "PASS", "FAIL"], key="qc_visual")

    st.markdown("#### Inspeksi Akhir")
    final_status = st.radio(
        "Sesuai kontrak?",
        ["Pilih...", "Ya", "Tidak"], index=0, key="qc_final"
    )

    if final_status == "Ya":
        st.markdown("#### BAST")
        c1, c2, c3 = st.columns(3)
        with c1:
            bast_date = st.date_input("Tanggal BAST", datetime(2024, 12, 20), key="qc_bd")
        with c2:
            bast_number = st.text_input("Nomor BAST", placeholder="BAST/KPOP/2024/047", key="qc_bn")
        with c3:
            approved_by = st.text_input("Disetujui oleh", placeholder="Nama lengkap", key="qc_ab")
        bast_notes = st.text_area("Catatan BAST", placeholder="Minimal 10 karakter...", height=80, key="qc_bnotes")
    else:
        bast_date = None
        bast_number = ""
        approved_by = ""
        bast_notes = ""

    inspeksi_valid = all(x in ["PASS", "FAIL"] for x in [mat_insp, ndt, hydro, visual])
    status_valid = final_status in ["Ya", "Tidak"]

    if status_valid and final_status == "Ya":
        bast_valid = (
            bast_number and len(bast_number.strip()) >= 5
            and approved_by and len(approved_by.strip()) >= 3
            and bast_notes and len(bast_notes.strip()) >= 10
        )
    else:
        bast_valid = True

    form_valid = inspeksi_valid and status_valid and bast_valid

    if not form_valid:
        missing = []
        if not inspeksi_valid:
            missing.append("semua inspeksi wajib dipilih")
        if not status_valid:
            missing.append("status akhir belum dipilih")
        if not bast_valid:
            missing.append("data BAST belum lengkap")
        st.warning("Belum lengkap — " + "; ".join(missing))

    if st.button("Simpan Data Quality Control", type="primary", key="btn_qc"):
        if not form_valid:
            st.error("Data belum lengkap.")
        else:
            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "material_inspection": mat_insp,
                "ndt": ndt,
                "hydrotest": hydro,
                "visual": visual,
                "final_status": final_status,
                "bast_date": str(bast_date) if bast_date else None,
                "bast_number": bast_number,
                "approved_by": approved_by,
                "bast_notes": bast_notes,
            }
            live_data["quality_control"] = payload
            st.success("Selesai. Project diselesaikan.")
            next_step()
            st.rerun()

    render_nav(show_next=stage_status(active, "quality_control"),
               next_label="Lihat Ringkasan")


# =========================================================
# STEP 5 — SELESAI
# =========================================================
elif step == 5:
    st.markdown("## Project Selesai")

    tender = live_data.get("tender", {})
    vs = live_data.get("vendor_selection", {})
    se = live_data.get("site_execution", {})
    qc = live_data.get("quality_control", {})

    if not qc or qc.get("final_status") != "Ya":
        st.warning("Project belum selesai.")
    else:
        st.success(f"Project **{meta['name']}** selesai.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Info Project**")
            st.markdown(f"- Nama: {meta['name']}")
            st.markdown(f"- Klien: {meta['client']}")
            st.markdown(f"- Lokasi: {meta['location']}")
            st.markdown(f"- Budget: {format_rupiah(meta['budget'])}")
            st.markdown(f"- RAB Awal: {format_rupiah(meta.get('rab_awal', 0))}")

        with c2:
            st.markdown("**Hasil**")
            if vs:
                st.markdown(f"- Vendor: {vs.get('selected_vendor', '-')}")
                st.markdown(f"- SPK: {format_rupiah(vs.get('selected_cost', 0))}")
                st.markdown(f"- Penghematan: {format_rupiah(vs.get('saving', 0))}")
            if se:
                st.markdown(f"- Realisasi: {se.get('actual_pct', 0):.1f}%")
                st.markdown(f"- SPI: {se.get('spi', 0):.3f}")
            if qc:
                st.markdown(f"- BAST: {qc.get('bast_number', '-')}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            all_data = {"tender": tender, "vendor_selection": vs,
                        "site_execution": se, "quality_control": qc}
            rows = []
            for tahap, d in all_data.items():
                if d:
                    row = {"Tahap": label(tahap)}
                    row.update({k: str(v) for k, v in d.items()})
                    rows.append(row)
            st.download_button(
                "Unduh data (CSV)",
                data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"project_{meta['name'].replace(' ', '_')}.csv",
                mime="text/csv",
            )
        with c2:
            if active["appointments"]:
                st.download_button(
                    "Unduh Appointment (CSV)",
                    data=pd.DataFrame(active["appointments"]).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"appointments_{meta['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Kembali ke QC", key="btn_bq"):
            goto_step(4)
            st.rerun()
    with c2:
        if st.button("Kembali ke Daftar Project", key="btn_bl"):
            st.session_state.step = 0
            st.rerun()

st.markdown("---")
st.caption("Vendor Selection & Project Control · Multi-project workspace · Data simulasi.")
