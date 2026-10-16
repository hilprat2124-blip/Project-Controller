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
    .sub-section {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .sub-section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f3f4f6;
    }
    .appointment-card {
        background: #eff6ff;
        border-left: 4px solid #1e40af;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .appointment-meta {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 4px;
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


def solve_vendor_lp(vendor_names, costs, budget):
    try:
        prob = pulp.LpProblem("Vendor_Selection", pulp.LpMinimize)
        x = _create_binary_vars(prob, "select", vendor_names)
        prob += pulp.lpSum(costs[n] * x[n] for n in vendor_names)
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


# URUTAN BARU: Tender → Vendor Qualification & Selection → Site Execution → QC → Selesai
STEPS = [
    {"key": "project_list", "label": "Daftar Project"},
    {"key": "tender", "label": "Tender"},
    {"key": "vendor_selection", "label": "Vendor Qualification & Selection"},
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
        "items_data": None,       # Item pekerjaan untuk klarifikasi teknis
        "appointments": [],       # Daftar appointment klarifikasi teknis
    }
    return pid


def get_active():
    if st.session_state.active_project is None:
        return None
    return st.session_state.projects.get(st.session_state.active_project)


def label(tahap):
    return {
        "tender": "Tender",
        "vendor_selection": "Vendor Qualification & Selection",
        "site_execution": "Site Execution",
        "quality_control": "Quality Control",
    }.get(tahap, tahap)


def stage_status(project, tahap):
    return bool(project["live_data"].get(tahap))


def project_progress(project):
    stages = ["tender", "vendor_selection", "site_execution", "quality_control"]
    return sum(1 for t in stages if stage_status(project, t)), len(stages)


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
        {"No": 7, "Item Pekerjaan": "Testing & Commissioning", "Bobot (%)": 5.0,
         "Mulai (minggu)": 12, "Selesai (minggu)": 15, "Realisasi (%)": 0.0},
        {"No": 8, "Item Pekerjaan": "Demobilisasi & Handover", "Bobot (%)": 2.0,
         "Mulai (minggu)": 14, "Selesai (minggu)": 15, "Realisasi (%)": 0.0},
    ])


def default_items():
    """Template item pekerjaan untuk klarifikasi teknis."""
    return pd.DataFrame([
        {"No": 1, "Item Pekerjaan": "Fabrikasi Pipa Carbon Steel", "Volume": "500 m",
         "Spesifikasi": "ASTM A106 Gr.B", "Kompleksitas": "Sedang"},
        {"No": 2, "Item Pekerjaan": "Welding SMAW + GTAW", "Volume": "300 joint",
         "Spesifikasi": "WPS sesuai ASME IX", "Kompleksitas": "Tinggi"},
        {"No": 3, "Item Pekerjaan": "NDT Radiografi", "Volume": "150 titik",
         "Spesifikasi": "ASME V Article 2", "Kompleksitas": "Tinggi"},
        {"No": 4, "Item Pekerjaan": "Hydrotest", "Volume": "3 line",
         "Spesifikasi": "ASME B31.3", "Kompleksitas": "Sedang"},
        {"No": 5, "Item Pekerjaan": "Support & Hanger Installation", "Volume": "80 unit",
         "Spesifikasi": "MSS SP-58", "Kompleksitas": "Rendah"},
        {"No": 6, "Item Pekerjaan": "Painting & Coating", "Volume": "800 m²",
         "Spesifikasi": "SSPC-SP10", "Kompleksitas": "Rendah"},
    ])


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
        help="Budget yang di-approve. Nilai RAB awal default = budget."
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
        min_performance = st.slider("Skor performa minimum", 0, 100, 75, key="min_perf")
        require_technical = st.checkbox("Wajib lolos teknis", value=True, key="req_tech")

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
        min_quality, min_hse, min_performance, require_technical = 82, 85, 75, True

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

# =========================================================
# VENDOR DATA — dengan kolom performa tambahan
# =========================================================
DEFAULT_VENDORS = pd.DataFrame([
    {"Vendor": "PT Baja Nusantara Konstruksi", "Harga (juta Rp)": 285.0, "Kualitas (0-100)": 91,
     "Durasi (hari)": 78, "Skor K3 (0-100)": 92, "Lolos teknis": True,
     "Pengalaman (tahun)": 18, "Karyawan": 240, "Proyek Selesai": 87,
     "Tingkat Keberhasilan (%)": 94, "Rata-rata Keterlambatan (hari)": 3,
     "Sertifikasi": "ISO 9001, ISO 45001, ASME", "Lokasi": "Jakarta"},
    {"Vendor": "CV Prima Pipa Sejahtera", "Harga (juta Rp)": 245.0, "Kualitas (0-100)": 78,
     "Durasi (hari)": 65, "Skor K3 (0-100)": 84, "Lolos teknis": True,
     "Pengalaman (tahun)": 9, "Karyawan": 120, "Proyek Selesai": 34,
     "Tingkat Keberhasilan (%)": 82, "Rata-rata Keterlambatan (hari)": 6,
     "Sertifikasi": "ISO 9001", "Lokasi": "Surabaya"},
    {"Vendor": "PT Karya Logam Presisi", "Harga (juta Rp)": 310.0, "Kualitas (0-100)": 95,
     "Durasi (hari)": 85, "Skor K3 (0-100)": 94, "Lolos teknis": True,
     "Pengalaman (tahun)": 22, "Karyawan": 310, "Proyek Selesai": 112,
     "Tingkat Keberhasilan (%)": 96, "Rata-rata Keterlambatan (hari)": 2,
     "Sertifikasi": "ISO 9001, ISO 45001, ASME, API", "Lokasi": "Cilegon"},
    {"Vendor": "PT Sinergi Teknik Mandiri", "Harga (juta Rp)": 220.0, "Kualitas (0-100)": 82,
     "Durasi (hari)": 92, "Skor K3 (0-100)": 80, "Lolos teknis": True,
     "Pengalaman (tahun)": 12, "Karyawan": 180, "Proyek Selesai": 56,
     "Tingkat Keberhasilan (%)": 88, "Rata-rata Keterlambatan (hari)": 5,
     "Sertifikasi": "ISO 9001", "Lokasi": "Bandung"},
    {"Vendor": "PT Anugerah Pipa Internasional", "Harga (juta Rp)": 340.0, "Kualitas (0-100)": 97,
     "Durasi (hari)": 70, "Skor K3 (0-100)": 96, "Lolos teknis": True,
     "Pengalaman (tahun)": 25, "Karyawan": 420, "Proyek Selesai": 156,
     "Tingkat Keberhasilan (%)": 98, "Rata-rata Keterlambatan (hari)": 1,
     "Sertifikasi": "ISO 9001, ISO 45001, ASME, API, DNV", "Lokasi": "Batam"},
    {"Vendor": "CV Mandiri Baja Perkasa", "Harga (juta Rp)": 195.0, "Kualitas (0-100)": 74,
     "Durasi (hari)": 88, "Skor K3 (0-100)": 76, "Lolos teknis": False,
     "Pengalaman (tahun)": 6, "Karyawan": 85, "Proyek Selesai": 18,
     "Tingkat Keberhasilan (%)": 72, "Rata-rata Keterlambatan (hari)": 9,
     "Sertifikasi": "ISO 9001", "Lokasi": "Semarang"},
])

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


# =========================================================
# JIKA BELUM ADA PROJECT DAN STEP > 0
# =========================================================
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
# STEP 1 — TENDER
# =========================================================
if step == 1:
    st.markdown("## Tahap 1 — Client & Tender")

    st.markdown(
        "<div class='flow-hint'>"
        "Alur: Klien membuka tender → dokumen gambar/RKS/BoQ diserahkan → "
        "Tim KPOP membuat RAB → Aanwijzing → Negosiasi → Kontrak. "
        "Setelah tahap ini selesai, lanjut ke <b>Vendor Qualification & Selection</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    live = live_data.get("tender", {})
    if live:
        c1, c2, c3 = st.columns(3)
        c1.metric("RAB Awal", format_rupiah(live.get('rab_awal', 0)))
        c2.metric("Budget", format_rupiah(live.get('budget', 0)))
        c3.metric("Selisih", format_rupiah(live.get('variance', 0)))
        st.caption(f"Terakhir disimpan: {live.get('timestamp', '-')}")
        st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Dokumen Tender")
        doc_drawing = st.checkbox("Gambar kerja", value=True, key="td_drawing")
        doc_rks = st.checkbox("RKS — Rencana Kerja & Syarat", value=True, key="td_rks")
        doc_spek = st.checkbox("Spesifikasi teknis", value=True, key="td_spek")
        doc_boq = st.checkbox("Bill of Quantities", value=True, key="td_boq")
        doc_k3 = st.checkbox("Dokumen K3 / HSE Plan", value=True, key="td_k3")

        st.markdown("#### Catatan Aanwijzing")
        catatan = st.text_area(
            "Ringkasan hasil aanwijzing",
            placeholder="Tulis ringkasan minimal 10 karakter...",
            height=120,
            key="td_catatan",
        )

    with col2:
        st.markdown("#### RAB Awal")
        st.markdown(
            f"<div class='hint-box'>"
            f"Budget project: <b>{format_rupiah(meta['budget'])}</b>. "
            f"Nilai RAB otomatis terisi sama dengan budget sebagai titik awal — "
            f"ubah manual kalau RAB dari tim KPOP berbeda."
            f"</div>",
            unsafe_allow_html=True,
        )
        rab_awal_input = st.number_input(
            "Nilai RAB (juta Rp)",
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
        variance_preview = budget_input - rab_awal_input
        st.metric("Selisih terhadap RAB", format_rupiah(variance_preview))

        st.markdown("#### Timeline")
        tgl_aanwijzing = st.date_input("Aanwijzing", datetime(2024, 9, 7), key="td_tgl_aan")
        tgl_penawaran = st.date_input("Batas penawaran", datetime(2024, 9, 15), key="td_tgl_pen")
        tgl_kontrak = st.date_input("Target kontrak", datetime(2024, 9, 30), key="td_tgl_kon")

    dokumen_lengkap = all([doc_drawing, doc_rks, doc_spek, doc_boq, doc_k3])
    catatan_lengkap = len(catatan.strip()) >= 10
    timeline_valid = tgl_aanwijzing < tgl_penawaran < tgl_kontrak
    form_valid = dokumen_lengkap and catatan_lengkap and timeline_valid

    if not form_valid:
        missing = []
        if not dokumen_lengkap:
            missing.append("semua dokumen wajib dicentang")
        if not catatan_lengkap:
            missing.append("catatan minimal 10 karakter")
        if not timeline_valid:
            missing.append("urutan: aanwijzing < penawaran < kontrak")
        st.warning("Belum lengkap — " + "; ".join(missing))

    if st.button("Simpan Data Tender", type="primary", key="btn_save_tender"):
        if not form_valid:
            st.error("Data belum lengkap. Perbaiki dulu.")
        else:
            meta["budget"] = float(budget_input)
            meta["rab_awal"] = float(rab_awal_input)
            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rab_awal": float(rab_awal_input),
                "budget": float(budget_input),
                "variance": float(variance_preview),
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
            }
            live_data["tender"] = payload
            st.success("Data Tender tersimpan. Melanjutkan ke Vendor Qualification & Selection...")
            next_step()
            st.rerun()

    render_nav(show_next=stage_status(active, "tender"),
               next_label="Lanjut ke Vendor Qualification")


# =========================================================
# STEP 2 — VENDOR QUALIFICATION & SELECTION
# =========================================================
elif step == 2:
    st.markdown("## Tahap 2 — Vendor Qualification & Selection")

    st.markdown(
        "<div class='flow-hint'>"
        "Tahap ini mencakup: (1) Evaluasi kelayakan & optimasi vendor, "
        "(2) Klarifikasi kemampuan teknis terhadap item pekerjaan, "
        "(3) Penilaian performa vendor, "
        "(4) Appointment klarifikasi teknis (tatap muka / Zoom)."
        "</div>",
        unsafe_allow_html=True,
    )

    # Tab internal untuk 4 sub-bagian
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "1. Evaluasi & Optimasi Vendor",
        "2. Kemampuan Teknis Item Pekerjaan",
        "3. Penilaian Performa Vendor",
        "4. Appointment Klarifikasi Teknis",
    ])

    # ---------------------------------------------------------
    # SUB-TAB 1: EVALUASI & OPTIMASI VENDOR
    # ---------------------------------------------------------
    with sub_tab1:
        uploaded = st.file_uploader("Unggah data vendor (CSV, opsional)", type=["csv"], key="up1")

        if uploaded is not None:
            try:
                vendors = pd.read_csv(uploaded)
                missing = [c for c in DEFAULT_VENDORS.columns if c not in vendors.columns]
                if missing:
                    st.warning("Kolom tidak ditemukan: " + ", ".join(missing))
                    vendors = DEFAULT_VENDORS.copy()
            except Exception as e:
                st.error(f"CSV tidak dapat dibaca: {e}")
                vendors = DEFAULT_VENDORS.copy()
        else:
            vendors = DEFAULT_VENDORS.copy()

        edited = st.data_editor(
            vendors, use_container_width=True, hide_index=True,
            num_rows="dynamic", key=f"vendor_editor_{meta['id']}",
            column_config={
                "Vendor": st.column_config.TextColumn("Vendor", required=True),
                "Harga (juta Rp)": st.column_config.NumberColumn("Harga (juta Rp)", min_value=0.0, format="%.0f"),
                "Kualitas (0-100)": st.column_config.NumberColumn("Kualitas", min_value=0, max_value=100, step=1),
                "Durasi (hari)": st.column_config.NumberColumn("Durasi (hari)", min_value=1, step=1),
                "Skor K3 (0-100)": st.column_config.NumberColumn("Skor K3", min_value=0, max_value=100, step=1),
                "Lolos teknis": st.column_config.CheckboxColumn("Lolos teknis"),
                "Pengalaman (tahun)": st.column_config.NumberColumn("Pengalaman (thn)", min_value=0, step=1),
                "Karyawan": st.column_config.NumberColumn("Karyawan", min_value=0, step=1),
                "Proyek Selesai": st.column_config.NumberColumn("Proyek selesai", min_value=0, step=1),
                "Tingkat Keberhasilan (%)": st.column_config.NumberColumn(
                    "Success Rate (%)", min_value=0, max_value=100, step=1,
                    help="Persentase proyek selesai tepat waktu."
                ),
                "Rata-rata Keterlambatan (hari)": st.column_config.NumberColumn(
                    "Avg Delay (hari)", min_value=0, step=1,
                    help="Rata-rata keterlambatan dari proyek sebelumnya."
                ),
                "Sertifikasi": st.column_config.TextColumn("Sertifikasi"),
                "Lokasi": st.column_config.TextColumn("Lokasi"),
            },
        )

        if not edited.empty:
            numeric_cols = ["Harga (juta Rp)", "Kualitas (0-100)", "Durasi (hari)", "Skor K3 (0-100)"]
            data = edited.copy()
            for c in numeric_cols:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            data["Vendor"] = data["Vendor"].fillna("").astype(str).str.strip()
            data["Lolos teknis"] = data["Lolos teknis"].fillna(False).astype(bool)
            data["Tingkat Keberhasilan (%)"] = pd.to_numeric(
                data["Tingkat Keberhasilan (%)"], errors="coerce").fillna(0)
            data["Rata-rata Keterlambatan (hari)"] = pd.to_numeric(
                data["Rata-rata Keterlambatan (hari)"], errors="coerce").fillna(0)

            # Hitung skor performa composite
            # 60% success rate + 40% (100 - delay*10) — cap delay 10 hari
            data["Skor Performa"] = (
                data["Tingkat Keberhasilan (%)"] * 0.6
                + (100 - data["Rata-rata Keterlambatan (hari)"] * 10).clip(lower=0) * 0.4
            ).round(1)

            valid = (
                data["Vendor"].ne("")
                & data[numeric_cols].notna().all(axis=1)
                & (data["Harga (juta Rp)"] >= 0)
                & (data["Kualitas (0-100)"].between(0, 100))
                & (data["Durasi (hari)"] > 0)
                & (data["Skor K3 (0-100)"].between(0, 100))
            )
            data = data.loc[valid].copy()

            if data.empty or data["Vendor"].duplicated().any():
                st.error("Data vendor tidak valid atau ada nama duplikat.")
            else:
                data["Status"] = "Lolos"
                data["Alasan"] = ""
                for idx, row in data.iterrows():
                    reasons = []
                    if row["Kualitas (0-100)"] < min_quality:
                        reasons.append(f"Kualitas < {min_quality}")
                    if row["Harga (juta Rp)"] > budget:
                        reasons.append(f"Harga > {format_rupiah(budget)}")
                    if row["Durasi (hari)"] > deadline:
                        reasons.append(f"Durasi > {deadline} hari")
                    if row["Skor K3 (0-100)"] < min_hse:
                        reasons.append(f"K3 < {min_hse}")
                    if row["Skor Performa"] < min_performance:
                        reasons.append(f"Performa < {min_performance}")
                    if require_technical and not row["Lolos teknis"]:
                        reasons.append("Tidak lolos teknis")
                    if reasons:
                        data.at[idx, "Status"] = "Tidak lolos"
                        data.at[idx, "Alasan"] = " · ".join(reasons)
                    else:
                        data.at[idx, "Alasan"] = "Memenuhi seluruh kriteria"

                eligible = data[data["Status"] == "Lolos"].copy()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total vendor", len(data))
                c2.metric("Lolos evaluasi", len(eligible))
                c3.metric("Tidak lolos", len(data) - len(eligible))
                c4.metric("Skor performa tertinggi",
                          f"{data['Skor Performa'].max():.1f}" if len(data) else "-")

                st.markdown("#### Matriks Kelayakan")
                display_df = data[["Vendor", "Harga (juta Rp)", "Kualitas (0-100)",
                                    "Durasi (hari)", "Skor K3 (0-100)",
                                    "Tingkat Keberhasilan (%)", "Rata-rata Keterlambatan (hari)",
                                    "Skor Performa", "Lolos teknis", "Status", "Alasan"]].copy()
                display_df["Harga (juta Rp)"] = display_df["Harga (juta Rp)"].apply(
                    lambda x: f"Rp {x:,.0f} jt"
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                if eligible.empty:
                    st.error("Tidak ada vendor yang memenuhi seluruh kriteria.")
                    st.markdown("#### Diagnostik")
                    all_reasons = []
                    for _, r in data.iterrows():
                        if r["Alasan"]:
                            parts = [x.strip() for x in r["Alasan"].split("·") if x.strip()]
                            all_reasons.extend(parts)
                    reason_count = Counter(all_reasons)
                    diag_df = pd.DataFrame(
                        [{"Alasan": k, "Jumlah vendor": v} for k, v in reason_count.most_common()]
                    )
                    st.dataframe(diag_df, use_container_width=True, hide_index=True)
                else:
                    vendor_names = eligible["Vendor"].tolist()
                    costs = eligible.set_index("Vendor")["Harga (juta Rp)"].to_dict()

                    selected_name, solver_status = solve_vendor_lp(vendor_names, costs, budget)

                    if selected_name is None:
                        st.error(f"Optimasi gagal. Status: {solver_status}")
                    else:
                        best = eligible[eligible["Vendor"] == selected_name].iloc[0]
                        selected_cost = float(best["Harga (juta Rp)"])

                        st.markdown("---")
                        st.markdown("#### Hasil Optimasi")
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Vendor terpilih", best["Vendor"])
                        r2.metric("Nilai kontrak", format_rupiah(selected_cost))
                        r3.metric("Kualitas / K3",
                                  f"{best['Kualitas (0-100)']:.0f} / {best['Skor K3 (0-100)']:.0f}")
                        r4.metric("Skor performa", f"{best['Skor Performa']:.1f}")

                        # Simpan hasil ke session state (untuk sub-tab lain)
                        st.session_state[f"selected_vendor_{pid}"] = best["Vendor"]
                        st.session_state[f"selected_cost_{pid}"] = selected_cost
                        st.session_state[f"eligible_{pid}"] = eligible

    # ---------------------------------------------------------
    # SUB-TAB 2: KEMAMPUAN TEKNIS ITEM PEKERJAAN
    # ---------------------------------------------------------
    with sub_tab2:
        st.markdown("### Klarifikasi Kemampuan Teknis")
        st.caption(
            "Tentukan item pekerjaan yang akan diklarifikasi, lalu minta setiap vendor "
            "menyatakan apakah mereka sanggup mengerjakan item tersebut."
        )

        selected_name = st.session_state.get(f"selected_vendor_{pid}", None)
        if selected_name:
            st.markdown(
                f"<div class='hint-box'>"
                f"<b>Vendor terpilih dari evaluasi:</b> {selected_name}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Item pekerjaan yang akan diklarifikasi
        st.markdown("#### Daftar Item Pekerjaan")
        if active.get("items_data") is None:
            active["items_data"] = default_items().to_dict("records")

        items_df = pd.DataFrame(active["items_data"])
        items_edited = st.data_editor(
            items_df, use_container_width=True, hide_index=True,
            num_rows="dynamic", key=f"items_editor_{pid}",
            column_config={
                "No": st.column_config.NumberColumn("No", min_value=1, step=1, width="small"),
                "Item Pekerjaan": st.column_config.TextColumn(
                    "Item Pekerjaan", required=True, width="large"
                ),
                "Volume": st.column_config.TextColumn("Volume"),
                "Spesifikasi": st.column_config.TextColumn("Spesifikasi"),
                "Kompleksitas": st.column_config.SelectboxColumn(
                    "Kompleksitas", options=["Rendah", "Sedang", "Tinggi"]
                ),
            },
        )
        active["items_data"] = items_edited.to_dict("records")

        if not items_edited.empty and selected_name:
            st.markdown("---")
            st.markdown(f"#### Pernyataan Kemampuan Teknis — {selected_name}")
            st.caption(
                "Setiap item dinilai: **Sanggup** / **Sanggup dengan Catatan** / **Tidak Sanggup**. "
                "Isi catatan jika ada keterbatasan khusus."
            )

            # Ambil vendor eligible untuk klarifikasi
            eligible = st.session_state.get(f"eligible_{pid}", pd.DataFrame())
            vendor_list = eligible["Vendor"].tolist() if not eligible.empty else [selected_name]

            # Struktur: kemampuan[vendor][item_no] = (status, catatan)
            key_kemampuan = f"kemampuan_{pid}"
            if key_kemampuan not in st.session_state:
                st.session_state[key_kemampuan] = {}

            for v in vendor_list:
                if v not in st.session_state[key_kemampuan]:
                    st.session_state[key_kemampuan][v] = {
                        item_no: {"Status": "Sanggup", "Catatan": ""}
                        for item_no in items_edited["No"].tolist()
                    }

            vendor_pick = st.selectbox(
                "Pilih vendor untuk diisi kemampuannya",
                options=vendor_list,
                key=f"vendor_pick_{pid}",
            )

            # Tampilkan sebagai form per-vendor
            with st.expander(f"Isi kemampuan: {vendor_pick}", expanded=True):
                for _, item in items_edited.iterrows():
                    item_no = int(item["No"])
                    item_name = item["Item Pekerjaan"]
                    spec = item["Spesifikasi"]
                    kompleks = item["Kompleksitas"]

                    c1, c2, c3 = st.columns([2, 1, 2])
                    with c1:
                        st.markdown(f"**{item_no}. {item_name}**")
                        st.caption(f"{spec} · Volume: {item['Volume']} · Kompleksitas: {kompleks}")
                    with c2:
                        status_val = st.selectbox(
                            f"Status item {item_no}",
                            ["Sanggup", "Sanggup dengan Catatan", "Tidak Sanggup"],
                            key=f"status_{vendor_pick}_{item_no}",
                            label_visibility="collapsed",
                        )
                    with c3:
                        catatan_val = st.text_input(
                            f"Catatan item {item_no}",
                            value=st.session_state[key_kemampuan][vendor_pick][item_no]["Catatan"],
                            key=f"catatan_{vendor_pick}_{item_no}",
                            placeholder="Catatan (opsional)",
                            label_visibility="collapsed",
                        )

                    st.session_state[key_kemampuan][vendor_pick][item_no] = {
                        "Status": status_val,
                        "Catatan": catatan_val,
                    }

            # Ringkasan semua vendor
            st.markdown("---")
            st.markdown("#### Ringkasan Kemampuan Seluruh Vendor")
            summary_kemampuan = []
            for v in vendor_list:
                for item_no in items_edited["No"].tolist():
                    entry = st.session_state[key_kemampuan][v][int(item_no)]
                    item_row = items_edited[items_edited["No"] == item_no].iloc[0]
                    summary_kemampuan.append({
                        "Vendor": v,
                        "Item": f"{int(item_no)}. {item_row['Item Pekerjaan']}",
                        "Status": entry["Status"],
                        "Catatan": entry["Catatan"],
                    })
            st.dataframe(pd.DataFrame(summary_kemampuan),
                         use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # SUB-TAB 3: PENILAIAN PERFORMA VENDOR
    # ---------------------------------------------------------
    with sub_tab3:
        st.markdown("### Penilaian Performa Vendor")
        st.caption(
            "Nilai performa vendor berdasarkan pengalaman proyek sebelumnya. "
            "Skor ini dipakai sebagai pertimbangan tambahan selain harga & kualitas."
        )

        eligible = st.session_state.get(f"eligible_{pid}", pd.DataFrame())

        if eligible.empty:
            st.info("Selesaikan evaluasi kelayakan di sub-tab 1 terlebih dahulu.")
        else:
            st.markdown("#### Performa dari Data Vendor")
            perf_display = eligible[[
                "Vendor", "Proyek Selesai", "Tingkat Keberhasilan (%)",
                "Rata-rata Keterlambatan (hari)", "Skor Performa",
                "Skor K3 (0-100)", "Kualitas (0-100)"
            ]].copy()
            st.dataframe(perf_display, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Penilaian Manual Tambahan")
            st.caption(
                "Nilai aspek berikut berdasarkan hasil klarifikasi & referensi dari proyek sebelumnya."
            )

            # Penilaian manual per vendor
            vendor_list = eligible["Vendor"].tolist()
            key_perf = f"perf_manual_{pid}"
            if key_perf not in st.session_state:
                st.session_state[key_perf] = {}

            for v in vendor_list:
                if v not in st.session_state[key_perf]:
                    st.session_state[key_perf][v] = {
                        "Kualitas Pekerjaan": 75,
                        "Ketepatan Waktu": 75,
                        "K3 & Lingkungan": 75,
                        "Koordinasi Tim": 75,
                        "Responsif terhadap Masalah": 75,
                        "Referensi Klien Sebelumnya": 75,
                    }

            with st.expander("Isi penilaian manual per vendor"):
                vendor_pick_perf = st.selectbox(
                    "Pilih vendor", options=vendor_list, key=f"perf_pick_{pid}"
                )
                c1, c2 = st.columns(2)
                for i, (aspek, val) in enumerate(st.session_state[key_perf][vendor_pick_perf].items()):
                    target_col = c1 if i % 2 == 0 else c2
                    with target_col:
                        new_val = st.slider(
                            aspek, 0, 100, val,
                            key=f"perf_{vendor_pick_perf}_{aspek}"
                        )
                        st.session_state[key_perf][vendor_pick_perf][aspek] = new_val

            # Ringkasan skor manual
            st.markdown("---")
            st.markdown("#### Skor Performa Manual")
            manual_rows = []
            for v in vendor_list:
                skor = st.session_state[key_perf][v]
                avg = sum(skor.values()) / len(skor)
                manual_rows.append({
                    "Vendor": v,
                    **skor,
                    "Rata-rata Manual": round(avg, 1),
                })
            st.dataframe(pd.DataFrame(manual_rows), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # SUB-TAB 4: APPOINTMENT KLARIFIKASI TEKNIS
    # ---------------------------------------------------------
    with sub_tab4:
        st.markdown("### Appointment Klarifikasi Teknis")
        st.caption(
            "Jadwalkan pertemuan klarifikasi teknis dengan vendor — "
            "bisa tatap muka di kantor/lokasi atau Zoom meeting."
        )

        eligible = st.session_state.get(f"eligible_{pid}", pd.DataFrame())

        if eligible.empty:
            st.info("Selesaikan evaluasi kelayakan di sub-tab 1 terlebih dahulu.")
        else:
            vendor_list = eligible["Vendor"].tolist()

            with st.expander("➕ Buat Appointment Baru", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    appt_vendor = st.selectbox(
                        "Vendor", options=vendor_list, key=f"appt_vendor_{pid}"
                    )
                    appt_mode = st.radio(
                        "Mode pertemuan",
                        ["Tatap muka (On-site)", "Zoom Meeting", "Tatap muka (Kantor)"],
                        key=f"appt_mode_{pid}"
                    )
                    appt_date = st.date_input(
                        "Tanggal", date.today(), key=f"appt_date_{pid}"
                    )
                    appt_time = st.time_input(
                        "Jam mulai", time(9, 0), key=f"appt_time_{pid}"
                    )
                    appt_duration = st.number_input(
                        "Durasi (menit)", min_value=30, max_value=480, value=90, step=15,
                        key=f"appt_duration_{pid}"
                    )
                with c2:
                    appt_location = st.text_input(
                        "Lokasi / Link Zoom",
                        placeholder="Contoh: Meeting Room A / https://zoom.us/j/...",
                        key=f"appt_location_{pid}"
                    )
                    appt_participants = st.text_area(
                        "Peserta dari kami",
                        placeholder="Contoh: Project Manager, Engineering, HSE, QC",
                        height=80,
                        key=f"appt_participants_{pid}"
                    )
                    appt_vendor_participants = st.text_area(
                        "Peserta dari vendor",
                        placeholder="Contoh: Site Manager, Welding Engineer, QC Inspector",
                        height=80,
                        key=f"appt_vendor_participants_{pid}"
                    )
                    appt_agenda = st.text_area(
                        "Agenda",
                        placeholder="Contoh: Klarifikasi WPS, jadwal NDT, metode hydrotest",
                        height=100,
                        key=f"appt_agenda_{pid}"
                    )

                if st.button("Simpan Appointment", type="primary", key=f"btn_appt_{pid}"):
                    if not appt_location.strip():
                        st.error("Lokasi / Link Zoom wajib diisi.")
                    elif len(appt_agenda.strip()) < 10:
                        st.error("Agenda minimal 10 karakter.")
                    else:
                        appointment = {
                            "id": str(uuid.uuid4())[:6],
                            "vendor": appt_vendor,
                            "mode": appt_mode,
                            "date": str(appt_date),
                            "time": str(appt_time),
                            "duration": int(appt_duration),
                            "location": appt_location,
                            "participants": appt_participants,
                            "vendor_participants": appt_vendor_participants,
                            "agenda": appt_agenda,
                            "status": "Terjadwal",
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        active["appointments"].append(appointment)
                        st.success("Appointment berhasil dibuat.")
                        st.rerun()

            st.markdown("---")
            st.markdown("#### Daftar Appointment")

            if not active["appointments"]:
                st.info("Belum ada appointment. Buat yang pertama di atas.")
            else:
                # Tampilkan sebagai kartu
                for i, appt in enumerate(active["appointments"]):
                    with st.expander(
                        f"[{appt['status']}] {appt['vendor']} · {appt['mode']} · "
                        f"{appt['date']} {appt['time']}",
                        expanded=(i == 0),
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Vendor:** {appt['vendor']}")
                            st.markdown(f"**Mode:** {appt['mode']}")
                            st.markdown(f"**Tanggal:** {appt['date']}")
                            st.markdown(f"**Jam:** {appt['time']} ({appt['duration']} menit)")
                            st.markdown(f"**Lokasi / Link:** {appt['location']}")
                        with c2:
                            st.markdown(f"**Peserta kami:** {appt['participants'] or '-'}")
                            st.markdown(f"**Peserta vendor:** {appt['vendor_participants'] or '-'}")
                            st.markdown(f"**Agenda:** {appt['agenda']}")
                            st.markdown(f"**Dibuat:** {appt['created_at']}")

                        # Ubah status
                        new_status = st.selectbox(
                            "Status",
                            ["Terjadwal", "Berlangsung", "Selesai", "Dibatalkan"],
                            index=["Terjadwal", "Berlangsung", "Selesai", "Dibatalkan"].index(appt["status"]),
                            key=f"appt_status_{i}_{appt['id']}",
                        )
                        if new_status != appt["status"]:
                            active["appointments"][i]["status"] = new_status
                            st.rerun()

                        # Hapus
                        if st.button(f"Hapus appointment ini", key=f"del_appt_{appt['id']}"):
                            active["appointments"].pop(i)
                            st.rerun()

            # Tombol submit tahap Vendor Selection
            st.markdown("---")
            st.markdown("### Finalisasi Tahap Vendor Qualification")

            selected_name = st.session_state.get(f"selected_vendor_{pid}", None)
            if not selected_name:
                st.warning("Selesaikan evaluasi & optimasi di sub-tab 1 terlebih dahulu.")
            elif not active["appointments"]:
                st.warning("Buat minimal 1 appointment klarifikasi teknis sebelum finalisasi.")
            else:
                st.markdown(
                    f"<div class='hint-box'>"
                    f"<b>Vendor terpilih:</b> {selected_name}<br>"
                    f"<b>Appointment terjadwal:</b> {len(active['appointments'])}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    approval_drawing = st.radio(
                        "Approval drawing vendor",
                        ["Belum", "Diajukan", "Disetujui", "Revisi"], index=0,
                        key="vs_approval",
                    )
                with col2:
                    kontrak_status = st.radio(
                        "Status kontrak / SPK",
                        ["Belum", "Proses", "Ditandatangani"], index=0,
                        key="vs_kontrak",
                    )

                approval_valid = approval_drawing == "Disetujui"
                kontrak_valid = kontrak_status == "Ditandatangani"
                form_valid = approval_valid and kontrak_valid

                if not form_valid:
                    missing = []
                    if not approval_valid:
                        missing.append("approval drawing harus 'Disetujui'")
                    if not kontrak_valid:
                        missing.append("kontrak/SPK harus 'Ditandatangani'")
                    st.warning("Belum lengkap — " + "; ".join(missing))

                if st.button("Simpan Keputusan Vendor Selection",
                             type="primary", key="btn_save_vs"):
                    if not form_valid:
                        st.error("Data belum lengkap. Perbaiki dulu.")
                    else:
                        eligible = st.session_state.get(f"eligible_{pid}", pd.DataFrame())
                        best_row = eligible[eligible["Vendor"] == selected_name]
                        selected_cost = float(best_row.iloc[0]["Harga (juta Rp)"]) if not best_row.empty else 0

                        payload = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "selected_vendor": selected_name,
                            "selected_cost": selected_cost,
                            "selected_duration": int(best_row.iloc[0]["Durasi (hari)"]) if not best_row.empty else 0,
                            "selected_quality": float(best_row.iloc[0]["Kualitas (0-100)"]) if not best_row.empty else 0,
                            "selected_hse": float(best_row.iloc[0]["Skor K3 (0-100)"]) if not best_row.empty else 0,
                            "selected_performance": float(best_row.iloc[0]["Skor Performa"]) if not best_row.empty else 0,
                            "total_vendors": len(eligible),
                            "eligible_count": len(eligible),
                            "appointment_count": len(active["appointments"]),
                            "approval_drawing": approval_drawing,
                            "kontrak_status": kontrak_status,
                        }
                        live_data["vendor_selection"] = payload
                        st.success("Vendor Selection tersimpan. Melanjutkan ke Site Execution...")
                        next_step()
                        st.rerun()

    render_nav(show_next=stage_status(active, "vendor_selection"),
               next_label="Lanjut ke Site Execution")


# =========================================================
# STEP 3 — SITE EXECUTION
# =========================================================
elif step == 3:
    st.markdown("## Tahap 3 — Site Execution")

    st.markdown(
        "<div class='flow-hint'>"
        "Alur: PCM → Mobilisasi material & alat → Pekerjaan konstruksi → "
        "Monitoring progress via S-Curve → Analisa kendala lapangan → Evaluasi KPI kontraktor. "
        "Setelah pekerjaan selesai, lanjut ke <b>Quality Control</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    vs = live_data.get("vendor_selection", {})
    selected_vendor = vs.get("selected_vendor", "-")
    selected_duration = vs.get("selected_duration", 78)

    st.caption(f"Kontraktor terpilih: {selected_vendor} · Durasi kontrak: {selected_duration} hari")

    st.markdown("### Template S-Curve Proyek")
    st.caption("Isi tabel WBS sesuai pekerjaan. Sistem akan menghitung S-Curve otomatis.")

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
            plan_weekly = np.zeros(max_week)
            for _, row in wbs_edited.iterrows():
                w_start = int(row["Mulai (minggu)"])
                w_end = int(row["Selesai (minggu)"])
                bobot = float(row["Bobot (%)"])
                span = w_end - w_start + 1
                if span > 0:
                    plan_weekly[w_start - 1:w_end] += bobot / span
            plan_cum = np.cumsum(plan_weekly)

            actual_weekly = np.zeros(max_week)
            for _, row in wbs_edited.iterrows():
                w_start = int(row["Mulai (minggu)"])
                w_end = int(row["Selesai (minggu)"])
                bobot = float(row["Bobot (%)"])
                realisasi_pct = float(row["Realisasi (%)"])
                if realisasi_pct <= 0:
                    continue
                target = bobot * (realisasi_pct / 100.0)
                w_actual_end = min(current_week, w_end)
                if w_actual_end < w_start:
                    continue
                span_actual = w_actual_end - w_start + 1
                actual_weekly[w_start - 1:w_actual_end] += target / span_actual
            actual_cum = np.cumsum(actual_weekly)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.fill_between(weeks, plan_cum, actual_cum, where=(actual_cum < plan_cum),
                            color="#dc2626", alpha=0.12, label="Di bawah rencana")
            ax.fill_between(weeks, plan_cum, actual_cum, where=(actual_cum >= plan_cum),
                            color="#047857", alpha=0.12, label="Di atas rencana")
            ax.plot(weeks, plan_cum, color="#1e40af", linewidth=2.5, marker="o", markersize=5, label="Rencana")
            ax.plot(weeks, actual_cum, color="#dc2626", linewidth=2.5, marker="s", markersize=5, label="Realisasi")
            ax.axvline(current_week, linestyle=":", color="#6b7280", linewidth=1.5, label=f"Minggu ini (W{current_week})")
            plan_now = plan_cum[current_week - 1]
            actual_now = actual_cum[current_week - 1]
            ax.scatter([current_week], [plan_now], color="#1e40af", s=80, zorder=5, edgecolor="white", linewidth=1.5)
            ax.scatter([current_week], [actual_now], color="#dc2626", s=80, marker="D", zorder=5, edgecolor="white", linewidth=1.5)
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
                    st.info("Tidak ada kendala — langkah penanganan tidak diperlukan.")
                    aksi = "Tidak diperlukan"
                    aksi_valid = True
                elif kendala == "Pilih kendala...":
                    st.caption("Pilih kendala dulu untuk melihat opsi penanganan.")
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
                    missing.append("catatan PCM minimal 10 karakter")
                if not kendala_valid:
                    missing.append("kendala belum dipilih")
                if not aksi_valid:
                    missing.append("aksi penanganan belum dipilih")
                if not kpi_valid:
                    missing.append("KPI harus > 0%")
                st.warning("Belum lengkap — " + "; ".join(missing))

            if st.button("Simpan Data Site Execution", type="primary", key="btn_save_se"):
                if not form_valid:
                    st.error("Data belum lengkap. Perbaiki dulu.")
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
                        "total_bobot": float(total_bobot),
                    }
                    live_data["site_execution"] = payload
                    st.success("Data Site Execution tersimpan. Melanjutkan ke Quality Control...")
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
        "Alur: Pemberitahuan pekerjaan selesai → Inspeksi mutu & test commissioning → "
        "Inspeksi akhir → Berita Acara Serah Terima (BAST) → Proyek selesai."
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

    st.markdown("#### Inspeksi Mutu & Test Commissioning")
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
        "Hasil inspeksi akhir sesuai kontrak?",
        ["Pilih...", "Ya", "Tidak"], index=0, key="qc_final"
    )

    if final_status == "Ya":
        st.markdown("#### Berita Acara Serah Terima")
        c1, c2, c3 = st.columns(3)
        with c1:
            bast_date = st.date_input("Tanggal BAST", datetime(2024, 12, 20), key="qc_bast_date")
        with c2:
            bast_number = st.text_input("Nomor BAST", placeholder="BAST/KPOP/2024/047", key="qc_bast_no")
        with c3:
            approved_by = st.text_input("Disetujui oleh", placeholder="Nama lengkap", key="qc_approver")
        bast_notes = st.text_area("Catatan BAST", placeholder="Catatan minimal 10 karakter...", height=80, key="qc_bast_notes")
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
            missing.append("semua hasil inspeksi wajib dipilih")
        if not status_valid:
            missing.append("status inspeksi akhir belum dipilih")
        if not bast_valid:
            missing.append("data BAST belum lengkap")
        st.warning("Belum lengkap — " + "; ".join(missing))

    if st.button("Simpan Data Quality Control", type="primary", key="btn_save_qc"):
        if not form_valid:
            st.error("Data belum lengkap. Perbaiki dulu.")
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
            st.success("Semua tahapan selesai. Project telah diselesaikan.")
            next_step()
            st.rerun()

    render_nav(show_next=stage_status(active, "quality_control"),
               next_label="Lihat Ringkasan")


# =========================================================
# STEP 5 — SELESAI / RINGKASAN
# =========================================================
elif step == 5:
    st.markdown("## Project Selesai")

    tender = live_data.get("tender", {})
    vs = live_data.get("vendor_selection", {})
    se = live_data.get("site_execution", {})
    qc = live_data.get("quality_control", {})

    if not qc or qc.get("final_status") != "Ya":
        st.warning("Project belum sepenuhnya selesai. Kembali ke tahap Quality Control.")
    else:
        st.success(f"Project **{meta['name']}** telah diselesaikan.")

        st.markdown("---")
        st.markdown("### Ringkasan Project")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Informasi Project**")
            st.markdown(f"- Nama: {meta['name']}")
            st.markdown(f"- Klien: {meta['client']}")
            st.markdown(f"- Lokasi: {meta['location']}")
            st.markdown(f"- Budget: {format_rupiah(meta['budget'])}")
            st.markdown(f"- RAB Awal: {format_rupiah(meta.get('rab_awal', 0))}")
            st.markdown(f"- Deadline: {meta['deadline']} hari")

        with c2:
            st.markdown("**Hasil Akhir**")
            if vs:
                st.markdown(f"- Vendor terpilih: {vs.get('selected_vendor', '-')}")
                st.markdown(f"- Nilai kontrak: {format_rupiah(vs.get('selected_cost', 0))}")
                st.markdown(f"- Appointment klarifikasi: {vs.get('appointment_count', 0)}")
            if se:
                st.markdown(f"- Realisasi progress: {se.get('actual_pct', 0):.1f}%")
                st.markdown(f"- SPI: {se.get('spi', 0):.3f}")
            if qc:
                st.markdown(f"- BAST: {qc.get('bast_number', '-')}")
                st.markdown(f"- Tanggal BAST: {qc.get('bast_date', '-')}")

        st.markdown("---")
        st.markdown("### Unduh Data")
        c1, c2, c3 = st.columns(3)
        with c1:
            all_data = {
                "tender": tender,
                "vendor_selection": vs,
                "site_execution": se,
                "quality_control": qc,
            }
            rows = []
            for tahap, d in all_data.items():
                if d:
                    row = {"Tahap": label(tahap)}
                    row.update({k: str(v) for k, v in d.items()})
                    rows.append(row)
            if rows:
                st.download_button(
                    "Unduh data project (CSV)",
                    data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"project_{meta['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                )
        with c2:
            if active.get("wbs_data"):
                wbs_df = pd.DataFrame(active["wbs_data"])
                if not wbs_df.empty:
                    st.download_button(
                        "Unduh WBS (CSV)",
                        data=wbs_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"wbs_{meta['name'].replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
        with c3:
            if active["appointments"]:
                st.download_button(
                    "Unduh Appointment (CSV)",
                    data=pd.DataFrame(active["appointments"]).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"appointments_{meta['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                )

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Kembali ke Quality Control", key="btn_back_qc"):
            goto_step(4)
            st.rerun()
    with c3:
        if st.button("Kembali ke Daftar Project", key="btn_back_list"):
            st.session_state.step = 0
            st.rerun()

st.markdown("---")
st.caption("Vendor Selection & Project Control · Multi-project workspace · Data simulasi.")
