import streamlit as st
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
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
    .stFormSubmitButton > button {
        background-color: #047857 !important; color: white !important; font-weight: 500 !important;
    }
    .stFormSubmitButton > button:hover { background-color: #065f46 !important; }
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

    /* Stepper di atas */
    .stepper {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .step-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #6b7280;
    }
    .step-chip.active {
        background: #1e40af;
        color: #ffffff;
        border-color: #1e40af;
        font-weight: 600;
    }
    .step-chip.done {
        background: #d1fae5;
        color: #065f46;
        border-color: #a7f3d0;
    }
    .step-chip .num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        font-size: 0.75rem;
        font-weight: 700;
    }
    .step-chip:not(.active):not(.done) .num {
        background: #f3f4f6;
        color: #6b7280;
    }
    .step-chip.active .num {
        background: rgba(255,255,255,0.25);
        color: #ffffff;
    }
    .step-chip.done .num {
        background: #047857;
        color: #ffffff;
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
    """Buat variabel biner — kompatibel PuLP 2.x, 3.x, 4.x."""
    try:
        return prob.add_variable_dicts(
            var_name, index_list, lowBound=0, upBound=1, cat="Binary"
        )
    except AttributeError:
        pass
    try:
        return pulp.LpVariable.dicts(
            var_name, index_list, lowBound=0, upBound=1, cat="Binary"
        )
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

# Step index untuk navigasi (0=single project screen, 1..5 = tahapan)
if "step" not in st.session_state:
    st.session_state.step = 1


STEPS = [
    {"key": "tender", "label": "Tender"},
    {"key": "project_mgmt", "label": "Project Management"},
    {"key": "vendor_selection", "label": "Vendor Selection"},
    {"key": "site_execution", "label": "Site Execution"},
    {"key": "quality_control", "label": "Quality Control"},
]


def goto_step(n):
    """Pindah ke step n (1..5)."""
    st.session_state.step = max(1, min(len(STEPS), n))


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
            "tender": {}, "project_mgmt": {}, "vendor_selection": {},
            "site_execution": {}, "quality_control": {},
        },
        "wbs_data": None,
    }
    return pid


def get_active():
    if st.session_state.active_project is None:
        return None
    return st.session_state.projects.get(st.session_state.active_project)


def label(tahap):
    return {
        "tender": "Tender",
        "project_mgmt": "Project Management",
        "vendor_selection": "Vendor Selection",
        "site_execution": "Site Execution",
        "quality_control": "Quality Control",
    }.get(tahap, tahap)


def stage_status(project, tahap):
    return bool(project["live_data"].get(tahap))


def project_progress(project):
    stages = ["tender", "project_mgmt", "vendor_selection",
              "site_execution", "quality_control"]
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

    with st.form("form_new_project", clear_on_submit=True):
        new_name = st.text_input("Nama project", placeholder="Contoh: Piping Area 5B")
        new_client = st.text_input("Klien", placeholder="Contoh: PT Pertamina")
        new_location = st.text_input("Lokasi", placeholder="Contoh: Balikpapan")
        new_budget = st.number_input(
            "Budget project (juta Rp)",
            min_value=10.0, value=300.0, step=10.0,
            help="Contoh: 300 berarti Rp 300.000.000."
        )
        new_deadline = st.number_input("Deadline project (hari)", min_value=1, value=90, step=5)
        submitted = st.form_submit_button("Tambah Project")
        if submitted:
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
        for i, t in enumerate(STEPS):
            ok = stage_status(active, t["key"])
            dot = "dot-ok" if ok else "dot-idle"
            text_color = "#047857" if ok else "#9ca3af"
            st.markdown(
                f"<div class='step-row'>"
                f"<span class='step-dot {dot}'></span>"
                f"<span style='color:{text_color};'>{i+1}. {t['label']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        done, total = project_progress(active)
        st.caption(f"{done} dari {total} tahap selesai")

        st.markdown("---")
        st.markdown("**Parameter Kriteria**")
        min_quality = st.slider("Kualitas minimum", 0, 100, 82)
        min_hse = st.slider("Skor K3 minimum", 0, 100, 85)
        require_technical = st.checkbox("Wajib lolos teknis", value=True)

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
        if st.button("Hapus Project Ini"):
            del st.session_state.projects[st.session_state.active_project]
            st.session_state.active_project = (
                list(st.session_state.projects.keys())[0]
                if st.session_state.projects else None
            )
            st.session_state.step = 1
            st.rerun()
    else:
        min_quality, min_hse, require_technical = 82, 85, True

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
# VENDOR DATA
# =========================================================
DEFAULT_VENDORS = pd.DataFrame([
    {"Vendor": "PT Baja Nusantara Konstruksi", "Harga (juta Rp)": 285.0, "Kualitas (0-100)": 91,
     "Durasi (hari)": 78, "Skor K3 (0-100)": 92, "Lolos teknis": True,
     "Pengalaman (tahun)": 18, "Karyawan": 240, "Proyek Selesai": 87,
     "Sertifikasi": "ISO 9001, ISO 45001, ASME", "Lokasi": "Jakarta"},
    {"Vendor": "CV Prima Pipa Sejahtera", "Harga (juta Rp)": 245.0, "Kualitas (0-100)": 78,
     "Durasi (hari)": 65, "Skor K3 (0-100)": 84, "Lolos teknis": True,
     "Pengalaman (tahun)": 9, "Karyawan": 120, "Proyek Selesai": 34,
     "Sertifikasi": "ISO 9001", "Lokasi": "Surabaya"},
    {"Vendor": "PT Karya Logam Presisi", "Harga (juta Rp)": 310.0, "Kualitas (0-100)": 95,
     "Durasi (hari)": 85, "Skor K3 (0-100)": 94, "Lolos teknis": True,
     "Pengalaman (tahun)": 22, "Karyawan": 310, "Proyek Selesai": 112,
     "Sertifikasi": "ISO 9001, ISO 45001, ASME, API", "Lokasi": "Cilegon"},
    {"Vendor": "PT Sinergi Teknik Mandiri", "Harga (juta Rp)": 220.0, "Kualitas (0-100)": 82,
     "Durasi (hari)": 92, "Skor K3 (0-100)": 80, "Lolos teknis": True,
     "Pengalaman (tahun)": 12, "Karyawan": 180, "Proyek Selesai": 56,
     "Sertifikasi": "ISO 9001", "Lokasi": "Bandung"},
    {"Vendor": "PT Anugerah Pipa Internasional", "Harga (juta Rp)": 340.0, "Kualitas (0-100)": 97,
     "Durasi (hari)": 70, "Skor K3 (0-100)": 96, "Lolos teknis": True,
     "Pengalaman (tahun)": 25, "Karyawan": 420, "Proyek Selesai": 156,
     "Sertifikasi": "ISO 9001, ISO 45001, ASME, API, DNV", "Lokasi": "Batam"},
    {"Vendor": "CV Mandiri Baja Perkasa", "Harga (juta Rp)": 195.0, "Kualitas (0-100)": 74,
     "Durasi (hari)": 88, "Skor K3 (0-100)": 76, "Lolos teknis": False,
     "Pengalaman (tahun)": 6, "Karyawan": 85, "Proyek Selesai": 18,
     "Sertifikasi": "ISO 9001", "Lokasi": "Semarang"},
])


# =========================================================
# JIKA BELUM ADA PROJECT
# =========================================================
if active is None:
    st.info("Belum ada project aktif. Tambahkan project baru dari sidebar.")
    st.stop()

live_data = active["live_data"]
meta = active["meta"]
budget = float(meta["budget"])
deadline = int(meta["deadline"])
pid = st.session_state.active_project
step = st.session_state.step


# =========================================================
# STEPPER DI ATAS
# =========================================================
stepper_html = "<div class='stepper'>"
for i, s in enumerate(STEPS):
    num = i + 1
    if num == step:
        cls = "active"
    elif stage_status(active, s["key"]):
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


# =========================================================
# NAVIGASI BAWAH (fungsi umum)
# =========================================================
def render_nav(show_next=True, next_label="Lanjut ke tahap berikutnya"):
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if step > 1:
            if st.button("← Sebelumnya", key=f"prev_{step}"):
                prev_step()
                st.rerun()
    with c3:
        if show_next and step < len(STEPS):
            if st.button(next_label + " →", key=f"next_{step}"):
                next_step()
                st.rerun()


# =========================================================
# STEP 1 — TENDER
# =========================================================
if step == 1:
    st.markdown("## Tahap 1 — Client & Tender")

    live = live_data.get("tender", {})
    if live:
        c1, c2, c3 = st.columns(3)
        c1.metric("RAB Awal", format_rupiah(live.get('rab_awal', 0)))
        c2.metric("Budget", format_rupiah(live.get('budget', 0)))
        c3.metric("Selisih", format_rupiah(live.get('variance', 0)))
        st.caption(f"Terakhir disimpan: {live.get('timestamp', '-')}")
        st.markdown("---")

    with st.form("form_tender"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Dokumen Tender")
            doc_drawing = st.checkbox("Gambar kerja", value=True)
            doc_rks = st.checkbox("RKS — Rencana Kerja & Syarat", value=True)
            doc_spek = st.checkbox("Spesifikasi teknis", value=True)
            doc_boq = st.checkbox("Bill of Quantities", value=True)
            doc_k3 = st.checkbox("Dokumen K3 / HSE Plan", value=True)

            st.markdown("#### Catatan Aanwijzing")
            catatan = st.text_area(
                "Ringkasan hasil aanwijzing",
                placeholder="Tulis ringkasan minimal 10 karakter...",
                height=120,
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("#### RAB Awal")
            st.caption("Nilai dalam juta Rupiah. Contoh: 300 berarti Rp 300.000.000.")
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
            tgl_aanwijzing = st.date_input("Aanwijzing", datetime(2024, 9, 7))
            tgl_penawaran = st.date_input("Batas penawaran", datetime(2024, 9, 15))
            tgl_kontrak = st.date_input("Target kontrak", datetime(2024, 9, 30))

        dokumen_lengkap = all([doc_drawing, doc_rks, doc_spek, doc_boq, doc_k3])
        catatan_lengkap = len(catatan.strip()) >= 10
        timeline_valid = tgl_aanwijzing < tgl_penawaran < tgl_kontrak

        if not (dokumen_lengkap and catatan_lengkap and timeline_valid):
            missing = []
            if not dokumen_lengkap:
                missing.append("semua dokumen wajib dicentang")
            if not catatan_lengkap:
                missing.append("catatan minimal 10 karakter")
            if not timeline_valid:
                missing.append("urutan: aanwijzing < penawaran < kontrak")
            st.warning("Belum lengkap — " + "; ".join(missing))

        submitted = st.form_submit_button("Simpan Data Tender")

        if submitted:
            if not (dokumen_lengkap and catatan_lengkap and timeline_valid):
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
                st.success("Data Tender tersimpan. Melanjutkan ke tahap berikutnya...")
                next_step()
                st.rerun()

    render_nav(show_next=stage_status(active, "tender"),
               next_label="Lanjut ke Project Management")


# =========================================================
# STEP 2 — PROJECT MANAGEMENT
# =========================================================
elif step == 2:
    st.markdown("## Tahap 2 — Project Management")

    live = live_data.get("project_mgmt", {})
    if live:
        c1, c2, c3 = st.columns(3)
        c1.metric("Approval Drawing", live.get("approval_status", "-"))
        c2.metric("Cross Tender", live.get("cross_tender_status", "-"))
        c3.metric("Progress", f"{live.get('progress_pct', 0)}%")
        st.caption(f"Terakhir disimpan: {live.get('timestamp', '-')}")
        st.markdown("---")

    with st.form("form_pm"):
        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.markdown("#### Approval Drawing")
            approval_status = st.radio(
                "Status approval",
                ["Belum diajukan", "Menunggu approval", "Disetujui", "Revisi"],
                index=0,
            )
            st.markdown("#### Dokumen Kontrol")
            doc_wom = st.checkbox("WOM", value=False)
            doc_boq = st.checkbox("BoQ", value=False)
            doc_rab = st.checkbox("RAB", value=False)
            doc_baseline = st.checkbox("Baseline Schedule", value=False)
            st.markdown("#### Cross Tender")
            cross_tender_status = st.selectbox(
                "Status cross tender",
                ["Belum dimulai", "Sedang proses", "Selesai"], index=0,
            )
            progress_pct = st.slider("Progress PM (%)", 0, 100, 0)

        with col2:
            st.markdown("#### Perbandingan Harga vs Baseline")
            baseline_ref = float(meta.get("rab_awal", 300.0))
            cross_data = pd.DataFrame({
                "Vendor": ["Baja Nusantara", "Prima Pipa", "Karya Logam",
                           "Sinergi Teknik", "Anugerah Pipa"],
                "Penawaran": [285, 245, 310, 220, 340],
                "Baseline": [baseline_ref] * 5,
            })

            fig, ax = plt.subplots(figsize=(8, 4))
            x_pos = np.arange(len(cross_data))
            ax.bar(x_pos, cross_data["Penawaran"], color="#1e40af", width=0.6, label="Penawaran")
            ax.plot(x_pos, cross_data["Baseline"], color="#dc2626", linewidth=2,
                    linestyle="--", marker="o", markersize=6, label="Baseline RAB")
            for i, (v, b) in enumerate(zip(cross_data["Penawaran"], cross_data["Baseline"])):
                variance = (v - b) / b * 100 if b > 0 else 0
                ax.text(i, v + 5, f"{variance:+.1f}%", ha="center", fontsize=9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cross_data["Vendor"], rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Harga (juta Rp)", fontsize=9)
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        dokumen_lengkap = all([doc_wom, doc_boq, doc_rab, doc_baseline])
        approval_valid = approval_status != "Belum diajukan"
        cross_valid = cross_tender_status != "Belum dimulai"
        progress_valid = progress_pct > 0
        form_valid = dokumen_lengkap and approval_valid and cross_valid and progress_valid

        if not form_valid:
            missing = []
            if not approval_valid:
                missing.append("approval drawing belum diisi")
            if not dokumen_lengkap:
                missing.append("semua dokumen wajib dicentang")
            if not cross_valid:
                missing.append("cross tender belum dimulai")
            if not progress_valid:
                missing.append("progress PM harus > 0%")
            st.warning("Belum lengkap — " + "; ".join(missing))

        submitted = st.form_submit_button("Simpan Data Project Management")

        if submitted:
            if not form_valid:
                st.error("Data belum lengkap. Perbaiki dulu.")
            else:
                payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "approval_status": approval_status,
                    "dokumen": [d for d, v in [
                        ("WOM", doc_wom), ("BoQ", doc_boq),
                        ("RAB", doc_rab), ("Baseline", doc_baseline),
                    ] if v],
                    "cross_tender_status": cross_tender_status,
                    "progress_pct": progress_pct,
                }
                live_data["project_mgmt"] = payload
                st.success("Data tersimpan. Melanjutkan ke tahap berikutnya...")
                next_step()
                st.rerun()

    render_nav(show_next=stage_status(active, "project_mgmt"),
               next_label="Lanjut ke Vendor Selection")


# =========================================================
# STEP 3 — VENDOR SELECTION
# =========================================================
elif step == 3:
    st.markdown("## Tahap 3 — Vendor Selection & Optimasi")

    uploaded = st.file_uploader("Unggah data vendor (CSV, opsional)", type=["csv"])

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
            "Harga (juta Rp)": st.column_config.NumberColumn(
                "Harga (juta Rp)", min_value=0.0, format="%.0f"
            ),
            "Kualitas (0-100)": st.column_config.NumberColumn(
                "Kualitas", min_value=0, max_value=100, step=1
            ),
            "Durasi (hari)": st.column_config.NumberColumn(
                "Durasi (hari)", min_value=1, step=1
            ),
            "Skor K3 (0-100)": st.column_config.NumberColumn(
                "Skor K3", min_value=0, max_value=100, step=1
            ),
            "Lolos teknis": st.column_config.CheckboxColumn("Lolos teknis"),
            "Pengalaman (tahun)": st.column_config.NumberColumn(
                "Pengalaman (thn)", min_value=0, step=1
            ),
            "Karyawan": st.column_config.NumberColumn("Karyawan", min_value=0, step=1),
            "Proyek Selesai": st.column_config.NumberColumn("Proyek selesai", min_value=0, step=1),
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
                if require_technical and not row["Lolos teknis"]:
                    reasons.append("Tidak lolos teknis")
                if reasons:
                    data.at[idx, "Status"] = "Tidak lolos"
                    data.at[idx, "Alasan"] = " · ".join(reasons)
                else:
                    data.at[idx, "Alasan"] = "Memenuhi seluruh kriteria"

            eligible = data[data["Status"] == "Lolos"].copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total vendor", len(data))
            c2.metric("Lolos evaluasi", len(eligible))
            c3.metric("Tidak lolos", len(data) - len(eligible))

            st.markdown("#### Matriks Kelayakan")
            display_df = data[["Vendor", "Harga (juta Rp)", "Kualitas (0-100)",
                                "Durasi (hari)", "Skor K3 (0-100)",
                                "Lolos teknis", "Status", "Alasan"]].copy()
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
                    r4.metric("Sisa budget", format_rupiah(budget - selected_cost))

                    chart_data = eligible.sort_values("Harga (juta Rp)")
                    colors = ["#047857" if v == selected_name else "#1e40af"
                              for v in chart_data["Vendor"]]

                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    bars = ax.bar(chart_data["Vendor"], chart_data["Harga (juta Rp)"],
                                  color=colors, width=0.55)
                    for bar, val in zip(bars, chart_data["Harga (juta Rp)"]):
                        ax.text(bar.get_x() + bar.get_width() / 2, val + 4,
                                f"Rp {val:,.0f} jt", ha="center", fontsize=8.5, color="#374151")
                    ax.axhline(budget, linestyle="--", color="#dc2626", linewidth=1.5,
                               label=f"Budget: {format_rupiah(budget)}")
                    ax.set_ylabel("Harga (juta Rp)", fontsize=9)
                    ax.tick_params(labelsize=9)
                    plt.xticks(rotation=15, ha="right")
                    ax.legend(fontsize=9)
                    ax.grid(axis="y", alpha=0.25)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown("---")
                    with st.form("form_vs"):
                        col1, col2 = st.columns(2)
                        with col1:
                            approval_drawing = st.radio(
                                "Approval drawing vendor",
                                ["Belum", "Diajukan", "Disetujui", "Revisi"], index=0
                            )
                        with col2:
                            kontrak_status = st.radio(
                                "Status kontrak / SPK",
                                ["Belum", "Proses", "Ditandatangani"], index=0
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

                        submitted = st.form_submit_button("Simpan Keputusan Vendor Selection")

                        if submitted:
                            if not form_valid:
                                st.error("Data belum lengkap. Perbaiki dulu.")
                            else:
                                payload = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "selected_vendor": best["Vendor"],
                                    "selected_cost": selected_cost,
                                    "selected_duration": int(best["Durasi (hari)"]),
                                    "selected_quality": float(best["Kualitas (0-100)"]),
                                    "selected_hse": float(best["Skor K3 (0-100)"]),
                                    "total_vendors": len(data),
                                    "eligible_count": len(eligible),
                                    "approval_drawing": approval_drawing,
                                    "kontrak_status": kontrak_status,
                                }
                                live_data["vendor_selection"] = payload
                                st.success("Data tersimpan. Melanjutkan ke tahap berikutnya...")
                                next_step()
                                st.rerun()

    render_nav(show_next=stage_status(active, "vendor_selection"),
               next_label="Lanjut ke Site Execution")


# =========================================================
# STEP 4 — SITE EXECUTION
# =========================================================
elif step == 4:
    st.markdown("## Tahap 4 — Site Execution")

    vs = live_data.get("vendor_selection", {})
    selected_vendor = vs.get("selected_vendor", "-")
    selected_duration = vs.get("selected_duration", 78)

    st.caption(
        f"Kontraktor terpilih: {selected_vendor} · Durasi kontrak: {selected_duration} hari"
    )

    st.markdown("### Template S-Curve Proyek")
    st.caption(
        "Isi tabel WBS sesuai pekerjaan. Sistem akan menghitung S-Curve rencana vs realisasi otomatis."
    )

    if active.get("wbs_data") is None:
        active["wbs_data"] = default_wbs().to_dict("records")

    wbs_df = pd.DataFrame(active["wbs_data"])
    wbs_edited = st.data_editor(
        wbs_df,
        use_container_width=True, hide_index=True,
        num_rows="dynamic", key=f"wbs_editor_{pid}",
        column_config={
            "No": st.column_config.NumberColumn("No", min_value=1, step=1, width="small"),
            "Item Pekerjaan": st.column_config.TextColumn(
                "Item Pekerjaan", required=True, width="large"
            ),
            "Bobot (%)": st.column_config.NumberColumn(
                "Bobot (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"
            ),
            "Mulai (minggu)": st.column_config.NumberColumn(
                "Mulai (minggu)", min_value=1, step=1, width="small"
            ),
            "Selesai (minggu)": st.column_config.NumberColumn(
                "Selesai (minggu)", min_value=1, step=1, width="small"
            ),
            "Realisasi (%)": st.column_config.NumberColumn(
                "Realisasi (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"
            ),
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
            ax.plot(weeks, plan_cum, color="#1e40af", linewidth=2.5,
                    marker="o", markersize=5, label="Rencana")
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
                df_scurve = pd.DataFrame({
                    "Minggu": weeks,
                    "Rencana (%)": np.round(plan_cum, 2),
                    "Realisasi (%)": np.round(actual_cum, 2),
                    "Deviasi (%)": np.round(actual_cum - plan_cum, 2),
                })
                st.dataframe(df_scurve, use_container_width=True, hide_index=True)
                st.download_button(
                    "Unduh S-Curve (CSV)",
                    data=df_scurve.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"s_curve_{meta['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            with st.form("form_se"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Pre Construction Meeting")
                    pcm_date = st.date_input("Tanggal PCM", datetime(2024, 10, 1))
                    pcm_notes = st.text_area(
                        "Catatan PCM",
                        placeholder="Tulis notulen PCM minimal 10 karakter...", height=100
                    )
                with col2:
                    st.markdown("#### Analisa Kendala")
                    kendala = st.selectbox(
                        "Kendala utama",
                        ["Pilih kendala...", "Tidak ada", "Perubahan desain",
                         "Keterlambatan material", "Cuaca ekstrem", "Akses terbatas"],
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
                            "Pilih aksi...",
                            "Evaluasi dengan klien",
                            "Revisi gambar kerja",
                            "Pengukuran ulang aktual",
                            "Konsultasi GSB & GSJ",
                        ])
                        aksi_valid = aksi != "Pilih aksi..."

                    kpi = st.slider("KPI kontraktor (%)", 0, 100, 0)

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

                submitted = st.form_submit_button("Simpan Data Site Execution")

                if submitted:
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
                        st.success("Data tersimpan. Melanjutkan ke tahap berikutnya...")
                        next_step()
                        st.rerun()

    render_nav(show_next=stage_status(active, "site_execution"),
               next_label="Lanjut ke Quality Control")


# =========================================================
# STEP 5 — QUALITY CONTROL
# =========================================================
elif step == 5:
    st.markdown("## Tahap 5 — Quality Control & Handover")

    live = live_data.get("quality_control", {})
    if live:
        c1, c2 = st.columns(2)
        c1.metric("Status akhir", live.get("final_status", "-"))
        c2.metric("BAST", live.get("bast_number", "-"))
        st.caption(f"Terakhir disimpan: {live.get('timestamp', '-')}")
        st.markdown("---")

    with st.form("form_qc"):
        st.markdown("#### Inspeksi Mutu & Test Commissioning")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mat_insp = st.selectbox("Material", ["Pilih...", "PASS", "FAIL"])
        with c2:
            ndt = st.selectbox("NDT / Weld", ["Pilih...", "PASS", "FAIL"])
        with c3:
            hydro = st.selectbox("Hydrotest", ["Pilih...", "PASS", "FAIL"])
        with c4:
            visual = st.selectbox("Visual", ["Pilih...", "PASS", "FAIL"])

        st.markdown("#### Inspeksi Akhir")
        final_status = st.radio(
            "Hasil inspeksi akhir sesuai kontrak?",
            ["Pilih...", "Ya", "Tidak"], index=0
        )

        if final_status == "Ya":
            st.markdown("#### Berita Acara Serah Terima")
            c1, c2, c3 = st.columns(3)
            with c1:
                bast_date = st.date_input("Tanggal BAST", datetime(2024, 12, 20))
            with c2:
                bast_number = st.text_input("Nomor BAST", placeholder="BAST/KPOP/2024/047")
            with c3:
                approved_by = st.text_input("Disetujui oleh", placeholder="Nama lengkap")
            bast_notes = st.text_area(
                "Catatan BAST", placeholder="Catatan minimal 10 karakter...", height=80
            )
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

        submitted = st.form_submit_button("Simpan Data Quality Control")

        if submitted:
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
                st.rerun()

    render_nav(show_next=False)


# =========================================================
# SECTION LAPORAN & PORTFOLIO (selalu di bawah stepper)
# =========================================================
st.markdown("---")
with st.expander("Laporan Project", expanded=False):
    tender = live_data.get("tender", {})
    vs = live_data.get("vendor_selection", {})
    se = live_data.get("site_execution", {})
    qc = live_data.get("quality_control", {})

    if tender:
        c1, c2 = st.columns(2)
        c1.metric("RAB Awal", format_rupiah(tender.get("rab_awal", 0)))
        c2.metric("Budget Approved", format_rupiah(tender.get("budget", 0)))

    if vs:
        c1, c2, c3 = st.columns(3)
        c1.metric("Vendor Terpilih", vs.get("selected_vendor", "-"))
        c2.metric("Nilai Kontrak", format_rupiah(vs.get("selected_cost", 0)))
        c3.metric("Durasi", f"{vs.get('selected_duration', 0)} hari")

    if se:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rencana", f"{se.get('plan_pct', 0):.1f}%")
        c2.metric("Realisasi", f"{se.get('actual_pct', 0):.1f}%")
        c3.metric("Deviasi", f"{se.get('deviation_pct', 0):+.1f}%")
        c4.metric("SPI", f"{se.get('spi', 0):.3f}")

    if qc:
        c1, c2 = st.columns(2)
        c1.metric("Status Akhir", qc.get("final_status", "-"))
        c2.metric("BAST", qc.get("bast_number", "-"))

with st.expander("Portfolio — Semua Project", expanded=False):
    projects = st.session_state.projects
    if projects:
        summary_rows = []
        for pid_, p in projects.items():
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
                "Budget": format_rupiah(m["budget"]),
                "Tahapan": f"{done}/{total}",
                "Vendor Terpilih": vs.get("selected_vendor", "-"),
                "Nilai Kontrak": format_rupiah(vs.get("selected_cost", 0))
                if vs.get("selected_cost") else "-",
                "Status": status,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Vendor Selection & Project Control · Multi-project workspace · Data simulasi.")
