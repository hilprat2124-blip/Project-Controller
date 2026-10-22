import io
import streamlit as st

try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
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
    /* S-Curve table */
    .sc-table {
        font-size: 0.78rem;
        border-collapse: collapse;
        width: 100%;
    }
    .sc-table th, .sc-table td {
        border: 1px solid #d1d5db;
        padding: 4px 8px;
        text-align: center;
    }
    .sc-table th {
        background: #1e40af;
        color: white;
        font-weight: 600;
    }
    .sc-table td.text-left {
        text-align: left;
    }
    .sc-table tr.total-row td {
        background: #fef3c7;
        font-weight: 600;
    }
    .sc-table tr.cum-row td {
        background: #dbeafe;
        font-weight: 600;
    }
    .sc-table tr.actual-cum-row td {
        background: #fecaca;
        font-weight: 600;
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


def _solve_direct(vendor_names, costs, budget, scores, weight_price, weight_perf):
    """Pemilihan langsung (tanpa solver) dengan fungsi tujuan yang sama dengan LP.
    Dipakai sebagai cadangan jika solver PuLP tidak tersedia di server."""
    cand = [n for n in vendor_names if costs[n] <= budget]
    if not cand:
        return None, "Infeasible"
    if scores is None:
        scores = {n: 100 for n in vendor_names}
    min_cost = min(costs[n] for n in vendor_names)
    max_cost = max(costs[n] for n in vendor_names)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1

    def objective(n):
        cost_score = 100 * (max_cost - costs[n]) / cost_range
        return weight_price * cost_score / 100 + weight_perf * scores[n] / 100

    best = max(cand, key=lambda n: (objective(n), -costs[n]))
    return best, "Optimal (direct)"


def solve_vendor_lp(vendor_names, costs, budget, scores=None,
                    weight_price=0.5, weight_perf=0.5):
    if not vendor_names:
        return None, "Tidak ada vendor"
    try:
        prob = pulp.LpProblem("Vendor_Selection", pulp.LpMaximize)
        x = _create_binary_vars(prob, "select", vendor_names)

        if scores is None:
            scores = {n: 100 for n in vendor_names}

        min_cost = min(costs.values())
        max_cost = max(costs.values())
        cost_range = max_cost - min_cost if max_cost > min_cost else 1

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

        # cari solver yang tersedia di server
        solver = None
        for solver_name in ("PULP_CBC_CMD", "HiGHS", "HiGHS_CMD", "COIN_CMD"):
            try:
                cand = getattr(pulp, solver_name)(msg=False)
                if cand.available():
                    solver = cand
                    break
            except Exception:
                continue
        if solver is None:
            return _solve_direct(vendor_names, costs, budget, scores,
                                 weight_price, weight_perf)

        prob.solve(solver)
        status = pulp.LpStatus.get(prob.status, "Unknown")
        if status == "Optimal":
            for n in vendor_names:
                val = pulp.value(x[n])
                if val is not None and val > 0.5:
                    return n, status
        return _solve_direct(vendor_names, costs, budget, scores,
                             weight_price, weight_perf)
    except Exception:
        return _solve_direct(vendor_names, costs, budget, scores,
                             weight_price, weight_perf)


# ---------------------------------------------------------
# HELPER IMPORT EXCEL BOQ
# ---------------------------------------------------------
BOQ_COLUMNS = ["No", "Item Pekerjaan", "Volume", "Satuan", "Spesifikasi", "Kompleksitas"]

BOQ_ALIASES = {
    "no": "No", "nomor": "No", "nomor item": "No",
    "item pekerjaan": "Item Pekerjaan", "item": "Item Pekerjaan",
    "uraian pekerjaan": "Item Pekerjaan", "uraian": "Item Pekerjaan",
    "pekerjaan": "Item Pekerjaan", "deskripsi": "Item Pekerjaan",
    "description": "Item Pekerjaan", "nama pekerjaan": "Item Pekerjaan",
    "volume": "Volume", "vol": "Volume", "qty": "Volume", "kuantitas": "Volume",
    "satuan": "Satuan", "sat": "Satuan", "unit": "Satuan", "uom": "Satuan",
    "spesifikasi": "Spesifikasi", "spek": "Spesifikasi", "specification": "Spesifikasi",
    "kompleksitas": "Kompleksitas", "complexity": "Kompleksitas",
}


def _clean_volume(v):
    s = str(v).strip()
    if s.lower() in ("nan", "none"):
        return ""
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def _norm_complexity(v):
    s = str(v).strip().lower()
    if s in ("rendah", "low", "r"):
        return "Rendah"
    if s in ("tinggi", "high", "t"):
        return "Tinggi"
    return "Sedang"


def _read_upload(file):
    """Baca file upload (Excel/CSV) -> DataFrame (semua kolom string)."""
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, dtype=str)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, dtype=str, encoding="latin-1")
    if not HAS_OPENPYXL and name.endswith(".xlsx"):
        raise ValueError(
            "Library 'openpyxl' belum terpasang di server. "
            "Tambahkan 'openpyxl' ke requirements.txt, atau upload file CSV."
        )
    try:
        return pd.read_excel(file, dtype=str)
    except ImportError:
        raise ValueError(
            "Library pembaca Excel belum terpasang (openpyxl untuk .xlsx, xlrd untuk .xls). "
            "Simpan file sebagai .xlsx atau CSV."
        )


VENDOR_ALIASES = ["vendor", "nama vendor", "nama perusahaan", "perusahaan",
                  "kontraktor", "supplier", "company", "nama"]


def parse_vendor_excel(file):
    """Baca daftar vendor dari Excel/CSV -> list nama unik. Raise ValueError jika kosong."""
    df = _read_upload(file)
    col = None
    for c in df.columns:
        if str(c).strip().lower() in VENDOR_ALIASES:
            col = c
            break
    if col is None:
        col = df.columns[0] if len(df.columns) else None  # fallback: kolom pertama
    if col is None:
        raise ValueError("File kosong.")
    names = []
    for v in df[col].dropna().astype(str):
        v = v.strip()
        if v and v.lower() != "nan" and v not in names:
            names.append(v)
    if not names:
        raise ValueError("Tidak ada nama vendor yang valid di file.")
    return names


def vendor_template_bytes():
    """Return (data, filename, mime) template daftar vendor."""
    df = pd.DataFrame({"Vendor": INVITED_VENDORS})
    if HAS_OPENPYXL:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Vendor")
            return (
                buf.getvalue(),
                "template_vendor.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            pass
    return df.to_csv(index=False).encode("utf-8-sig"), "template_vendor.csv", "text/csv"


def parse_boq_excel(file):
    """Baca file Excel/CSV BOQ -> list of dict. Mendukung multi-sheet dan format RAB kompleks."""
    name = getattr(file, "name", "").lower()
    all_parsed_items = []
    
    if name.endswith(".csv"):
        # Jika CSV, langsung baca sebagai satu sheet
        try:
            df_raw = pd.read_csv(file, dtype=str, header=None)
        except UnicodeDecodeError:
            file.seek(0)
            df_raw = pd.read_csv(file, dtype=str, encoding="latin-1", header=None)
        sheets = {"CSV": df_raw}
    else:
        if not HAS_OPENPYXL:
            raise ValueError("Library 'openpyxl' belum terpasang. Gunakan CSV atau install openpyxl.")
        try:
            # Baca semua sheet sekaligus
            sheets = pd.read_excel(file, sheet_name=None, dtype=str, header=None)
        except Exception as e:
            raise ValueError(f"Gagal membaca file Excel: {e}")

    # Iterasi setiap sheet
    for sheet_name, df_raw in sheets.items():
        if df_raw.empty:
            continue

        # 1. Deteksi Baris Header
        header_idx = None
        for i, row in df_raw.iterrows():
            # Ubah semua nilai baris menjadi string huruf kecil, abaikan yang kosong
            row_str = [str(x).strip().lower() for x in row.values if pd.notna(x)]
            
            # Cari baris yang memiliki 'volume' DAN ('satuan'/'unit') DAN ('uraian'/'deskripsi'/'pekerjaan')
            has_vol = any("volume" in s or "vol" in s for s in row_str)
            has_unit = any("satuan" in s or "unit" in s for s in row_str)
            has_desc = any("uraian" in s or "deskripsi" in s or "description" in s or "pekerjaan" in s for s in row_str)
            
            if has_vol and has_unit and has_desc:
                header_idx = i
                break
                
        # Jika header tidak ditemukan di sheet ini, lewati
        if header_idx is None:
            continue
            
        # Set header ke baris yang terdeteksi (paksa jadi list of string)
        df = df_raw.iloc[header_idx + 1:].copy()
        raw_cols = df_raw.iloc[header_idx].values
        
        # Bersihkan nama kolom dari float/int (misal: 1.0 -> "1", NaN -> "Unnamed")
        clean_cols = []
        for c in raw_cols:
            if pd.isna(c):
                clean_cols.append("Unnamed")
            else:
                clean_cols.append(str(c).strip())
                
        # Tangani duplikat nama kolom dengan menambahkan suffix
        final_cols = []
        seen = set()
        for c in clean_cols:
            base_c = c
            counter = 1
            while c in seen:
                c = f"{base_c}_{counter}"
                counter += 1
            seen.add(c)
            final_cols.append(c)
            
        df.columns = final_cols

        # 2. Samakan nama kolom menggunakan alias
        rename = {}
        for c in df.columns:
            key = str(c).strip().lower()
            if key in BOQ_ALIASES:
                rename[c] = BOQ_ALIASES[key]
            else:
                for alias, target in BOQ_ALIASES.items():
                    if alias in key:
                        rename[c] = target
                        break
                        
        df = df.rename(columns=rename)

        # 3. Validasi kolom wajib (Item Pekerjaan)
        if "Item Pekerjaan" not in df.columns:
            # Fallback: cari kolom teks terpanjang (kemungkinan nama pekerjaan)
            best_col = None
            max_len = 0
            for c in df.columns:
                # Hitung rata-rata panjang teks di kolom ini
                try:
                    avg_len = df[c].dropna().astype(str).str.len().mean()
                    if pd.notna(avg_len) and avg_len > max_len:
                        max_len = avg_len
                        best_col = c
                except Exception:
                    pass
            
            if best_col and max_len > 5:
                df = df.rename(columns={best_col: "Item Pekerjaan"})
            else:
                continue # Jika tetap tidak ada, lewati sheet ini

        # Pastikan semua kolom standar ada
        for col in BOQ_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # 4. Bersihkan data
        # Ganti NaN dengan string kosong
        for col in BOQ_COLUMNS:
            df[col] = df[col].fillna("").astype(str).str.strip()
            
        # Hapus baris yang kosong melompong (tidak ada item DAN tidak ada volume)
        df = df[~((df["Item Pekerjaan"] == "") & (df["Volume"] == ""))]
        
        # Format Volume
        df["Volume"] = df["Volume"].apply(_clean_volume)
        
        # Filter: Hanya ambil baris yang memiliki nama pekerjaan
        df = df[df["Item Pekerjaan"].str.len() > 0]
        
        # Normalisasi Kompleksitas
        df["Kompleksitas"] = df["Kompleksitas"].apply(_norm_complexity)
        
        # Tambahkan ke list global
        if not df.empty:
            all_parsed_items.extend(df.to_dict("records"))

    if not all_parsed_items:
        raise ValueError(
            "Tidak ada data BOQ yang bisa dibaca. Pastikan file memiliki baris Header "
            "yang mengandung kata 'Uraian/Deskripsi', 'Volume', dan 'Satuan'."
        )

    # 5. Penomoran Ulang untuk semua item
    for i, item in enumerate(all_parsed_items, start=1):
        item["No"] = i
        
    # Ambil hanya kolom standar
    final_items = [{k: item.get(k, "") for k in BOQ_COLUMNS} for item in all_parsed_items]
    
    return final_items


def boq_template_bytes():
    """Return (data, filename, mime) template RAB sesuai standar Excel."""
    # Membuat DataFrame template yang menyerupai format RAB R.05
    template_data = [
        {"No": 1, "Item Pekerjaan": "BAB I. PEKERJAAN PENDAHULUAN", "Volume": "", "Satuan": "", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 2, "Item Pekerjaan": "Mobilisasi dan Demobilisasi", "Volume": "1", "Satuan": "ls", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 3, "Item Pekerjaan": "Site Management Proyek", "Volume": "1", "Satuan": "ls", "Spesifikasi": "", "Kompleksitas": "Sedang"},
        {"No": 4, "Item Pekerjaan": "BAB II. BANGUNAN OFFICE", "Volume": "", "Satuan": "", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 5, "Item Pekerjaan": "II.1 PEKERJAAN STRUKTUR", "Volume": "", "Satuan": "", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 6, "Item Pekerjaan": "Pekerjaan Pondasi Footplate (FP1)", "Volume": "5.5", "Satuan": "m³", "Spesifikasi": "Beton K-300", "Kompleksitas": "Tinggi"},
        {"No": 7, "Item Pekerjaan": "Pembesian Footplate", "Volume": "1560.75", "Satuan": "kg", "Spesifikasi": "BJTS U42", "Kompleksitas": "Tinggi"},
        {"No": 8, "Item Pekerjaan": "Pekerjaan Kolom K1", "Volume": "5.47", "Satuan": "m³", "Spesifikasi": "Beton K-300", "Kompleksitas": "Tinggi"},
        {"No": 9, "Item Pekerjaan": "II.2 PEKERJAAN ARSITEKTUR", "Volume": "", "Satuan": "", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 10, "Item Pekerjaan": "Pek. Pasangan Dinding Bata Ringan", "Volume": "251.64", "Satuan": "m²", "Spesifikasi": "Tebal 10 cm", "Kompleksitas": "Sedang"},
        {"No": 11, "Item Pekerjaan": "Pek. Plasteran", "Volume": "398.11", "Satuan": "m²", "Spesifikasi": "Mortar MU-302", "Kompleksitas": "Sedang"},
        {"No": 12, "Item Pekerjaan": "BAB III. CALIBRATION ROOM", "Volume": "", "Satuan": "", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 13, "Item Pekerjaan": "III.1 PEKERJAAN STRUKTUR", "Volume": "", "Satuan": "", "Spesifikasi": "", "Kompleksitas": "Rendah"},
        {"No": 14, "Item Pekerjaan": "Pekerjaan GWT (Ground Water Tank)", "Volume": "3.03", "Satuan": "m³", "Spesifikasi": "Beton K-300", "Kompleksitas": "Tinggi"}
    ]
    df = pd.DataFrame(template_data)
    
    if HAS_OPENPYXL:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Template BOQ")
                
                # Atur lebar kolom agar rapi saat dibuka di Excel
                worksheet = writer.sheets["Template BOQ"]
                worksheet.column_dimensions['A'].width = 5
                worksheet.column_dimensions['B'].width = 50
                worksheet.column_dimensions['C'].width = 10
                worksheet.column_dimensions['D'].width = 10
                worksheet.column_dimensions['E'].width = 30
                worksheet.column_dimensions['F'].width = 15
                
            return (
                buf.getvalue(),
                "template_boq_rab.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            pass
            
    return (
        df.to_csv(index=False).encode("utf-8-sig"),
        "template_boq_rab.csv",
        "text/csv",
    )


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
        "items_ver": 0,
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
    # Default 11 item pekerjaan dengan harga, bobot, dan rentang minggu
    return pd.DataFrame([
        {"No": "A", "Item Pekerjaan": "Pekerjaan Persiapan", "Harga (juta Rp)": 28.0,
         "Mulai": 1, "Selesai": 4, "Realisasi (%)": 100.0},
        {"No": "B", "Item Pekerjaan": "Pekerjaan Tanah", "Harga (juta Rp)": 38.0,
         "Mulai": 2, "Selesai": 6, "Realisasi (%)": 100.0},
        {"No": "C", "Item Pekerjaan": "Pekerjaan Struktur", "Harga (juta Rp)": 60.0,
         "Mulai": 4, "Selesai": 9, "Realisasi (%)": 85.0},
        {"No": "D", "Item Pekerjaan": "Pekerjaan Dinding", "Harga (juta Rp)": 32.0,
         "Mulai": 5, "Selesai": 10, "Realisasi (%)": 60.0},
        {"No": "E", "Item Pekerjaan": "Pekerjaan Atap", "Harga (juta Rp)": 20.0,
         "Mulai": 6, "Selesai": 11, "Realisasi (%)": 45.0},
        {"No": "F", "Item Pekerjaan": "Pekerjaan Plesteran", "Harga (juta Rp)": 16.0,
         "Mulai": 8, "Selesai": 12, "Realisasi (%)": 20.0},
        {"No": "G", "Item Pekerjaan": "Pekerjaan Acian", "Harga (juta Rp)": 12.0,
         "Mulai": 9, "Selesai": 13, "Realisasi (%)": 10.0},
        {"No": "H", "Item Pekerjaan": "Pekerjaan Pintu", "Harga (juta Rp)": 24.0,
         "Mulai": 10, "Selesai": 14, "Realisasi (%)": 0.0},
        {"No": "I", "Item Pekerjaan": "Pekerjaan Plafond", "Harga (juta Rp)": 20.0,
         "Mulai": 10, "Selesai": 14, "Realisasi (%)": 0.0},
        {"No": "J", "Item Pekerjaan": "Pekerjaan Lantai", "Harga (juta Rp)": 20.0,
         "Mulai": 11, "Selesai": 15, "Realisasi (%)": 0.0},
        {"No": "K", "Item Pekerjaan": "Pekerjaan Pengecatan", "Harga (juta Rp)": 16.0,
         "Mulai": 12, "Selesai": 16, "Realisasi (%)": 0.0},
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
# =========================================================with st.sidebar:
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
                new_name.strip(), new_client.strip() or "-",
                new_location.strip() or "-", new_budget, new_deadline,
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
        st.markdown("**Budget & Deadline Project**")
        new_budget_edit = st.number_input(
            "Budget project (juta Rp)", min_value=10.0,
            value=float(active["meta"]["budget"]),
            step=10.0, key=f"edit_budget_{active['meta']['id']}",
        )
        new_deadline_edit = st.number_input(
            "Deadline project (hari)", min_value=1,
            value=int(active["meta"]["deadline"]),
            step=5, key=f"edit_deadline_{active['meta']['id']}"
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
        pass

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
    st.info("Belum ada project aktif.")
    st.stop()

live_data = active["live_data"]
meta = active["meta"]
budget = float(meta["budget"])
deadline = int(meta["deadline"])
pid = st.session_state.active_project

# =========================================================
# STEPPER
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
        "Alur: Klien membuka tender → kita siapkan BOQ → vendor diundang untuk isi penawaran harga & durasi."
        "</div>",
        unsafe_allow_html=True,
    )

    sub1, sub2 = st.tabs(["1. Dokumen & Timeline Tender", "2. BOQ — Bill of Quantities"])

    with sub1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Dokumen Tender")
            doc_drawing = st.checkbox("Gambar kerja", value=True, key="td_drawing")
            doc_rks = st.checkbox("RKS", value=True, key="td_rks")
            doc_spek = st.checkbox("Spesifikasi teknis", value=True, key="td_spek")
            doc_boq = st.checkbox("Bill of Quantities", value=True, key="td_boq_check")
            doc_k3 = st.checkbox("Dokumen K3 / HSE Plan", value=True, key="td_k3")

            st.markdown("#### Catatan Aanwijzing")
            catatan = st.text_area(
                "Ringkasan", placeholder="Minimal 10 karakter...",
                height=120, key="td_catatan",
            )
        with col2:
            st.markdown("#### RAB Awal (Estimasi Kita)")
            st.markdown(
                f"<div class='hint-box'>Budget project: <b>{format_rupiah(meta['budget'])}</b>.</div>",
                unsafe_allow_html=True,
            )
            rab_awal_input = st.number_input(
                "RAB Estimasi (juta Rp)", min_value=10.0,
                value=float(meta.get("rab_awal", meta.get("budget", 300.0))),
                step=10.0, key=f"rab_awal_{meta['id']}",
            )
            budget_input = st.number_input(
                "Budget approved (juta Rp)", min_value=10.0,
                value=float(meta.get("budget", 300.0)),
                step=10.0, key=f"budget_tender_{meta['id']}",
            )
            st.metric("Selisih", format_rupiah(budget_input - rab_awal_input))

            st.markdown("#### Timeline")
            tgl_aanwijzing = st.date_input("Aanwijzing", datetime(2024, 9, 7), key="td_tgl_aan")
            tgl_penawaran = st.date_input("Batas penawaran", datetime(2024, 9, 15), key="td_tgl_pen")
            tgl_kontrak = st.date_input("Target kontrak", datetime(2024, 9, 30), key="td_tgl_kon")

    with sub2:
        st.markdown("### BOQ — Bill of Quantities")

        if active.get("items_data") is None:
            active["items_data"] = default_items().to_dict("records")
        if "items_ver" not in active:
            active["items_ver"] = 0

        # ---------- IMPORT EXCEL ----------
        with st.expander("📥 Import BOQ dari Excel", expanded=False):
            if not HAS_OPENPYXL:
                st.warning(
                    "Library 'openpyxl' belum terpasang, jadi import Excel (.xlsx) dinonaktifkan. "
                    "Gunakan CSV, atau tambahkan 'openpyxl' ke requirements.txt lalu redeploy."
                )
            c_up, c_tpl = st.columns([3, 1])
            with c_tpl:
                tpl_data, tpl_name, tpl_mime = boq_template_bytes()
                st.download_button(
                    "Unduh Template",
                    data=tpl_data,
                    file_name=tpl_name,
                    mime=tpl_mime,
                    key=f"tpl_boq_{pid}",
                )
            with c_up:
                up_types = ["xlsx", "xls", "csv"] if HAS_OPENPYXL else ["csv"]
                up_file = st.file_uploader(
                    "Upload file (" + " / ".join("." + t for t in up_types) + ")",
                    type=up_types,
                    key=f"boq_upload_{pid}_{active['items_ver']}",
                )

            if up_file is not None:
                try:
                    parsed = parse_boq_excel(up_file)
                    st.success(f"Terbaca {len(parsed)} item. Preview:")
                    st.dataframe(pd.DataFrame(parsed), use_container_width=True, hide_index=True)

                    mode = st.radio(
                        "Mode impor",
                        ["Ganti semua BOQ", "Tambahkan ke BOQ yang ada"],
                        horizontal=True, key=f"boq_mode_{pid}",
                    )
                    if st.button("Terapkan ke BOQ", type="primary", key=f"boq_apply_{pid}"):
                        if mode == "Ganti semua BOQ":
                            new_items = parsed
                        else:
                            new_items = list(active["items_data"]) + parsed
                        for i, it in enumerate(new_items, start=1):
                            it["No"] = i
                        active["items_data"] = new_items
                        active["items_ver"] += 1   # reset editor & uploader
                        st.rerun()
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

        st.caption(
            "📝 **Tabel ini bisa diedit langsung** — tambah baris, hapus, ubah nama item, "
            "volume, satuan, spesifikasi, dan kompleksitas."
        )

        items_df = pd.DataFrame(active["items_data"])
        items_edited = st.data_editor(
            items_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=f"items_editor_{pid}_{active['items_ver']}",
            column_config={
                "No": st.column_config.NumberColumn("No", min_value=1, step=1, width="small"),
                "Item Pekerjaan": st.column_config.TextColumn("Item Pekerjaan", required=True, width="large"),
                "Volume": st.column_config.TextColumn("Volume", width="small"),
                "Satuan": st.column_config.TextColumn("Satuan", width="small"),
                "Spesifikasi": st.column_config.TextColumn("Spesifikasi", width="medium"),
                "Kompleksitas": st.column_config.SelectboxColumn(
                    "Kompleksitas", options=["Rendah", "Sedang", "Tinggi"], width="small"
                ),
            },
        )
        active["items_data"] = items_edited.to_dict("records")
        st.caption(f"Total item: **{len(items_edited)}**")

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
        "Alur: Vendor diundang → mereka isi penawaran harga & durasi → "
        "klarifikasi teknis → optimasi → deal harga & SPK."
        "</div>",
        unsafe_allow_html=True,
    )

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "1. BOQ & Vendor Diundang",
        "2. Input Penawaran Harga & Durasi",
        "3. Appointment Klarifikasi Teknis",
        "4. Optimasi Setelah Klarifikasi",
        "5. Deal Harga & SPK",
    ])

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

        if "invited_ver" not in active:
            active["invited_ver"] = 0

        # ---------- IMPORT EXCEL VENDOR ----------
        with st.expander("📥 Import Vendor dari Excel", expanded=False):
            if not HAS_OPENPYXL:
                st.warning(
                    "Library 'openpyxl' belum terpasang, jadi import Excel (.xlsx) dinonaktifkan. "
                    "Gunakan CSV, atau tambahkan 'openpyxl' ke requirements.txt lalu redeploy."
                )
            cv_up, cv_tpl = st.columns([3, 1])
            with cv_tpl:
                vt_data, vt_name, vt_mime = vendor_template_bytes()
                st.download_button(
                    "Unduh Template",
                    data=vt_data, file_name=vt_name, mime=vt_mime,
                    key=f"tpl_vendor_{pid}",
                )
            with cv_up:
                v_types = ["xlsx", "xls", "csv"] if HAS_OPENPYXL else ["csv"]
                v_file = st.file_uploader(
                    "Upload file (" + " / ".join("." + t for t in v_types) + ")",
                    type=v_types,
                    key=f"vendor_upload_{pid}_{active['invited_ver']}",
                )

            if v_file is not None:
                try:
                    v_parsed = parse_vendor_excel(v_file)
                    st.success(f"Terbaca {len(v_parsed)} vendor. Preview:")
                    st.dataframe(pd.DataFrame({"Vendor": v_parsed}),
                                 use_container_width=True, hide_index=True)
                    v_mode = st.radio(
                        "Mode impor",
                        ["Ganti semua vendor", "Tambahkan ke daftar yang ada"],
                        horizontal=True, key=f"vendor_mode_{pid}",
                    )
                    if st.button("Terapkan ke Daftar Vendor", type="primary",
                                 key=f"vendor_apply_{pid}"):
                        if v_mode == "Ganti semua vendor":
                            new_list = v_parsed
                        else:
                            new_list = list(active["invited_vendors"])
                            for nm in v_parsed:
                                if nm not in new_list:
                                    new_list.append(nm)
                        active["invited_vendors"] = new_list
                        active["invited_ver"] += 1  # reset editor & uploader
                        st.rerun()
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

        invited_df = pd.DataFrame({"Vendor": active["invited_vendors"]})
        invited_edited = st.data_editor(
            invited_df, use_container_width=True, hide_index=True,
            num_rows="dynamic", key=f"invited_{pid}_{active['invited_ver']}",
            column_config={"Vendor": st.column_config.TextColumn("Vendor", required=True)},
        )
        active["invited_vendors"] = invited_edited["Vendor"].dropna().tolist()

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
                        "catatan_vendor": "",
                    }

            vendor_pick = st.selectbox("Pilih vendor", options=invited, key=f"vp_{pid}")
            v_data = active["vendor_data"][vendor_pick]

            st.markdown(f"#### A. Penawaran Harga & Durasi — {vendor_pick}")
            c1, c2 = st.columns(2)
            with c1:
                v_data["harga_rab"] = st.number_input(
                    "Penawaran Harga (juta Rp)", min_value=0.0,
                    value=float(v_data["harga_rab"]), step=5.0,
                    key=f"rab_{vendor_pick}",
                )
            with c2:
                v_data["durasi"] = st.number_input(
                    "Durasi (hari)", min_value=1,
                    value=int(v_data["durasi"]) if v_data["durasi"] > 0 else 60,
                    step=1, key=f"dur_{vendor_pick}",
                )

            st.markdown(f"#### B. Catatan — {vendor_pick}")
            v_data["catatan_vendor"] = st.text_area(
                "Catatan", value=v_data.get("catatan_vendor", ""), height=100,
                key=f"nt_{vendor_pick}",
            )

            active["vendor_data"][vendor_pick] = v_data

            st.markdown("---")
            st.markdown("#### Ringkasan")
            rows = []
            for v in invited:
                vd = active["vendor_data"][v]
                rows.append({
                    "Vendor": v,
                    "Penawaran (jt)": vd["harga_rab"],
                    "Durasi (hari)": vd["durasi"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
                    appt_agenda = st.text_area("Agenda", placeholder="Klarifikasi WPS, NDT...", height=100, key=f"apa_{pid}")

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

    with sub4:
        st.markdown("### Optimasi Setelah Klarifikasi")
        invited = active.get("invited_vendors", [])
        vendor_data = active.get("vendor_data", {})

        prasyarat = []
        if not invited:
            prasyarat.append("vendor diundang")
        if not vendor_data or all(v["harga_rab"] == 0 for v in vendor_data.values()):
            prasyarat.append("penawaran harga vendor")
        if not active["appointments"]:
            prasyarat.append("appointment klarifikasi")

        if prasyarat:
            st.warning("Belum bisa optimasi. Selesaikan: " + ", ".join(prasyarat))
        else:
            st.caption("Optimasi memilih vendor dengan **penawaran harga terendah** yang masih dalam budget.")

            harga = {}
            for v in invited:
                vd = vendor_data.get(v, {})
                harga[v] = vd.get("harga_rab", 0)

            st.markdown("#### Data untuk Optimasi")
            rows = []
            for v in invited:
                rows.append({
                    "Vendor": v,
                    "Penawaran (juta Rp)": format_rupiah(harga[v]),
                    "Durasi (hari)": vendor_data.get(v, {}).get("durasi", 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            eligible = [
                v for v in invited
                if harga[v] > 0 and harga[v] <= budget
            ]

            if not eligible:
                st.error("Tidak ada vendor yang lolos kriteria.")
            else:
                cc1, cc2 = st.columns(2)
                cc1.metric("Budget", format_rupiah(budget))
                cc2.metric("Vendor lolos budget", f"{len(eligible)} dari {len(invited)}")

                st.markdown("---")
                selected_name, solver_status = solve_vendor_lp(
                    eligible, harga, budget, None, 1.0, 0.0
                )

                if selected_name is None:
                    st.error(f"Optimasi gagal: {solver_status}")
                else:
                    selected_cost = harga[selected_name]
                    selected_duration = vendor_data.get(selected_name, {}).get("durasi", 0)

                    st.success(f"**Vendor terbaik: {selected_name}**")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Vendor", selected_name)
                    r2.metric("Harga Penawaran", format_rupiah(selected_cost))
                    r3.metric("Durasi", f"{selected_duration} hari")
                    r4.metric("Sisa budget", format_rupiah(budget - selected_cost))

                    st.session_state[f"optimized_vendor_{pid}"] = selected_name
                    st.session_state[f"optimized_cost_{pid}"] = selected_cost

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
                f"<b>Harga penawaran awal:</b> {format_rupiah(optimized_cost)}"
                f"</div>",
                unsafe_allow_html=True,
            )

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

            negosiasi_notes = st.text_area("Catatan Negosiasi", height=100, key=f"ngn_{pid}")

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
# STEP 3 — SITE EXECUTION (S-CURVE TABEL + GRAFIK)
# =========================================================
elif step == 3:
    st.markdown("## Tahap 3 — Site Execution")
    st.markdown(
        "<div class='flow-hint'>"
        "Isi tabel WBS item pekerjaan — tiap item punya harga, bobot, dan rentang minggu. "
        "Sistem akan hitung distribusi bobot per minggu, total per minggu, dan progress kumulatif "
        "dalam format tabel seperti Excel, plus grafik S-Curve."
        "</div>",
        unsafe_allow_html=True,
    )

    vs = live_data.get("vendor_selection", {})
    selected_vendor = vs.get("selected_vendor", "-")
    final_cost = vs.get("selected_cost", 0)

    st.caption(
        f"Kontraktor: {selected_vendor} · Nilai kontrak: {format_rupiah(final_cost)}"
    )

    # ---------------------------------------------------------
    # TABEL WBS dengan harga (bisa diedit)
    # ---------------------------------------------------------
    st.markdown("### Tabel Rencana Kerja (WBS)")

    if active.get("wbs_data") is None:
        active["wbs_data"] = default_wbs().to_dict("records")

    wbs_df = pd.DataFrame(active["wbs_data"])
    wbs_edited = st.data_editor(
        wbs_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=f"wbs_editor_{pid}",
        column_config={
            "No": st.column_config.TextColumn("No", width="small", help="Nomor/abjad urut (A, B, C atau 1, 2, 3)"),
            "Item Pekerjaan": st.column_config.TextColumn("Uraian Pekerjaan", required=True, width="large"),
            "Harga (juta Rp)": st.column_config.NumberColumn(
                "Harga (juta Rp)", min_value=0.0, step=1.0, format="%.1f",
            ),
            "Mulai": st.column_config.NumberColumn("Mulai (M)", min_value=1, step=1, width="small"),
            "Selesai": st.column_config.NumberColumn("Selesai (M)", min_value=1, step=1, width="small"),
            "Realisasi (%)": st.column_config.NumberColumn(
                "Real. (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
            ),
        },
    )
    active["wbs_data"] = wbs_edited.to_dict("records")

    if wbs_edited.empty:
        st.warning("Tabel WBS kosong.")
        render_nav(show_next=False)
        st.stop()

    # ---------------------------------------------------------
    # HITUNG BOBOT OTOMATIS dari harga
    # ---------------------------------------------------------
    total_harga = float(wbs_edited["Harga (juta Rp)"].sum())
    if total_harga > 0:
        wbs_edited["Bobot (%)"] = (wbs_edited["Harga (juta Rp)"] / total_harga * 100).round(2)
    else:
        wbs_edited["Bobot (%)"] = 0.0

    # ---------------------------------------------------------
    # HITUNG DISTRIBUSI BOBOT PER MINGGU (format tabel)
    # ---------------------------------------------------------
    max_week = int(wbs_edited["Selesai"].max())
    weeks = list(range(1, max_week + 1))

    # matrix distribusi: rows = item, cols = minggu
    distrib_matrix = []
    for _, row in wbs_edited.iterrows():
        w_start = int(row["Mulai"])
        w_end = int(row["Selesai"])
        bobot = float(row["Bobot (%)"])
        span = w_end - w_start + 1

        week_vals = [0.0] * max_week
        if span > 0 and bobot > 0:
            ts = np.linspace(0, 1, span + 1)
            s_vals = np.array([sigmoid_kernel(t, k=4.0) for t in ts])
            weekly = np.diff(s_vals) * bobot
            if weekly.sum() > 0:
                weekly = weekly / weekly.sum() * bobot
            for i, w in enumerate(range(w_start, w_end + 1)):
                week_vals[w - 1] = round(float(weekly[i]), 2)
        distrib_matrix.append(week_vals)

    # total per minggu
    total_per_week = np.sum(distrib_matrix, axis=0).round(2)
    cum_plan = np.cumsum(total_per_week).round(2)

    # ---------------------------------------------------------
    # REALISASI PER MINGGU (pakai kurva S yang sama, cut-off)
    # ---------------------------------------------------------
    current_week = st.number_input(
        "Minggu saat ini (untuk realisasi)",
        min_value=1, max_value=max_week,
        value=min(max_week, max(1, int(max_week * 0.6))),
        step=1, key=f"current_week_{pid}",
    )

    distrib_actual = []
    for idx, row in wbs_edited.iterrows():
        w_start = int(row["Mulai"])
        w_end = int(row["Selesai"])
        bobot = float(row["Bobot (%)"])
        realisasi_pct = float(row["Realisasi (%)"])
        span = w_end - w_start + 1

        week_vals = [0.0] * max_week
        if span > 0 and bobot > 0 and realisasi_pct > 0:
            # bentuk S penuh
            ts = np.linspace(0, 1, span + 1)
            s_vals = np.array([sigmoid_kernel(t, k=4.0) for t in ts])
            weekly_plan = np.diff(s_vals) * bobot
            if weekly_plan.sum() > 0:
                weekly_plan = weekly_plan / weekly_plan.sum() * bobot

            # realisasi = bentuk S yang sama × (realisasi_pct/100)
            weekly_actual_full = weekly_plan * (realisasi_pct / 100.0)

            # cut-off di current_week
            w_cut = min(current_week, w_end)
            if w_cut >= w_start:
                for i, w in enumerate(range(w_start, w_cut + 1)):
                    week_vals[w - 1] = round(float(weekly_actual_full[i]), 2)
        distrib_actual.append(week_vals)

    total_actual_per_week = np.sum(distrib_actual, axis=0).round(2)
    cum_actual = np.cumsum(total_actual_per_week).round(2)

    # ---------------------------------------------------------
    # TAMPILKAN TABEL S-CURVE (HTML custom untuk rapi)
    # ---------------------------------------------------------
    st.markdown("### Tabel S-Curve Mingguan")

    # Header
    header_cells = (
        "<th style='text-align:left;'>No</th>"
        "<th style='text-align:left;'>Uraian Pekerjaan</th>"
        "<th>Harga</th>"
        "<th>Bobot</th>"
    )
    for w in weeks:
        header_cells += f"<th>M{w}</th>"

    # Baris item
    body_rows = ""
    for i, row in wbs_edited.iterrows():
        body_rows += "<tr>"
        body_rows += f"<td class='text-left'>{row['No']}</td>"
        body_rows += f"<td class='text-left'>{row['Item Pekerjaan']}</td>"
        body_rows += f"<td>Rp {row['Harga (juta Rp)']:,.0f}</td>"
        body_rows += f"<td>{row['Bobot (%)']:.2f}%</td>"
        for w_idx in range(max_week):
            val = distrib_matrix[i][w_idx]
            if val > 0:
                body_rows += f"<td>{val:.2f}%</td>"
            else:
                body_rows += "<td>-</td>"
        body_rows += "</tr>"

    # Baris Total
    body_rows += "<tr class='total-row'>"
    body_rows += "<td class='text-left'></td>"
    body_rows += "<td class='text-left'>Total</td>"
    body_rows += f"<td>Rp {total_harga:,.0f}</td>"
    body_rows += "<td>100.00%</td>"
    for val in total_per_week:
        body_rows += f"<td>{val:.2f}%</td>" if val > 0 else "<td>-</td>"
    body_rows += "</tr>"

    # Baris Progress Rencana Kumulatif
    body_rows += "<tr class='cum-row'>"
    body_rows += "<td class='text-left'></td>"
    body_rows += "<td class='text-left'>Progress Rencana Kumulatif</td>"
    body_rows += "<td></td><td></td>"
    for val in cum_plan:
        body_rows += f"<td>{val:.2f}%</td>"
    body_rows += "</tr>"

    # Baris Total Realisasi per Minggu
    body_rows += "<tr>"
    body_rows += "<td class='text-left'></td>"
    body_rows += "<td class='text-left'>Total Realisasi per Minggu</td>"
    body_rows += "<td></td><td></td>"
    for val in total_actual_per_week:
        body_rows += f"<td>{val:.2f}%</td>" if val > 0 else "<td>-</td>"
    body_rows += "</tr>"

    # Baris Progress Realisasi Kumulatif
    body_rows += "<tr class='actual-cum-row'>"
    body_rows += "<td class='text-left'></td>"
    body_rows += "<td class='text-left'>Progress Realisasi Kumulatif</td>"
    body_rows += "<td></td><td></td>"
    for val in cum_actual:
        body_rows += f"<td>{val:.2f}%</td>"
    body_rows += "</tr>"

    table_html = f"""
    <div style='overflow-x:auto;'>
        <table class='sc-table'>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # GRAFIK S-CURVE
    # ---------------------------------------------------------
    st.markdown("### Grafik S-Curve")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(weeks, cum_plan, cum_actual,
                    where=(cum_actual < cum_plan),
                    color="#dc2626", alpha=0.12, label="Deviasi negatif")
    ax.fill_between(weeks, cum_plan, cum_actual,
                    where=(cum_actual >= cum_plan),
                    color="#047857", alpha=0.12, label="Deviasi positif")

    ax.plot(weeks, cum_plan, color="#1e40af", linewidth=2.5,
            marker="o", markersize=6, label="Rencana (S-Curve)")
    ax.plot(weeks, cum_actual, color="#dc2626", linewidth=2.5,
            marker="s", markersize=6, label="Realisasi")

    ax.axvline(current_week, linestyle=":", color="#6b7280",
               linewidth=1.5, label=f"Minggu ini (M{current_week})")

    plan_now = cum_plan[current_week - 1]
    actual_now = cum_actual[current_week - 1]

    ax.scatter([current_week], [plan_now], color="#1e40af", s=100,
               zorder=5, edgecolor="white", linewidth=2)
    ax.scatter([current_week], [actual_now], color="#dc2626", s=100,
               marker="D", zorder=5, edgecolor="white", linewidth=2)

    ax.annotate(f"Rencana: {plan_now:.2f}%",
                xy=(current_week, plan_now), xytext=(10, 10),
                textcoords="offset points", color="#1e40af",
                fontsize=9, fontweight="600")
    ax.annotate(f"Realisasi: {actual_now:.2f}%",
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

    # ---------------------------------------------------------
    # KPI
    # ---------------------------------------------------------
    deviasi = actual_now - plan_now
    spi = actual_now / plan_now if plan_now > 0 else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(f"Rencana (M{current_week})", f"{plan_now:.2f}%")
    sc2.metric(f"Realisasi (M{current_week})", f"{actual_now:.2f}%")
    sc3.metric("Deviasi", f"{deviasi:+.2f}%")
    sc4.metric("SPI", f"{spi:.3f}")

    if deviasi < -5:
        st.error(f"Deviasi {deviasi:.2f}% — berpotensi terlambat.")
    elif deviasi < 0:
        st.warning(f"Deviasi {deviasi:.2f}% — perlu perhatian.")
    else:
        st.success(f"Progress sesuai/lebih cepat {deviasi:+.2f}%.")

    # ---------------------------------------------------------
    # UNDUH CSV
    # ---------------------------------------------------------
    with st.expander("Unduh Data S-Curve"):
        # bentuk dataframe dengan kolom M1..Mn
        cols = ["No", "Uraian Pekerjaan", "Harga (juta Rp)", "Bobot (%)"] + [f"M{w}" for w in weeks]
        rows_export = []
        for i, row in wbs_edited.iterrows():
            r = [row["No"], row["Item Pekerjaan"], row["Harga (juta Rp)"], row["Bobot (%)"]]
            r += distrib_matrix[i]
            rows_export.append(r)
        # total
        rows_export.append(["", "Total", total_harga, 100.00] + list(total_per_week))
        rows_export.append(["", "Progress Rencana Kumulatif", "", ""] + list(cum_plan))
        rows_export.append(["", "Total Realisasi per Minggu", "", ""] + list(total_actual_per_week))
        rows_export.append(["", "Progress Realisasi Kumulatif", "", ""] + list(cum_actual))

        df_export = pd.DataFrame(rows_export, columns=cols)
        st.dataframe(df_export, use_container_width=True, hide_index=True)
        st.download_button(
            "Unduh S-Curve (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"s_curve_{meta['name'].replace(' ', '_')}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # FORM SITE EXECUTION
    # ---------------------------------------------------------
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
                st.markdown(f"- Realisasi: {se.get('actual_pct', 0):.2f}%")
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
