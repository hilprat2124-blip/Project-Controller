import io
import re
import csv
import html
import json
import base64
import hashlib
import uuid
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

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
from supabase import create_client

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vendor Selection & Project Control",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

KPDP_DOMAIN = str(st.secrets.get("KPDP_DOMAIN", "@kpdp.co.id")).lower()

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
    div[data-testid="stButton"] button[kind="primary"] { background-color: #047857 !important; }
    div[data-testid="stButton"] button[kind="primary"]:hover { background-color: #065f46 !important; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    .stDataFrame { border: 1px solid #e5e7eb; border-radius: 6px; }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
    .pill-ok { background: #d1fae5; color: #065f46; }
    .pill-warn { background: #fef3c7; color: #92400e; }
    .pill-fail { background: #fee2e2; color: #991b1b; }
    .pill-info { background: #dbeafe; color: #1e40af; }
    [data-testid="stMetricDelta"] svg { display: none; }
    .step-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; font-size: 0.85rem; }
    .step-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .dot-ok { background: #047857; }
    .dot-idle { background: #d1d5db; }
    .stepper { display: flex; align-items: center; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
    .step-chip {
        display: flex; align-items: center; gap: 6px; padding: 8px 14px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 500; border: 1px solid #e5e7eb;
        background: #ffffff; color: #6b7280;
    }
    .step-chip.active { background: #1e40af; color: #ffffff; border-color: #1e40af; font-weight: 600; }
    .step-chip.done { background: #d1fae5; color: #065f46; border-color: #a7f3d0; }
    .step-chip .num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 20px; height: 20px; border-radius: 50%;
        background: rgba(255,255,255,0.2); font-size: 0.75rem; font-weight: 700;
    }
    .step-chip:not(.active):not(.done) .num { background: #f3f4f6; color: #6b7280; }
    .step-chip.active .num { background: rgba(255,255,255,0.25); color: #ffffff; }
    .step-chip.done .num { background: #047857; color: #ffffff; }
    .hint-box {
        background: #eff6ff; border-left: 3px solid #1e40af; padding: 8px 12px;
        border-radius: 4px; margin: 6px 0; font-size: 0.85rem; color: #1e40af;
    }
    .flow-hint {
        background: #f9fafb; border: 1px dashed #d1d5db; padding: 10px 14px;
        border-radius: 6px; font-size: 0.82rem; color: #4b5563; margin-bottom: 12px;
    }
    .sc-table { font-size: 0.78rem; border-collapse: collapse; width: 100%; }
    .sc-table th, .sc-table td { border: 1px solid #d1d5db; padding: 4px 8px; text-align: center; }
    .sc-table th { background: #1e40af; color: white; font-weight: 600; }
    .sc-table td.text-left { text-align: left; }
    .sc-table tr.total-row td { background: #fef3c7; font-weight: 600; }
    .sc-table tr.cum-row td { background: #dbeafe; font-weight: 600; }
    .sc-table tr.actual-cum-row td { background: #fecaca; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER UMUM
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


def user_tz():
    """Zona waktu browser pengguna (otomatis). Fallback ke Asia/Jakarta."""
    try:
        tz = st.context.timezone
        if tz:
            ZoneInfo(tz)  # validasi
            return tz
    except Exception:
        pass
    return "Asia/Jakarta"


def now_local():
    """Waktu sekarang sesuai zona waktu pengguna (bukan zona waktu server)."""
    return datetime.now(ZoneInfo(user_tz()))


def fmt_ts(ts):
    """Format timestamp ISO (UTC) dari database ke zona waktu pengguna."""
    if not ts:
        return "-"
    try:
        return pd.to_datetime(ts).tz_convert(user_tz()).strftime("%d %b %Y %H:%M:%S")
    except Exception:
        return str(ts)[:19].replace("T", " ")


# =========================================================
# SOLVER OPTIMASI VENDOR
# =========================================================
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
    """Pemilihan langsung (tanpa solver) dengan fungsi tujuan yang sama dengan LP."""
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
        cost_scores = {n: 100 * (max_cost - costs[n]) / cost_range for n in vendor_names}

        prob += pulp.lpSum(
            (weight_price * cost_scores[n] / 100 + weight_perf * scores[n] / 100) * x[n]
            for n in vendor_names
        )
        prob += pulp.lpSum(x[n] for n in vendor_names) == 1
        prob += pulp.lpSum(costs[n] * x[n] for n in vendor_names) <= budget

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
            return _solve_direct(vendor_names, costs, budget, scores, weight_price, weight_perf)

        prob.solve(solver)
        status = pulp.LpStatus.get(prob.status, "Unknown")
        if status == "Optimal":
            for n in vendor_names:
                val = pulp.value(x[n])
                if val is not None and val > 0.5:
                    return n, status
        return _solve_direct(vendor_names, costs, budget, scores, weight_price, weight_perf)
    except Exception:
        return _solve_direct(vendor_names, costs, budget, scores, weight_price, weight_perf)


# =========================================================
# HELPER IMPORT EXCEL BOQ
# =========================================================
BOQ_COLUMNS = ["No", "Description", "Specification", "Volume", "Unit",
               "Unit Price (Rp)", "Total Price (Rp)"]
BOQ_NUM_COLS = ["Unit Price (Rp)", "Total Price (Rp)"]
WBS_COLUMNS = ["No", "Item Pekerjaan", "Harga (juta Rp)", "Mulai", "Selesai", "Realisasi (%)"]


def _cell(v):
    try:
        if v is None or pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).replace("\u00a0", " ").strip()
    if s.lower() in ("nan", "none", "nat"):
        return ""
    return s


def _norm_text(v):
    return re.sub(r"\s+", " ", _cell(v).lower()).strip()


def _to_number(v):
    s = _cell(v)
    if not s:
        return None
    s = re.sub(r"[^\d,\.\-]", "", s)
    if s in ("", "-", ".", ",", "-."):
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if s.count(",") > 1:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _id_number(n, decimals):
    s = f"{n:,.{decimals}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _fmt_volume(v):
    s = _cell(v)
    if not s or s in ("-", "–", "—"):
        return ""
    if not re.fullmatch(r"[\s\d\.,\-]+", s):
        return s
    n = _to_number(s)
    if n is None:
        return s
    return _id_number(round(n, 2), 2)


def _fmt_price(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        try:
            n = None if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            n = None
    else:
        s = _cell(v)
        if not s or s in ("-", "–", "—"):
            return "-"
        if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):
            n = float(s.replace(".", "").replace(",", "."))
        else:
            n = _to_number(s)
    if n is None or n == 0:
        return "-"
    if abs(n - round(n)) < 0.005:
        return _id_number(round(n), 0)
    return _id_number(n, 2)


def _clean_no(v):
    s = _cell(v)
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


_PRICE_WORDS = {"harga", "jumlah", "total", "price", "amount", "biaya", "upah", "bahan", "rp"}
_VOLUME_WORDS = {"volume", "vol", "qty", "kuantitas", "quantity"}
_UNIT_WORDS = {"satuan", "sat", "unit", "uom"}
_SPEC_WORDS = {"spesifikasi", "spek", "specification", "specifications",
               "spesification", "spesifications", "spec"}
_ITEM_WORDS = {"uraian", "deskripsi", "description", "pekerjaan"}
_ITEM_WEAK_WORDS = {"item", "nama"}
_NO_WORDS = {"no", "nomor"}


def _match_target(cell, weak=False):
    key = _norm_text(cell)
    if not key:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", key))
    if not tokens or "bobot" in tokens:
        return None
    if tokens & _PRICE_WORDS:
        if tokens & {"total", "jumlah"}:
            return "TotalPrice"
        if tokens & {"harga", "price"}:
            return "UnitPrice"
        return None
    if tokens & _VOLUME_WORDS:
        return "Volume"
    if tokens & _UNIT_WORDS:
        return "Unit"
    if tokens & _SPEC_WORDS:
        return "Specification"
    if tokens & _ITEM_WORDS:
        return "Description"
    if tokens & _NO_WORDS:
        return "No"
    if weak and (tokens & _ITEM_WEAK_WORDS):
        return "Description"
    return None


def _build_mapping(cells):
    mapping = {}
    for j, c in enumerate(cells):
        t = _match_target(c)
        if t and t not in mapping:
            mapping[t] = j
    if "Description" not in mapping:
        used = set(mapping.values())
        for j, c in enumerate(cells):
            if j in used:
                continue
            if _match_target(c, weak=True) == "Description":
                mapping["Description"] = j
                break
    return mapping


def _header_ok(mapping):
    return all(k in mapping for k in ("Description", "Volume", "Unit"))


def _find_header(df_raw, max_scan=80):
    n = len(df_raw)
    for i in range(min(n, max_scan)):
        cells = list(df_raw.iloc[i].values)
        m = _build_mapping(cells)
        if _header_ok(m):
            return i, 1, m
        if i + 1 < n:
            nxt = list(df_raw.iloc[i + 1].values)
            comb = [(_cell(a) + " " + _cell(b)).strip() for a, b in zip(cells, nxt)]
            m = _build_mapping(comb)
            if _header_ok(m):
                return i, 2, m
    return None


def _column_spans(mapping, ncols):
    cols = sorted(set(mapping.values()))
    spans = {}
    for name, j in mapping.items():
        nxt = [c for c in cols if c > j]
        end = nxt[0] if nxt else ncols
        spans[name] = list(range(j, max(end, j + 1)))
    return spans


def _extract_items(df_raw):
    found = _find_header(df_raw)
    if found is None:
        return []
    header_idx, nrows, mapping = found
    data = df_raw.iloc[header_idx + nrows:].values
    ncols = data.shape[1] if data.ndim == 2 else 0
    spans = _column_spans(mapping, ncols)

    def pick(row, name):
        for j in spans.get(name, []):
            if j < ncols and _cell(row[j]):
                return row[j]
        return ""

    items = []
    for row in data:
        no = _clean_no(pick(row, "No"))
        desc = re.sub(r"\s+", " ", _cell(pick(row, "Description")))
        spec = re.sub(r"\s+", " ", _cell(pick(row, "Specification")))
        volume = _fmt_volume(pick(row, "Volume"))
        unit = _cell(pick(row, "Unit"))
        unit_price = _fmt_price(pick(row, "UnitPrice"))
        total_price = _fmt_price(pick(row, "TotalPrice"))

        if not any([no, desc, spec, volume, unit]):
            continue
        if not volume and len(_build_mapping(list(row))) >= 3:
            continue
        if desc and re.fullmatch(r"[\d\.\s,]+", desc) and not volume and not unit:
            continue

        items.append({
            "No": no, "Description": desc, "Specification": spec,
            "Volume": volume, "Unit": unit,
            "Unit Price (Rp)": unit_price, "Total Price (Rp)": total_price,
        })
    return items


_OLD_KEYS = {"Item Pekerjaan": "Description", "Spesifikasi": "Specification", "Satuan": "Unit"}


def _normalize_items(records):
    out = []
    for r in records:
        d = {}
        for k, v in dict(r).items():
            d[_OLD_KEYS.get(k, k)] = v
        row = {}
        for c in BOQ_COLUMNS:
            if c in BOQ_NUM_COLS:
                row[c] = _fmt_price(d.get(c))
            elif c == "No":
                row[c] = _clean_no(d.get(c))
            elif c == "Volume":
                row[c] = _fmt_volume(d.get(c))
            else:
                row[c] = _cell(d.get(c))
        out.append(row)
    return out


def _sample_boq_rows():
    def r(no, desc, spec="", vol="", unit=""):
        return {"No": no, "Description": desc, "Specification": spec,
                "Volume": vol, "Unit": unit,
                "Unit Price (Rp)": "-", "Total Price (Rp)": "-"}
    return [
        r("BAB. II", "BANGUNAN OFFICE"),
        r("II.1", "PEKERJAAN STRUKTUR"),
        r("II.1.A", "PEKERJAAN PERSIAPAN"),
        r("1", "Pek. Bouwplank", "", "88,00", "m'"),
        r("2", "Pek. Clearing area; termasuk bongkar dan pemindahan Kontainer eksisting",
          "termasuk pemindahan furniture ke ruangan sementara", "315,00", "m2"),
        r("3", "Pek. Bongkaran Dinding Eksisting", "", "129,40", "m2"),
        r("4", "Pek. Bongkaran Kloset & Wastafel Eksisting", "", "7,00", "unit"),
        r("5", "Pek. Bobokan Slab Eksisting untuk install Footplate & Tiebeam Baru", "", "1,00", "Ls"),
        r("6", "Pek. Bongkar Keramik Eksisting", "", "246,77", "m2"),
        r("7", "Pek. Bongkar Plafond Eksisting", "", "246,20", "m2"),
        r("8", "Pek. Bongkar Rangka Atap Eksisting", "", "266,30", "m2"),
        r("9", "Pek. Buang Puing Ex Bongkaran", "", "1,00", "Ls"),
    ]


def _decode_csv_bytes(data):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _csv_to_frame(data):
    text = _decode_csv_bytes(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, dtype=object)


@st.cache_data(show_spinner="Membaca file BOQ...")
def _parse_boq_cached(data, name):
    name = (name or "").lower()
    if name.endswith(".csv"):
        sheets = {"CSV": _csv_to_frame(data)}
    else:
        if name.endswith(".xlsx") and not HAS_OPENPYXL:
            raise ValueError("Library 'openpyxl' belum terpasang. Gunakan CSV atau install openpyxl.")
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str, header=None)
        except ImportError:
            raise ValueError(
                "Library pembaca Excel belum terpasang (openpyxl untuk .xlsx, xlrd untuk .xls). "
                "Simpan file sebagai .xlsx atau CSV."
            )
        except Exception as e:
            raise ValueError(f"Gagal membaca file Excel: {e}")

    per_sheet = []
    for _sheet_name, df_raw in sheets.items():
        if df_raw is None or df_raw.empty:
            continue
        try:
            its = _extract_items(df_raw)
        except Exception:
            continue
        if its:
            per_sheet.append(its)

    good = [its for its in per_sheet if sum(1 for i in its if i["Volume"]) >= 3]
    chosen = good or per_sheet
    all_items = [it for its in chosen for it in its]

    if not all_items:
        raise ValueError(
            "Tidak ada data BOQ yang bisa dibaca. Pastikan file memiliki tabel dengan "
            "kolom Description/Uraian, Volume, dan Unit/Satuan."
        )
    return all_items


def parse_boq_excel(file):
    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        try:
            file.seek(0)
        except Exception:
            pass
        data = file.read()
    return _parse_boq_cached(data, getattr(file, "name", ""))


def _read_upload(file):
    try:
        file.seek(0)
    except Exception:
        pass
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


INVITED_VENDORS = [
    "PT Baja Nusantara Konstruksi",
    "CV Prima Pipa Sejahtera",
    "PT Karya Logam Presisi",
    "PT Sinergi Teknik Mandiri",
    "PT Anugerah Pipa Internasional",
]

VENDOR_ALIASES = ["vendor", "nama vendor", "nama perusahaan", "perusahaan",
                  "kontraktor", "supplier", "company", "nama"]
EMAIL_ALIASES = ["email", "e-mail", "surel", "alamat email", "email vendor"]


def _norm_invited(lst):
    """Rapikan daftar vendor ke bentuk [{'Vendor':..., 'Email':...}] (mendukung data lama berupa list nama)."""
    out = []
    for v in lst or []:
        if isinstance(v, dict):
            name = _cell(v.get("Vendor"))
            email = _cell(v.get("Email")).lower()
        else:
            name, email = _cell(v), ""
        if name:
            out.append({"Vendor": name, "Email": email})
    return out


def parse_vendor_excel(file):
    """Baca daftar vendor (nama + email) dari Excel/CSV -> list of dict."""
    df = _read_upload(file)
    if len(df.columns) == 0:
        raise ValueError("File kosong.")
    name_idx, email_idx = 0, None
    name_found = False
    for j, c in enumerate(df.columns):
        key = _norm_text(c)
        if key in EMAIL_ALIASES and email_idx is None:
            email_idx = j
        elif key in VENDOR_ALIASES and not name_found:
            name_idx = j
            name_found = True
    out, seen = [], set()
    for _, row in df.iterrows():
        name = _cell(row.iloc[name_idx])
        email = _cell(row.iloc[email_idx]).lower() if email_idx is not None else ""
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"Vendor": name, "Email": email})
    if not out:
        raise ValueError("Tidak ada nama vendor yang valid di file.")
    return out


def vendor_template_bytes():
    df = pd.DataFrame({"Vendor": INVITED_VENDORS, "Email": [""] * len(INVITED_VENDORS)})
    if HAS_OPENPYXL:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Vendor")
            return (buf.getvalue(), "template_vendor.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass
    return df.to_csv(index=False).encode("utf-8-sig"), "template_vendor.csv", "text/csv"


def boq_template_bytes():
    rows = []
    for r in _sample_boq_rows():
        rows.append({
            "NO": r["No"], "DESCRIPTION": r["Description"],
            "SPESIFICATION": r["Specification"], "VOLUME": _to_number(r["Volume"]),
            "UNIT": r["Unit"], "UNIT PRICE (Rp.)": None, "TOTAL PRICE (Rp.)": None,
        })
    df = pd.DataFrame(rows)
    if HAS_OPENPYXL:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="BOQ")
                ws = writer.sheets["BOQ"]
                for col, w in zip("ABCDEFG", (10, 60, 38, 12, 8, 18, 18)):
                    ws.column_dimensions[col].width = w
            return (buf.getvalue(), "template_boq.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass
    return df.to_csv(index=False).encode("utf-8-sig"), "template_boq.csv", "text/csv"


# =========================================================
# HELPER S-CURVE VENDOR
# (file vendor disimpan apa adanya, tidak dibaca isinya; KPDP yang mengunduh)
# =========================================================
SCURVE_FIXED = ["No", "Uraian Pekerjaan", "Harga (juta Rp)", "Bobot (%)"]


def scurve_expected_headers(max_week):
    """Header baku: kolom tetap + M1..Mn (n = durasi project dalam minggu)."""
    return SCURVE_FIXED + [f"M{w}" for w in range(1, int(max_week) + 1)]


def scurve_format_example(max_week):
    """Contoh format S-Curve (dipakai untuk template yang diunduh vendor)."""
    cols = scurve_expected_headers(max_week)

    def row(no, name, harga, bobot, start, vals):
        r = {c: "" for c in cols}
        r.update({"No": no, "Uraian Pekerjaan": name,
                  "Harga (juta Rp)": harga, "Bobot (%)": bobot})
        for k, v in enumerate(vals):
            if start + k <= max_week:
                r[f"M{start + k}"] = v
        return r

    return pd.DataFrame([
        row("A", "Pekerjaan Persiapan", "40", "40.00", 1, ["5.00", "10.00", "15.00", "10.00"]),
        row("B", "Pekerjaan Struktur", "60", "60.00", 3, ["10.00", "20.00", "20.00", "10.00"]),
    ], columns=cols)


def scurve_template_bytes(max_week):
    """Template S-Curve (header baku + contoh) untuk diunduh vendor."""
    df = scurve_format_example(max_week)
    if HAS_OPENPYXL:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="S-Curve")
                ws = writer.sheets["S-Curve"]
                ws.column_dimensions["A"].width = 8
                ws.column_dimensions["B"].width = 40
                ws.column_dimensions["C"].width = 18
                ws.column_dimensions["D"].width = 12
            return (buf.getvalue(), "template_scurve.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass
    return df.to_csv(index=False).encode("utf-8-sig"), "template_scurve.csv", "text/csv"


_MIME_BY_EXT = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
}


def _guess_mime(filename):
    fn = (filename or "").lower()
    for ext, mime in _MIME_BY_EXT.items():
        if fn.endswith(ext):
            return mime
    return "application/octet-stream"


def scurve_file_from_row(row):
    """Ambil (bytes, filename, mime) dari baris tabel scurve, atau None."""
    if not row:
        return None
    d = row.get("data")
    if isinstance(d, dict) and d.get("file_b64"):
        try:
            raw = base64.b64decode(d["file_b64"])
        except Exception:
            return None
        fn = row.get("filename") or "s_curve"
        return raw, fn, d.get("mime") or _guess_mime(fn)
    return None


# =========================================================
# HELPER GRAFIK S-CURVE DARI EXCEL (hanya kurva, tanpa tabel)
# =========================================================
_WEEK_RE = re.compile(r"(?:minggu(?:\s*ke)?|week|mg|m|w)\s*[-_.]*\s*(\d+)")
_CUM_WORDS = ("kumulatif", "komulatif", "kumulatip", "cumulative")
_ACT_RE = re.compile(r"\b(realisasi|aktual|actual|real)\b")
_PLAN_RE = re.compile(r"\b(rencana|plan|target)\b")
_TOTAL_RE = re.compile(r"\b(total|jumlah)\b")


def _week_no(v):
    s = _norm_text(v)
    if not s:
        return None
    m = _WEEK_RE.fullmatch(s)
    return int(m.group(1)) if m else None


def _find_week_header(df_raw, max_scan=80):
    """Cari baris header minggu (M1, M2, ... / Minggu 1 / W1 / 1,2,3,...)."""
    n = min(len(df_raw), max_scan)
    for i in range(n):
        row = list(df_raw.iloc[i].values)
        cols = [(j, _week_no(c)) for j, c in enumerate(row)]
        cols = [(j, w) for j, w in cols if w is not None]
        if len(cols) >= 3:
            return i, cols
    for i in range(n):
        row = list(df_raw.iloc[i].values)
        cols = []
        for j, c in enumerate(row):
            s = _cell(c)
            if re.fullmatch(r"\d+(\.0+)?", s):
                cols.append((j, int(float(s))))
        if len(cols) >= 4 and [w for _, w in cols] == list(range(cols[0][1], cols[0][1] + len(cols))):
            return i, cols
    return None


def _classify_curve_row(label):
    is_cum = any(k in label for k in _CUM_WORDS)
    is_act = bool(_ACT_RE.search(label))
    is_plan = bool(_PLAN_RE.search(label))
    if is_cum and is_act:
        return "actual_cum"
    if is_cum:
        return "plan_cum"
    if is_act:
        return "actual_week"
    if _TOTAL_RE.search(label) or (is_plan and "progress" in label):
        return "plan_week"
    return "item"


def _ffill(vs):
    out, prev = [], 0.0
    for v in vs:
        if v is not None:
            prev = v
        out.append(prev)
    return out


def _cumsum_none(vs):
    out, t = [], 0.0
    for v in vs:
        t += (v or 0.0)
        out.append(t)
    return out


def _extract_curve_weekly(df_raw):
    found = _find_week_header(df_raw)
    if found is None:
        return None
    hdr, cols = found
    seen, wc = set(), []
    for j, w in cols:
        if w not in seen:
            seen.add(w)
            wc.append((j, w))
    wc.sort(key=lambda x: x[1])
    weeks = [w for _, w in wc]
    first_col = min(j for j, _ in wc)

    series = {"plan_cum": None, "actual_cum": None, "plan_week": None, "actual_week": None}
    items = []
    for row in df_raw.iloc[hdr + 1:].values:
        vals = [_curve_num(row[j]) if j < len(row) else None for j, _ in wc]
        if all(v is None for v in vals):
            continue
        label = _norm_text(" ".join(_cell(row[k]) for k in range(min(first_col, len(row)))))
        kind = _classify_curve_row(label)
        if kind == "item":
            if any((v or 0) > 0 for v in vals):
                items.append(vals)
        elif series[kind] is None:
            series[kind] = vals

    # --- kurva rencana ---
    if series["plan_cum"] is not None:
        plan = _ffill(series["plan_cum"])
    elif series["plan_week"] is not None:
        plan = _cumsum_none(series["plan_week"])
    elif items:
        plan = _cumsum_none([sum((v or 0.0) for v in col) for col in zip(*items)])
    else:
        return None

    # --- kurva realisasi (opsional) ---
    actual = None
    if series["actual_cum"] is not None:
        src, mode = series["actual_cum"], "cum"
    elif series["actual_week"] is not None:
        src, mode = series["actual_week"], "week"
    else:
        src, mode = None, None
    if src is not None:
        filled = [i for i, v in enumerate(src) if v is not None]
        if filled:
            last = filled[-1]
            actual = [None] * len(src)
            run, prev = 0.0, 0.0
            for i in range(last + 1):
                v = src[i]
                if mode == "cum":
                    prev = v if v is not None else prev
                    actual[i] = prev
                else:
                    run += (v or 0.0)
                    actual[i] = run

    # sel berformat persen di Excel tersimpan sebagai pecahan (0-1) -> jadikan persen
    factor = 100.0 if max(plan) <= 1.5 else 1.0
    plan = [round(v * factor, 2) for v in plan]
    if actual is not None:
        actual = [None if v is None else round(v * factor, 2) for v in actual]
    return {"weeks": weeks, "plan": plan, "actual": actual}


_PLAN_TAGS = {"plan", "pan", "rencana", "rcn", "target"}
_ACT_TAGS = {"aktual", "actual", "akt", "real", "realisasi"}


def _curve_num(v):
    """Angka dari sel; aman untuk notasi ilmiah (1e-05) dan format Indonesia."""
    s = _cell(v)
    if re.fullmatch(r"-?\d*\.?\d+[eE][-+]?\d+", s):
        try:
            return float(s)
        except ValueError:
            return None
    return _to_number(s)


def _find_day_header(df_raw, max_scan=60):
    """Cari baris nomor hari (1,2,3,...) di bawah label 'Week N'.
    Pilih baris dengan deret terpanjang, supaya baris tanggal (21..30,1..31) tidak terpilih."""
    n = min(len(df_raw), max_scan)
    best_overall, best_row = [], None
    for i in range(n):
        row = list(df_raw.iloc[i].values)
        nums = []
        for j, c in enumerate(row):
            s = _cell(c)
            if re.fullmatch(r"\d+(\.0+)?", s):
                nums.append((j, int(float(s))))
        run, best = [], []
        for j, d in nums:
            if run and d == run[-1][1] + 1:
                run.append((j, d))
            else:
                run = [(j, d)]
            if len(run) > len(best):
                best = list(run)
        if len(best) < 14 or best[0][1] != 1:
            continue
        wk = 0
        for k in range(max(0, i - 4), min(len(df_raw), i + 2)):
            wk = max(wk, sum(1 for c in df_raw.iloc[k].values if _week_no(c) is not None))
        if wk >= 2 and len(best) >= 3 * wk and len(best) > len(best_overall):
            best_overall, best_row = best, i
    if best_row is None:
        return None
    return best_row, best_overall


def _row_tag(left_cells):
    """Return (tag, is_cum, is_sum); tag = 'plan' / 'actual' / None."""
    joined = _norm_text(" ".join(left_cells))
    tag = None
    for t in left_cells:
        toks = set(re.findall(r"[a-z]+", t.lower()))
        if not toks or len(toks) > 3:
            continue
        if toks & _PLAN_TAGS:
            tag = "plan"
            break
        if toks & _ACT_TAGS:
            tag = "actual"
            break
    if tag is None:
        return None, False, False
    is_cum = any(k in joined for k in _CUM_WORDS)
    is_sum = bool(_TOTAL_RE.search(joined)) or "progress" in joined or "subtotal" in joined
    return tag, is_cum, is_sum


def _is_total_of(vals, rows):
    """True jika vals = jumlah semua baris sebelumnya (baris total, bukan item)."""
    if len(rows) < 2:
        return False
    tot = [sum((r[i] or 0.0) for r in rows) for i in range(len(vals))]
    if not any(t > 0 for t in tot):
        return False
    return all(abs((v or 0.0) - t) <= 1e-6 + 1e-4 * abs(t) for v, t in zip(vals, tot))


def _extract_curve_daily(df_raw, found):
    """Format harian: kolom = hari (1..N), baris 'Plan'/'Aktual' per item + baris kumulatif.
    Diringkas per minggu (7 hari)."""
    hdr, day_cols = found
    first_col = min(j for j, _ in day_cols)
    dc = {}
    for j, d in day_cols:
        dc.setdefault(d, j)
    day_cols = [(dc[d], d) for d in sorted(dc)]
    ndays = len(day_cols)

    plan_items, act_items = [], []
    sum_rows = {"plan": None, "actual": None}
    cum_rows = {"plan": None, "actual": None}
    for row in df_raw.iloc[hdr + 1:].values:
        left = [_cell(row[k]) for k in range(min(first_col, len(row)))]
        tag, is_cum, is_sum = _row_tag(left)
        if tag is None:
            continue
        vals = [_curve_num(row[j]) if j < len(row) else None for j, _ in day_cols]
        if all(v is None for v in vals):
            continue
        items = plan_items if tag == "plan" else act_items
        if is_cum:
            if cum_rows[tag] is None:
                cum_rows[tag] = vals
        elif is_sum or _is_total_of(vals, items):
            if sum_rows[tag] is None:
                sum_rows[tag] = vals
        else:
            items.append(vals)

    def daily_total(rows):
        tot = [0.0] * ndays
        for vals in rows:
            for i, v in enumerate(vals):
                tot[i] += (v or 0.0)
        return tot

    # --- rencana kumulatif ---
    if cum_rows["plan"] is not None:
        plan_cum = _ffill(cum_rows["plan"])
    elif sum_rows["plan"] is not None:
        plan_cum = _cumsum_none(sum_rows["plan"])
    elif plan_items:
        plan_cum = _cumsum_none(daily_total(plan_items))
    else:
        return None

    # --- realisasi kumulatif ---
    actual_cum_full = None
    if cum_rows["actual"] is not None:
        actual_cum_full = _ffill(cum_rows["actual"])
    elif sum_rows["actual"] is not None:
        actual_cum_full = _cumsum_none(sum_rows["actual"])
    elif act_items:
        actual_cum_full = _cumsum_none(daily_total(act_items))

    actual_cum, last_act = None, -1
    if actual_cum_full is not None:
        # hari terakhir yang benar-benar ada progres (kumulatif berhenti naik setelahnya)
        for i in range(len(actual_cum_full) - 1, -1, -1):
            prev = actual_cum_full[i - 1] if i > 0 else 0.0
            if actual_cum_full[i] > prev + 1e-9:
                last_act = i
                break
        if last_act >= 0:
            actual_cum = actual_cum_full[:last_act + 1]

    factor = 100.0 if max(plan_cum) <= 1.5 else 1.0
    days = [d for _, d in day_cols]
    plan = [round(v * factor, 2) for v in plan_cum]
    actual = None
    if actual_cum is not None:
        actual = ([round(v * factor, 2) for v in actual_cum]
                  + [None] * (ndays - len(actual_cum)))

    warning = ""
    if abs(plan[-1] - 100.0) > 2.0:
        warning = (f"Total rencana di akhir proyek terbaca {plan[-1]:.2f}% (seharusnya sekitar 100%). "
                   "Kemungkinan ada baris subtotal yang ikut terjumlah atau sebagian kolom hari tidak terbaca. "
                   "Periksa file / sheet yang dipilih.")
    return {"days": days, "weeks": list(range(1, (ndays + 6) // 7 + 1)),
            "plan": plan, "actual": actual,
            "mode": f"harian ({ndays} hari), sama seperti grafik di Excel", "warning": warning}


def _extract_curve(df_raw):
    found = _find_day_header(df_raw)
    if found is not None:
        return _extract_curve_daily(df_raw, found)
    return _extract_curve_weekly(df_raw)


def _curve_sheet_match(name):
    return "scurve" in re.sub(r"[^a-z]", "", str(name).lower())


@st.cache_data(show_spinner=False)
def scurve_sheet_names(data, name):
    """Daftar nama sheet di file (untuk pilihan sheet di UI)."""
    name = (name or "").lower()
    if name.endswith(".csv"):
        return ["CSV"]
    try:
        return list(pd.ExcelFile(io.BytesIO(data)).sheet_names)
    except Exception:
        return []


def parse_scurve_chart(data, name, sheet=None):
    """Baca kurva S-Curve (rencana & realisasi kumulatif) dari file Excel/CSV.
    Otomatis mencari sheet yang namanya mengandung 'S-Curve' kecuali sheet dipilih manual."""
    name = (name or "").lower()
    if name.endswith(".csv"):
        names = ["CSV"]
        xl = None
    else:
        if name.endswith(".xlsx") and not HAS_OPENPYXL:
            raise ValueError("Library 'openpyxl' belum terpasang. Gunakan CSV atau install openpyxl.")
        try:
            xl = pd.ExcelFile(io.BytesIO(data))
        except ImportError:
            raise ValueError("Library pembaca Excel belum terpasang (openpyxl untuk .xlsx).")
        except Exception as e:
            raise ValueError(f"Gagal membaca file Excel: {e}")
        names = list(xl.sheet_names)

    if sheet and sheet in names:
        cands = [sheet]
    else:
        match = [n for n in names if _curve_sheet_match(n)]
        cands = match or names

    named = bool(not sheet and cands and all(_curve_sheet_match(c) for c in cands))
    found_all = []
    for sh in cands:
        try:
            if xl is None:
                df_raw = _csv_to_frame(data)
            else:
                df_raw = xl.parse(sh, header=None, dtype=str)
        except Exception:
            continue
        if df_raw is None or df_raw.empty:
            continue
        try:
            res = _extract_curve(df_raw)
        except Exception:
            res = None
        if res:
            res["sheet"] = str(sh)
            if sheet or named:
                return res          # sheet dipilih manual / bernama S-Curve: langsung pakai
            found_all.append(res)   # tidak ada sheet bernama S-Curve: ambil yang datanya terpanjang
    if found_all:
        return max(found_all, key=lambda r: len(r["weeks"]))
    tried = ", ".join(str(c) for c in cands[:6])
    raise ValueError(
        f"Kurva S-Curve tidak ditemukan di sheet: {tried}. Sheet harus punya header hari/minggu "
        "(1,2,3,... di bawah label 'Week N', atau M1, M2, ...) dan baris 'Plan'/'Aktual' atau "
        "'Progress Rencana Kumulatif'. Pastikan nilainya sudah terhitung (bukan rumus yang belum tersimpan)."
    )


# =========================================================
# SUPABASE: CLIENT & AUTH
# =========================================================
def get_client():
    """Satu client PER SESI browser (jangan di-cache global, supaya login antar user tidak tercampur)."""
    if "sb" not in st.session_state:
        st.session_state.sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    return st.session_state.sb


def db():
    return get_client()


def get_role(email):
    return "kpdp" if str(email).lower().endswith(KPDP_DOMAIN) else "vendor"


def do_logout():
    try:
        get_client().auth.sign_out()
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def handle_email_confirmation():
    """Tangani klik link konfirmasi email dari Supabase (?token_hash=...&type=signup)."""
    qp = st.query_params
    token_hash = qp.get("token_hash")
    typ = qp.get("type")
    if token_hash and typ in ("signup", "email"):
        try:
            get_client().auth.verify_otp({"token_hash": token_hash, "type": typ})
            try:
                get_client().auth.sign_out()  # user tetap login manual
            except Exception:
                pass
            st.session_state.just_confirmed = True
            st.session_state.pop("confirm_failed", None)
        except Exception:
            # link sudah dipakai / kedaluwarsa
            st.session_state.confirm_failed = True
            st.session_state.pop("just_confirmed", None)
        st.query_params.clear()


def login_page():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        if st.session_state.get("just_confirmed"):
            st.success("✅ Akun Anda sudah terdaftar dan email terkonfirmasi. "
                       "Silakan masuk dengan email dan password Anda.")
        elif st.session_state.get("confirm_failed"):
            st.warning("Link konfirmasi sudah dipakai atau kedaluwarsa. "
                       "Jika akun sudah terdaftar, silakan langsung masuk.")
        st.markdown("# Vendor Selection & Project Control")
        st.caption("Masuk dengan email Anda. Email KPDP masuk ke dashboard KPDP, email lain masuk ke portal vendor.")
        tab_login, tab_daftar = st.tabs(["Masuk", "Daftar (Vendor)"])

        with tab_login:
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Masuk", key="btn_login", type="primary"):
                if not email.strip() or not pw:
                    st.error("Email dan password wajib diisi.")
                else:
                    try:
                        res = get_client().auth.sign_in_with_password(
                            {"email": email.strip().lower(), "password": pw})
                        st.session_state.user_email = res.user.email.lower()
                        st.session_state.pop("just_confirmed", None)
                        st.session_state.pop("confirm_failed", None)
                        st.rerun()
                    except Exception:
                        st.error("Email atau password salah (atau email belum dikonfirmasi).")

        with tab_daftar:
            st.caption("Khusus vendor. Akun KPDP dibuat oleh admin.")
            r_email = st.text_input("Email perusahaan", key="reg_email")
            r_pw = st.text_input("Password (min. 8 karakter)", type="password", key="reg_pw")
            if st.button("Daftar", key="btn_reg"):
                e = r_email.strip().lower()
                if not e or "@" not in e:
                    st.error("Email tidak valid.")
                elif e.endswith(KPDP_DOMAIN):
                    st.error("Akun KPDP tidak bisa didaftarkan di sini. Hubungi admin.")
                elif len(r_pw) < 8:
                    st.error("Password minimal 8 karakter.")
                else:
                    try:
                        get_client().auth.sign_up({"email": e, "password": r_pw})
                        st.success("Pendaftaran berhasil. Cek email untuk konfirmasi, lalu login. "
                                   "Pastikan email ini sudah didaftarkan KPDP sebagai vendor yang diundang.")
                    except Exception as ex:
                        st.error(f"Gagal mendaftar: {ex}")


# =========================================================
# SUPABASE: DATA LAYER
# =========================================================
def _clean(o):
    """Bersihkan objek agar aman jadi JSON (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def _hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _fill_defaults(p):
    p.setdefault("meta", {})
    p.setdefault("live_data", {})
    for k in ("tender", "vendor_selection", "site_execution", "quality_control"):
        p["live_data"].setdefault(k, {})
    p.setdefault("wbs_data", None)
    p.setdefault("scurve_chart", None)
    p.setdefault("items_data", None)
    p.setdefault("items_ver", 0)
    p.setdefault("invited_vendors", [])
    p.setdefault("invited_ver", 0)
    return p


def load_projects():
    rows = db().table("projects").select("id,data").execute().data or []
    projects = {}
    for r in rows:
        p = _fill_defaults(r["data"])
        projects[r["id"]] = p
        st.session_state.saved_hash[r["id"]] = _hash(_clean(p))
    return projects


def save_project(pid):
    p = st.session_state.projects.get(pid)
    if not p:
        return
    data = _clean(p)
    h = _hash(data)
    if st.session_state.saved_hash.get(pid) == h:
        return
    try:
        m = p["meta"]
        db().table("projects").upsert({
            "id": pid, "name": m.get("name"), "data": data,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }).execute()
        # versi publik untuk vendor (TANPA budget / RAB)
        vs_ = (data.get("live_data") or {}).get("vendor_selection") or {}
        sel_email_ = str(vs_.get("selected_vendor_email") or "").lower() or None
        db().table("project_public").upsert({
            "id": pid, "name": m.get("name"), "client": m.get("client"),
            "location": m.get("location"), "deadline": int(m.get("deadline", 0)),
            "selected_vendor_email": sel_email_,
            "boq": data.get("items_data") or [],
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }).execute()
        # sinkron daftar vendor diundang
        inv = _norm_invited(p.get("invited_vendors"))
        db().table("project_vendors").delete().eq("project_id", pid).execute()
        rows = [{"project_id": pid, "vendor_name": v["Vendor"], "vendor_email": v["Email"]}
                for v in inv if v["Email"]]
        if rows:
            db().table("project_vendors").insert(rows).execute()
        st.session_state.saved_hash[pid] = h
    except Exception as e:
        st.error(f"Gagal menyimpan ke database: {e}")


def persist():
    for pid_ in list(st.session_state.get("projects", {}).keys()):
        save_project(pid_)


def delete_project_db(pid):
    for t, col in (("appointments", "project_id"), ("penawaran", "project_id"),
                   ("scurve", "project_id"),
                   ("project_vendors", "project_id"), ("project_public", "id"),
                   ("projects", "id")):
        try:
            db().table(t).delete().eq(col, pid).execute()
        except Exception as e:
            st.error(f"Gagal menghapus dari {t}: {e}")
    st.session_state.saved_hash.pop(pid, None)


def fetch_penawaran(pid):
    """Return dict {email_vendor: row}."""
    try:
        rows = db().table("penawaran").select("*").eq("project_id", pid).execute().data or []
    except Exception as e:
        st.error(f"Gagal membaca penawaran: {e}")
        rows = []
    return {str(r["vendor_email"]).lower(): r for r in rows}


def fetch_appointments(pid, vendor_email=None):
    try:
        q = db().table("appointments").select("*").eq("project_id", pid)
        if vendor_email:
            q = q.eq("vendor_email", vendor_email)
        return q.order("created_at", desc=True).execute().data or []
    except Exception as e:
        st.error(f"Gagal membaca appointment: {e}")
        return []


def fetch_scurve(pid, vendor_email):
    """Return baris S-Curve terakhir dari vendor, atau None."""
    if not vendor_email:
        return None
    try:
        rows = db().table("scurve").select("*").eq("project_id", pid) \
            .eq("vendor_email", str(vendor_email).lower()).execute().data or []
    except Exception as e:
        st.error(f"Gagal membaca S-Curve: {e}")
        return None
    return rows[0] if rows else None


# Helper agar data tersimpan sebelum rerun / stop
def rerun():
    persist()
    st.rerun()


def stop_app():
    persist()
    st.stop()


# =========================================================
# HALAMAN VENDOR
# =========================================================
@st.fragment(run_every=5)
def vendor_appointments_live(pid, email):
    appts = fetch_appointments(pid, email)
    if not appts:
        st.info("Belum ada jadwal klarifikasi dari KPDP.")
        return
    for a in appts:
        with st.expander(f"[{a.get('status')}] {a.get('mode')} · {a.get('date')} {a.get('time')}"):
            st.markdown(f"**Lokasi / Link:** {a.get('location') or '-'}")
            st.markdown(f"**Durasi:** {a.get('duration')} menit")
            st.markdown(f"**Agenda:** {a.get('agenda') or '-'}")
            st.markdown(f"**Peserta KPDP:** {a.get('participants') or '-'}")
            st.markdown(f"**Peserta vendor:** {a.get('vendor_participants') or '-'}")
            if a.get("hasil"):
                st.markdown(f"**Hasil klarifikasi:** {a['hasil']}")
    st.caption(f"Diperbarui otomatis · {now_local():%H:%M:%S}")


def vendor_page(email):
    st.markdown("# Portal Vendor")
    st.caption(f"Login sebagai {email}")

    try:
        inv = db().table("project_vendors").select("project_id,vendor_name") \
            .eq("vendor_email", email).execute().data or []
    except Exception as e:
        st.error(f"Gagal membaca undangan: {e}")
        return
    if not inv:
        st.info("Belum ada project yang mengundang email ini. Hubungi KPDP untuk mendaftarkan email Anda.")
        return

    ids = [i["project_id"] for i in inv]
    name_by_pid = {i["project_id"]: i["vendor_name"] for i in inv}
    try:
        pubs = db().table("project_public").select("*").in_("id", ids).execute().data or []
    except Exception as e:
        st.error(f"Gagal membaca project: {e}")
        return
    if not pubs:
        st.info("Data project belum tersedia.")
        return
    pub_map = {p["id"]: p for p in pubs}

    pid_sel = st.selectbox("Project", options=list(pub_map.keys()),
                           format_func=lambda x: pub_map[x]["name"], key="vendor_proj_sel")
    pub = pub_map[pid_sel]
    vendor_name = name_by_pid.get(pid_sel, email.split("@")[0])
    st.caption(f"{pub['name']} · {pub.get('client') or '-'} · {pub.get('location') or '-'} · "
               f"Deadline: {pub.get('deadline')} hari · Terdaftar sebagai: {vendor_name}")

    is_selected = str(pub.get("selected_vendor_email") or "").lower() == email.lower()
    tab_labels = ["1. BOQ Project", "2. Kirim Penawaran", "3. Jadwal Klarifikasi"]
    if is_selected:
        tab_labels.append("4. Upload S-Curve")
    tabs_v = st.tabs(tab_labels)
    t1, t2, t3 = tabs_v[0], tabs_v[1], tabs_v[2]
    t4 = tabs_v[3] if is_selected else None

    with t1:
        boq = pub.get("boq") or []
        if boq:
            v = pd.DataFrame(_normalize_items(boq), columns=BOQ_COLUMNS)
            v.columns = ["NO", "DESCRIPTION", "SPESIFICATION", "VOLUME", "UNIT",
                         "UNIT PRICE (Rp.)", "TOTAL PRICE (Rp.)"]
            st.dataframe(v, use_container_width=True, hide_index=True, height=480)
        else:
            st.info("BOQ belum tersedia.")

    with t2:
        try:
            ex = db().table("penawaran").select("*").eq("project_id", pid_sel) \
                .eq("vendor_email", email).execute().data or []
        except Exception:
            ex = []
        cur = ex[0] if ex else {}
        # Durasi ditetapkan KPDP (deadline project), vendor tidak bisa mengubah
        durasi = int(pub.get("deadline") or 0)
        # Harga terkunci begitu KPDP menjadwalkan klarifikasi (selama tidak dibatalkan)
        my_appts = [a for a in fetch_appointments(pid_sel, email)
                    if a.get("status") != "Dibatalkan"]
        locked = len(my_appts) > 0
        if locked:
            st.warning("KPDP sudah menjadwalkan klarifikasi untuk Anda, jadi penawaran "
                       "dikunci dan tidak bisa diubah lagi.")
        if cur:
            st.success(f"Penawaran terakhir dikirim: {fmt_ts(cur.get('updated_at'))} "
                       f"({format_rupiah(cur.get('harga_rab', 0))}, {durasi} hari). "
                       "Anda bisa mengubahnya selama KPDP belum menutup tender.")

        c1, c2 = st.columns(2)
        with c1:
            harga = st.number_input("Penawaran Harga (juta Rp)", min_value=0.0, step=5.0,
                                    value=float(cur.get("harga_rab") or 0.0),
                                    disabled=locked, key=f"v_harga_{pid_sel}")
        with c2:
            st.number_input("Durasi (hari) — ditetapkan KPDP", value=durasi,
                            disabled=True, key=f"v_dur_{pid_sel}_{durasi}")
        catatan = st.text_area("Catatan", value=cur.get("catatan") or "", height=100,
                               disabled=locked, key=f"v_cat_{pid_sel}")

        if st.button("Kirim Penawaran", type="primary", disabled=locked,
                     key=f"v_send_{pid_sel}"):
            if locked:
                st.error("Penawaran sudah dikunci oleh KPDP.")
            elif harga <= 0:
                st.error("Harga penawaran harus lebih dari 0.")
            else:
                try:
                    db().table("penawaran").upsert({
                        "project_id": pid_sel, "vendor_email": email,
                        "vendor_name": vendor_name, "harga_rab": float(harga),
                        "durasi": int(durasi), "catatan": catatan,
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                    }, on_conflict="project_id,vendor_email").execute()
                    st.success("Terkirim. KPDP langsung bisa melihat penawaran Anda.")
                except Exception as e:
                    st.error(f"Gagal mengirim: {e}")

    with t3:
        vendor_appointments_live(pid_sel, email)

    if t4 is not None:
        with t4:
            st.markdown("### Upload S-Curve (Wajib)")
            max_week = max(1, (int(pub.get("deadline") or 0) + 6) // 7)
            st.info(f"Anda terpilih sebagai kontraktor project ini. Wajib upload S-Curve. "
                    f"Durasi project {pub.get('deadline')} hari (maksimal minggu ke-{max_week}). "
                    f"File yang Anda upload akan langsung diteruskan ke KPDP apa adanya.")

            sc_cur = fetch_scurve(pid_sel, email)
            if sc_cur:
                st.success(f"S-Curve terakhir diupload: {fmt_ts(sc_cur.get('uploaded_at'))} · "
                           f"file {sc_cur.get('filename') or '-'}")
            else:
                st.warning("Belum ada S-Curve yang diupload.")

            sc_types = ["xlsx", "xls", "csv"]
            sc_file = st.file_uploader(
                "Upload S-Curve (" + " / ".join("." + t for t in sc_types) + ")",
                type=sc_types, key=f"sc_upload_{pid_sel}",
            )
            if sc_file is not None:
                raw = sc_file.getvalue()
                size_kb = len(raw) / 1024
                if len(raw) > 5 * 1024 * 1024:
                    st.error("Ukuran file maksimal 5 MB.")
                else:
                    st.success(f"File siap dikirim: {sc_file.name} ({size_kb:,.0f} KB)")
                    if st.button("Kirim S-Curve", type="primary", key=f"sc_send_{pid_sel}"):
                        try:
                            db().table("scurve").upsert({
                                "project_id": pid_sel, "vendor_email": email,
                                "vendor_name": vendor_name,
                                "filename": sc_file.name,
                                "data": {
                                    "file_b64": base64.b64encode(raw).decode("ascii"),
                                    "mime": _guess_mime(sc_file.name),
                                    "size": len(raw),
                                },
                                "total_harga": 0.0,
                                "uploaded_at": datetime.utcnow().isoformat() + "Z",
                            }, on_conflict="project_id,vendor_email").execute()
                            st.toast("S-Curve terkirim ke KPDP.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal mengirim S-Curve: {e}")


# =========================================================
# ROUTER LOGIN
# =========================================================
if "saved_hash" not in st.session_state:
    st.session_state.saved_hash = {}

if "user_email" not in st.session_state:
    handle_email_confirmation()
    login_page()
    st.stop()

user_email = st.session_state.user_email
role = get_role(user_email)

if role == "vendor":
    with st.sidebar:
        st.markdown("### Akun")
        st.caption(user_email)
        st.markdown("<span class='pill pill-info'>Vendor</span>", unsafe_allow_html=True)
        if st.button("Keluar", key="btn_logout_v"):
            do_logout()
    vendor_page(user_email)
    st.stop()

# =========================================================
# ====================  DASHBOARD KPDP  ===================
# =========================================================
if "projects" not in st.session_state:
    try:
        st.session_state.projects = load_projects()
    except Exception as e:
        st.session_state.projects = {}
        st.error(f"Gagal memuat project dari database: {e}")
if "active_project" not in st.session_state:
    st.session_state.active_project = None
if st.session_state.active_project not in st.session_state.projects:
    st.session_state.active_project = (
        next(iter(st.session_state.projects)) if st.session_state.projects else None
    )
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
    st.session_state.projects[pid] = _fill_defaults({
        "meta": {
            "id": pid, "name": name, "client": client, "location": location,
            "budget": float(budget), "deadline": int(deadline),
            "rab_awal": float(budget),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    })
    return pid


def get_active():
    if st.session_state.active_project is None:
        return None
    return st.session_state.projects.get(st.session_state.active_project)


def label(tahap):
    return {
        "tender": "Tender & BOQ", "vendor_selection": "Vendor Qualification",
        "site_execution": "Site Execution", "quality_control": "Quality Control",
    }.get(tahap, tahap)


def stage_status(project, tahap):
    return bool(project["live_data"].get(tahap))


def project_progress(project):
    stages = ["tender", "vendor_selection", "site_execution", "quality_control"]
    return sum(1 for t in stages if stage_status(project, t)), len(stages)


def default_items():
    return pd.DataFrame(_sample_boq_rows(), columns=BOQ_COLUMNS)


def default_wbs():
    return pd.DataFrame([
        {"No": "A", "Item Pekerjaan": "Pekerjaan Persiapan", "Harga (juta Rp)": 28.0, "Mulai": 1, "Selesai": 4, "Realisasi (%)": 100.0},
        {"No": "B", "Item Pekerjaan": "Pekerjaan Tanah", "Harga (juta Rp)": 38.0, "Mulai": 2, "Selesai": 6, "Realisasi (%)": 100.0},
        {"No": "C", "Item Pekerjaan": "Pekerjaan Struktur", "Harga (juta Rp)": 60.0, "Mulai": 4, "Selesai": 9, "Realisasi (%)": 85.0},
        {"No": "D", "Item Pekerjaan": "Pekerjaan Dinding", "Harga (juta Rp)": 32.0, "Mulai": 5, "Selesai": 10, "Realisasi (%)": 60.0},
        {"No": "E", "Item Pekerjaan": "Pekerjaan Atap", "Harga (juta Rp)": 20.0, "Mulai": 6, "Selesai": 11, "Realisasi (%)": 45.0},
        {"No": "F", "Item Pekerjaan": "Pekerjaan Plesteran", "Harga (juta Rp)": 16.0, "Mulai": 8, "Selesai": 12, "Realisasi (%)": 20.0},
        {"No": "G", "Item Pekerjaan": "Pekerjaan Acian", "Harga (juta Rp)": 12.0, "Mulai": 9, "Selesai": 13, "Realisasi (%)": 10.0},
        {"No": "H", "Item Pekerjaan": "Pekerjaan Pintu", "Harga (juta Rp)": 24.0, "Mulai": 10, "Selesai": 14, "Realisasi (%)": 0.0},
        {"No": "I", "Item Pekerjaan": "Pekerjaan Plafond", "Harga (juta Rp)": 20.0, "Mulai": 10, "Selesai": 14, "Realisasi (%)": 0.0},
        {"No": "J", "Item Pekerjaan": "Pekerjaan Lantai", "Harga (juta Rp)": 20.0, "Mulai": 11, "Selesai": 15, "Realisasi (%)": 0.0},
        {"No": "K", "Item Pekerjaan": "Pekerjaan Pengecatan", "Harga (juta Rp)": 16.0, "Mulai": 12, "Selesai": 16, "Realisasi (%)": 0.0},
    ])


# =========================================================
# LIVE: PENAWARAN VENDOR (auto-refresh tiap 3 detik)
# =========================================================
@st.fragment(run_every=3)
def live_penawaran(pid, invited, deadline_days):
    pen = fetch_penawaran(pid)
    rows = []
    sent = 0
    for v in invited:
        em = (v.get("Email") or "").lower()
        r = pen.get(em)
        if r:
            sent += 1
        rows.append({
            "Vendor": v["Vendor"],
            "Email": em or "(belum diisi)",
            "Status": "Sudah mengirim" if r else "Belum mengirim",
            "Penawaran (juta Rp)": float(r["harga_rab"]) if r else None,
            "Durasi (hari)": int(deadline_days) if r else None,
            "Catatan": (r.get("catatan") or "") if r else "",
            "Update terakhir": fmt_ts(r.get("updated_at")) if r else "-",
        })
    c1, c2 = st.columns(2)
    c1.metric("Vendor sudah mengirim", f"{sent} dari {len(invited)}")
    c2.metric("Diperbarui", now_local().strftime("%H:%M:%S"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    no_email = [v["Vendor"] for v in invited if not v.get("Email")]
    if no_email:
        st.warning("Vendor berikut belum punya email, jadi belum bisa login/mengisi penawaran: "
                   + ", ".join(no_email))


# =========================================================
# LIVE: RANKING PENAWARAN (urut harga terendah, auto-refresh)
# =========================================================
@st.fragment(run_every=3)
def live_ranking(pid, invited, budget, deadline_days):
    pen = fetch_penawaran(pid)
    data = []
    for v in invited:
        em = (v.get("Email") or "").lower()
        r = pen.get(em) if em else None
        harga_v = float(r["harga_rab"]) if r and r.get("harga_rab") else 0.0
        data.append({
            "Vendor": v["Vendor"],
            "harga": harga_v,
            "durasi": int(deadline_days) if harga_v > 0 else 0,
            "updated": r.get("updated_at") if r else None,
        })

    sudah = sorted([d for d in data if d["harga"] > 0],
                   key=lambda d: (d["harga"], str(d["updated"] or "")))
    belum = [d for d in data if d["harga"] <= 0]

    rows = []
    low = sudah[0]["harga"] if sudah else 0.0
    for i, d in enumerate(sudah, start=1):
        if d["harga"] > budget:
            status = "Melebihi budget"
        elif i == 1:
            status = "Terendah"
        else:
            status = "Dalam budget"
        rows.append({
            "Peringkat": i,
            "Vendor": d["Vendor"],
            "Penawaran": format_rupiah(d["harga"]),
            "Selisih dari terendah": "-" if i == 1 else f"+{format_rupiah(d['harga'] - low)}",
            "Durasi (hari)": d["durasi"],
            "Status": status,
            "Update terakhir": fmt_ts(d["updated"]),
        })
    for d in belum:
        rows.append({
            "Peringkat": None,
            "Vendor": d["Vendor"],
            "Penawaran": "-",
            "Selisih dari terendah": "-",
            "Durasi (hari)": None,
            "Status": "Belum mengirim",
            "Update terakhir": "-",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Diurutkan otomatis dari harga terendah · diperbarui {now_local():%H:%M:%S}")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Akun")
    st.caption(user_email)
    st.markdown("<span class='pill pill-ok'>KPDP</span>", unsafe_allow_html=True)
    if st.button("Keluar", key="btn_logout_k"):
        do_logout()
    st.markdown("---")
    st.markdown("### Project Manager")

    if st.session_state.projects:
        options = list(st.session_state.projects.keys())
        current_idx = 0
        if st.session_state.active_project in options:
            current_idx = options.index(st.session_state.active_project)
        selected = st.selectbox(
            "Project aktif", options=options, index=current_idx,
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
    new_budget = st.number_input("Budget project (juta Rp)", min_value=10.0, value=300.0, step=10.0, key="new_budget")
    new_deadline = st.number_input("Deadline project (hari)", min_value=1, value=90, step=5, key="new_deadline")
    if st.button("Tambah Project", key="btn_add_project"):
        if new_name.strip():
            pid_new = create_project(
                new_name.strip(), new_client.strip() or "-",
                new_location.strip() or "-", new_budget, new_deadline,
            )
            st.session_state.active_project = pid_new
            st.session_state.step = 1
            rerun()
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
                f"<div class='step-row'><span class='step-dot {dot}'></span>"
                f"<span style='color:{text_color};'>{i}. {t['label']}</span></div>",
                unsafe_allow_html=True,
            )
        done, total = project_progress(active)
        st.caption(f"{done} dari {total} tahap selesai")

        st.markdown("---")
        st.markdown("**Budget & Deadline Project**")
        new_budget_edit = st.number_input(
            "Budget project (juta Rp)", min_value=10.0,
            value=max(10.0, float(active["meta"]["budget"])),
            step=10.0, key=f"edit_budget_{active['meta']['id']}",
        )
        new_deadline_edit = st.number_input(
            "Deadline project (hari)", min_value=1,
            value=int(active["meta"]["deadline"]),
            step=5, key=f"edit_deadline_{active['meta']['id']}",
        )
        if (new_budget_edit != active["meta"]["budget"]
                or new_deadline_edit != active["meta"]["deadline"]):
            active["meta"]["budget"] = float(new_budget_edit)
            active["meta"]["deadline"] = int(new_deadline_edit)
            rerun()

        st.markdown("---")
        if st.button("Hapus Project Ini", key="btn_delete_project"):
            del_id = st.session_state.active_project
            delete_project_db(del_id)
            del st.session_state.projects[del_id]
            st.session_state.active_project = (
                list(st.session_state.projects.keys())[0]
                if st.session_state.projects else None
            )
            st.session_state.step = 0
            rerun()

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
        f"<span class='pill pill-info'>{total_projects} project</span></div>",
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
        stop_app()

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
            "Project": m["name"], "Klien": m["client"], "Lokasi": m["location"],
            "Budget": format_rupiah(m["budget"]), "Tahapan": f"{done}/{total}",
            "Vendor Terpilih": vs.get("selected_vendor", "-"),
            "Nilai Kontrak": format_rupiah(vs.get("selected_cost", 0)) if vs.get("selected_cost") else "-",
            "Status": status,
        })

    df_summary = pd.DataFrame(summary_rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total project", len(df_summary))
    total_budget = sum(p["meta"]["budget"] for p in st.session_state.projects.values())
    c2.metric("Total budget", format_rupiah(total_budget))
    c3.metric("Project selesai", int((df_summary["Status"] == "Selesai").sum()))

    st.markdown("---")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Buka Project")
    project_options = list(st.session_state.projects.keys())
    selected_pid = st.selectbox(
        "Pilih project untuk dibuka", options=project_options,
        format_func=lambda x: st.session_state.projects[x]["meta"]["name"],
        key="open_project_select",
    )
    if st.button("Buka Project", key="btn_open"):
        st.session_state.active_project = selected_pid
        st.session_state.step = 1
        rerun()
    stop_app()


if active is None:
    st.info("Belum ada project aktif.")
    stop_app()

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
    if i == step:
        cls = "active"
    elif i > 0 and i < step:
        cls = "done"
    else:
        cls = ""
    stepper_html += (
        f"<div class='step-chip {cls}'><span class='num'>{i}</span><span>{s['label']}</span></div>"
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
                rerun()
    with c3:
        if show_next and step < len(STEPS) - 1:
            if st.button(next_label + " →", key=f"next_{step}"):
                next_step()
                rerun()


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
            catatan = st.text_area("Ringkasan", placeholder="Minimal 10 karakter...",
                                   height=120, key="td_catatan")
        with col2:
            st.markdown("#### RAB Awal (Estimasi Kita)")
            st.markdown(
                f"<div class='hint-box'>Budget project: <b>{format_rupiah(meta['budget'])}</b>.</div>",
                unsafe_allow_html=True,
            )
            rab_awal_input = st.number_input(
                "RAB Estimasi (juta Rp)", min_value=10.0,
                value=max(10.0, float(meta.get("rab_awal", meta.get("budget", 300.0)))),
                step=10.0, key=f"rab_awal_{meta['id']}",
            )
            budget_input = st.number_input(
                "Budget approved (juta Rp)", min_value=10.0,
                value=max(10.0, float(meta.get("budget", 300.0))),
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

        with st.expander("📥 Import BOQ dari Excel", expanded=False):
            if not HAS_OPENPYXL:
                st.warning("Library 'openpyxl' belum terpasang, jadi import Excel (.xlsx) dinonaktifkan. "
                           "Gunakan CSV, atau tambahkan 'openpyxl' ke requirements.txt lalu redeploy.")
            c_up, c_tpl = st.columns([3, 1])
            with c_tpl:
                tpl_data, tpl_name, tpl_mime = boq_template_bytes()
                st.download_button("Unduh Template", data=tpl_data, file_name=tpl_name,
                                   mime=tpl_mime, key=f"tpl_boq_{pid}")
            with c_up:
                up_types = ["xlsx", "xls", "csv"] if HAS_OPENPYXL else ["csv"]
                up_file = st.file_uploader(
                    "Upload file (" + " / ".join("." + t for t in up_types) + ")",
                    type=up_types, key=f"boq_upload_{pid}_{active['items_ver']}",
                )

            if up_file is not None:
                try:
                    parsed = parse_boq_excel(up_file)
                    n_vol = sum(1 for it in parsed if it["Volume"])
                    st.success(f"Terbaca {len(parsed)} baris ({n_vol} item dengan volume). Preview:")
                    st.dataframe(pd.DataFrame(parsed, columns=BOQ_COLUMNS),
                                 use_container_width=True, hide_index=True, height=420)
                    mode = st.radio("Mode impor", ["Ganti semua BOQ", "Tambahkan ke BOQ yang ada"],
                                    horizontal=True, key=f"boq_mode_{pid}")
                    if st.button("Terapkan ke BOQ", type="primary", key=f"boq_apply_{pid}"):
                        if mode == "Ganti semua BOQ":
                            new_items = [dict(it) for it in parsed]
                        else:
                            new_items = list(active["items_data"]) + [dict(it) for it in parsed]
                        active["items_data"] = new_items
                        active["items_ver"] += 1
                        rerun()
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

        st.caption(
            "📝 **Tabel ini bisa diedit langsung** — tambah baris, hapus, ubah nama item, "
            "volume, satuan, dan spesifikasi. Isi tabel mengikuti file Excel BOQ apa adanya."
        )

        active["items_data"] = _normalize_items(active["items_data"])
        items_df = pd.DataFrame(active["items_data"], columns=BOQ_COLUMNS)
        items_edited = st.data_editor(
            items_df, use_container_width=True, hide_index=True, num_rows="dynamic",
            height=560, key=f"items_editor_{pid}_{active['items_ver']}",
            column_config={
                "No": st.column_config.TextColumn("NO", width="small"),
                "Description": st.column_config.TextColumn("DESCRIPTION", required=True, width="large"),
                "Specification": st.column_config.TextColumn("SPESIFICATION", width="medium"),
                "Volume": st.column_config.TextColumn("VOLUME", width="small"),
                "Unit": st.column_config.TextColumn("UNIT", width="small"),
                "Unit Price (Rp)": st.column_config.TextColumn("UNIT PRICE (Rp.)", width="small"),
                "Total Price (Rp)": st.column_config.TextColumn("TOTAL PRICE (Rp.)", width="small"),
            },
        )
        active["items_data"] = _normalize_items(items_edited.to_dict("records"))
        st.caption(f"Total baris: **{len(items_edited)}**")

    dokumen_lengkap = all([doc_drawing, doc_rks, doc_spek, doc_boq, doc_k3])
    catatan_lengkap = len(catatan.strip()) >= 10
    timeline_valid = tgl_aanwijzing < tgl_penawaran < tgl_kontrak
    items_ok = bool(active.get("items_data")) and len(active["items_data"]) > 0
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
            rerun()

    render_nav(show_next=stage_status(active, "tender"), next_label="Lanjut ke Vendor Qualification")


# =========================================================
# STEP 2 — VENDOR QUALIFICATION
# =========================================================
elif step == 2:
    st.markdown("## Tahap 2 — Vendor Qualification & Selection")
    st.markdown(
        "<div class='flow-hint'>"
        "Alur: Vendor diundang (pakai email) → vendor login & isi penawaran sendiri → "
        "KPDP melihat langsung (realtime) → klarifikasi teknis → optimasi → deal harga & SPK."
        "</div>",
        unsafe_allow_html=True,
    )

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "1. BOQ & Vendor Diundang",
        "2. Penawaran Masuk (Live)",
        "3. Appointment Klarifikasi Teknis",
        "4. Optimasi Setelah Klarifikasi",
        "5. Deal Harga & SPK",
    ])

    # ---------- 1. BOQ & VENDOR DIUNDANG ----------
    with sub1:
        st.markdown("### Item Pekerjaan (BOQ)")
        items_data = active.get("items_data", [])
        if not items_data:
            st.warning("BOQ belum diisi. Kembali ke Tahap 1.")
        else:
            boq_view = pd.DataFrame(_normalize_items(items_data), columns=BOQ_COLUMNS)
            boq_view.columns = ["NO", "DESCRIPTION", "SPESIFICATION", "VOLUME", "UNIT",
                                "UNIT PRICE (Rp.)", "TOTAL PRICE (Rp.)"]
            st.dataframe(boq_view, use_container_width=True, hide_index=True, height=480)

        st.markdown("---")
        st.markdown("### Vendor yang Diundang")
        st.caption("Isi **Email** tiap vendor. Vendor login dengan email ini untuk mengisi penawaran. "
                   "Perubahan daftar ini otomatis tersimpan ke database.")

        active["invited_vendors"] = _norm_invited(active.get("invited_vendors"))
        if not active["invited_vendors"]:
            active["invited_vendors"] = [{"Vendor": n, "Email": ""} for n in INVITED_VENDORS]

        with st.expander("📥 Import Vendor dari Excel", expanded=False):
            if not HAS_OPENPYXL:
                st.warning("Library 'openpyxl' belum terpasang, jadi import Excel (.xlsx) dinonaktifkan.")
            cv_up, cv_tpl = st.columns([3, 1])
            with cv_tpl:
                vt_data, vt_name, vt_mime = vendor_template_bytes()
                st.download_button("Unduh Template", data=vt_data, file_name=vt_name,
                                   mime=vt_mime, key=f"tpl_vendor_{pid}")
            with cv_up:
                v_types = ["xlsx", "xls", "csv"] if HAS_OPENPYXL else ["csv"]
                v_file = st.file_uploader(
                    "Upload file (" + " / ".join("." + t for t in v_types) + ") — kolom: Vendor, Email",
                    type=v_types, key=f"vendor_upload_{pid}_{active['invited_ver']}",
                )

            if v_file is not None:
                try:
                    v_parsed = parse_vendor_excel(v_file)
                    st.success(f"Terbaca {len(v_parsed)} vendor. Preview:")
                    st.dataframe(pd.DataFrame(v_parsed), use_container_width=True, hide_index=True)
                    v_mode = st.radio("Mode impor",
                                      ["Ganti semua vendor", "Tambahkan ke daftar yang ada"],
                                      horizontal=True, key=f"vendor_mode_{pid}")
                    if st.button("Terapkan ke Daftar Vendor", type="primary", key=f"vendor_apply_{pid}"):
                        if v_mode == "Ganti semua vendor":
                            new_list = list(v_parsed)
                        else:
                            new_list = list(active["invited_vendors"])
                            existing = {x["Vendor"].lower() for x in new_list}
                            for nm in v_parsed:
                                if nm["Vendor"].lower() not in existing:
                                    new_list.append(nm)
                        active["invited_vendors"] = new_list
                        active["invited_ver"] += 1
                        rerun()
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

        invited_df = pd.DataFrame(active["invited_vendors"], columns=["Vendor", "Email"])
        invited_edited = st.data_editor(
            invited_df, use_container_width=True, hide_index=True, num_rows="dynamic",
            key=f"invited_{pid}_{active['invited_ver']}",
            column_config={
                "Vendor": st.column_config.TextColumn("Vendor", required=True),
                "Email": st.column_config.TextColumn("Email login vendor"),
            },
        )
        active["invited_vendors"] = _norm_invited(invited_edited.to_dict("records"))

    invited = active.get("invited_vendors", [])
    invited_names = [v["Vendor"] for v in invited]

    # ---------- 2. PENAWARAN MASUK (LIVE) ----------
    with sub2:
        st.markdown("### Penawaran Masuk dari Vendor")
        st.caption("Tabel ini **update otomatis tiap 3 detik** saat vendor mengirim / mengubah penawaran "
                   "dari portal vendor. Tidak perlu refresh halaman.")
        if not invited:
            st.warning("Belum ada vendor diundang.")
        else:
            save_project(pid)  # pastikan undangan vendor sudah di database
            live_penawaran(pid, invited, deadline)

    # ---------- 3. APPOINTMENT ----------
    with sub3:
        st.markdown("### Appointment Klarifikasi Teknis")
        if not invited:
            st.warning("Belum ada vendor.")
        else:
            with st.expander("➕ Buat Appointment", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    appt_vendor = st.selectbox("Vendor", options=invited_names, key=f"apv_{pid}")
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
                        v_email = next((v["Email"] for v in invited if v["Vendor"] == appt_vendor), "")
                        try:
                            db().table("appointments").insert({
                                "id": str(uuid.uuid4())[:6], "project_id": pid,
                                "vendor": appt_vendor, "vendor_email": v_email.lower(),
                                "mode": appt_mode, "date": str(appt_date), "time": str(appt_time),
                                "duration": int(appt_duration), "location": appt_location,
                                "participants": appt_participants,
                                "vendor_participants": appt_vendor_participants,
                                "agenda": appt_agenda, "status": "Terjadwal", "hasil": "",
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }).execute()
                            st.success("Appointment dibuat.")
                            rerun()
                        except Exception as e:
                            st.error(f"Gagal menyimpan appointment: {e}")

            st.markdown("---")
            appointments = fetch_appointments(pid)
            if not appointments:
                st.info("Belum ada appointment.")
            else:
                status_options = ["Terjadwal", "Berlangsung", "Selesai", "Dibatalkan"]
                for i, appt in enumerate(appointments):
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
                        cur_status = appt["status"] if appt["status"] in status_options else "Terjadwal"
                        new_status = st.selectbox("Status", status_options,
                                                  index=status_options.index(cur_status),
                                                  key=f"as_{appt['id']}")
                        hasil = st.text_area("Hasil klarifikasi", value=appt.get("hasil") or "",
                                             height=120, key=f"ah_{appt['id']}")
                        cs, cd = st.columns([3, 1])
                        with cs:
                            if st.button("Simpan", key=f"sv_{appt['id']}"):
                                try:
                                    db().table("appointments").update(
                                        {"status": new_status, "hasil": hasil}
                                    ).eq("id", appt["id"]).execute()
                                    st.success("Tersimpan.")
                                    rerun()
                                except Exception as e:
                                    st.error(f"Gagal menyimpan: {e}")
                        with cd:
                            if st.button("Hapus", key=f"dl_{appt['id']}"):
                                try:
                                    db().table("appointments").delete().eq("id", appt["id"]).execute()
                                    rerun()
                                except Exception as e:
                                    st.error(f"Gagal menghapus: {e}")

    # ---------- data penawaran untuk optimasi & SPK ----------
    pen_now = fetch_penawaran(pid)
    harga, durasi_map = {}, {}
    for v in invited:
        r = pen_now.get(v["Email"].lower()) if v["Email"] else None
        harga[v["Vendor"]] = float(r["harga_rab"]) if r else 0.0
        durasi_map[v["Vendor"]] = int(deadline) if r else 0

    # ---------- 4. OPTIMASI ----------
    with sub4:
        st.markdown("### Optimasi Setelah Klarifikasi")
        n_appt = len(fetch_appointments(pid))

        prasyarat = []
        if not invited:
            prasyarat.append("vendor diundang")
        if not any(h > 0 for h in harga.values()):
            prasyarat.append("penawaran harga vendor")
        if n_appt == 0:
            prasyarat.append("appointment klarifikasi")

        if prasyarat:
            st.warning("Belum bisa optimasi. Selesaikan: " + ", ".join(prasyarat))
        else:
            st.caption("Optimasi memilih vendor dengan **penawaran harga terendah** yang masih dalam budget.")

            st.markdown("#### Data untuk Optimasi (Ranking Penawaran)")
            live_ranking(pid, invited, budget, deadline)

            eligible = [n for n in invited_names if harga[n] > 0 and harga[n] <= budget]

            if not eligible:
                st.error("Tidak ada vendor yang lolos kriteria.")
            else:
                cc1, cc2 = st.columns(2)
                cc1.metric("Budget", format_rupiah(budget))
                cc2.metric("Vendor lolos budget", f"{len(eligible)} dari {len(invited)}")

                st.markdown("---")
                selected_name, solver_status = solve_vendor_lp(
                    eligible, {n: harga[n] for n in eligible}, budget, None, 1.0, 0.0
                )

                if selected_name is None:
                    st.error(f"Optimasi gagal: {solver_status}")
                else:
                    selected_cost = harga[selected_name]
                    selected_duration = durasi_map.get(selected_name, 0)

                    st.success(f"**Vendor terbaik: {selected_name}**")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Vendor", selected_name)
                    r2.metric("Harga Penawaran", format_rupiah(selected_cost))
                    r3.metric("Durasi", f"{selected_duration} hari")
                    r4.metric("Sisa budget", format_rupiah(budget - selected_cost))

                    st.session_state[f"optimized_vendor_{pid}"] = selected_name
                    st.session_state[f"optimized_cost_{pid}"] = selected_cost

    # ---------- 5. DEAL & SPK ----------
    with sub5:
        st.markdown("### Deal Harga & SPK")
        optimized_vendor = st.session_state.get(f"optimized_vendor_{pid}")
        optimized_cost = st.session_state.get(f"optimized_cost_{pid}", 0)

        if not optimized_vendor:
            st.warning("Selesaikan optimasi dulu (sub-tab 4).")
        else:
            st.markdown(
                f"<div class='hint-box'><b>Vendor:</b> {html.escape(str(optimized_vendor))}<br>"
                f"<b>Harga penawaran awal:</b> {format_rupiah(optimized_cost)}</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                final_price = st.number_input(
                    "Harga kesepakatan (juta Rp)", min_value=1.0,
                    value=max(1.0, float(optimized_cost)), step=5.0, key=f"fp_{pid}",
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
                ["Belum", "Diajukan", "Disetujui", "Revisi"], index=0, key=f"vsa_{pid}",
            )
            negosiasi_notes = st.text_area("Catatan Negosiasi", height=100, key=f"ngn_{pid}")

            valid = bool(
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
                    sel_email = next((v["Email"] for v in invited if v["Vendor"] == optimized_vendor), "")
                    payload = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "selected_vendor": optimized_vendor,
                        "selected_vendor_email": sel_email,
                        "rab_awal_vendor": float(optimized_cost),
                        "selected_cost": float(final_price),
                        "saving": float(saving),
                        "selected_duration": int(durasi_map.get(optimized_vendor, 0)),
                        "spk_number": spk_number,
                        "spk_date": str(spk_date),
                        "spk_signer": spk_signer,
                        "approval_drawing": approval_drawing,
                        "negosiasi_notes": negosiasi_notes,
                        "appointment_count": len(fetch_appointments(pid)),
                        "invited_vendors": len(invited),
                    }
                    live_data["vendor_selection"] = payload
                    st.success("Keputusan & SPK tersimpan. Lanjut ke Site Execution...")
                    next_step()
                    rerun()

    render_nav(show_next=stage_status(active, "vendor_selection"), next_label="Lanjut ke Site Execution")


# =========================================================
# STEP 3 — SITE EXECUTION (S-CURVE TABEL + GRAFIK)
# =========================================================
elif step == 3:
    st.markdown("## Tahap 3 — Site Execution")
    st.markdown(
        "<div class='flow-hint'>"
        "Unduh S-Curve yang diupload vendor, atau upload file Excel S-Curve. Sistem hanya membuat "
        "grafik S-Curve (rencana vs realisasi) dari file tersebut, tanpa menampilkan tabel."
        "</div>",
        unsafe_allow_html=True,
    )

    vs = live_data.get("vendor_selection", {})
    selected_vendor = vs.get("selected_vendor", "-")
    final_cost = vs.get("selected_cost", 0)

    st.caption(f"Kontraktor: {selected_vendor} · Nilai kontrak: {format_rupiah(final_cost)}")

    st.markdown("### S-Curve dari Vendor")
    sel_email_v = vs.get("selected_vendor_email", "")
    sc_sub = fetch_scurve(pid, sel_email_v) if sel_email_v else None
    vendor_file = None
    if not sel_email_v:
        st.info("Email vendor terpilih belum tersimpan, jadi S-Curve vendor tidak bisa dicari.")
    elif not sc_sub:
        st.warning("Vendor terpilih belum mengupload S-Curve.")
    else:
        vendor_file = scurve_file_from_row(sc_sub)
        st.success(f"S-Curve sudah diupload {fmt_ts(sc_sub.get('uploaded_at'))} · "
                   f"file {sc_sub.get('filename') or '-'}")
        if vendor_file:
            sc_bytes, sc_fname, sc_mime = vendor_file
            st.download_button("Unduh S-Curve dari Vendor", data=sc_bytes,
                               file_name=sc_fname, mime=sc_mime, key=f"dl_sc_{pid}")
        else:
            st.warning("File S-Curve tidak bisa dibaca (kemungkinan data lama dengan format berbeda). "
                       "Minta vendor upload ulang.")

    st.markdown("### Grafik S-Curve dari Excel")
    st.caption("Aplikasi mencari sheet bernama S-Curve di dalam file Excel, lalu membaca baris Plan dan "
               "Aktual (atau Progress Rencana/Realisasi Kumulatif) untuk membuat grafik. Data harian "
               "dijumlahkan per minggu.")

    VENDOR_OPT = "Pakai file S-Curve dari vendor"
    UPLOAD_OPT = "Upload file Excel baru"
    src_opts = ([VENDOR_OPT] if vendor_file else []) + [UPLOAD_OPT]
    src_choice = st.radio("Sumber file", src_opts, horizontal=True, key=f"sc_src_{pid}")

    chart_bytes, chart_name = None, ""
    if src_choice == VENDOR_OPT and vendor_file:
        chart_bytes, chart_name = vendor_file[0], vendor_file[1]
    else:
        gc_types = ["xlsx", "xls", "csv"] if HAS_OPENPYXL else ["csv"]
        gc_file = st.file_uploader(
            "Upload S-Curve (" + " / ".join("." + t for t in gc_types) + ")",
            type=gc_types, key=f"sc_chart_upload_{pid}",
        )
        if gc_file is not None:
            chart_bytes, chart_name = gc_file.getvalue(), gc_file.name

    sheet_choice = None
    if chart_bytes is not None:
        sheets_avail = scurve_sheet_names(chart_bytes, chart_name)
        if len(sheets_avail) > 1:
            AUTO_OPT = "Otomatis (cari sheet bernama S-Curve)"
            pick = st.selectbox("Sheet yang dibaca", [AUTO_OPT] + sheets_avail, key=f"sc_sheet_{pid}")
            sheet_choice = None if pick == AUTO_OPT else pick
        if st.button("Buat / Perbarui Grafik dari File Ini", type="primary", key=f"sc_make_{pid}"):
            try:
                res = parse_scurve_chart(chart_bytes, chart_name, sheet_choice)
                res["filename"] = chart_name
                active["scurve_chart"] = res
                st.success("Grafik berhasil dibuat dari file.")
            except Exception as e:
                st.error(f"Gagal membuat grafik: {e}")

    chart = active.get("scurve_chart")
    current_week = 0
    plan_now = actual_now = deviasi = spi = 0.0
    weeks = []

    if not chart:
        st.info("Belum ada grafik. Pilih sumber file lalu klik tombol \"Buat / Perbarui Grafik\".")
    else:
        is_daily = bool(chart.get("days"))
        weeks = [int(v) for v in (chart["days"] if is_daily else chart["weeks"])]  # posisi sumbu-x
        unit = "Hari" if is_daily else "Minggu"
        us = "H" if is_daily else "M"
        plan = np.array([np.nan if v is None else float(v) for v in chart["plan"]])
        if chart.get("actual"):
            actual = np.array([np.nan if v is None else float(v) for v in chart["actual"]])
        else:
            actual = np.full(len(weeks), np.nan)
        has_actual = bool(np.isfinite(actual).any())
        x = np.array(weeks)

        st.caption(f"Sumber grafik: {chart.get('filename', '-')} · sheet {chart.get('sheet', '-')} · "
                   f"{len(weeks)} {unit.lower()}" + (f" · {chart['mode']}" if chart.get("mode") else ""))
        if chart.get("warning"):
            st.warning(chart["warning"])

        if has_actual:
            default_week = weeks[int(np.where(np.isfinite(actual))[0][-1])]
        else:
            default_week = weeks[min(len(weeks) - 1, int(len(weeks) * 0.6))]
        chart_sig = hashlib.md5(json.dumps(chart, sort_keys=True, default=str).encode()).hexdigest()[:8]
        pos_now = int(st.number_input(
            f"{unit} saat ini (default: {unit.lower()} terakhir yang ada realisasinya)",
            min_value=weeks[0], max_value=weeks[-1],
            value=int(default_week), step=1, key=f"current_week_{pid}_{chart_sig}",
        ))
        idx = int(np.argmin([abs(w - pos_now) for w in weeks]))
        current_week = (weeks[idx] + 6) // 7 if is_daily else weeks[idx]
        lbl_now = f"{us}{weeks[idx]}" + (f" / Minggu {current_week}" if is_daily else "")

        plan_now = float(plan[idx])
        if has_actual:
            fin = np.where(np.isfinite(actual[:idx + 1]))[0]
            actual_now = float(actual[fin[-1]]) if len(fin) else 0.0
        else:
            actual_now = 0.0

        st.markdown("### Grafik S-Curve")
        fig, ax = plt.subplots(figsize=(13, 5.5))
        valid = np.isfinite(actual)
        mk = 0 if is_daily else 6
        if has_actual and valid.sum() >= 2:
            ax.fill_between(x[valid], plan[valid], actual[valid], where=(actual[valid] < plan[valid]),
                            color="#dc2626", alpha=0.12, label="Deviasi negatif")
            ax.fill_between(x[valid], plan[valid], actual[valid], where=(actual[valid] >= plan[valid]),
                            color="#047857", alpha=0.12, label="Deviasi positif")
        ax.plot(x, plan, color="#1e40af", linewidth=2.5, marker="o", markersize=mk, label="Rencana (S-Curve)")
        if has_actual:
            ax.plot(x, actual, color="#dc2626", linewidth=2.5, marker="s", markersize=mk, label="Realisasi")
        ax.axvline(weeks[idx], linestyle=":", color="#6b7280", linewidth=1.5, label=f"Saat ini ({us}{weeks[idx]})")

        ax.scatter([weeks[idx]], [plan_now], color="#1e40af", s=100, zorder=5, edgecolor="white", linewidth=2)
        ax.annotate(f"Rencana: {plan_now:.2f}%", xy=(weeks[idx], plan_now), xytext=(10, 10),
                    textcoords="offset points", color="#1e40af", fontsize=9, fontweight="600")
        if has_actual and np.isfinite(actual[idx]):
            ax.scatter([weeks[idx]], [actual_now], color="#dc2626", s=100, marker="D", zorder=5,
                       edgecolor="white", linewidth=2)
            ax.annotate(f"Realisasi: {actual_now:.2f}%", xy=(weeks[idx], actual_now), xytext=(10, -20),
                        textcoords="offset points", color="#dc2626", fontsize=9, fontweight="600")

        if is_daily:
            wk_starts = [d for d in weeks if (d - 1) % 7 == 0]
            for d in wk_starts[1:]:
                ax.axvline(d, color="#9ca3af", linewidth=0.6, alpha=0.5)
            ax.set_xticks(wk_starts)
            ax.set_xticklabels([f"M{(d - 1) // 7 + 1}\nH{d}" for d in wk_starts])
            ax.set_xlabel("Minggu ke- (H = hari ke-)", fontsize=10)
        else:
            ax.set_xticks(x[::max(1, len(x) // 20)])
            ax.set_xlabel("Minggu ke-", fontsize=10)
        ax.set_ylabel("Progress Kumulatif (%)", fontsize=10)
        ax.set_title(f"S-Curve — {selected_vendor}", fontsize=12, fontweight="600")
        ax.set_ylim(0, 105)
        ax.set_xlim(weeks[0], max(weeks[-1], weeks[0] + 1))
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric(f"Rencana ({lbl_now})", f"{plan_now:.2f}%")
        if has_actual:
            deviasi = actual_now - plan_now
            spi = actual_now / plan_now if plan_now > 0 else 0.0
            sc2.metric(f"Realisasi ({lbl_now})", f"{actual_now:.2f}%")
            sc3.metric("Deviasi", f"{deviasi:+.2f}%")
            sc4.metric("SPI", f"{spi:.3f}")
            if deviasi < -5:
                st.error(f"Deviasi {deviasi:.2f}% — berpotensi terlambat.")
            elif deviasi < 0:
                st.warning(f"Deviasi {deviasi:.2f}% — perlu perhatian.")
            else:
                st.success(f"Progress sesuai/lebih cepat {deviasi:+.2f}%.")
        else:
            sc2.metric(f"Realisasi ({lbl_now})", "-")
            sc3.metric("Deviasi", "-")
            sc4.metric("SPI", "-")
            st.info("File ini tidak memiliki baris realisasi, jadi hanya kurva rencana yang digambar.")

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
    sc_ok = bool(chart)
    form_valid = sc_ok and pcm_valid and kendala_valid and aksi_valid and kpi_valid

    if not form_valid:
        missing = []
        if not sc_ok:
            missing.append("grafik S-Curve belum dibuat")
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
                "current_pos": int(weeks[idx]) if weeks else 0,
                "current_unit": "hari" if (chart or {}).get("days") else "minggu",
                "plan_pct": float(plan_now),
                "actual_pct": float(actual_now),
                "deviation_pct": float(deviasi),
                "spi": float(spi),
                "pcm_date": str(pcm_date),
                "pcm_notes": pcm_notes,
                "kendala": kendala,
                "aksi": aksi,
                "kpi": kpi,
                "weeks_count": len(weeks),
                "source_file": (chart or {}).get("filename", ""),
            }
            live_data["site_execution"] = payload
            st.success("Site Execution tersimpan. Lanjut ke Quality Control...")
            next_step()
            rerun()

    render_nav(show_next=stage_status(active, "site_execution"), next_label="Lanjut ke Quality Control")


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
        c2.metric("BAST", live.get("bast_number", "-") or "-")
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
    final_status = st.radio("Sesuai kontrak?", ["Pilih...", "Ya", "Tidak"], index=0, key="qc_final")

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
        bast_valid = bool(
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
                "material_inspection": mat_insp, "ndt": ndt, "hydrotest": hydro,
                "visual": visual, "final_status": final_status,
                "bast_date": str(bast_date) if bast_date else None,
                "bast_number": bast_number, "approved_by": approved_by,
                "bast_notes": bast_notes,
            }
            live_data["quality_control"] = payload
            st.success("Selesai. Project diselesaikan.")
            next_step()
            rerun()

    render_nav(show_next=stage_status(active, "quality_control"), next_label="Lihat Ringkasan")


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
            appts_final = fetch_appointments(pid)
            if appts_final:
                st.download_button(
                    "Unduh Appointment (CSV)",
                    data=pd.DataFrame(appts_final).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"appointments_{meta['name'].replace(' ', '_')}.csv",
                    mime="text/csv",
                )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Kembali ke QC", key="btn_bq"):
            goto_step(4)
            rerun()
    with c2:
        if st.button("Kembali ke Daftar Project", key="btn_bl"):
            st.session_state.step = 0
            rerun()

st.markdown("---")
st.caption("Vendor Selection & Project Control · Multi-project workspace · Data tersimpan di database.")

# simpan semua perubahan ke database di akhir setiap run
persist()
