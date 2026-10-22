import io
import re
import csv
import html
import json
import hashlib
import uuid
from datetime import datetime, date, time

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


def fmt_ts(ts):
    """Format timestamp ISO (UTC) dari database ke WIB."""
    if not ts:
        return "-"
    try:
        return pd.to_datetime(ts).tz_convert("Asia/Jakarta").strftime("%d %b %Y %H:%M:%S")
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
# HELPER S-CURVE VENDOR (template + parser)
# =========================================================
SCURVE_COLUMNS = ["No", "Uraian Pekerjaan", "Harga (juta Rp)", "Mulai (Minggu)", "Selesai (Minggu)"]
_SC_ALIASES = {
    "No": ["no", "nomor"],
    "Uraian Pekerjaan": ["uraian pekerjaan", "item pekerjaan", "uraian", "pekerjaan",
                         "deskripsi", "description"],
    "Harga (juta Rp)": ["harga (juta rp)", "harga (juta rp.)", "harga (juta)", "harga",
                        "nilai (juta rp)", "nilai"],
    "Mulai (Minggu)": ["mulai (minggu)", "mulai (m)", "mulai", "start"],
    "Selesai (Minggu)": ["selesai (minggu)", "selesai (m)", "selesai", "finish", "end"],
}


def scurve_template_bytes(max_week):
    """Template S-Curve yang harus diisi vendor terpilih."""
    mw = max(1, int(max_week))
    spans = [("A", "Pekerjaan Persiapan", 0.0, 0.2),
             ("B", "Pekerjaan Struktur", 0.15, 0.6),
             ("C", "Pekerjaan Arsitektur", 0.5, 0.9),
             ("D", "Pekerjaan Finishing", 0.8, 1.0)]
    rows = []
    for no, name, a, b in spans:
        s = max(1, int(round(a * mw)))
        e = max(s, min(mw, int(round(b * mw))))
        rows.append({"No": no, "Uraian Pekerjaan": name, "Harga (juta Rp)": None,
                     "Mulai (Minggu)": s, "Selesai (Minggu)": e})
    df = pd.DataFrame(rows, columns=SCURVE_COLUMNS)
    guide = pd.DataFrame({"Petunjuk": [
        "1. Isi sheet 'S-Curve'. Jangan ubah nama kolom di baris pertama.",
        "2. Hapus/ubah baris contoh, lalu isi semua item pekerjaan Anda.",
        "3. Harga (juta Rp): nilai tiap item dalam juta rupiah, harus lebih dari 0.",
        f"4. Mulai/Selesai: nomor minggu (bilangan bulat), antara 1 dan {mw}.",
        "5. Selesai tidak boleh lebih kecil dari Mulai.",
        "6. Total harga seluruh item sebaiknya sama dengan nilai kontrak (SPK).",
        "7. Simpan sebagai .xlsx lalu upload di Portal Vendor.",
    ]})
    if HAS_OPENPYXL:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="S-Curve")
                guide.to_excel(writer, index=False, sheet_name="Petunjuk")
                ws = writer.sheets["S-Curve"]
                for col, w in zip("ABCDE", (8, 42, 18, 16, 16)):
                    ws.column_dimensions[col].width = w
                writer.sheets["Petunjuk"].column_dimensions["A"].width = 90
            return (buf.getvalue(), "template_scurve.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass
    return df.to_csv(index=False).encode("utf-8-sig"), "template_scurve.csv", "text/csv"


def parse_scurve_file(file, max_week):
    """Baca & validasi file S-Curve vendor. Return (rows, errors)."""
    df = _read_upload(file)
    colmap = {}
    for j, c in enumerate(df.columns):
        key = _norm_text(c)
        for target, aliases in _SC_ALIASES.items():
            if key in aliases and target not in colmap:
                colmap[target] = j
                break
    missing = [t for t in SCURVE_COLUMNS if t != "No" and t not in colmap]
    if missing:
        return [], ["Kolom tidak ditemukan: " + ", ".join(missing)
                    + ". Gunakan template yang diberikan dan jangan ubah nama kolom."]

    def raw(row, name):
        j = colmap.get(name)
        return _cell(row.iloc[j]) if j is not None else ""

    def as_int(s):
        n = _to_number(s)
        if n is None or abs(n - round(n)) > 1e-9:
            return None
        return int(round(n))

    rows, errors = [], []
    for idx, (_, row) in enumerate(df.iterrows()):
        line = idx + 2  # baris di Excel (baris 1 = header)
        item = raw(row, "Uraian Pekerjaan")
        h_raw, m_raw, s_raw = raw(row, "Harga (juta Rp)"), raw(row, "Mulai (Minggu)"), raw(row, "Selesai (Minggu)")
        if not (item or h_raw or m_raw or s_raw):
            continue
        if item.lower() in ("total", "jumlah", "grand total"):
            continue
        if not item:
            errors.append(f"Baris {line}: uraian pekerjaan kosong.")
            continue
        harga_v = _to_number(h_raw)
        if harga_v is None or harga_v <= 0:
            errors.append(f"Baris {line} ({item}): harga harus berupa angka lebih dari 0.")
            continue
        mulai, selesai = as_int(m_raw), as_int(s_raw)
        if mulai is None or selesai is None:
            errors.append(f"Baris {line} ({item}): Mulai/Selesai harus bilangan bulat (nomor minggu).")
            continue
        if mulai < 1 or selesai > max_week:
            errors.append(f"Baris {line} ({item}): minggu harus antara 1 dan {max_week}.")
            continue
        if selesai < mulai:
            errors.append(f"Baris {line} ({item}): Selesai lebih kecil dari Mulai.")
            continue
        rows.append({"No": raw(row, "No"), "Uraian Pekerjaan": item,
                     "Harga (juta Rp)": float(harga_v),
                     "Mulai (Minggu)": mulai, "Selesai (Minggu)": selesai})
    if not rows and not errors:
        errors.append("Tidak ada baris data di file.")
    return rows, errors


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


def login_page():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
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
    st.caption(f"Diperbarui otomatis · {datetime.now():%H:%M:%S}")


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
            st.info(f"Anda terpilih sebagai kontraktor project ini. Wajib upload S-Curve memakai "
                    f"template dari KPDP. Durasi project {pub.get('deadline')} hari "
                    f"(maksimal minggu ke-{max_week}).")

            tpl_bytes, tpl_name, tpl_mime = scurve_template_bytes(max_week)
            st.download_button("Unduh Template S-Curve", data=tpl_bytes, file_name=tpl_name,
                               mime=tpl_mime, key=f"tpl_sc_{pid_sel}")

            sc_cur = fetch_scurve(pid_sel, email)
            if sc_cur:
                st.success(f"S-Curve terakhir diupload: {fmt_ts(sc_cur.get('uploaded_at'))} · "
                           f"file {sc_cur.get('filename') or '-'} · "
                           f"total {format_rupiah(sc_cur.get('total_harga') or 0)}")
                with st.expander("Lihat S-Curve yang sudah diupload"):
                    st.dataframe(pd.DataFrame(sc_cur.get("data") or []),
                                 use_container_width=True, hide_index=True)
            else:
                st.warning("Belum ada S-Curve yang diupload.")

            sc_types = ["xlsx", "xls", "csv"] if HAS_OPENPYXL else ["csv"]
            sc_file = st.file_uploader(
                "Upload S-Curve (" + " / ".join("." + t for t in sc_types) + ")",
                type=sc_types, key=f"sc_upload_{pid_sel}",
            )
            if sc_file is not None:
                try:
                    sc_rows, sc_errors = parse_scurve_file(sc_file, max_week)
                except Exception as e:
                    sc_rows, sc_errors = [], [f"File tidak bisa dibaca: {e}"]

                if sc_errors:
                    st.error("Format file belum sesuai template:\n\n"
                             + "\n".join(f"- {x}" for x in sc_errors[:20]))
                else:
                    sc_total = sum(r["Harga (juta Rp)"] for r in sc_rows)
                    st.success(f"Terbaca {len(sc_rows)} item pekerjaan, total {format_rupiah(sc_total)}. Preview:")
                    st.dataframe(pd.DataFrame(sc_rows, columns=SCURVE_COLUMNS),
                                 use_container_width=True, hide_index=True)
                    if st.button("Kirim S-Curve", type="primary", key=f"sc_send_{pid_sel}"):
                        try:
                            db().table("scurve").upsert({
                                "project_id": pid_sel, "vendor_email": email,
                                "vendor_name": vendor_name,
                                "filename": getattr(sc_file, "name", ""),
                                "data": sc_rows, "total_harga": float(sc_total),
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
    c2.metric("Diperbarui", datetime.now().strftime("%H:%M:%S"))
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
    st.caption(f"Diurutkan otomatis dari harga terendah · diperbarui {datetime.now():%H:%M:%S}")


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
