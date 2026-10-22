# =========================================================
# HELPER BOQ — mengikuti format Excel RAB (NO | DESCRIPTION | SPESIFICATION |
# VOLUME | UNIT | UNIT PRICE | TOTAL PRICE) dengan hierarki Bab > Sub Bab > Item > Rincian
#
# Ganti blok lama di app.py mulai dari "# HELPER IMPORT EXCEL BOQ" sampai
# sebelum fungsi parse_vendor_excel / VENDOR_ALIASES TIDAK perlu diubah.
# Fungsi yang diganti: BOQ_COLUMNS, BOQ_ALIASES, _clean_volume, _norm_complexity,
# parse_boq_excel, boq_template_bytes, default_items (+ fungsi baru di bawah).
# =========================================================
import io
import re
import pandas as pd

try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

BOQ_LEVELS = ["Bab", "Sub Bab", "Item", "Rincian", "Catatan"]
BOQ_COLUMNS = ["No", "Uraian Pekerjaan", "Spesifikasi", "Volume", "Satuan",
               "Harga Satuan (Rp)", "Level"]
BOQ_TOTAL_COL = "Total Harga (Rp)"

BOQ_ALIASES = {
    "no": "No", "nomor": "No",
    "uraian pekerjaan": "Uraian Pekerjaan", "item pekerjaan": "Uraian Pekerjaan",
    "description": "Uraian Pekerjaan", "uraian": "Uraian Pekerjaan",
    "item": "Uraian Pekerjaan", "pekerjaan": "Uraian Pekerjaan",
    "deskripsi": "Uraian Pekerjaan",
    "spesifikasi": "Spesifikasi", "spesification": "Spesifikasi",
    "specification": "Spesifikasi", "spek": "Spesifikasi",
    "volume": "Volume", "vol": "Volume", "qty": "Volume", "kuantitas": "Volume",
    "satuan": "Satuan", "unit": "Satuan", "sat": "Satuan", "uom": "Satuan",
    "harga satuan (rp)": "Harga Satuan (Rp)", "harga satuan": "Harga Satuan (Rp)",
    "unit price": "Harga Satuan (Rp)", "unit price (rp.)": "Harga Satuan (Rp)",
    "level": "Level", "tipe": "Level",
}


def _txt(v):
    """Teks bersih satu baris (newline & spasi ganda dirapikan)."""
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        v = int(v)
    return " ".join(str(v).split())


def _num(v):
    """Angka -> float, selain itu None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        s = str(v).strip().replace(",", "")
        if not s or s.lower() in ("nan", "none"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    if f != f:  # NaN
        return None
    return f


def _norm_level(v):
    s = _txt(v).lower()
    if s == "bab":
        return "Bab"
    if s in ("sub bab", "subbab", "sub-bab", "sub"):
        return "Sub Bab"
    if s in ("rincian", "detail", "-"):
        return "Rincian"
    if s in ("catatan", "note", "keterangan"):
        return "Catatan"
    return "Item"


def _find_rab_header(ws):
    """Cari baris header format RAB. Return dict kolom atau None."""
    for r in range(1, min(ws.max_row, 40) + 1):
        if _txt(ws.cell(r, 1).value).upper() != "NO":
            continue
        cols = {}
        for c in range(1, min(ws.max_column, 20) + 1):
            h = _txt(ws.cell(r, c).value).upper()
            if h in ("DESCRIPTION", "URAIAN PEKERJAAN", "URAIAN"):
                cols["desc"] = c
            elif h in ("SPESIFICATION", "SPECIFICATION", "SPESIFIKASI"):
                cols["spec"] = c
            elif h == "VOLUME":
                cols["vol"] = c
            elif h in ("UNIT", "SATUAN"):
                cols["unit"] = c
            elif h in ("UNIT PRICE", "HARGA SATUAN"):
                cols["price"] = c
        if {"desc", "spec", "vol", "unit"} <= cols.keys():
            cols["row"] = r
            cols["no"] = 1
            cols.setdefault("price", cols["unit"] + 1)
            return cols
    return None


def _parse_rab_sheet(ws, h):
    items = []
    desc_cols = range(h["desc"], h["spec"])
    for r in range(h["row"] + 1, ws.max_row + 1):
        a_raw = ws.cell(r, h["no"]).value
        a = _txt(a_raw)
        texts = [t for t in (_txt(ws.cell(r, c).value) for c in desc_cols) if t]
        spec = _txt(ws.cell(r, h["spec"]).value)
        vol = _num(ws.cell(r, h["vol"]).value)
        unit = _txt(ws.cell(r, h["unit"]).value)
        price_raw = ws.cell(r, h["price"]).value

        if not (a or texts or spec or vol is not None):
            continue
        if a.lower().startswith(("sub total", "total", "pembulatan")):
            continue  # subtotal dihitung otomatis

        price = _num(price_raw) or None  # 0 / kosong = belum diisi
        price_note = _txt(price_raw) if (price is None and isinstance(price_raw, str)) else ""
        if price_note:  # mis. "By Alindo", "Exsisting" -> simpan di spesifikasi
            spec = (spec + " " if spec else "") + f"[{price_note}]"

        is_item_no = (isinstance(a_raw, (int, float)) and not isinstance(a_raw, bool)) or a.isdigit()
        if re.match(r"^BAB\b", a, re.I):
            level, no, uraian = "Bab", a, (texts[0] if texts else "")
        elif is_item_no:
            level, no, uraian = "Item", str(int(float(a))), (texts[0] if texts else "(tanpa uraian)")
        elif a:
            level, no, uraian = "Sub Bab", a, (texts[0] if texts else "")
        elif texts and texts[0] == "-":
            level, no, uraian = "Rincian", "-", " ".join(texts[1:])
        elif vol is not None:
            level, no, uraian = "Item", "", " ".join(texts)
        else:
            level, no, uraian = "Catatan", "", " ".join(texts)

        items.append({
            "No": no, "Uraian Pekerjaan": uraian, "Spesifikasi": spec,
            "Volume": round(vol, 4) if vol is not None else None,
            "Satuan": unit,
            "Harga Satuan (Rp)": price,
            "Level": level,
        })
    return items


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


def parse_boq_excel(file):
    """Baca BOQ dari:
    1) Excel format RAB (multi-sheet; header NO | DESCRIPTION | SPESIFICATION | VOLUME | UNIT | UNIT PRICE)
    2) Tabel datar Excel/CSV (kolom: No, Uraian Pekerjaan, Spesifikasi, Volume, Satuan, Harga Satuan, Level)
    Return list of dict (kunci = BOQ_COLUMNS). Raise ValueError jika format salah."""
    name = getattr(file, "name", "").lower()

    # ---- 1) format RAB (xlsx) ----
    if HAS_OPENPYXL and name.endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        try:
            wb = load_workbook(file, data_only=True)
        except Exception:
            wb = None
        if wb is not None:
            items = []
            for ws in wb.worksheets:
                h = _find_rab_header(ws)
                if h:
                    items.extend(_parse_rab_sheet(ws, h))
            if items:
                return items
        try:
            file.seek(0)
        except Exception:
            pass

    # ---- 2) tabel datar ----
    df = _read_upload(file)
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in BOQ_ALIASES:
            rename[c] = BOQ_ALIASES[key]
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]

    if "Uraian Pekerjaan" not in df.columns:
        raise ValueError(
            "Kolom 'DESCRIPTION' / 'Uraian Pekerjaan' tidak ditemukan. "
            "Gunakan template yang disediakan."
        )
    had_no = "No" in df.columns
    had_level = "Level" in df.columns
    for col in BOQ_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[BOQ_COLUMNS].copy()
    df = df[df["Uraian Pekerjaan"].notna() & (df["Uraian Pekerjaan"].astype(str).str.strip() != "")]
    if df.empty:
        raise ValueError("Tidak ada baris data yang valid di file.")

    out, counter = [], 0
    for _, r in df.iterrows():
        level = _norm_level(r["Level"]) if had_level else "Item"
        no = _txt(r["No"]) if had_no else ""
        if not had_no and level == "Item":
            counter += 1
            no = str(counter)
        vol = _num(r["Volume"])
        out.append({
            "No": no,
            "Uraian Pekerjaan": _txt(r["Uraian Pekerjaan"]),
            "Spesifikasi": _txt(r["Spesifikasi"]),
            "Volume": round(vol, 4) if vol is not None else None,
            "Satuan": _txt(r["Satuan"]),
            "Harga Satuan (Rp)": _num(r["Harga Satuan (Rp)"]),
            "Level": level,
        })
    return out


# ---------------------------------------------------------
# Hitung total & rekap
# ---------------------------------------------------------
def boq_records(df):
    """DataFrame hasil st.data_editor -> list of dict bersih (tanpa NaN, tanpa kolom Total)."""
    recs = []
    for r in df.to_dict("records"):
        rec = {c: r.get(c) for c in BOQ_COLUMNS}
        for k in ("Volume", "Harga Satuan (Rp)"):
            rec[k] = _num(rec[k])
        for k in ("No", "Uraian Pekerjaan", "Spesifikasi", "Satuan"):
            v = rec[k]
            rec[k] = "" if (v is None or v != v) else str(v)
        rec["Level"] = _norm_level(rec["Level"])
        recs.append(rec)
    return recs


def boq_with_total(items):
    """DataFrame BOQ + kolom Total Harga (Volume x Harga Satuan) untuk baris Item/Rincian."""
    df = pd.DataFrame(items, columns=BOQ_COLUMNS)
    for c in ("Volume", "Harga Satuan (Rp)"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    lvl = df["Level"].apply(_norm_level)
    priced = lvl.isin(["Item", "Rincian"])
    df[BOQ_TOTAL_COL] = (df["Volume"] * df["Harga Satuan (Rp)"]).where(priced)
    return df


def boq_line_count(items):
    """Jumlah baris pekerjaan yang punya volume (bukan judul/catatan)."""
    df = boq_with_total(items)
    return int(df["Volume"].notna().sum()) if len(df) else 0


def boq_rekap(items):
    """Rekap per Bab dan Sub Bab tingkat-1 (II.1, III.2, ...), seperti sheet 'GS'."""
    df = boq_with_total(items)
    rows, cur_bab, cur_sub = {}, "-", None
    order = []
    for _, r in df.iterrows():
        lvl = _norm_level(r["Level"])
        no, ur = _txt(r["No"]), _txt(r["Uraian Pekerjaan"])
        if lvl == "Bab":
            cur_bab, cur_sub = f"{no} {ur}".strip(), None
            order.append((cur_bab, None))
            rows.setdefault((cur_bab, None), 0.0)
            continue
        if lvl == "Sub Bab" and re.match(r"^[IVX]+\.\d+$", no):
            cur_sub = f"{no} {ur}".strip()
            order.append((cur_bab, cur_sub))
            rows.setdefault((cur_bab, cur_sub), 0.0)
            continue
        tot = r[BOQ_TOTAL_COL]
        if tot == tot and tot is not None:  # bukan NaN
            rows.setdefault((cur_bab, cur_sub), 0.0)
            rows[(cur_bab, cur_sub)] += float(tot)
    out, seen = [], set()
    babs = list(dict.fromkeys(b for b, _ in order)) or list(dict.fromkeys(b for b, _ in rows))
    for b in babs:
        subs = [s for (bb, s) in rows if bb == b and s is not None]
        bab_total = sum(v for (bb, _), v in rows.items() if bb == b)
        out.append({"Bagian": b, "Total (Rp)": bab_total})
        for s in subs:
            out.append({"Bagian": "      " + s, "Total (Rp)": rows[(b, s)]})
    return pd.DataFrame(out, columns=["Bagian", "Total (Rp)"])


# ---------------------------------------------------------
# Export / template Excel (layout sama dengan file RAB)
# ---------------------------------------------------------
def boq_excel_bytes(items, meta=None):
    """Tulis BOQ ke xlsx dengan layout RAB: NO | DESCRIPTION | SPESIFICATION | VOLUME |
    UNIT | UNIT PRICE | TOTAL PRICE (rumus =H*J). Return bytes."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

    meta = meta or {}
    wb = Workbook()
    ws = wb.active
    ws.title = "BOQ"
    f_norm = Font(name="Times New Roman", size=11)
    f_bold = Font(name="Times New Roman", size=11, bold=True)
    f_ital = Font(name="Times New Roman", size=11, italic=True)
    thin = Side(style="thin", color="999999")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill_hdr = PatternFill("solid", fgColor="D9E1F2")
    fill_bab = PatternFill("solid", fgColor="F2F2F2")
    acc2 = '_(* #,##0.00_);_(* \\(#,##0.00\\);_(* "-"??_);_(@_)'
    acc0 = '_(* #,##0_);_(* \\(#,##0\\);_(* "-"_);_(@_)'

    for col, w in {"A": 11.6, "B": 2.5, "C": 3, "D": 3, "E": 3, "F": 100,
                   "G": 36.7, "H": 12.7, "I": 8, "J": 18.7, "K": 20.7}.items():
        ws.column_dimensions[col].width = w

    ws.merge_cells("A1:K1")
    ws["A1"] = "RENCANA ANGGARAN BIAYA"
    ws["A1"].font = Font(name="Times New Roman", size=14, bold=True)
    ws["A1"].alignment = Alignment(horizontal="center")
    for i, (k, v) in enumerate([
        ("Owner", meta.get("client", "")),
        ("Project Name", meta.get("name", "")),
        ("Scope", meta.get("scope", "")),
        ("Location", meta.get("location", "")),
    ]):
        r = 3 + i
        ws.cell(r, 1, k).font = f_norm
        ws.cell(r, 6, f": {v}" if v and v != "-" else ":").font = f_norm

    heads = {1: "NO", 2: "DESCRIPTION", 7: "SPESIFICATION", 8: "VOLUME",
             9: "UNIT", 10: "UNIT PRICE", 11: "TOTAL PRICE"}
    for c, t in heads.items():
        ws.cell(8, c, t)
    ws["J9"], ws["K9"] = "(Rp.)", "(Rp.)"
    for c in (1, 7, 8, 9):
        ws.merge_cells(start_row=8, start_column=c, end_row=9, end_column=c)
    ws.merge_cells("B8:F9")
    for r in (8, 9):
        for c in range(1, 12):
            cell = ws.cell(r, c)
            cell.font = f_bold
            cell.fill = fill_hdr
            cell.border = box
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    r = 10
    first = r
    for it in items:
        lvl = _norm_level(it.get("Level"))
        no, ur = _txt(it.get("No")), _txt(it.get("Uraian Pekerjaan"))
        spec, unit = _txt(it.get("Spesifikasi")), _txt(it.get("Satuan"))
        vol, price = _num(it.get("Volume")), _num(it.get("Harga Satuan (Rp)"))

        if lvl in ("Bab", "Sub Bab"):
            ws.cell(r, 1, no).font = f_bold
            ws.cell(r, 2, ur).font = f_bold
            if lvl == "Bab":
                for c in range(1, 12):
                    ws.cell(r, c).fill = fill_bab
            if vol is not None:
                ws.cell(r, 8, vol).font = f_norm
                ws.cell(r, 9, unit).font = f_norm
        elif lvl == "Catatan":
            ws.cell(r, 3, ur).font = f_ital
        else:
            if lvl == "Item":
                ws.cell(r, 1, int(no) if no.isdigit() else no).font = f_norm
                ws.cell(r, 3, ur).font = f_bold if vol is None else f_norm
            else:
                ws.cell(r, 3, "-").font = f_norm
                ws.cell(r, 4, ur).font = f_norm
            ws.cell(r, 7, spec).font = f_norm
            ws.cell(r, 7).alignment = Alignment(wrap_text=True, vertical="top")
            if vol is not None:
                ws.cell(r, 8, vol).number_format = acc2
                ws.cell(r, 9, unit)
                if price is not None:
                    ws.cell(r, 10, price)
                ws.cell(r, 10).number_format = acc2
                ws.cell(r, 11, f"=H{r}*J{r}").number_format = acc0
                for c in (8, 9, 10, 11):
                    ws.cell(r, c).font = f_norm
        r += 1

    ws.cell(r + 1, 1, "Sub Total BOQ =").font = f_bold
    ws.cell(r + 1, 11, f"=SUM(K{first}:K{r})").number_format = acc0
    ws.cell(r + 1, 11).font = f_bold
    ws.freeze_panes = "A10"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def boq_template_bytes():
    """Return (data, filename, mime). Fallback ke CSV jika openpyxl tidak ada."""
    if HAS_OPENPYXL:
        try:
            return (
                boq_excel_bytes(default_items().to_dict("records")),
                "template_boq.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            pass
    return (
        default_items().to_csv(index=False).encode("utf-8-sig"),
        "template_boq.csv",
        "text/csv",
    )


def default_items():
    """BOQ contoh = BAB I Pekerjaan Pendahuluan dari file RAB Alindo Cilegon."""
    rows = [
        ("BAB. I", "PEKERJAAN PENDAHULUAN", "", 8, "Bulan", "Bab"),
        ("1", "Mobilisasi dan Demobilisasi Pekerja dan Alat Kerja", "", 1, "ls", "Item"),
        ("2", "Site Management Proyek", "", 1, "ls", "Item"),
        ("3", "Pengadaan Kantor Site/Direksikeet uk. 4.00 x 4.00 m", "", 16, "m2", "Item"),
        ("4", "Pembuatan Gudang dan Bedeng Pekerja uk. 4.00 x 6.00 m", "", 24, "m2", "Item"),
        ("5", "Papan Nama Proyek", "", 1, "ls", "Item"),
        ("6", "Pembuatan Pagar Sementara setinggi 1.80 m", "", 104, "m'", "Item"),
        ("7", "Pembuatan Laporan Kemajuan Pekerjaan, Shop Drawing, Asbuild Drawing", "", 1, "ls", "Item"),
        ("8", "Pengadaan Sample dan Pengujian Material", "", 1, "ls", "Item"),
        ("9", "Pengadaan Air Kerja", "[By Alindo]", 1, "ls", "Item"),
        ("10", "Pengadaan Listrik Kerja", "[By Alindo]", 1, "ls", "Item"),
        ("11", "Penyiapan RKK K3 dan APD", "", 1, "ls", "Item"),
        ("12", "Asuransi Tenaga Pekerja", "", 1, "ls", "Item"),
    ]
    return pd.DataFrame([
        {"No": n, "Uraian Pekerjaan": u, "Spesifikasi": s,
         "Volume": float(v) if v != "" else None, "Satuan": sat,
         "Harga Satuan (Rp)": None, "Level": lv}
        for n, u, s, v, sat, lv in rows
    ], columns=BOQ_COLUMNS)
