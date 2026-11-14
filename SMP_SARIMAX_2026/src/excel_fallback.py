"""Minimal standard-library XLSX reader for environments without openpyxl.

Reads cached values from the first worksheet. It does not recalculate formulas and
does not support legacy .xls files.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import re

import pandas as pd

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    value = 0
    for char in letters.group(0):
        value = value * 26 + ord(char) - 64
    return value - 1


def read_first_sheet(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        files = set(zf.namelist())
        strings = []
        if "xl/sharedStrings.xml" in files:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in item.findall(".//m:t", NS)) for item in root.findall("m:si", NS)]
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in files:
            raise ValueError("Workbook has no first worksheet XML")
        root = ET.fromstring(zf.read(sheet_path))
        rows, width = [], 0
        for row in root.findall(".//m:sheetData/m:row", NS):
            cells = {}
            for cell in row.findall("m:c", NS):
                pos = _col_number(cell.attrib["r"]); width = max(width, pos + 1)
                kind = cell.attrib.get("t", "n")
                node = cell.find("m:v", NS)
                value = node.text if node is not None else None
                if kind == "s" and value is not None: value = strings[int(value)]
                elif kind == "inlineStr": value = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                elif kind == "b" and value is not None: value = value == "1"
                elif kind in {"n", "str"} and value is not None:
                    try: value = float(value)
                    except ValueError: pass
                cells[pos] = value
            if cells: rows.append([cells.get(i) for i in range(width)])
    if not rows: return pd.DataFrame()
    max_width = max(len(row) for row in rows)
    rows = [row + [None] * (max_width - len(row)) for row in rows]
    headers = [str(v).strip() if v is not None else f"unnamed_{i}" for i, v in enumerate(rows[0])]
    frame = pd.DataFrame(rows[1:], columns=headers)
    date_cols = {"기간", "날짜", "date", "일자", "거래일자", "거래일"}
    for col in frame.columns:
        if str(col).strip().lower() in date_cols:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            serial_mask = numeric.between(30000, 60000)
            if serial_mask.mean() > .8:
                frame[col] = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return frame
