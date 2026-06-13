from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence


def norm_col(value: str) -> str:
    return "".join(str(value).strip().lower().split())


@lru_cache(maxsize=256)
def detect_delimiter(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    sample = "\n".join(text.splitlines()[:6])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def find_header_row(path: Path, required_cols: Sequence[str], delimiter: str, max_scan_rows: int = 40) -> int:
    required = {norm_col(c) for c in required_cols}
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i >= max_scan_rows:
                break
            seen = {norm_col(c) for c in row}
            if required.issubset(seen):
                return i
    raise ValueError("CSV 헤더를 찾지 못했습니다: 파일 경로")


@dataclass(slots=True)
class CsvTable:
    path: Path
    required_cols: Sequence[str]
    max_scan_rows: int = 40
    delimiter: str = field(init=False)
    header_row: int = field(init=False)
    header: list[str] = field(init=False)
    header_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.delimiter = detect_delimiter(str(self.path.resolve()))
        self.header_row = find_header_row(self.path, self.required_cols, self.delimiter, self.max_scan_rows)
        with self.path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            for _ in range(self.header_row):
                next(reader, None)
            header = next(reader)
        self.header = [h.strip() for h in header]
        self.header_index = {norm_col(name): i for i, name in enumerate(self.header)}

    def iter_rows(self) -> Iterator[list[str]]:
        with self.path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            for _ in range(self.header_row + 1):
                next(reader, None)
            for row in reader:
                if row and any(str(c).strip() for c in row):
                    yield row

    def get(self, row: Sequence[str], column: str, default: str = "") -> str:
        idx = self.header_index.get(norm_col(column))
        if idx is None or idx >= len(row):
            return default
        return row[idx]
