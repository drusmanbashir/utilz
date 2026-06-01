from __future__ import annotations

import math
import os
import select
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PageGeometry:
    rows: int
    cols: int
    term_columns: int
    term_lines: int


@dataclass(frozen=True)
class ArrayPage:
    text: str
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    total_rows: int
    total_cols: int
    row_page: int
    col_page: int
    n_row_pages: int


def geometry_from_terminal_size(
    term_columns: int,
    term_lines: int,
    *,
    reserve_lines: int = 8,
    min_rows: int = 3,
    min_cols: int = 20,
) -> PageGeometry:
    if term_columns < 1 or term_lines < 1:
        raise ValueError("term_columns and term_lines must be >= 1")
    rows = max(min_rows, term_lines - reserve_lines)
    cols = max(min_cols, term_columns)
    return PageGeometry(
        rows=rows,
        cols=cols,
        term_columns=term_columns,
        term_lines=term_lines,
    )


def terminal_page_geometry(
    *,
    reserve_lines: int = 8,
    fallback_columns: int = 120,
    fallback_lines: int = 40,
    min_rows: int = 3,
    min_cols: int = 20,
) -> PageGeometry:
    term = shutil.get_terminal_size(fallback=(fallback_columns, fallback_lines))
    return geometry_from_terminal_size(
        term.columns,
        term.lines,
        reserve_lines=reserve_lines,
        min_rows=min_rows,
        min_cols=min_cols,
    )


def geometry_from_pixels(
    width_px: int,
    height_px: int,
    *,
    char_width_px: int,
    char_height_px: int,
    reserve_lines: int = 8,
    min_rows: int = 3,
    min_cols: int = 20,
) -> PageGeometry:
    if min(width_px, height_px, char_width_px, char_height_px) <= 0:
        raise ValueError("pixel geometry values must be > 0")
    term_columns = max(1, width_px // char_width_px)
    term_lines = max(1, height_px // char_height_px)
    return geometry_from_terminal_size(
        term_columns,
        term_lines,
        reserve_lines=reserve_lines,
        min_rows=min_rows,
        min_cols=min_cols,
    )


def _coerce_2d(data: Any) -> tuple[np.ndarray, list[str] | None, list[Any], tuple[int, ...]]:
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1), None, ["value"], arr.shape
    if arr.ndim == 1:
        return arr.reshape(-1, 1), None, ["value"], arr.shape
    if arr.ndim == 2:
        return arr, None, list(range(arr.shape[1])), arr.shape

    arr2 = arr.reshape(-1, arr.shape[-1])
    row_labels = []
    for flat_idx in range(arr2.shape[0]):
        coord = np.unravel_index(flat_idx, arr.shape[:-1])
        row_labels.append(",".join(str(v) for v in coord))
    return arr2, row_labels, list(range(arr.shape[-1])), arr.shape


def _render_df(df: pd.DataFrame, precision: int) -> str:
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
        "display.float_format",
        lambda x: f"{x:.{precision}f}",
    ):
        return df.to_string()


def _fit_col_stop(
    arr2: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    col_start: int,
    max_text_width: int,
    precision: int,
    row_labels: list[str] | None,
    col_labels: list[Any],
) -> int:
    best_stop = col_start + 1
    for col_stop in range(col_start + 1, arr2.shape[1] + 1):
        index = None if row_labels is None else row_labels[row_start:row_stop]
        df = pd.DataFrame(
            arr2[row_start:row_stop, col_start:col_stop],
            index=index,
            columns=col_labels[col_start:col_stop],
        )
        if index is not None:
            df.index.name = "idx"
        rendered = _render_df(df, precision=precision)
        width = max(len(line) for line in rendered.splitlines()) if rendered else 0
        if width > max_text_width:
            break
        best_stop = col_stop
    return best_stop


def _advance_page(page: ArrayPage, row_page: int, col_page: int) -> tuple[int, int]:
    next_row = row_page + 1
    next_col = col_page
    if next_row < page.n_row_pages:
        return next_row, next_col

    next_row = 0
    next_col += 1
    page_width = max(1, page.col_stop - page.col_start)
    if next_col * page_width >= page.total_cols:
        next_col = 0
    return next_row, next_col


def page(
    data: Any,
    *,
    row_page: int = 0,
    col_page: int = 0,
    precision: int = 4,
    geometry: PageGeometry | None = None,
    reserve_lines: int = 8,
) -> ArrayPage:
    arr2, row_labels, col_labels, _ = _coerce_2d(data)

    if geometry is None:
        geometry = terminal_page_geometry(reserve_lines=reserve_lines)
    if row_page < 0 or col_page < 0:
        raise ValueError("row_page and col_page must be >= 0")

    row_start = row_page * geometry.rows
    row_stop = min(row_start + geometry.rows, arr2.shape[0])
    if row_start >= arr2.shape[0]:
        raise IndexError(f"row_page {row_page} out of range for {arr2.shape[0]} rows")

    col_start = 0
    col_stop = 0
    for idx in range(col_page + 1):
        if col_start >= arr2.shape[1]:
            raise IndexError(f"col_page {col_page} out of range for {arr2.shape[1]} cols")
        col_stop = _fit_col_stop(
            arr2,
            row_start=row_start,
            row_stop=row_stop,
            col_start=col_start,
            max_text_width=geometry.term_columns,
            precision=precision,
            row_labels=row_labels,
            col_labels=col_labels,
        )
        if col_stop <= col_start:
            raise RuntimeError("Could not fit even one column into requested width")
        if idx == col_page:
            break
        col_start = col_stop

    index = None if row_labels is None else row_labels[row_start:row_stop]
    df = pd.DataFrame(
        arr2[row_start:row_stop, col_start:col_stop],
        index=index,
        columns=col_labels[col_start:col_stop],
    )
    if index is not None:
        df.index.name = "idx"

    text = _render_df(df, precision=precision)

    return ArrayPage(
        text=text,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        total_rows=arr2.shape[0],
        total_cols=arr2.shape[1],
        row_page=row_page,
        col_page=col_page,
        n_row_pages=math.ceil(arr2.shape[0] / geometry.rows),
    )


def view(
    data: Any,
    *,
    row_page: int = 0,
    col_page: int = 0,
    precision: int = 4,
    geometry: PageGeometry | None = None,
    reserve_lines: int = 8,
    clear_screen: bool = False,
    return_text: bool = False,
):
    rendered_page = page(
        data,
        row_page=row_page,
        col_page=col_page,
        precision=precision,
        geometry=geometry,
        reserve_lines=reserve_lines,
    )
    if clear_screen:
        os.system("clear")
    print(rendered_page.text)
    if return_text:
        return rendered_page.text
    return rendered_page


def browse(
    data: Any,
    *,
    precision: int = 4,
    geometry: PageGeometry | None = None,
    reserve_lines: int = 8,
    autoplay_seconds: float | None = None,
):
    row_page = 0
    col_page = 0
    while True:
        rendered_page = view(
            data,
            row_page=row_page,
            col_page=col_page,
            precision=precision,
            geometry=geometry,
            reserve_lines=reserve_lines,
            clear_screen=True,
        )
        if autoplay_seconds is None:
            cmd = input(
                "[enter]/j next rows, k prev rows, l next cols, h prev cols, q quit: "
            ).strip().lower()
        else:
            print(
                f"[auto {autoplay_seconds}s | enter/j next | k prev | l next col | h prev col | q quit]: ",
                end="",
                flush=True,
            )
            ready, _, _ = select.select([sys.stdin], [], [], autoplay_seconds)
            if ready:
                cmd = sys.stdin.readline().strip().lower()
            else:
                print()
                row_page, col_page = _advance_page(rendered_page, row_page, col_page)
                continue
        if cmd in {"q", "quit"}:
            return
        if cmd in {"", "j"}:
            row_page, col_page = _advance_page(rendered_page, row_page, col_page)
        elif cmd == "k":
            row_page = max(0, row_page - 1)
        elif cmd == "l":
            col_page += 1
        elif cmd == "h":
            col_page = max(0, col_page - 1)
