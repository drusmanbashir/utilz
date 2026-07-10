from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageGrab

STREAM_BEFORE = "<<<STREAM_BEFORE>>>"
STREAM_AFTER = "<<<STREAM_AFTER>>>"
BATCH_BEFORE = "<<<BATCH_BEFORE>>>"
BATCH_AFTER = "<<<BATCH_AFTER>>>"
SLICE_BEFORE = "<<<SLICE_BEFORE>>>"
SLICE_AFTER = "<<<SLICE_AFTER>>>"
PAGE_BEFORE = "<<<PAGE_BEFORE>>>"
PAGE_AFTER = "<<<PAGE_AFTER>>>"

BARCO_MONITOR_W = 1536
BARCO_MONITOR_H = 2048
BARCO_DUAL_W = BARCO_MONITOR_W * 2
BARCO_DUAL_H = BARCO_MONITOR_H
BARCO_REFRESH_HZ = 50.0

DEFAULT_LINE_PX = 16
DEFAULT_CHARS_PER_LINE = 220

_SLICE_BEFORE_RE = re.compile(
    rf"{re.escape(SLICE_BEFORE)}\s+z=(\d+)(?:\s+shape=(\d+),(\d+))?"
)
_BATCH_BEFORE_RE = re.compile(rf"{re.escape(BATCH_BEFORE)}\s+b=(\d+)")
_PAGE_BEFORE_RE = re.compile(rf"{re.escape(PAGE_BEFORE)}\s+p=(\d+)")
_STREAM_BEFORE_RE = re.compile(
    rf"{re.escape(STREAM_BEFORE)}\s+"
    r"(?:n=(\d+)\s+)?"
    r"shape=([\d,]+)\s+dtype=(\w+)"
    r"(?:\s+canvas=(\d+),(\d+))?"
    r"(?:\s+refresh_hz=([\d.]+))?"
)


def barco_frame_period(refresh_hz=BARCO_REFRESH_HZ):
    return 1.0 / refresh_hz


def barco_dual_canvas():
    """Return dual-Barco stretched canvas size (width, height)."""
    return BARCO_DUAL_W, BARCO_DUAL_H


def get_display_canvas():
    """Return full X11 desktop canvas size (width, height) in pixels."""
    out = subprocess.check_output(["xrandr", "--query"], text=True)
    match = re.search(r"current (\d+) x (\d+)", out)
    if match is None:
        raise RuntimeError("Could not parse xrandr canvas dimensions")
    return int(match.group(1)), int(match.group(2))


def get_barco_canvas():
    """
    Return the dual-Barco stretched region as (x, y, width, height).

    Each monitor is 1536x2048 at 50 Hz; together they span 3072x2048.
    Screen offset (x, y) comes from xrandr; width/height use BARCO_* constants.
    """
    out = subprocess.check_output(["xrandr", "--query"], text=True)
    boxes = []
    for line in out.splitlines():
        if " connected " not in line:
            continue
        match = re.search(r"(\d+)x(\d+)\+(\d+)\+(\d+)", line)
        if match is None:
            continue
        w, h, x, y = (int(match.group(i)) for i in range(1, 5))
        if w == BARCO_MONITOR_W and h == BARCO_MONITOR_H:
            boxes.append((x, y, w, h))
    if len(boxes) == 0:
        return 0, 0, BARCO_DUAL_W, BARCO_DUAL_H
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    return x0, y0, BARCO_DUAL_W, BARCO_DUAL_H


def _rows_per_page(canvas_h, line_px):
    usable = canvas_h - 4 * DEFAULT_LINE_PX
    return max(1, usable // line_px)


def _iter_batch_volumes(tensor):
    t = tensor.detach().cpu()
    if t.dtype == torch.float16:
        t = t.float()
    if t.ndim == 5:
        if t.shape[1] != 1:
            raise ValueError(f"5D tensor must have C==1, got shape {tuple(t.shape)}")
        t = t[:, 0]
    if t.ndim == 4:
        return [t[i] for i in range(t.shape[0])]
    if t.ndim == 3:
        return [t]
    raise ValueError(f"Expected 3D/4D/5D tensor, got shape {tuple(t.shape)}")


def _format_row(row):
    return " ".join(f"{v:.6g}" for v in row)


def _paginate_lines(lines, rows_per_page):
    pages = []
    for start in range(0, len(lines), rows_per_page):
        pages.append(lines[start : start + rows_per_page])
    if len(pages) == 0:
        pages.append([])
    return pages


def stream_tensor(  #AI
    tensor,
    print_fn=print,
    include_values=True,
    canvas=None,
    line_px=DEFAULT_LINE_PX,
    rows_per_page=None,
):
    """
    Print a delimited pytorch stream for a single tensor or batch.

    Uses STREAM/BATCH/SLICE/PAGE BEFORE/AFTER markers. Canvas dimensions
    (dual Barco stretch by default) control page breaks so each screen fill
    fits the monitor for OCR capture.
    """
    if canvas is None:
        canvas = barco_dual_canvas()
    if rows_per_page is None:
        rows_per_page = _rows_per_page(canvas[1], line_px)

    volumes = _iter_batch_volumes(tensor)
    n_batch = len(volumes)
    shape = tuple(volumes[0].shape)
    dtype = str(volumes[0].dtype).replace("torch.", "")
    print_fn(
        f"{STREAM_BEFORE} n={n_batch} shape={','.join(str(x) for x in shape)} "
        f"dtype={dtype} canvas={canvas[0]},{canvas[1]} "
        f"refresh_hz={BARCO_REFRESH_HZ:g}"
    )
    for b, volume in enumerate(volumes):
        vol = volume.numpy()
        print_fn(f"{BATCH_BEFORE} b={b}")
        for z in range(vol.shape[0]):
            slice_2d = vol[z]
            print_fn(f"{SLICE_BEFORE} z={z} shape={slice_2d.shape[0]},{slice_2d.shape[1]}")
            if include_values:
                rows = [_format_row(row) for row in slice_2d]
                for p, page in enumerate(_paginate_lines(rows, rows_per_page)):
                    print_fn(f"{PAGE_BEFORE} p={p}")
                    for line in page:
                        print_fn(line)
                    print_fn(f"{PAGE_AFTER} p={p}")
            print_fn(f"{SLICE_AFTER} z={z}")
        print_fn(f"{BATCH_AFTER} b={b}")
    print_fn(STREAM_AFTER)


def grab_barco_frame():
    """Capture the dual-Barco screen region as an RGB numpy array."""
    x, y, w, h = get_barco_canvas()
    return np.asarray(ImageGrab.grab(bbox=(x, y, x + w, y + h)).convert("RGB"))


def save_barco_frame(out_path):
    from PIL import Image

    Image.fromarray(grab_barco_frame()).save(out_path)


def _crop_barco(image):
    from PIL import Image

    x, y, w, h = get_barco_canvas()
    return image.crop((x, y, x + w, y + h))


def _load_tensor(tensor_or_path):
    if isinstance(tensor_or_path, (str, Path)):
        return torch.load(str(tensor_or_path), map_location="cpu", weights_only=False)
    return tensor_or_path


def _slice_2d(volume, sl, slice_axis):
    if slice_axis == 0:
        return volume[sl]
    if slice_axis == 1:
        return volume[:, sl]
    if slice_axis == 2:
        return volume[:, :, sl]
    raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}")


def _n_slices(volume, slice_axis):
    return volume.shape[slice_axis]


def _volume_numpy(tensor_or_path):
    tensor = _load_tensor(tensor_or_path)
    arr = tensor.detach().cpu().numpy()
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    return arr


def _barco_capture_loop(out_dir, stop_event, refresh_hz, counter):
    frame_period = barco_frame_period(refresh_hz)
    while not stop_event.is_set():
        t0 = time.time()
        save_barco_frame(out_dir / f"{counter[0]:06d}.png")
        counter[0] += 1
        elapsed = time.time() - t0
        wait = frame_period - elapsed
        if wait > 0:
            time.sleep(wait)


def stream_volume_rows(  #AI
    tensor_or_path,
    print_fn=print,
    slice_axis=2,
    max_slices=None,
):
    """Print a 3D volume slice-by-slice, one row per line, with stream markers."""
    volume = _volume_numpy(tensor_or_path)
    n_slices = _n_slices(volume, slice_axis)
    if max_slices is not None:
        n_slices = min(n_slices, max_slices)
    shape = volume.shape
    dtype = str(volume.dtype)
    print_fn(
        f"{STREAM_BEFORE} n=1 shape={shape[0]},{shape[1]},{shape[2]} "
        f"dtype={dtype} canvas={BARCO_DUAL_W},{BARCO_DUAL_H} "
        f"refresh_hz={BARCO_REFRESH_HZ:g} slice_axis={slice_axis}"
    )
    print_fn(f"{BATCH_BEFORE} b=0")
    torch.set_printoptions(threshold=float("inf"), linewidth=220)
    for sl in range(n_slices):
        slice_2d = _slice_2d(volume, sl, slice_axis)
        print_fn(
            f"{SLICE_BEFORE} z={sl} shape={slice_2d.shape[0]},{slice_2d.shape[1]}"
        )
        for ri, row in enumerate(slice_2d):
            print_fn(f"{sl} {ri}", row)
        print_fn(f"{SLICE_AFTER} z={sl}")
        sys.stdout.flush()
    print_fn(f"{BATCH_AFTER} b=0")
    print_fn(STREAM_AFTER)
    sys.stdout.flush()


def record_barco_screen(  #AI
    out_dir,
    refresh_hz=BARCO_REFRESH_HZ,
    stop_event=None,
):
    """
    Capture the dual-Barco region at refresh_hz until stop_event is set.

    Intended to run while the user manually prints tensor rows and positions
    the terminal across the Barco stretch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stop_event is None:
        stop_event = threading.Event()
    counter = [0]
    _barco_capture_loop(out_dir, stop_event, refresh_hz, counter)
    return out_dir, counter[0]


def _needs_barco_crop(image):
    return image.size[0] > BARCO_DUAL_W


def _active_gray_from_image(image):
    gray = np.asarray(image.convert("L"))
    row_mean = gray.mean(axis=1)
    active_rows = np.where(row_mean > 5)[0]
    if active_rows.size == 0:
        return None
    y0, y1 = int(active_rows[0]), int(active_rows[-1])
    crop = gray[y0 : y1 + 1]
    col_mean = crop.mean(axis=0)
    active_cols = np.where(col_mean > 5)[0]
    if active_cols.size == 0:
        return crop
    x0, x1 = int(active_cols[0]), int(active_cols[-1])
    return crop[:, x0 : x1 + 1]


def _active_gray_from_path(path):
    image = Image.open(path)
    return _active_gray_from_image(image)


def _best_row_overlap(prev, nxt, min_k=20, max_k=1200):
    limit = min(max_k, prev.shape[0], nxt.shape[0])
    best_k, best_score = 0, -1.0
    for k in range(limit, min_k - 1, -1):
        score = 1.0 - np.mean(np.abs(prev[-k:].astype(np.int16) - nxt[:k].astype(np.int16))) / 255.0
        if score > best_score:
            best_k, best_score = k, float(score)
    if best_score < 0.55:
        return 0
    return best_k


def _is_content_frame(path):
    return Path(path).stat().st_size > 100_000


def build_scrolled_composite(image_paths, crop_barco=True):
    """Merge scrolled terminal frames, dropping repeated overlap rows."""
    grays = []
    for path in image_paths:
        if not _is_content_frame(path):
            continue
        image = Image.open(path)
        if crop_barco and _needs_barco_crop(image):
            image = _crop_barco(image)
        gray = _active_gray_from_image(image)
        if gray is not None:
            grays.append(gray)
    if len(grays) == 0:
        raise ValueError("No content frames found in captures")
    composite = grays[0]
    for gray in grays[1:]:
        overlap = _best_row_overlap(composite, gray)
        composite = np.vstack([composite, gray[overlap:]])
    return composite


def _line_bands(gray):
    row_energy = gray.std(axis=1)
    active = row_energy > 2.0
    bands = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        if not on and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(active) - 1))
    return bands


def _ocr_line_band(gray, y0, y1):
    import pytesseract
    from PIL import Image

    band = gray[y0 : y1 + 1]
    if band.shape[0] < 4:
        return ""
    scale = 2 if band.shape[1] < 2400 else 1
    if scale > 1:
        band = np.repeat(np.repeat(band, scale, axis=0), scale, axis=1)
    image = Image.fromarray(band)
    config = "--psm 7 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(image, config=config).strip()


def _ocr_composite_lines(gray):
    lines = []
    for y0, y1 in _line_bands(gray):
        text = _ocr_line_band(gray, y0, y1)
        if text:
            lines.append(text)
    return _dedupe_consecutive_lines(lines)


def _dedupe_consecutive_lines(lines):
    out = []
    for line in lines:
        if out and line == out[-1]:
            continue
        out.append(line)
    return out


def _stitch_frame_texts(frame_texts):
    """Drop repeated scroll overlap between OCR'd frames."""
    if len(frame_texts) == 0:
        return []
    lines = [ln for ln in frame_texts[0].splitlines() if ln.strip()]
    for text in frame_texts[1:]:
        nxt = [ln for ln in text.splitlines() if ln.strip()]
        best_k, best_score = 0, -1.0
        limit = min(len(lines), len(nxt), 400)
        for k in range(limit, 0, -1):
            if lines[-k:] == nxt[:k]:
                best_k, best_score = k, 1.0
                break
        if best_k == 0:
            lines.extend(nxt)
        else:
            lines.extend(nxt[best_k:])
    return _dedupe_consecutive_lines(lines)


def _ocr_image(path, crop_barco=True):
    import pytesseract
    from PIL import Image

    image = Image.open(path)
    if crop_barco and _needs_barco_crop(image):
        image = _crop_barco(image)
    gray = _active_gray_from_image(image)
    if gray is None:
        return ""
    return "\n".join(_ocr_composite_lines(gray))


def _ocr_paths(image_paths, crop_barco=True):
    frame_texts = []
    for path in image_paths:
        if not _is_content_frame(path):
            continue
        frame_texts.append(_ocr_image(path, crop_barco=crop_barco))
    if len(frame_texts) == 0:
        raise ValueError("No content frames found in captures")
    return "\n".join(_stitch_frame_texts(frame_texts))


def _is_numeric_row(line):
    if not line or line.startswith("<<<"):
        return False
    if "tensor" in line or "metatensor" in line:
        return False
    tokens = line.split()
    if len(tokens) < 2:
        return False
    for tok in tokens:
        try:
            float(tok.replace(",", ""))
        except ValueError:
            return False
    return True


def _parse_float_row(line):
    return [float(tok.replace(",", "")) for tok in line.split()]


def _parse_enumerated_rows(text, expected_cols):
    row_map = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("<<<"):
            continue
        tokens = line.split()
        if len(tokens) < 2 + expected_cols:
            continue
        try:
            z = int(tokens[0])
            ri = int(tokens[1])
            vals = [float(tok.replace(",", "")) for tok in tokens[2 : 2 + expected_cols]]
        except ValueError:
            continue
        if len(vals) == expected_cols:
            row_map[(z, ri)] = vals
    if len(row_map) == 0:
        raise ValueError("No enumerated rows parsed")
    return row_map


def _volume_from_enumerated(row_map, shape, slice_axis=2):
    if slice_axis == 2:
        rows_per_slice, cols, depth = shape[0], shape[1], shape[2]
    elif slice_axis == 0:
        depth, rows_per_slice, cols = shape[0], shape[1], shape[2]
    else:
        rows_per_slice, depth, cols = shape[0], shape[1], shape[2]
    volume = np.full(shape, np.nan, dtype=np.float32)
    for (z, ri), vals in row_map.items():
        if z < 0 or z >= depth or ri < 0 or ri >= rows_per_slice:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        if slice_axis == 2:
            volume[ri, :, z] = arr
        elif slice_axis == 0:
            volume[z, ri, :] = arr
        else:
            volume[ri, z, :] = arr
    if np.isnan(volume).any():
        missing = int(np.isnan(volume).sum())
        raise ValueError(f"Enumerated rows left {missing} missing voxels")
    return torch.from_numpy(volume)


def _parse_float_rows(text, expected_cols=None):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not _is_numeric_row(line):
            continue
        row = _parse_float_row(line)
        if expected_cols is not None and len(row) != expected_cols:
            continue
        rows.append(row)
    if len(rows) == 0:
        raise ValueError("No numeric rows found")
    if expected_cols is None:
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise ValueError(f"Inconsistent row widths: {widths}")
    return np.asarray(rows, dtype=np.float32)


def _expected_cols(shape, slice_axis):
    if slice_axis == 2:
        return shape[1]
    if slice_axis == 0:
        return shape[2]
    return shape[2]


def _reconstruct_from_row_blocks(rows, shape, slice_axis=2):
    if slice_axis == 2:
        rows_per_slice, cols, depth = shape[0], shape[1], shape[2]
    elif slice_axis == 0:
        depth, rows_per_slice, cols = shape[0], shape[1], shape[2]
    elif slice_axis == 1:
        rows_per_slice, depth, cols = shape[0], shape[1], shape[2]
    else:
        raise ValueError(f"slice_axis must be 0, 1, or 2, got {slice_axis}")
    rows = rows[: rows_per_slice * depth]
    usable = (len(rows) // rows_per_slice) * rows_per_slice
    rows = rows[:usable]
    n_slices = usable // rows_per_slice
    slices = []
    for sl in range(n_slices):
        block = np.asarray(rows[sl * rows_per_slice : (sl + 1) * rows_per_slice], dtype=np.float32)
        if block.shape != (rows_per_slice, cols):
            raise ValueError(f"Slice {sl} shape {block.shape} != {(rows_per_slice, cols)}")
        slices.append(block)
    if slice_axis == 2:
        volume = np.stack(slices, axis=2)
    elif slice_axis == 0:
        volume = np.stack(slices, axis=0)
    else:
        volume = np.stack(slices, axis=1)
    return torch.from_numpy(volume)


def reconstruct_tensor_from_text(text, expected_shape=None, slice_axis=2):  #AI
    """
    Rebuild tensor from OCR text (or raw captured terminal text).

    Parses STREAM/BATCH/SLICE/PAGE markers and float rows between
    SLICE_BEFORE and SLICE_AFTER. Without markers, uses expected_shape
    to group fixed-size row blocks per slice.
    """
    stream_match = _STREAM_BEFORE_RE.search(text)
    if stream_match is None:
        if expected_shape is None:
            raise ValueError(f"Missing {STREAM_BEFORE} header; pass expected_shape")
        cols = _expected_cols(expected_shape, slice_axis)
        try:
            row_map = _parse_enumerated_rows(text, cols)
            return _volume_from_enumerated(row_map, expected_shape, slice_axis=slice_axis)
        except ValueError:
            rows = _parse_float_rows(text, expected_cols=cols)
            return _reconstruct_from_row_blocks(rows, expected_shape, slice_axis=slice_axis)
    n_batch = int(stream_match.group(1) or 1)
    shape_tokens = [int(x) for x in stream_match.group(2).split(",")]
    if len(shape_tokens) != 3:
        raise ValueError(f"Expected Z,Y,X shape in stream header, got {shape_tokens}")

    batches = {}
    current_b = None
    current_z = None
    body_lines = []

    def _flush_slice():
        nonlocal body_lines, current_b, current_z
        if current_b is None or current_z is None:
            return
        slice_text = "\n".join(body_lines)
        batches.setdefault(current_b, {})[current_z] = _parse_float_rows(slice_text)
        body_lines = []

    for line in text.splitlines():
        batch_match = _BATCH_BEFORE_RE.search(line)
        if batch_match is not None:
            _flush_slice()
            current_b = int(batch_match.group(1))
            current_z = None
            continue
        slice_match = _SLICE_BEFORE_RE.search(line)
        if slice_match is not None:
            _flush_slice()
            current_z = int(slice_match.group(1))
            continue
        if SLICE_AFTER in line:
            _flush_slice()
            current_z = None
            continue
        if PAGE_BEFORE in line or PAGE_AFTER in line:
            continue
        if current_z is not None:
            body_lines.append(line)
    _flush_slice()

    if len(batches) == 0:
        raise ValueError("No slice blocks parsed from OCR text")
    if len(batches) != n_batch:
        raise ValueError(f"Expected {n_batch} batches, parsed {len(batches)}")

    out = []
    for b in range(n_batch):
        slice_map = batches[b]
        depth = shape_tokens[0]
        if len(slice_map) != depth:
            missing = [z for z in range(depth) if z not in slice_map]
            raise ValueError(f"Batch {b} missing slices: {missing[:8]}")
        volume = np.stack([slice_map[z] for z in range(depth)], axis=0)
        if tuple(volume.shape) != tuple(shape_tokens):
            raise ValueError(f"Batch {b} shape {volume.shape} != header {tuple(shape_tokens)}")
        out.append(volume)
    stacked = np.stack(out, axis=0) if n_batch > 1 else out[0]
    return torch.from_numpy(stacked)


def reconstruct_tensor_from_ocr_images(  #AI
    image_paths,
    crop_barco=True,
    expected_shape=None,
    slice_axis=2,
    use_composite=True,
):
    """OCR screen captures from the Barco canvas and reconstruct the tensor."""
    paths = [Path(p) for p in image_paths]
    if use_composite:
        gray = build_scrolled_composite(paths, crop_barco=crop_barco)
        text = "\n".join(_ocr_composite_lines(gray))
    else:
        text = _ocr_paths(paths, crop_barco=crop_barco)
    return reconstruct_tensor_from_text(
        text,
        expected_shape=expected_shape,
        slice_axis=slice_axis,
    )
