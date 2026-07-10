import importlib.util
import os
from pathlib import Path

import matplotlib
import numpy as np
import SimpleITK as sitk
import torch
from matplotlib.widgets import RangeSlider, Slider

def _module_available(name):
    return importlib.util.find_spec(name) is not None


def _gui_display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _select_matplotlib_backend():
    if os.environ.get("MPLBACKEND"):
        return
    backend = matplotlib.get_backend().lower()
    if "tk" in backend:
        return
    if not _gui_display_available():
        return
    if _module_available("tkinter") and _module_available("matplotlib.backends.backend_tkagg"):
        matplotlib.use("tkagg")


_select_matplotlib_backend()

import matplotlib.pyplot as plt

plt.ion()

__all__ = [
    "ImageBBoxViewer",
    "ImageMaskBboxViewer",
    "ImageMaskViewer",
    "discrete_cmap",
    "fix_labels",
    "get_window_level_numpy_array",
    "view",
    "view_3d_np",
    "view_5d_torch",
    "view_sitk",
    "viewer",
]

IMAGE_DTYPES = {"i", "img", "image"}
MASK_DTYPES = {"m", "mask", "label", "lm", "labelimage"}
DISPLAY_ORIENTATION = "LPS"
ORIENTATION_SPECS = {
    "xyz": {
        "axial": {"slice_axis": 2, "rot_k": 1, "axis_name": "z"},
        "coronal": {"slice_axis": 1, "rot_k": 1, "axis_name": "y"},
        "sagittal": {"slice_axis": 0, "rot_k": 1, "axis_name": "x"},
        "sag": {"slice_axis": 0, "rot_k": 1, "axis_name": "x"},
    },
    "zyx": {
        "axial": {"slice_axis": 0, "rot_k": 1, "axis_name": "z"},
        "coronal": {"slice_axis": 1, "rot_k": 1, "axis_name": "y"},
        "sagittal": {"slice_axis": 2, "rot_k": 1, "axis_name": "x"},
        "sag": {"slice_axis": 2, "rot_k": 1, "axis_name": "x"},
    },
}
BBOX_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]


def discrete_cmap(n_bins, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    colors = base(np.linspace(0, 1, n_bins))
    return base.from_list(f"{base.name}{n_bins}", colors, n_bins)


def fix_labels(image):
    if isinstance(image, sitk.Image) and image.GetPixelID() == 22:
        return sitk.Cast(image, sitk.sitkUInt8)
    return image


def _is_nifti_path(path):
    name = path.name
    return name.endswith(".nii") or name.endswith(".nii.gz") or name.endswith(".nrrd")


def _load_path(path):
    if _is_nifti_path(path):
        return sitk.ReadImage(str(path))
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unsupported image input path: {path}")


def _normalize_dtype(dtype):
    token = dtype.lower()
    if token in IMAGE_DTYPES:
        return "image"
    if token in MASK_DTYPES:
        return "mask"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _normalize_dtypes(dtypes, n_images):
    if isinstance(dtypes, str):
        lowered = dtypes.lower()
        tokens = [dtypes] if lowered in IMAGE_DTYPES | MASK_DTYPES else list(dtypes)
    else:
        tokens = list(dtypes)
    if len(tokens) != n_images:
        raise ValueError(f"Expected {n_images} dtypes, received {len(tokens)}")
    return [_normalize_dtype(token) for token in tokens]


def _oriented_sitk_array(image):
    oriented = sitk.DICOMOrient(fix_labels(image), DISPLAY_ORIENTATION)
    return sitk.GetArrayFromImage(oriented)


def _tensor_has_affine_meta(image):
    return hasattr(image, "meta") and image.meta is not None and "affine" in image.meta


def _oriented_tensor_array(image):
    from utilz.itk_sitk import monai_to_sitk_image

    sitk_image, _ = monai_to_sitk_image(image)
    return _oriented_sitk_array(sitk_image)


def _squeeze_volume(image):
    while image.ndim > 3:
        image = image[0]
    if image.ndim == 2:
        image = image[None]
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image after normalization, got shape {image.shape}")
    return image


def _resolve_orientation(orientation, coord_order):
    orient_key = orientation.lower()
    if orient_key == "sag":
        orient_key = "sagittal"
    if coord_order == "auto":
        raise ValueError("coord_order='auto' must be resolved before calling _resolve_orientation")
    if coord_order not in ORIENTATION_SPECS:
        raise ValueError(f"Unsupported coord_order: {coord_order}")
    if orient_key not in ORIENTATION_SPECS[coord_order]:
        raise ValueError(
            f"Unsupported orientation {orientation!r}; "
            f"expected one of {list(ORIENTATION_SPECS[coord_order])}"
        )
    return ORIENTATION_SPECS[coord_order][orient_key]


def _make_view_state(orientation, coord_order, apply_transpose=True):
    view_spec = _resolve_orientation(orientation, coord_order)
    rot_k = view_spec["rot_k"] if apply_transpose else 0
    return view_spec, view_spec["slice_axis"], rot_k


def _volume_slice(volume, slice_axis, slice_idx):
    if slice_axis == 0:
        return volume[slice_idx]
    if slice_axis == 1:
        return volume[:, slice_idx]
    return volume[:, :, slice_idx]


def _display_slice(volume, slice_axis, slice_idx, rot_k):
    sl = _volume_slice(volume, slice_axis, slice_idx)
    if rot_k % 4:
        sl = np.rot90(sl, k=rot_k % 4)
    return sl


def _viewer_volumes(image_list):
    oriented = [_load_oriented_volume(image) for image in image_list]
    coord_orders = [item[1] for item in oriented]
    if len(set(coord_orders)) == 1:
        return [item[0] for item in oriented], coord_orders[0]
    return [_load_raw_volume(image) for image in image_list], "xyz"


def _load_oriented_volume(image):
    if isinstance(image, (str, Path)):
        image = _load_path(Path(image))
    if isinstance(image, sitk.Image):
        return _oriented_sitk_array(image), "zyx"
    if isinstance(image, torch.Tensor) and _tensor_has_affine_meta(image):
        return _oriented_tensor_array(image), "zyx"
    if isinstance(image, torch.Tensor):
        image = _tensor_to_numpy(image)
        if image.dtype == np.float16:
            image = image.astype(np.float32)
        return _squeeze_volume(image), "xyz"
    if isinstance(image, np.ndarray):
        return _squeeze_volume(image), "xyz"
    raise TypeError(f"Unsupported image input type: {type(image)}")


def _load_raw_volume(image):
    if isinstance(image, (str, Path)):
        image = _load_path(Path(image))
    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(fix_labels(image))
    elif isinstance(image, torch.Tensor):
        image = _tensor_to_numpy(image)
        if image.dtype == np.float16:
            image = image.astype(np.float32)
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image input type: {type(image)}")
    return _squeeze_volume(image)


def _tensor_meta_dict(image):
    if isinstance(image, torch.Tensor) and hasattr(image, "meta") and image.meta is not None:
        return dict(image.meta)
    return {}


def _tensor_to_numpy(tensor):  #AI
    return tensor.detach().cpu().numpy()


def _normalize_boxes(bbox):
    if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], torch.Tensor):
        bbox = bbox[0]
    if isinstance(bbox, torch.Tensor):
        boxes = _tensor_to_numpy(bbox)
    else:
        boxes = np.asarray(bbox, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[None]
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError(f"Expected bbox shape (N, 6) xyzxyz, got {boxes.shape}")
    return boxes


def _bbox_colors(n_boxes):
    if n_boxes == 1:
        return [BBOX_COLORS[0]]
    return [BBOX_COLORS[i % len(BBOX_COLORS)] for i in range(n_boxes)]


def _box_visible(box, slice_axis, slice_idx):
    x0, y0, z0, x1, y1, z1 = box
    if slice_axis == 0:
        return x0 <= slice_idx <= x1
    if slice_axis == 1:
        return y0 <= slice_idx <= y1
    return z0 <= slice_idx <= z1


def _box_index_ranges(box):
    x0, y0, z0, x1, y1, z1 = box
    return (
        int(np.floor(x0)),
        int(np.floor(x1)),
        int(np.floor(y0)),
        int(np.floor(y1)),
        int(np.floor(z0)),
        int(np.floor(z1)),
    )


def _box_slice_mask(shape_2d, box, slice_axis):
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = _box_index_ranges(box)
    h, w = shape_2d
    if slice_axis == 0:
        row_lo, row_hi, col_lo, col_hi = y_lo, y_hi, z_lo, z_hi
    elif slice_axis == 1:
        row_lo, row_hi, col_lo, col_hi = x_lo, x_hi, z_lo, z_hi
    else:
        row_lo, row_hi, col_lo, col_hi = x_lo, x_hi, y_lo, y_hi
    row_lo = max(0, min(row_lo, h - 1))
    row_hi = max(0, min(row_hi, h - 1))
    col_lo = max(0, min(col_lo, w - 1))
    col_hi = max(0, min(col_hi, w - 1))
    if row_lo > row_hi or col_lo > col_hi:
        return np.zeros(shape_2d, dtype=bool)
    mask = np.zeros(shape_2d, dtype=bool)
    mask[row_lo : row_hi + 1, col_lo : col_hi + 1] = True
    return mask


def _box_display_edges(box, volume, slice_axis, slice_idx, rot_k):
    if not _box_visible(box, slice_axis, slice_idx):
        return None
    sl = _volume_slice(volume, slice_axis, slice_idx)
    mask = _box_slice_mask(sl.shape, box, slice_axis)
    if rot_k % 4:
        mask = np.rot90(mask, k=rot_k % 4)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def _box_span(box, slice_axis):
    x0, y0, z0, x1, y1, z1 = box
    if slice_axis == 0:
        return x0, x1
    if slice_axis == 1:
        return y0, y1
    return z0, z1


def _draw_box(ax, box, color, line_style, linewidth, volume, slice_axis, slice_idx, rot_k):
    edges = _box_display_edges(box, volume, slice_axis, slice_idx, rot_k)
    if edges is None:
        return []
    r0, r1, c0, c1 = edges
    segments = [
        ([c0, c1], [r0, r0]),
        ([c0, c1], [r1, r1]),
        ([c0, c0], [r0, r1]),
        ([c1, c1], [r0, r1]),
    ]
    lines = []
    for xs, ys in segments:
        line, = ax.plot(xs, ys, color=color, linestyle=line_style, linewidth=linewidth)
        lines.append(line)
    return lines


def _box_summary_lines(box, index, color, slice_idx, slice_axis, axis_name):
    x0, y0, z0, x1, y1, z1 = box
    center = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    size = x1 - x0, y1 - y0, z1 - z0
    span_lo, span_hi = _box_span(box, slice_axis)
    visible = _box_visible(box, slice_axis, slice_idx)
    state = "ON" if visible else "off"
    return [
        f"[{index}] {color}  {state}",
        f"  xyz=({x0:g},{y0:g},{z0:g})-({x1:g},{y1:g},{z1:g})",
        f"  size=({size[0]:g},{size[1]:g},{size[2]:g})",
        f"  ctr=({center[0]:g},{center[1]:g},{center[2]:g})",
        f"  {axis_name} span [{span_lo:g}, {span_hi:g}]",
    ]


def _image_info_lines(image_array, meta, slice_idx, orientation, coord_order, view_spec):
    n_slices = image_array.shape[view_spec["slice_axis"]] - 1
    lines = [
        "image",
        f"  shape: {tuple(image_array.shape)}",
        f"  view: {orientation} ({coord_order})",
    ]
    if "spacing" in meta:
        lines.append(f"  spacing: {np.asarray(meta['spacing']).tolist()}")
    if "spatial_shape" in meta:
        lines.append(f"  spatial_shape: {np.asarray(meta['spatial_shape']).tolist()}")
    if "space" in meta:
        lines.append(f"  space: {meta['space']}")
    if "filename_or_obj" in meta:
        lines.append(f"  file: {meta['filename_or_obj']}")
    if "src_filename" in meta:
        lines.append(f"  src: {meta['src_filename']}")
    lines.append("")
    lines.append(
        f"slice {slice_idx} / {n_slices}  "
        f"({view_spec['axis_name']}, axis {view_spec['slice_axis']})"
    )
    return lines


def _image_wl_limits(image, intensity_slider_range_percentile):
    limits = np.percentile(image.reshape(-1), intensity_slider_range_percentile)
    return limits[0], limits[1]


def _to_numpy_array(image):
    return _load_oriented_volume(image)[0]


def get_window_level_numpy_array(
    image_list,
    intensity_slider_range_percentile=(2, 98),
    data_types=("img", "mask"),
):
    npa_list, coord_order = _viewer_volumes(image_list)
    dtypes = _normalize_dtypes(data_types, len(npa_list))
    wl_range = []
    wl_init = []
    for image, dtype in zip(npa_list, dtypes):
        if dtype == "image":
            limits = np.percentile(image.reshape(-1), intensity_slider_range_percentile)
        else:
            limits = [image.min(), image.max()]
        wl_range.append((limits[0], limits[1]))
        wl_init.append((limits[0], limits[1]))
    return npa_list, wl_range, wl_init, coord_order


def _figure_axes(n_images, figure_size):
    fig, axes = plt.subplots(1, n_images, figsize=figure_size)
    axes = np.atleast_1d(axes).tolist()
    return fig, axes


def _show_figure():
    plt.show(block=False)


def view_sitk(img, mask, dtypes="im", data_types=None, **kwargs):
    ImageMaskViewer([img, mask], dtypes=dtypes, data_types=data_types, **kwargs)


def view_3d_np(x):
    ImageMaskViewer([np.expand_dims(x[0], 0), np.expand_dims(x[1], 0)], dtypes="im")


def view_5d_torch(x, n=0):
    ImageMaskViewer([x[0][n, 0], x[1][n, 0]], dtypes="im")


def view(*arrays, n=0, cmap_img="Greys_r", cmap_mask="RdPu_r", dtypes=None):
    if hasattr(arrays[0], "ndim") and arrays[0].ndim > 4:
        arrays = [array[n] for array in arrays]
    if dtypes is None:
        dtypes = "im" if len(arrays) == 2 else "i" * len(arrays)
    ImageMaskViewer(arrays, dtypes=dtypes, cmap_img=cmap_img, cmap_mask=cmap_mask)


def viewer(*arrays):
    dtypes = "im" if len(arrays) == 2 else "i" * len(arrays)
    return ImageMaskViewer(list(arrays), dtypes=dtypes)


class _SliceViewerBase:
    def _init_bbox_state(self, bbox, line_style, linewidth):
        self.boxes = _normalize_boxes(bbox) if bbox is not None else None
        self.colors = _bbox_colors(len(self.boxes)) if self.boxes is not None else []
        self.line_style = line_style
        self.linewidth = linewidth
        self.box_lines_by_ax = {}

    def _attach_slice_slider(self, fig, rect, volume_shape):
        valmax = max(0, volume_shape[self.slice_axis] - 1)
        slider = Slider(
            ax=fig.add_axes(rect),
            label="slice",
            valmin=0,
            valmax=valmax,
            valinit=0,
            valstep=1,
        )
        slider.drawon = False
        slider.eventson = False
        return slider

    def _attach_wl_slider(self, fig, rect, wl_range, wl_init):
        slider = RangeSlider(
            ax=fig.add_axes(rect),
            label="Window level",
            valmin=wl_range[0],
            valmax=wl_range[1],
            valinit=wl_init,
        )
        slider.drawon = False
        slider.eventson = False
        return slider

    def _connect_view_sliders(self, slice_cb, wl_cb):
        self.slider.eventson = True
        self.slider.on_changed(slice_cb)
        self.slider_wl.eventson = True
        self.slider_wl.on_changed(wl_cb)

    def _imshow_slice(self, ax, volume, slice_idx, cmap, vmin, vmax):
        return ax.imshow(
            _display_slice(volume, self.slice_axis, slice_idx, self.rot_k),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    def _clear_ax_boxes(self, ax):
        for line in self.box_lines_by_ax.get(ax, []):
            line.remove()
        self.box_lines_by_ax[ax] = []

    def _draw_boxes_on_ax(self, ax, volume, slice_idx):
        self._clear_ax_boxes(ax)
        if self.boxes is None:
            return
        lines = []
        for box, color in zip(self.boxes, self.colors):
            lines.extend(
                _draw_box(
                    ax,
                    box,
                    color,
                    self.line_style,
                    self.linewidth,
                    volume,
                    self.slice_axis,
                    slice_idx,
                    self.rot_k,
                )
            )
        self.box_lines_by_ax[ax] = lines

    def _bbox_info_lines(self, slice_idx):
        if self.boxes is None:
            return []
        lines = [f"boxes ({len(self.boxes)})"]
        for index, (box, color) in enumerate(zip(self.boxes, self.colors)):
            lines.extend(
                _box_summary_lines(
                    box,
                    index,
                    color,
                    slice_idx,
                    self.slice_axis,
                    self.view_spec["axis_name"],
                )
            )
            lines.append("")
        return lines


class ImageMaskViewer(_SliceViewerBase):
    def __init__(
        self,
        image_list,
        dtypes="im",
        data_types=None,
        figure_size=(10, 8),
        intensity_slider_range_percentile=(2, 98),
        cmap_img="Greys_r",
        cmap_mask=None,
        apply_transpose=True,
        orientation="axial",
        coord_order="auto",
    ):
        self.cmap_img = cmap_img
        self.cmap_mask = cmap_mask or "nipy_spectral"
        self.orientation = orientation
        dtypes = data_types or dtypes
        self.npa_list, self.wl_range, self.wl_init, viewer_coord_order = get_window_level_numpy_array(
            image_list,
            intensity_slider_range_percentile=intensity_slider_range_percentile,
            data_types=dtypes,
        )
        self.dtypes = _normalize_dtypes(dtypes, len(self.npa_list))
        self.coord_order = viewer_coord_order if coord_order == "auto" else coord_order
        self.view_spec, self.slice_axis, self.rot_k = _make_view_state(
            orientation, self.coord_order, apply_transpose
        )
        self._init_bbox_state(None, "solid", 1.5)
        self.fig, self.axises = _figure_axes(len(self.npa_list), figure_size)
        self.slider = self._attach_slice_slider(
            self.fig, [0.1, 0.05, 0.8, 0.03], self.npa_list[0].shape
        )
        self.slider_wl = self._attach_wl_slider(
            self.fig, [0.1, 0.0, 0.8, 0.03], self.wl_range[0], self.wl_init[0]
        )
        self.ax_imgs = self.create_images()
        self.fig.subplots_adjust(bottom=0.14)
        self._connect_view_sliders(self.update_slice, self.update_window_level)
        _show_figure()

    def create_images(self):
        ax_imgs = []
        for axis, image, dtype in zip(self.axises, self.npa_list, self.dtypes):
            if dtype == "mask":
                ax_img = self._imshow_slice(axis, image, 0, self.cmap_mask, image.min(), image.max())
            else:
                ax_img = self._imshow_slice(
                    axis, image, 0, self.cmap_img, self.slider_wl.val[0], self.slider_wl.val[1]
                )
            ax_imgs.append(ax_img)
        return ax_imgs

    def update_slice(self, value):
        n_slices = self.npa_list[0].shape[self.slice_axis]
        index = max(0, min(int(round(value)), n_slices - 1))
        for ax_img, image in zip(self.ax_imgs, self.npa_list):
            ax_img.set_array(_display_slice(image, self.slice_axis, index, self.rot_k))
        self.fig.canvas.draw_idle()

    def update_window_level(self, values):
        for ax_img, dtype in zip(self.ax_imgs, self.dtypes):
            if dtype == "image":
                ax_img.set_clim(*values)
        self.fig.canvas.draw_idle()


class ImageBBoxViewer(_SliceViewerBase):
    def __init__(
        self,
        image,
        bbox=None,
        figure_size=(12, 8),
        intensity_slider_range_percentile=(2, 98),
        cmap_img="Greys_r",
        line_style="solid",
        linewidth=1.5,
        orientation="axial",
        coord_order="xyz",
    ):
        self.image_input = image
        self.meta = _tensor_meta_dict(image)
        self.image = _load_raw_volume(image)
        self.orientation = orientation
        self.coord_order = coord_order
        self.view_spec, self.slice_axis, self.rot_k = _make_view_state(orientation, coord_order)
        self.cmap_img = cmap_img
        self._init_bbox_state(bbox, line_style, linewidth)
        wl_lo, wl_hi = _image_wl_limits(self.image, intensity_slider_range_percentile)
        self.wl_range = (wl_lo, wl_hi)
        self.wl_init = (wl_lo, wl_hi)

        self.fig = plt.figure(figsize=figure_size)
        self.ax_img = self.fig.add_axes([0.06, 0.15, 0.62, 0.80])
        self.ax_info = self.fig.add_axes([0.70, 0.15, 0.28, 0.80])
        self.ax_info.axis("off")
        self.slider = self._attach_slice_slider(
            self.fig, [0.06, 0.05, 0.62, 0.03], self.image.shape
        )
        self.slider_wl = self._attach_wl_slider(
            self.fig, [0.06, 0.0, 0.62, 0.03], self.wl_range, self.wl_init
        )
        self.ax_im = self._imshow_slice(
            self.ax_img, self.image, 0, self.cmap_img, self.slider_wl.val[0], self.slider_wl.val[1]
        )
        self.info_text = self.ax_info.text(
            0.0, 1.0, "", transform=self.ax_info.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
        )
        self.update_slice(0)
        self._connect_view_sliders(self.update_slice, self.update_window_level)
        _show_figure()

    def _update_info(self, slice_idx):
        lines = _image_info_lines(
            self.image, self.meta, slice_idx, self.orientation, self.coord_order, self.view_spec
        )
        lines.append("")
        lines.extend(self._bbox_info_lines(slice_idx))
        self.info_text.set_text("\n".join(lines).rstrip())

    def update_slice(self, value):
        n_slices = self.image.shape[self.slice_axis]
        slice_idx = max(0, min(int(round(value)), n_slices - 1))
        self.ax_im.set_array(_display_slice(self.image, self.slice_axis, slice_idx, self.rot_k))
        self._draw_boxes_on_ax(self.ax_img, self.image, slice_idx)
        self._update_info(slice_idx)
        self.fig.canvas.draw_idle()

    def update_window_level(self, values):
        self.ax_im.set_clim(*values)
        self.fig.canvas.draw_idle()


class ImageMaskBboxViewer(_SliceViewerBase):
    def __init__(
        self,
        image,
        mask,
        bbox,
        dtypes="im",
        figure_size=(14, 8),
        intensity_slider_range_percentile=(2, 98),
        cmap_img="Greys_r",
        cmap_mask=None,
        orientation="axial",
        coord_order="xyz",
        line_style="solid",
        linewidth=1.5,
    ):
        self.image_input = image
        self.meta = _tensor_meta_dict(image)
        self.image = _load_raw_volume(image)
        self.mask = _load_raw_volume(mask)
        self.orientation = orientation
        self.coord_order = coord_order
        self.view_spec, self.slice_axis, self.rot_k = _make_view_state(orientation, coord_order)
        self.cmap_img = cmap_img
        self.cmap_mask = cmap_mask or "nipy_spectral"
        self._init_bbox_state(bbox, line_style, linewidth)
        wl_lo, wl_hi = _image_wl_limits(self.image, intensity_slider_range_percentile)
        self.wl_range = (wl_lo, wl_hi)
        self.wl_init = (wl_lo, wl_hi)

        self.fig = plt.figure(figsize=figure_size)
        self.ax_img = self.fig.add_axes([0.04, 0.15, 0.28, 0.80])
        self.ax_mask = self.fig.add_axes([0.34, 0.15, 0.28, 0.80])
        self.ax_info = self.fig.add_axes([0.70, 0.15, 0.28, 0.80])
        self.ax_info.axis("off")
        self.slider = self._attach_slice_slider(
            self.fig, [0.04, 0.05, 0.58, 0.03], self.image.shape
        )
        self.slider_wl = self._attach_wl_slider(
            self.fig, [0.04, 0.0, 0.58, 0.03], self.wl_range, self.wl_init
        )
        self.ax_im = self._imshow_slice(
            self.ax_img, self.image, 0, self.cmap_img, self.slider_wl.val[0], self.slider_wl.val[1]
        )
        self.ax_m = self._imshow_slice(
            self.ax_mask, self.mask, 0, self.cmap_mask, self.mask.min(), self.mask.max()
        )
        self.info_text = self.ax_info.text(
            0.0, 1.0, "", transform=self.ax_info.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
        )
        self.update_slice(0)
        self._connect_view_sliders(self.update_slice, self.update_window_level)
        _show_figure()

    def _update_info(self, slice_idx):
        lines = _image_info_lines(
            self.image, self.meta, slice_idx, self.orientation, self.coord_order, self.view_spec
        )
        lines.append("")
        lines.extend(self._bbox_info_lines(slice_idx))
        self.info_text.set_text("\n".join(lines).rstrip())

    def update_slice(self, value):
        n_slices = self.image.shape[self.slice_axis]
        slice_idx = max(0, min(int(round(value)), n_slices - 1))
        self.ax_im.set_array(_display_slice(self.image, self.slice_axis, slice_idx, self.rot_k))
        self.ax_m.set_array(_display_slice(self.mask, self.slice_axis, slice_idx, self.rot_k))
        self._draw_boxes_on_ax(self.ax_img, self.image, slice_idx)
        self._draw_boxes_on_ax(self.ax_mask, self.image, slice_idx)
        self._update_info(slice_idx)
        self.fig.canvas.draw_idle()

    def update_window_level(self, values):
        self.ax_im.set_clim(*values)
        self.fig.canvas.draw_idle()
