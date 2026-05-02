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
    if "qt" in backend or "tk" in backend:
        return
    if not _gui_display_available():
        return
    if _module_available("PyQt6") and _module_available("matplotlib.backends.backend_qtagg"):
        matplotlib.use("qtagg")
        return
    if _module_available("tkinter") and _module_available("matplotlib.backends.backend_tkagg"):
        matplotlib.use("tkagg")


_select_matplotlib_backend()

import matplotlib.pyplot as plt

plt.ion()

__all__ = [
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


def _to_numpy_array(image):
    if isinstance(image, (str, Path)):
        image = _load_path(Path(image))
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(fix_labels(image))
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image input type: {type(image)}")

    while image.ndim > 3:
        image = image[0]
    if image.ndim == 2:
        image = image[None]
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image after normalization, got shape {image.shape}")
    return image


def get_window_level_numpy_array(
    image_list,
    intensity_slider_range_percentile=(2, 98),
    data_types=("img", "mask"),
):
    npa_list = [_to_numpy_array(image) for image in image_list]
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
    return npa_list, wl_range, wl_init


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


class ImageMaskViewer:
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
    ):
        self.cmap_img = cmap_img
        self.cmap_mask = cmap_mask or "nipy_spectral"
        self.apply_transpose = apply_transpose
        dtypes = data_types or dtypes
        self.npa_list, self.wl_range, self.wl_init = get_window_level_numpy_array(
            image_list,
            intensity_slider_range_percentile=intensity_slider_range_percentile,
            data_types=dtypes,
        )
        self.dtypes = _normalize_dtypes(dtypes, len(self.npa_list))
        self.fig, self.axises = _figure_axes(len(self.npa_list), figure_size)
        self.axamp = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.axamp_wl = plt.axes([0.1, 0.0, 0.8, 0.03])
        self.slider = Slider(
            ax=self.axamp,
            label="slice",
            valmin=0,
            valmax=self.npa_list[0].shape[0] - 1,
            valinit=0,
            valstep=1,
        )
        self.slider_wl = RangeSlider(
            ax=self.axamp_wl,
            label="Window level",
            valmin=self.wl_range[0][0],
            valmax=self.wl_range[0][1],
            valinit=self.wl_init[0],
        )
        self.slider.drawon = False
        self.slider_wl.drawon = False
        self.ax_imgs = self.create_images()
        self.slider.on_changed(self.update_fig_fast)
        self.slider_wl.on_changed(self.update_window_level)
        self.fig.subplots_adjust(bottom=0.14)
        _show_figure()

    def create_images(self):
        ax_imgs = []
        for axis, image, dtype in zip(self.axises, self.npa_list, self.dtypes):
            image_slice = image[0]
            if dtype == "mask":
                ax_img = axis.imshow(
                    image_slice,
                    cmap=self.cmap_mask,
                    vmin=image.min(),
                    vmax=image.max(),
                )
            else:
                ax_img = axis.imshow(
                    image_slice,
                    cmap=self.cmap_img,
                    vmin=self.slider_wl.val[0],
                    vmax=self.slider_wl.val[1],
                )
            ax_imgs.append(ax_img)
        return ax_imgs

    def update_fig_fast(self, value):
        index = int(round(value))
        for ax_img, image in zip(self.ax_imgs, self.npa_list):
            ax_img.set_array(image[index])
        self.fig.canvas.draw_idle()

    def update_window_level(self, values):
        for ax_img, dtype in zip(self.ax_imgs, self.dtypes):
            if dtype == "image":
                ax_img.set_clim(*values)
        self.fig.canvas.draw_idle()
