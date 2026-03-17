from __future__ import annotations

import SimpleITK as sitk
import torch
import math
import numpy as np

from typing import List, TYPE_CHECKING, Any
from pathlib import Path

# Optional ITK import (Slicer often does not ship the PyPI `itk` package)
try:
    import itk as _itk  # type: ignore
except Exception:
    _itk = None  # type: ignore

if TYPE_CHECKING:
    import itk  # for type checkers only


def _require_itk() -> Any:
    if _itk is None:
        raise ImportError(
            "Python package 'itk' is not available in this environment. "
            "Install into Slicer with: PythonSlicer -m pip install itk"
        )
    return _itk




def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _extract_tensor_and_meta(obj):
    meta = {}
    tensor = obj
    if hasattr(obj, "meta"):
        try:
            meta = dict(obj.meta)
        except Exception:
            meta = {}
    elif isinstance(obj, dict):
        if "meta" in obj and isinstance(obj["meta"], dict):
            meta = dict(obj["meta"])
        for key in ("lm", "label", "labels", "image", "img", "tensor"):
            if key in obj:
                tensor = obj[key]
                break
        if hasattr(tensor, "meta") and not meta:
            try:
                meta = dict(tensor.meta)
            except Exception:
                meta = {}
    return tensor, meta


def monai_to_sitk_image(li):
    src = None
    payload = li
    if isinstance(li, (str, Path)) and str(li).lower().endswith(".pt"):
        src = str(li)
        payload = torch.load(str(li), map_location="cpu", weights_only=False)

    tensor, meta = _extract_tensor_and_meta(payload)
    arr = _as_numpy(tensor)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected channel-first tensor with one channel, got shape {arr.shape}"
            )
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D tensor/image, got shape {arr.shape}")

    # MONAI tensor convention is XYZ (or CXYZ); SITK array constructor expects ZYX.
    sitk_img = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)))

    affine = meta.get("affine", None)
    if affine is not None:
        aff = _as_numpy(affine).astype(np.float64)
        if aff.shape != (4, 4):
            raise ValueError(f"Expected 4x4 affine, got shape {aff.shape}")
        linear_ras = aff[:3, :3]
        spacing = np.linalg.norm(linear_ras, axis=0)
        spacing = np.where(spacing == 0, 1.0, spacing)
        direction_ras = linear_ras / spacing
        ras_to_lps = np.diag([-1.0, -1.0, 1.0])
        direction_lps = ras_to_lps @ direction_ras
        origin_lps = ras_to_lps @ aff[:3, 3]

        sitk_img.SetSpacing(tuple(float(x) for x in spacing))
        sitk_img.SetDirection(tuple(float(x) for x in direction_lps.reshape(-1)))
        sitk_img.SetOrigin(tuple(float(x) for x in origin_lps))
    else:
        spacing = meta.get("spacing", (1.0, 1.0, 1.0))
        spacing = _as_numpy(spacing).tolist()
        if len(spacing) >= 3:
            spacing = spacing[:3]
        else:
            spacing = [1.0, 1.0, 1.0]
        sitk_img.SetSpacing(tuple(float(x) for x in spacing))
        sitk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        sitk_img.SetOrigin((0.0, 0.0, 0.0))

    if src is None:
        src = meta.get("filename_or_obj", None)
    return sitk_img, src


def monai_to_itk_image(li, pixel_type=None):
    itk = _require_itk()
    sitk_img, src = monai_to_sitk_image(li)
    if pixel_type is None:
        pixel_type = itk.UC
    itk_img = ConvertSimpleItkImageToItkImage(sitk_img, pixel_type)
    return itk_img, src

def ConvertItkImageToSimpleItkImage(
    _itk_image: "itk.Image",
    _pixel_id_value: int,
) -> sitk.Image:
    """
    Converts ITK image to SimpleITK image
    """
    itk = _require_itk()
    direction_array = np.array(_itk_image.GetDirection())
    direction = tuple(direction_array.flatten())
    array: np.ndarray = itk.GetArrayFromImage(_itk_image)
    sitk_image: sitk.Image = sitk.GetImageFromArray(array)
    sitk_image = CopyImageMetaInformationFromItkImageToSimpleItkImage(
        sitk_image, _itk_image, _pixel_id_value, direction
    )
    return sitk_image


def ConvertSimpleItkImageToItkImage(_sitk_image: sitk.Image, _pixel_id_value):
    """
    Converts SimpleITK image to ITK image
    """
    itk = _require_itk()
    array: np.ndarray = sitk.GetArrayFromImage(_sitk_image)
    itk_image: "itk.Image" = itk.GetImageFromArray(array)
    itk_image = CopyImageMetaInformationFromSimpleItkImageToItkImage(
        itk_image, _sitk_image, _pixel_id_value
    )
    return itk_image


def CopyImageMetaInformationFromSimpleItkImageToItkImage(
    _itk_image: "itk.Image",
    _reference_sitk_image: sitk.Image,
    _output_pixel_type,
) -> "itk.Image":
    """
    Copies the meta information from SimpleITK image to ITK image
    """
    itk = _require_itk()

    _itk_image.SetOrigin(_reference_sitk_image.GetOrigin())
    _itk_image.SetSpacing(_reference_sitk_image.GetSpacing())

    # Setting direction
    reference_image_direction: np.ndarray = np.eye(3)
    np_dir_vnl = itk.GetVnlMatrixFromArray(reference_image_direction)
    itk_image_direction = _itk_image.GetDirection()
    itk_image_direction.GetVnlMatrix().copy_in(np_dir_vnl.data_block())

    dimension: int = _itk_image.GetImageDimension()
    input_image_type = type(_itk_image)
    output_image_type = itk.Image[_output_pixel_type, dimension]

    castImageFilter = itk.CastImageFilter[input_image_type, output_image_type].New()
    castImageFilter.SetInput(_itk_image)
    castImageFilter.Update()
    result_itk_image: "itk.Image" = castImageFilter.GetOutput()
    return result_itk_image


def CopyImageMetaInformationFromItkImageToSimpleItkImage(
    _sitk_image: sitk.Image,
    _reference_itk_image: "itk.Image",
    _pixel_id_value: int,
    _direction: List[float],
) -> sitk.Image:
    """
    Copies the meta information from ITK image to SimpleITK image
    """
    reference_image_origin: List[float] = list(_reference_itk_image.GetOrigin())
    _sitk_image.SetOrigin(reference_image_origin)

    reference_image_spacing: List[float] = list(_reference_itk_image.GetSpacing())
    _sitk_image.SetSpacing(reference_image_spacing)

    _sitk_image.SetDirection(_direction)
    result_sitk_image: sitk.Image = sitk.Cast(_sitk_image, _pixel_id_value)
    return result_sitk_image


def get_sitk_target_size_from_spacings(sitk_array, spacing_dest):
    sz_source, spacing_source = sitk_array.GetSize(), sitk_array.GetSpacing()
    sz_dest, _ = get_scale_factor_from_spacings(sz_source, spacing_source, spacing_dest)
    return sz_dest


def get_scale_factor_from_spacings(sz_source, spacing_source, spacing_dest):
    scale_factor = [a / b for a, b in zip(spacing_source, spacing_dest)]
    sz_dest = [round(a * b) for a, b in zip(sz_source, scale_factor)]
    return sz_dest, scale_factor


def rescale_bbox(scale_factor, bbox):
    bbox_out = []
    for a, b in zip(scale_factor, bbox):
        bbox_neo = slice(int(b.start * a), int(np.ceil(b.stop * a)), b.step)
        bbox_out.append(bbox_neo)
    return tuple(bbox_out)


def apply_threshold(input_img, threshold):
    input_img[input_img < threshold] = 0
    input_img[input_img >= threshold] = 1
    return input_img


def get_amount_to_pad(img_shape, patch_size):
    pad_deficits = np.maximum(0, np.array(patch_size) - img_shape)
    padding = (
        (math.floor(pad_deficits[0] / 2), math.ceil(pad_deficits[0] / 2)),
        (math.floor(pad_deficits[1] / 2), math.ceil(pad_deficits[1] / 2)),
        (math.floor(pad_deficits[2] / 2), math.ceil(pad_deficits[2] / 2)),
    )
    return padding


def aip(niifn, niifn_out, max_thickness=3.0):
    from torch import nn

    print(f"Processing file {niifn}")
    im_ni = sitk.ReadImage(niifn)

    st_org = im_ni.GetSpacing()[-1]

    if st_org >= max_thickness:
        print(f"Already thick slice-image ({st_org}mm). Skipping {niifn}")
        return

    if 0.9 < st_org < 1.5:
        step = 3
    elif st_org < 0.9:
        step = 5
    elif 1.5 <= st_org < max_thickness:
        step = 2
    else:
        step = 2

    im_np = sitk.GetArrayFromImage(im_ni)
    im = torch.tensor(im_np).float()

    n_slice = im_np.shape[0]
    in_plane = im_np.shape[1:]
    im1d = im.view(n_slice, -1).unsqueeze(1)

    av = nn.Conv1d(1, 1, 3, padding=1)
    av.weight = nn.parameter.Parameter(torch.ones_like(av.weight))

    imthic = av(im1d).view(-1, *in_plane)
    imthic = imthic[::step, :]

    im_out = sitk.GetImageFromArray(imthic.detach().numpy())

    # NOTE: align_sitk_imgs must exist in your environment; unchanged here.
    im_out = align_sitk_imgs(im_out, im_ni)

    outthickness = np.minimum(max_thickness, st_org * step)
    spacing = im_out.GetSpacing()
    im_out.SetSpacing((spacing[0], spacing[1], float(outthickness)))

    print(f"Starting nslices: {n_slice}. Final nslices: {imthic.shape[0]}")
    sitk.WriteImage(im_out, niifn_out)
