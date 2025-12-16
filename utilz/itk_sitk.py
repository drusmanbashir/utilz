from __future__ import annotations

import SimpleITK as sitk
import torch
import math
import numpy as np

from typing import List, TYPE_CHECKING, Any

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


def ConvertItkImageToSimpleItkImage(
    _itk_image: "itk.Image",
    _pixel_id_value: int,
    _direction: List[float],
) -> sitk.Image:
    """
    Converts ITK image to SimpleITK image
    """
    itk = _require_itk()
    array: np.ndarray = itk.GetArrayFromImage(_itk_image)
    sitk_image: sitk.Image = sitk.GetImageFromArray(array)
    sitk_image = CopyImageMetaInformationFromItkImageToSimpleItkImage(
        sitk_image, _itk_image, _pixel_id_value, _direction
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

