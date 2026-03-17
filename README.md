# utilz
Shared Python utilities used across projects in `~/code`.

## Module Function Index

### `utilz/__init__.py`
- `create_nifti_overlay_grid_gif`: Re-export for grid GIF overlay generation.
- `create_case_crop_overlay_gif`: Re-export for single-case cropped overlay GIF creation.
- `create_folder_crop_overlay_gifs`: Re-export for batch cropped overlay GIF creation.

### `utilz/cprint.py`
- `cprint`: Prints colored and styled terminal text with ANSI codes.

### `utilz/dictopts.py`
- `key_from_value`: Finds all keys that match a given value.
- `assimilate_dict`: Copies dict keys/values to attributes on a target object.
- `list_unique`: Returns unique values from a list.
- `dic_lists_to_sets`: Converts list values inside a dict into sets.
- `dic_in_list`: Checks if a normalized dict exists in a list of dicts.
- `fix_ast`: Applies `ast.literal_eval` to selected dict fields.

### `utilz/fileio.py`
- `is_sitk_file`: Checks if a file looks like a SimpleITK image file.
- `is_image_file`: Checks if a file looks like an image/tensor file.
- `str_replace`: Replaces a substring in a `Path`.
- `convert_uint16_to_uint8`: Converts `.npy` arrays from uint16 to uint8 in place.
- `save_file_wrapper`: Decorator handling overwrite and directory creation for save functions.
- `save_np`: Saves NumPy arrays.
- `load_file`: Decorator for opening files with a specific mode before loading.
- `load_json`: Loads JSON files.
- `load_pickle`: Loads pickle files.
- `load_yaml`: Loads YAML files.
- `load_text`: Loads text files into stripped lines.
- `load_xml`: Loads XML with BeautifulSoup.
- `save_xml`: Saves XML/text to disk.
- `save_json`: Saves dicts to JSON.
- `save_sitk`: Saves tensors/arrays/SimpleITK images via SimpleITK.
- `np_to_ni`: Converts NumPy arrays to NIfTI files.
- `is_filename`: Checks if a string is a filename-like token.
- `maybe_makedirs`: Creates missing directories for one or more paths.
- `dump`: Decorator helper for serializer-style save functions.
- `load_dict`: Loads dicts from JSON or pickle by extension.
- `save_dict`: Saves dicts to JSON, with pickle fallback.
- `sitk_filename_to_numpy`: Reads image file into a NumPy array.
- `load_image`: Unified loader for `.npy`, `.pt`, and image files.
- `rename_and_move_images_and_masks`: Renames and organizes image/mask pairs into dataset layout.
- `save_list`: Saves nested list content as formatted text.

### `utilz/helpers.py`
- `set_autoreload_ray`: Enables notebook autoreload in Ray-friendly conditions.
- `set_autoreload`: Enables notebook autoreload.
- `test_modified`: Returns whether a file was modified in the last N days.
- `timing`: Decorator that prints execution time.
- `range_inclusive`: Inclusive range helper.
- `multiply_lists`: Elementwise multiplication of two lists.
- `abs_list`: Returns absolute values for each list item.
- `slice_list`: Slices a list from start/end indices.
- `chunks`: Splits a list into roughly equal chunks.
- `merge_dicts`: Recursively merges two dictionaries.
- `maybe_to_torch`: Converts compatible values to torch tensors.
- `to_cuda`: Moves tensors/collections to CUDA.
- `folder_name_from_list`: Builds folder names from list elements.
- `spacing_from_folder_name`: Extracts spacing info from folder naming.
- `pp`: Pretty-prints objects.
- `str_to_list`: Decorator to coerce string input into list form.
- `ask_proceed`: Prompts interactively to continue or abort.
- `str_to_list_float`: Parses float lists from string input.
- `str_to_list_int`: Parses int lists from string input.
- `purge_cuda`: Clears CUDA memory/cache state.
- `get_list_input`: Reads and parses list input interactively.
- `_available_cpus`: Returns available CPU count.
- `_limit_threads_for_io`: Sets strict low-thread config for I/O-heavy jobs.
- `_limit_threads_for_numeric`: Sets balanced thread config for numeric workloads.
- `multiprocess_multiarg`: Multiprocess runner for `func(*args)` argument tuples.
- `get_available_device`: Selects a GPU under memory-use threshold.
- `resolve_device`: Normalizes user device input to a torch device target.
- `set_cuda_device`: Sets active CUDA device.
- `get_last_n_dims`: Returns the trailing N dimensions of a shape/tensor.
- `sigmoid`: NumPy sigmoid helper.
- `convert_sigmoid_to_label`: Thresholds probabilities into labels.
- `match_filename_with_case_id`: Matches filename against case-id convention.
- `project_title_from_folder`: Extracts project title from folder name.
- `get_train_valid_test_lists_from_json`: Loads dataset split lists from JSON.
- `find_matching_fn`: Finds matching filename using parsed filename metadata.
- `create_df_from_folder`: Builds a dataframe from files in a folder.
- `get_fileslist_from_path`: Lists files in a path filtered by extension.
- `make_channels`: Generates channel progression list for model depth.
- `is_notebook`: Detects notebook runtime.
- `write_file_or_not`: Checks whether a single output should be written.
- `write_files_or_not`: Checks whether multiple outputs should be written.

### `utilz/image_utils.py`
- `convert_float16_to_float32`: Converts `.npy` image arrays from float16 to float32.
- `convert_np_to_tensor`: Converts `.npy` arrays to torch tensors and saves `.pt`.
- `resize_tensor_3d`: Resizes 3D tensors with interpolation.
- `resize_multilabel_mask_torch`: Resizes multi-label masks with nearest-neighbor torch interpolation.
- `resize_multilabel_mask_sitk`: Resizes multi-label masks by per-label boolean overlays.
- `get_bbox_from_mask`: Computes 3D bounding box slices for non-background voxels.
- `crop_to_bbox`: Crops arrays/tensors to bbox with optional padding/stride.
- `is_standard_orientation`: Checks whether an image direction is standard orientation.
- `get_img_mask_from_nii`: Loads image/mask arrays and basic properties from NIfTI files.
- `retrieve_properties_from_nii`: Returns key NIfTI geometry/properties for a case.

### `utilz/imageviewers.py`
- `discrete_cmap`: Creates a discrete colormap.
- `fix_labels`: Casts unsupported label pixel types for viewing.
- `view_sitk`: Displays SITK image/mask pair in interactive viewer.
- `view_3d_np`: Displays 3D NumPy image/mask pair.
- `view_5d_torch`: Displays one sample from 5D torch image/mask tensors.
- `view`: General image/mask display wrapper for torch/NumPy inputs.
- `get_window_level_numpy_array`: Computes display arrays and window/level ranges.
- `viewer`: Convenience wrapper for opening viewer with raw arrays.

### `utilz/itk_sitk.py`
- `_require_itk`: Raises a clear error if Python `itk` is unavailable.
- `ConvertItkImageToSimpleItkImage`: Converts ITK image to SimpleITK and preserves metadata.
- `ConvertSimpleItkImageToItkImage`: Converts SimpleITK image to ITK and preserves metadata.
- `CopyImageMetaInformationFromSimpleItkImageToItkImage`: Copies origin/spacing/direction to ITK image.
- `CopyImageMetaInformationFromItkImageToSimpleItkImage`: Copies origin/spacing/direction to SimpleITK image.
- `get_sitk_target_size_from_spacings`: Computes target image size from desired spacing.
- `get_scale_factor_from_spacings`: Computes spacing scale factor and destination size.
- `rescale_bbox`: Rescales bbox slices using scale factors.
- `apply_threshold`: Binarizes image by threshold.
- `get_amount_to_pad`: Computes symmetric padding to reach patch size.
- `aip`: Creates an average-intensity-projection-like output.

### `utilz/itk_sitk_bk.py`
- `ConvertItkImageToSimpleItkImage`: Legacy ITK-to-SimpleITK conversion helper.
- `ConvertSimpleItkImageToItkImage`: Legacy SimpleITK-to-ITK conversion helper.
- `CopyImageMetaInformationFromSimpleItkImageToItkImage`: Legacy metadata copy to ITK image.
- `CopyImageMetaInformationFromItkImageToSimpleItkImage`: Legacy metadata copy to SimpleITK image.
- `get_sitk_target_size_from_spacings`: Legacy target-size-from-spacing helper.
- `get_scale_factor_from_spacings`: Legacy spacing scale-factor helper.
- `rescale_bbox`: Legacy bbox rescaling helper.
- `apply_threshold`: Legacy thresholding helper.
- `get_amount_to_pad`: Legacy patch padding helper.
- `aip`: Legacy average-intensity-projection helper.

### `utilz/stringz.py`
- `ast_literal_eval`: Safely parses literal string values/lists.
- `regex_matcher`: Decorator that applies regex and returns a selected match group.
- `dec_to_str`: Converts decimal to compact digit string.
- `int_to_str`: Left-pads integers with zeros.
- `headline`: Prints a separator-style headline.
- `append_time`: Appends timestamp suffix to strings.
- `infer_dataset_name`: Extracts dataset name token from filename.
- `strip_extension`: Removes chained imaging/annotation extensions.
- `replace_extension`: Replaces extension with a new one.
- `strip_slicer_strings`: Removes common Slicer suffix tokens.
- `str_to_path`: Decorator converting selected args to `Path`.
- `path_to_str`: Decorator converting path args/returns to strings.
- `cleanup_fname`: Normalizes filename into stable project/case/date format.
- `get_extension`: Extracts file extension.
- `drop_digit_suffix`: Removes numeric suffix from case/patch names.
- `info_from_filename`: Parses `(project, case_id, date, description)` from filename.
- `match_filenames`: Compares filenames by parsed metadata.
- `find_file`: Finds file(s) containing a substring in name.

### `utilz/overlay_grid_gif.py`
- `create_nifti_overlay_grid_gif`: Builds an animated grid GIF with image/label overlays.
- `_optimize_gif_palette`: Re-quantizes GIF frames to reduce size.
- `_iter_progress`: Iterates with tqdm-like progress feedback.
- `_list_case_dirs`: Lists case directories under dataset root.
- `_pick_label_file`: Chooses label file for a case directory.
- `_base_stem`: Returns stable stem for multi-extension files.
- `_list_supported_files`: Lists supported volume files in a folder.
- `_to_numpy`: Converts supported input object to NumPy array.
- `_normalize_to_3d`: Normalizes input arrays into 3D volume shape.
- `_infer_slice_axis`: Chooses slice axis if not explicitly provided.
- `_read_zyx`: Reads/normalizes volume as ZYX array.
- `_normalize_slice`: Intensity-normalizes a 2D slice.
- `_window_normalize_slice`: Applies preset window normalization.
- `_rotate_clockwise_2d`: Rotates 2D arrays clockwise by fixed angle.
- `_pad_center_2d`: Pads a 2D array centrally to target size.
- `_stencil_edges`: Computes label edge stencil.
- `_label_color`: Returns RGB color for a label value.
- `_labels_present`: Lists non-zero labels present in slice.
- `_overlay_stencil_rgb`: Renders colored label edges over grayscale slice.
- `_load_cases_from_case_dirs`: Loads paired cases from case-folder structure.
- `_match_flat_pairs`: Matches image/label pairs in flat folders.
- `_load_cases_from_images_lms`: Loads paired cases from `images/` and `lms/`.
- `_load_cases`: Unified case-loading entrypoint.
- `_build_parser`: CLI parser builder.
- `main`: CLI entrypoint.

### `utilz/pred_label_crop_gif.py`
- `create_paginated_gif_grid`: Combines GIFs into paginated grid outputs.
- `create_case_crop_overlay_gif`: Creates one cropped overlay GIF for a case.
- `create_folder_crop_overlay_gifs`: Batch-creates cropped overlay GIFs for a folder.
- `_list_nifti_files`: Lists NIfTI files in folder.
- `_resample_like`: Resamples moving image to reference geometry.
- `_bbox_from_mask_xyz`: Computes XYZ bounding box from mask.
- `_axis_aligned_crop_for_label`: Computes crop around one label with margin.
- `_axis_aligned_crop_for_overlay_union`: Computes crop from union of overlay masks.
- `_window_to_uint8`: Window-level converts image to uint8.
- `_sample_slice_indices`: Selects evenly spaced slice indices.
- `_contour_from_mask_np`: Extracts contour mask from binary mask.
- `_normalize_overlay_sources`: Normalizes overlay source config structure.
- `_parse_overlay_source_arg`: Parses CLI overlay-source argument.
- `_upscale_rgb`: Upscales RGB frames by integer scale factor.
- `_add_case_id_footer`: Adds case-id footer text to a frame.
- `_read_gif_frames`: Loads frames from existing GIF.
- `_build_parser`: CLI parser builder.
- `main`: CLI entrypoint.

### `utilz/bone_grid_viewer.py`
- `parse_args`: CLI argument parser for bone grid viewer.
- `is_volume_file`: Checks supported volume extensions.
- `case_id_from_name`: Extracts case ID from filename.
- `discover_labels`: Indexes label files by case ID.
- `discover_images`: Indexes image files by case ID.
- `choose_best_image`: Picks best image candidate for a case.
- `build_pairs`: Builds matched image/label case pairs.
- `sitk_to_numpy_xyz`: Converts SITK image to NumPy XYZ.
- `numpy_xyz_to_sitk`: Converts NumPy XYZ back to SITK image.
- `resample_xyz`: Resamples volume to target shape.
- `normalize_slice`: Normalizes grayscale slice for display.
- `edge_from_mask`: Computes mask edges for overlay.
- `overlay_rgb`: Overlays mask edges onto image slice.
- `chunked`: Splits sequence into fixed-size chunks/pages.
- `load_and_resample_pair`: Loads and resamples one image/label pair.
- `infer_target_shape`: Infers canonical target shape from first pair.
- `main`: CLI entrypoint for interactive 8x8 viewer.
