# %%
from datetime import datetime
import re
from pathlib import Path
from fastcore.basics import Union, listify
import ipdb
import numpy as np

tr = ipdb.set_trace
import ast

def ast_literal_eval(str_list):
    """Safely evaluate string list using ast.literal_eval."""
    if isinstance(str_list, str):
        str_list = ast.literal_eval(str_list)
    return str_list


def regex_matcher(indx=0):
    """Decorator to match regex patterns and return specific group."""
    def _outer(func):

        def _inner(*args, **kwargs):
            pat, string = func(*args, **kwargs)
            pat = re.compile(pat, re.IGNORECASE)
            answer = re.search(pat, string)
            return answer[indx] if answer else None

        return _inner

    return _outer


def dec_to_str(val: float, trailing_zeros=3):
    """Convert decimal to string with specified trailing zeros."""
    val2 = str(round(val, 2))
    val2 = val2.replace(".", "")
    trailing_zeros = (
        np.maximum(trailing_zeros - len(val2), 0) if trailing_zeros > 0 else 0
    )
    val2 = val2 + "0" * trailing_zeros  # ensure 3 digits
    return val2


def int_to_str(val: int, total_length=5):
    """Convert integer to string with leading zeros."""
    val = str(val)
    precending_zeros = total_length - len(val)
    return "0" * precending_zeros + val


def headline(inp: str):
    """Print text with decorative lines around it."""
    print("=" * 100)
    print(inp)
    print("=" * 100)


def append_time(input_str, now=True):
    """Append current timestamp to input string."""
    now = datetime.now()
    dt_string = now.strftime("_%d%m%y_%H%M")
    return input_str + dt_string


def infer_dataset_name(filename):
    """Extract dataset name from filename using regex pattern."""
    pat = "^([^-_]*)"
    return pat, filename.name


def strip_extension(fname: str):
    """Remove file extension from filename."""
    exts = ".mrk.json .npy .seg.nrrd .nii.gz.nrrd .nii.gz .nii .nrrd .pt ".split(" ")
    for e in exts:
        pat = r"{}$".format(e)
        fname_stripped = re.sub(pat, "", fname)
        if fname_stripped != fname:
            return fname_stripped
    fname_stripped = fname.split(".")[0]
    return fname_stripped


def replace_extension(fname: str, new_ext: str):
    """Replace file extension with new extension (new_ext has no dot)."""
    fname_base = strip_extension(fname)
    fname_out = ".".join([fname_base, new_ext])
    return fname_out


def strip_slicer_strings(fname: str):
    """Remove Slicer-specific strings from filename."""
    pt = re.compile(r"(_\\d)?$", re.IGNORECASE)
    pt2 = re.compile(r"(_\\d)?-segment.*$", re.IGNORECASE)
    fname = fname.replace("-label", "")
    fname = fname.replace("-test", "")
    fname_cl1 = fname.replace("-tissue", "")
    fname_cl2 = re.sub(pt, "", fname_cl1)
    fname_cl3 = re.sub(pt2, "", fname_cl2)
    return fname_cl3

# %%

def str_to_path(arg_inds=None):
    """Decorator to convert string arguments to Path objects."""
    arg_inds = listify(arg_inds)

    def wrapper(func):
        def inner(*args, **kwargs):
            if len(arg_inds) == 0:
                args = [Path(arg) for arg in args]
                kwargs = {key: Path(val) for key, val in kwargs.items()}
            else:
                args = list(args)
                all_inds = range(len(args))
                args = [
                    Path(arg) if ind in arg_inds else arg
                    for ind, arg in zip(all_inds, args)
                ]
            return func(*args, **kwargs)

        return inner

    return wrapper


def path_to_str(fnc):
    """Decorator to convert Path objects to strings."""
    def inner(*args, **kwargs):
        args = map(str, args)
        for k, v in kwargs.items():
            kwargs[k] = str(v) if isinstance(v, Path) else v
        output = fnc(*args, **kwargs)
        return output

    return inner


def cleanup_fname(fname: str):
    """Clean up filename by removing extensions and slicer strings if needed."""
    fname = strip_extension(fname)

    pt_token = "(_[a-z0-9]*)"
    tokens = re.findall(pt_token, fname)
    if (
        len(tokens) > 1
    ):  # this will by pass short filenames with single digit pt id confusing with slicer suffix _\\d
        fname = strip_slicer_strings(fname)
    return fname


@str_to_path(0)
@regex_matcher(1)
def get_extension(fn):
    """Extract file extension from filename."""
    pat = r"[^\\.]*\\.(.*)"
    return pat, fn.name


def drop_digit_suffix(fname: str):
    """Remove digit suffix from filename to identify case patches."""
    pat = r"(_\\\\d{1,3})?$"
    fname_cl = re.sub(pat, "", fname)
    return fname_cl


def info_from_filename(fname: str, full_caseid=False):
    """Extract project info from filename (proj_title, case_id, date, desc)."""
    tags = ["proj_title", "case_id", "date", "desc"]
    name = cleanup_fname(fname)

    parts = name.split("_")
    output_dic = {}
    for key, val in zip(tags, parts):
        output_dic[key] = val
    if full_caseid == True:
        output_dic["case_id"] = output_dic["proj_title"] + "_" + output_dic["case_id"]
    return output_dic


def match_filenames(fname1: str, fname2: str):
    """Check if two filenames match based on their extracted info."""
    info1 = info_from_filename(fname1)
    info2 = info_from_filename(fname2)
    matched = all([val1 == val2 for val1, val2 in zip(info1.values(), info2.values())])
    return matched


def find_file(substring: str, filenames: Union[list, Path]):
    """Find file(s) containing substring in their name."""
    if isinstance(filenames, Path) and filenames.is_dir():
        filenames = filenames.glob("*")

    matching_fn = [fn for fn in filenames if substring in fn.name]
    if len(matching_fn) == 1:
        return matching_fn[0]
    elif len(matching_fn) == 0:
        raise ValueError(f"Found {len(matching_fn)} matches for {substring}")
    else:
        print("Multiple matches found")
        return matching_fn


# %%
# %%
if __name__ == "__main__":
    name = "lits_11_20111509.nii"
    name2 = "lits_11.nii"
    name3 = "lits_11_20111509_jacsde3d_thick.nii"
    pt = r"(-?label(_\\\\d)?)|(_\\\\d$)"
    name = "drli_005-label.nrrd"
    nm = strip_extension(name)
    print(nm)
    nm = strip_slicer_strings(nm)
    print(nm)

    pt = r"(-?label(_\\d)?)|(_\\d$)"
    re.sub(pt, nm)
    rpt_pt = "([a-z0-9]*_)"
    st = "litq_11_20190927_2"
    re.findall(rpt_pt, st)
    re.sub(pt, "", st)
# %%
    fname = "litq_40_20171117_1-label"
    pt_token = "(_[a-z0-9]*)"
    re.findall(pt_token, fname)
    pt = re.compile(r"(([a-z0-9]*_)*)((-label)?(_\\d)?)$", re.IGNORECASE)
    jj = re.sub(pt, "", fname)
    print(jj)
    pt = re.compile(r"(-?label(_\\d)?)|_.*(_\\d$)", re.IGNORECASE)
