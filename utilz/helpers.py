# %%
from typing import Union
import pandas as pd
import logging, os
import collections
import pprint
import re
from multiprocessing import Pool
from pathlib import Path

import ipdb
from ipdb.__main__ import get_ipython
import numpy as np
import torch
from tqdm import tqdm as tqdm_ip

from utilz.dictopts import *
from utilz.fileio import is_img_file, load_dict, str_to_path
from utilz.string import cleanup_fname, dec_to_str, info_from_filename, path_to_str, regex_matcher
from functools import wraps
from time import time
tr = ipdb.set_trace
import gc, ray
# from utilz.fileio import *
# %%

def set_autoreload():
    # gals = globals()
    # print(gals.keys)
    if "get_ipython" in globals() and not any(['APPLAUNCHER_0_PATH' in os.environ, ray.is_initialized()]): 
        print("setting autoreload")
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.run_line_magic("load_ext", "autoreload")
            ipython.run_line_magic("autoreload", "2")

@str_to_path(0)
def create_df_from_folder(folder):
    images_fldr = folder/("images")
    lms_fldr = folder/("lms")
    image_fns = list(images_fldr.glob("*"))
    lm_fns =  list(lms_fldr.glob("*"))
    dicis = []
    for img_fn in image_fns:
        lm_fn = find_matching_fn(img_fn,lm_fns,tags='case_id')
        case_id = info_from_filename(lm_fn.name,full_caseid=True)['case_id']
        dici= {"image":img_fn, "lm":lm_fn, "case_id":case_id}
        dicis.append(dici)
    df = pd.DataFrame(dicis)
    return df



def test_modified(filename,ndays:int= 1):
    '''
    returns true if file was modified in last ndays
    '''
    
    delta = time() - os.path.getmtime(filename)
    delta = delta / (60*60*24)
    if delta < ndays:
        return True
    return False

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


def get_pbar():
    if 'get_ipython' in globals():
        if get_ipython().__class__.__name__ == 'TerminalInteractiveShell' :
            return tqdm_ip
        else:
            return  tqdm_ip


pbar=get_pbar()
def range_inclusive(start, end):
     return range(start, end+1)
def multiply_lists(a,b):
    return [aa*bb for aa, bb in zip(a,b)]
 
class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass   

class LazyDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def abs_list(numlist:list):
    return [abs(num) for num in numlist]


def slice_list(listi, start_end: list):
    return listi[start_end[0] : start_end[1]]

def chunks (listi, n):
    chunk_size = round(len(listi)/n)
    sld = [[x*chunk_size,(x+1)*chunk_size] for x in range(0,n-1)]
    sld.append([(n-1)*chunk_size,None])
    for s in sld:
        yield (listi[s[0]:s[1]])


def merge_dicts(d1, d2):
    outdict={}
    for k, v in d1.items():
        if not k in d2.keys():
            outdict[k]=d1[k]
        else:
            if isinstance(v, collections.abc.Mapping):
                outdict[k] = merge_dicts(d1[k],d2[k])
            else:
                outdict[k] = d1[k]+ d2[k]
    return outdict

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


def folder_name_from_list(prefix:str,parent_folder:Path,values_list=None, suffix=None):
        assert prefix in ("spc","dim", "sze"), "Choose a valid prefix between spc(for spacings) and dim(for patch size)"
        add_zeros=3 if prefix=='spc' else 0
        output= [dec_to_str(val,add_zeros) for val in values_list]
        subfolder =  "_".join([prefix]+output)
        if suffix:
            subfolder += "_"+suffix
        return parent_folder/subfolder

def spacing_from_folder_name(prefix:str,folder_name:str):
    assert prefix in ("spc","dim"), "Choose a valid prefix between spc(for spacings) and dim(for patch size)"
    name = Path(folder_name).name
    name = name.replace(prefix+"_","")
    pcs = name.split("_")
    output= [float(pc)*1e-2 for pc in pcs]
    return output

    
# ========================== wrapper functions ====================


def pp(obj, *args,**kwargs):
    pp = pprint.PrettyPrinter(*args,**kwargs)
    pp.pprint(obj)

def str_to_list(fnc):
    def inner(input:str):
        tmp = input.split(",")
        final = [fnc(g) for g in tmp]
        return final
    return inner

def ask_proceed(statement=None):
    def wrapper(func):
        def inner(*args, **kwargs):
            if statement:
                print(statement)

            def _ask():
                decision = input("Proceed (Y/y) or Skip (N/n)?: ")
                return decision

            decision = _ask()
            if not decision in "YyNn":
                decision = _ask()
            if decision.lower() == "y":
                func(*args, **kwargs)

        return inner

    return wrapper

# ====================================================================================================
@str_to_list
def str_to_list_float(input):
    return float(input)

@str_to_list
def str_to_list_int(input):
    return int(input)

def purge_cuda(learn):
    gc.collect()
    torch.cuda.empty_cache()

def get_list_input(text:str="",fnc=str_to_list_float):
    str_list = input (text)
    return fnc(str_list)

def multiprocess_multiarg(func,arguments, num_processes=8,multiprocess=True,debug=False,progress_bar=True, logname = None):
    results=[]
    if multiprocess==False or debug==True:
        for res in pbar(arguments,total=len(arguments)):
            if debug==True:
                if logname:

                    logging.basicConfig(filename=logname,
                                        filemode='w',
                                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                        datefmt='%H:%M:%S',
                                        level=logging.DEBUG)
                    logging.info(" Processing {} .".format(res))
                # tr()
            results.append(func(*res,))
    else:
        p = Pool(num_processes)
        jobs = [p.apply_async(func=func, args=(*argument, )) for argument in arguments]
        p.close()
        pbar_fnc = get_pbar() if progress_bar==True else lambda x: x
        for job in pbar_fnc(jobs):
                results.append(job.get())
    return results

  

def get_available_device(max_memory=0.8)->int:
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered availablj

    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available=-1
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available 

def resolve_device (device):
    assert device in ['cuda', 'cpu',0,1], "Device has to be either 'cpu' , 'gpu' or an integer"
    if device == 'cpu': return 'cpu'
    if device == 'cuda': device = 0
    device = torch.device('cuda:{}'.format(device))
    return device


def set_cuda_device(device_id=None):
        if device_id == None:        device_id= get_available_device()
        torch.cuda.set_device(device_id)



def get_last_n_dims(img,n):
    '''
    n = number of dimensions to return
    '''
    if img.ndim <=n:
        return img
    slc = (0,)*(img.ndim-n)
    slc+=(slice(None),)*n
    return img[slc]

def sigmoid(x:np.ndarray):
    return 1/(1+np.exp(-x))


def convert_sigmoid_to_label(y,threshold = 0.5):
    y[y>=threshold]=1
    y[y<threshold]=0
    return y.int()

@path_to_str
@regex_matcher()
def match_filename_with_case_id(project_title, case_id,filename):
                pat = project_title+"_"+case_id+r"_?\."
                return pat,filename

def project_title_from_folder(folder_name):
    if isinstance(folder_name,Path): folder_name = folder_name.name
    pat = re.compile(r"(Task\d{0,5}_)?(\w*)",re.IGNORECASE)
    result= re.search(pat,folder_name)
    return result.groups()[-1]



#
# def FileSplitter_ub(json_filename,fold=0):
#     "Split `items` by providing file `fname` (contains names of valid items separated by newline)."
#     with open(json_filename,"r") as f:
#         validation_folds=json.load(f)
#     valid = validation_folds['fold_'+str(int(fold))]
#     def _func(x): return x.name in valid
#     def _inner(o): 
#         return FuncSplitter(_func)(o)
#     return _inner
#
# def EndSplitterShort(size = 48,valid_pct=0.2, valid_last=True):
#     "Create function that splits `items` between train/val with `valid_pct` at the end if `valid_last` else at the start. Useful for ordered data."
#     assert 0<valid_pct<1, "valid_pct must be in (0,1)"
#     def _inner(o):
#         o = o[:size]
#         idxs = np.arange(len(o))
#         cut = int(valid_pct * len(o))
#         return (idxs[:-cut], idxs[-cut:]) if valid_last else (idxs[cut:],idxs[:cut])
#     return _inner
#

# def create_train_valid_test_lists_from_filenames(project_title, files_list, json_filename,pct_test=0.1,pct_valid=0.2,shuffle=False):
#     pct_valid = 0.2
#     case_ids = [get_case_id_from_filename(project_title,fn) for fn in files_list]
#     if shuffle==True: 
#         print("Shuffling all files")
#         random.shuffle(case_ids)
#     else:    
#         print("Files are in sorted order")
#         case_ids.sort()
#     final_dict= {"all_cases":case_ids} 
#     n_test= int(pct_test*len(case_ids))
#     n_valid = int(pct_valid*len(case_ids))
#     len(case_ids)-n_test-n_valid
#     cases_test =case_ids[:n_test]
#     final_dict.update({"test_cases": cases_test})
#     cases_train_valid = case_ids[n_test:]
#     folds = int(1/pct_valid)
#     n_valid_per_fold =  math.ceil(len(cases_train_valid)*pct_valid)
#     print("Given proportion {0} of validation files yield {1} folds".format(pct_valid,folds))
#     slices = [slice(fold*n_valid_per_fold,(fold+1)*n_valid_per_fold) for fold in range(folds)]
#     val_cases_per_fold = [cases_train_valid[slice] for slice in slices]
#     for n in range(folds):
#         train_cases_fold = list(set(cases_train_valid)-set(val_cases_per_fold[n]))
#         fold = {'fold_{}'.format(n):{'train':train_cases_fold, 'valid':val_cases_per_fold[n]}}
#         final_dict.update(fold)
#     print("Saving folds to {}  ..".format(json_filename))
#     save_dict(final_dict,json_filename)

def get_train_valid_test_lists_from_json(project_title, fold, json_fname, image_folder,ext=".pt"):
        all_folds= load_dict(json_fname)
        train_case_ids, validation_case_ids, test_case_ids,= all_folds['fold_'+str(int(fold))]['train'], all_folds['fold_'+str(int(fold))]['valid'],all_folds['test_cases']
        all_files =list(image_folder.glob("*{}".format(ext)))
        train_files , valid_files ,test_files=[],[], []
        for cases_list,output_list in zip([train_case_ids,validation_case_ids,test_case_ids],[train_files,valid_files,test_files]):
            for case_id in cases_list:
                    matched_files = [fn for fn in all_files if match_filename_with_case_id(project_title,case_id,fn)]
                    if len(matched_files)>1:
                        tr()
                    output_list.append(matched_files)
        return train_files,valid_files,test_files



@str_to_path(0)
def find_matching_fn(src_fn:Path,mask_fnames:Union[list,Path],tags='case_id'):
        allowed_tags = [ "case_id", "all"] # all means identical filename
        assert tags in allowed_tags, "Allowed tags are {0}".format(allowed_tags)
        if isinstance(mask_fnames,Path) and mask_fnames.is_dir():
            mask_fnames = list(mask_fnames.glob("*"))
            mask_fnames = [fn for fn in mask_fnames if is_img_file(fn)]
        assert (len(mask_fnames) > 0), "List of candidate filenames is empty"
        src_fn = cleanup_fname(src_fn.name)
        matching_mask_fns=[]
        for mask_fn in mask_fnames:
            if tags == 'all':
                mask_fn_clean = cleanup_fname(mask_fn.name)
                if mask_fn_clean==src_fn:
                    matching_mask_fns.append(mask_fn)
            else:
                cid = info_from_filename(src_fn,full_caseid=True)['case_id']
                cid_mask = info_from_filename(mask_fn.name,full_caseid=True)['case_id']
                if cid_mask == cid:
                    matching_mask_fns.append(mask_fn)
        if len(matching_mask_fns)==1: return matching_mask_fns[0]
        # elif len(matching_mask_fns)==0: return None
        else: raise Exception("Multiple or None files matching {0} found:\n{1}".format(src_fn, matching_mask_fns))


def get_fileslist_from_path(path:Path, ext:str = ".pt"):
    return list(path.glob("*"+ext))

def make_channels (base_ch=16,depth=5):
    base_multiplier = int(np.log2(base_ch))
    return [1,]+[2**n for n in range(base_multiplier,base_multiplier+depth)]

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False 


def write_file_or_not(output_filename, overwrite=True):
    if Path(output_filename).exists() and overwrite==False:
                print("File exists. Skipping {}".format(output_filename))
                return False
    else:
        return True

def write_files_or_not(output_filenames, overwrite=True):
    return list(map(write_file_or_not,output_filenames,[overwrite,]*len(output_filenames)))

# %%
if __name__=="__main__":

    from label_analysis.helpers import get_labels, relabel
    dd = load_dict("/home/ub/datasets/preprocessed/lits/patches/spc_100_100_200/dim_256_256_128/bboxes_info.pkl")

    fn = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/crc_CRC154_20140712_ABDOMEN.nii.gz")
    fldr = Path("/s/xnat_shadow/crc/lms_staging/")
    fn_out = fldr/fn.name
    import SimpleITK as sitk
    lm = sitk.ReadImage(str(fn))
    get_labels(lm)
    lm = relabel(lm,{3:2})
    sitk.WriteImage(lm,str(fn_out))

    dd[0]
    from fran.utils.common import *

