from contextlib import nullcontext

import ray
from tqdm.auto import tqdm


def collect_ray_refs(
    refs,
    *,
    weights=None,
    use_tqdm=True,
    total=None,
    desc=None,
    unit="task",
    tqdm_cls=None,
):
    ref_list = list(refs)
    weights = [1] * len(ref_list) if weights is None else list(weights)
    total = sum(weights) if total is None else total
    results = [None] * len(ref_list)
    ref_to_index = {ref: idx for idx, ref in enumerate(ref_list)}
    pending_refs = ref_list.copy()
    tqdm_cls = tqdm if tqdm_cls is None else tqdm_cls
    progress = (
        tqdm_cls(total=total, desc=desc, unit=unit)
        if use_tqdm and ref_list
        else nullcontext()
    )
    with progress as pbar:
        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1)
            ready_ref = ready_refs[0]
            idx = ref_to_index[ready_ref]
            results[idx] = ray.get(ready_ref)
            if use_tqdm:
                pbar.update(weights[idx])
    return results


def shutdown_actors(actors, timeout: float = 5) -> None:
    actor_list = [actor for actor in actors if actor is not None]
    if len(actor_list) == 0:
        return
    shutdown_refs = [actor.__ray_terminate__.remote() for actor in actor_list]
    _, pending_refs = ray.wait(
        shutdown_refs,
        num_returns=len(shutdown_refs),
        timeout=timeout,
    )
    if len(pending_refs) == 0:
        return
    pending_ids = {ref.hex() for ref in pending_refs}
    for actor, shutdown_ref in zip(actor_list, shutdown_refs):
        if shutdown_ref.hex() in pending_ids:
            ray.kill(actor, no_restart=True)
