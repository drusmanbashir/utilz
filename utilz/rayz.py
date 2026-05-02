import ray


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
