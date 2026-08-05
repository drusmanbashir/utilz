"""utilz — flat modules; import from submodules (utilz.stringz, utilz.api_keys, …).

Keep this package init free of heavy imports so agent-env / LSP can load
neutral modules (api_keys, stringz, listify, …) without pulling SimpleITK/torch.
Imaging / ML modules (fileio, imageviewers, overlay_grid_gif, …) stay at the
same paths — import them only from the dl/fran env.
"""
