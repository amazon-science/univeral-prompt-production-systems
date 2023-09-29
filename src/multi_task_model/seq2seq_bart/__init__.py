import os
import pkgutil

dirs = [x[0] for x in os.walk(__path__[0]) if "__pycache__" not in x[0]]
__all__ = []
for module_dir in dirs:
    for loader, module_name, is_pkg in pkgutil.walk_packages([module_dir]):
        if module_name != "model":
            continue
        __all__.append(module_name)
        _module = loader.find_module(module_name).load_module(module_name)
        globals()[module_name] = _module
