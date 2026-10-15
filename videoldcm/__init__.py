"""VideoLDCM depth completion models and refinement utilities."""

import sys as _sys
from types import ModuleType as _ModuleType

__all__ = ["videoldcm"]


def __getattr__(name):
    # Poisson-only imports do not need the optional refinement model packages.
    if name == "videoldcm":
        from .videoldcm import videoldcm
        globals()[name] = videoldcm
        return videoldcm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _VideoLDCMPackage(_ModuleType):
    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        # Import machinery assigns the same-named submodule to its parent.
        # Preserve the public class export even when that submodule loads first.
        if name == "videoldcm" and isinstance(value, _ModuleType):
            value = value.videoldcm
            super().__setattr__(name, value)
        return value


_sys.modules[__name__].__class__ = _VideoLDCMPackage
