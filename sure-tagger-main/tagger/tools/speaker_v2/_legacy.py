"""Load the frozen speaker_v1 implementation without importing its package.

``speaker_v1`` is the release snapshot used by ``legacy-shadow``.  Its package
initializer intentionally points at speaker_v2 schema constants, so importing
the package directly while speaker_v2 is being initialized would create a
cycle.  This loader executes individual, dependency-light snapshot modules
under private names and lets v2 override only the components that changed.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "speaker_v1"
_CACHE = {}


def load_legacy_module(stem):
    stem = str(stem)
    if stem in _CACHE:
        return _CACHE[stem]
    path = _SNAPSHOT_ROOT / (stem + ".py")
    if not path.is_file():
        raise ImportError("speaker_v1 snapshot module does not exist: %s" % path)
    module_name = "tagger.tools.speaker_v2._legacy_%s" % stem
    spec = spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load speaker_v1 snapshot module: %s" % path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _CACHE[stem] = module
    return module


def install_legacy_alias(public_name, stem):
    """Install one snapshot module at a speaker_v2 compatibility import path."""

    module = load_legacy_module(stem)
    sys.modules[str(public_name)] = module
    parent_name, attribute = str(public_name).rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attribute, module)
    return module

