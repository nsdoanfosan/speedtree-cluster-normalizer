bl_info = {
    "name": "SpeedTree Cluster Normalizer",
    "author": "Codex for PARK",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Cluster Normalize",
    "description": "Normalize skinned SpeedTree cluster parts and build covering plans.",
    "category": "Object",
}

import importlib
import sys

import bpy


_SUBMODULE_NAMES = ("normalization", "props", "atlas_handoff", "operators")


def _load_submodules():
    loaded = {}
    for name in _SUBMODULE_NAMES:
        full_name = f"{__name__}.{name}"
        if full_name in sys.modules:
            loaded[name] = importlib.reload(sys.modules[full_name])
        else:
            loaded[name] = importlib.import_module(f".{name}", __name__)
    return loaded


_modules = _load_submodules()
STCLUSTER_Properties = _modules["props"].STCLUSTER_Properties
STCLUSTER_OT_build = _modules["operators"].STCLUSTER_OT_build
STCLUSTER_PT_panel = _modules["operators"].STCLUSTER_PT_panel


classes = (STCLUSTER_Properties, STCLUSTER_OT_build, STCLUSTER_PT_panel)


def unregister_class_if_registered(class_name):
    cls = getattr(bpy.types, class_name, None)
    if cls is not None:
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


def register():
    if hasattr(bpy.types.Scene, "speedtree_cluster_normalizer"):
        del bpy.types.Scene.speedtree_cluster_normalizer
    for cls in classes:
        unregister_class_if_registered(cls.__name__)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.speedtree_cluster_normalizer = bpy.props.PointerProperty(
        type=STCLUSTER_Properties
    )


def unregister():
    if hasattr(bpy.types.Scene, "speedtree_cluster_normalizer"):
        del bpy.types.Scene.speedtree_cluster_normalizer
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
