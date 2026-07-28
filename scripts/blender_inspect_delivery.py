import json

import bpy


def object_row(obj):
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "hide_render": bool(obj.hide_render),
        "vertices": len(obj.data.vertices) if obj.type == "MESH" else None,
        "polygons": len(obj.data.polygons) if obj.type == "MESH" else None,
        "materials": (
            [material.name for material in obj.data.materials if material]
            if obj.type == "MESH"
            else []
        ),
        "custom": {
            key: obj[key]
            for key in obj.keys()
            if isinstance(obj[key], (str, int, float, bool))
            and (
                key.startswith("speedtree_")
                or key.startswith("atlas_")
                or key.startswith("nanite_")
            )
        },
    }


def main():
    collections = {}
    for name in (
        "SpeedTree_Source",
        "Export",
        "Atlas_Cluster_Cards",
        "Atlas_Camera_Reference",
        "Atlas_Capture_Sync",
        "Atlas_Auto_Capture",
        "Cluster_Source_Reference",
    ):
        collection = bpy.data.collections.get(name)
        collections[name] = (
            {
                "hide_render": bool(collection.hide_render),
                "objects": [object_row(obj) for obj in collection.all_objects],
            }
            if collection
            else None
        )
    scene_custom = {}
    for key in bpy.context.scene.keys():
        value = bpy.context.scene[key]
        if isinstance(value, (str, int, float, bool)) and (
            key.startswith("speedtree_")
            or key.startswith("atlas_")
            or key.startswith("nanite_")
        ):
            scene_custom[key] = value
    print(
        "STCLUSTER_DELIVERY="
        + json.dumps(
            {
                "blend": bpy.data.filepath,
                "collections": collections,
                "scene_custom": scene_custom,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
