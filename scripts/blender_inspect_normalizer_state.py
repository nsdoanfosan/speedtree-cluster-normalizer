import json

import bpy


def main():
    props = getattr(bpy.context.scene, "speedtree_cluster_normalizer", None)
    values = {}
    if props is not None:
        for item in props.bl_rna.properties:
            if item.identifier == "rna_type":
                continue
            value = getattr(props, item.identifier)
            if isinstance(value, bpy.types.ID):
                value = value.name
            elif not isinstance(value, (str, int, float, bool, type(None))):
                continue
            values[item.identifier] = value
    if not values:
        stored = bpy.context.scene.get("speedtree_cluster_normalizer")
        if stored is not None:
            for key in stored.keys():
                value = stored[key]
                if hasattr(value, "name"):
                    value = value.name
                if isinstance(value, (str, int, float, bool, type(None))):
                    values[key] = value
    print(
        "STCLUSTER_STATE="
        + json.dumps(
            {
                "blend": bpy.data.filepath,
                "scene": bpy.context.scene.name,
                "props": values,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
