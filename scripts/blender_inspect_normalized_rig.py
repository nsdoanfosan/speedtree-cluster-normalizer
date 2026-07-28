"""Write a compact normalized-rig report for one already-open Blender file."""

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    return parser.parse_args(values)


def local_bounds(obj):
    points = [vertex.co for vertex in obj.data.vertices]
    if not points:
        return None
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "center": [
            (minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)
        ],
        "size": [
            maximum[axis] - minimum[axis] for axis in range(3)
        ],
    }


def main():
    args = parse_args()
    rows = []
    for obj in bpy.data.objects:
        role = obj.get("speedtree_cluster_asset_role")
        if role not in {
            "skeletal_armature",
            "skeletal_mesh",
            "speedtree_plan",
            "send2ue_pivot",
        }:
            continue
        row = {
            "name": obj.name,
            "type": obj.type,
            "role": role,
            "location": list(obj.location),
            "parent": obj.parent.name if obj.parent else None,
            "collections": [collection.name for collection in obj.users_collection],
            "matrix_world": [list(values) for values in obj.matrix_world],
        }
        if obj.type == "ARMATURE":
            row["bones"] = [
                {
                    "name": bone.name,
                    "head": list(bone.head_local),
                    "tail": list(bone.tail_local),
                    "length": bone.length,
                }
                for bone in obj.data.bones
            ]
        if obj.type == "MESH":
            row["bounds"] = local_bounds(obj)
            row["groups"] = [group.name for group in obj.vertex_groups]
            row["prototype_index"] = obj.get(
                "speedtree_cluster_prototype_index"
            )
            row["source_bone"] = obj.get("speedtree_cluster_source_bone")
            row["xml_attachment"] = obj.get(
                "speedtree_cluster_xml_attachment"
            )
            row["root_lock"] = obj.get(
                "speedtree_cluster_plan_root_lock"
            )
        rows.append(row)
    payload = {
        "blend": bpy.data.filepath,
        "rows": rows,
        "collections": [
            {
                "name": collection.name,
                "hide_viewport": bool(collection.hide_viewport),
                "hide_render": bool(collection.hide_render),
                "object_count": len(collection.all_objects),
            }
            for collection in bpy.data.collections
        ],
        "armatures": [
            {
                "name": obj.name,
                "bone_count": len(obj.data.bones),
                "collections": [
                    collection.name for collection in obj.users_collection
                ],
                "generated": bool(obj.get("speedtree_cluster_generated")),
            }
            for obj in bpy.data.objects
            if obj.type == "ARMATURE"
        ],
    }
    report = Path(args.report).expanduser().absolute()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
