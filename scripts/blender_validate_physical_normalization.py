"""Read-only validation for a saved physical cluster normalization blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import addon_utils
import bpy


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--plan-base", required=True)
    parser.add_argument("--skeletal-base", required=True)
    parser.add_argument("--card-count", type=int, required=True)
    parser.add_argument("--prototype-count", type=int, required=True)
    parser.add_argument(
        "--source-reference-collection",
        default="Cluster_Source_Reference",
    )
    return parser.parse_args(values)


def main():
    args = parse_args()
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    from speedtree_cluster_normalizer.delivery_validation import (
        validate_cluster_delivery,
    )

    expected_export_names = {
        f"{args.skeletal_base}_{index:02d}{suffix}"
        for index in range(1, args.prototype_count + 1)
        for suffix in ("", "_Armature", "_Mesh")
    }
    validation = validate_cluster_delivery(
        bpy.context.scene,
        args.plan_collection,
        args.material,
        plan_base=args.plan_base,
        expected_export_names=expected_export_names,
        expected_card_count=args.card_count,
        expected_prototype_count=args.prototype_count,
    )
    rig_rows = []
    for index in range(1, args.prototype_count + 1):
        armature_name = f"{args.skeletal_base}_{index:02d}_Armature"
        mesh_name = f"{args.skeletal_base}_{index:02d}_Mesh"
        armature = bpy.data.objects.get(armature_name)
        mesh = bpy.data.objects.get(mesh_name)
        if armature is None or armature.type != "ARMATURE":
            raise RuntimeError(f"Normalized armature is missing: {armature_name}")
        if mesh is None or mesh.type != "MESH":
            raise RuntimeError(f"Normalized mesh is missing: {mesh_name}")
        bones = list(armature.data.bones)
        groups = [group.name for group in mesh.vertex_groups]
        if len(bones) != 1 or bones[0].name != "part_root":
            raise RuntimeError(
                f"{armature_name} must contain exactly one part_root bone"
            )
        if groups != ["part_root"]:
            raise RuntimeError(
                f"{mesh_name} must contain exactly one part_root vertex group"
            )
        minimum_z = min(float(vertex.co.z) for vertex in mesh.data.vertices)
        if abs(minimum_z) > 1.0e-6:
            raise RuntimeError(
                f"{mesh_name} does not start at its normalized attachment plane"
            )
        rig_rows.append(
            {
                "armature": armature_name,
                "mesh": mesh_name,
                "bone_count": 1,
                "bone": "part_root",
                "bone_length": float(bones[0].length),
                "mesh_minimum_z": minimum_z,
            }
        )

    source_reference = bpy.data.collections.get(
        args.source_reference_collection
    )
    if (
        source_reference is None
        or not source_reference.hide_viewport
        or not source_reference.hide_render
    ):
        raise RuntimeError("Source reference collection is not hidden")
    payload = {
        "kind": "speedtree_cluster_physical_normalization_validation",
        "version": 1,
        "status": "pass",
        "blend": bpy.data.filepath,
        "delivery": validation,
        "rigs": rig_rows,
        "source_reference": {
            "collection": source_reference.name,
            "hide_viewport": bool(source_reference.hide_viewport),
            "hide_render": bool(source_reference.hide_render),
            "object_count": len(source_reference.all_objects),
        },
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STCLUSTER_NORMALIZATION_VALIDATION=" + str(report_path))


if __name__ == "__main__":
    main()
