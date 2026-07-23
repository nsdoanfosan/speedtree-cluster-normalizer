from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import addon_utils
import bpy


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-blend", required=True)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--atlas-albedo", required=True)
    parser.add_argument("--camera-spm", required=True)
    parser.add_argument(
        "--camera-name",
        default="Dropped XY plane camera 2",
    )
    parser.add_argument("--target-spm", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--plan-base", required=True)
    parser.add_argument("--skeletal-base", required=True)
    parser.add_argument("--variant-count", required=True, type=int)
    parser.add_argument(
        "--prototype-count",
        type=int,
        help="Unique normalized SK prototypes/subparts; may exceed card count for composite clusters",
    )
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--generated-material", required=True)
    parser.add_argument("--source-material", required=True)
    parser.add_argument("--source-material-id", required=True, type=int)
    parser.add_argument("--source-reference-collection", default="Cluster_Source_Reference")
    return parser.parse_args(values)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    normalized_blend = Path(args.normalized_blend).expanduser().resolve()
    save_to = Path(args.save_to).expanduser().resolve()
    albedo = Path(args.atlas_albedo).expanduser().resolve()
    camera_spm = Path(args.camera_spm).expanduser().resolve()
    target_spm = Path(args.target_spm).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if Path(bpy.data.filepath).resolve() != normalized_blend:
        raise RuntimeError("Blender did not open the requested normalized blend")
    if not albedo.is_file() or not camera_spm.is_file() or not target_spm.is_file():
        raise RuntimeError("Atlas albedo, explicit camera SPM, and target SPM must already exist")
    if not args.camera_name.strip():
        raise RuntimeError("Explicit camera name cannot be empty")
    if args.variant_count < 1:
        raise RuntimeError("Variant count must be greater than zero")
    prototype_count = args.prototype_count or args.variant_count
    if prototype_count < 1:
        raise RuntimeError("Prototype count must be greater than zero")
    if args.generated_material != args.source_material:
        raise RuntimeError(
            "Delivery must adopt the existing source material name in place"
        )

    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    from speedtree_cluster_normalizer.delivery_validation import (
        validate_camera_uv_delivery,
    )
    export_collection = bpy.data.collections.get("Export")
    if export_collection is None:
        raise RuntimeError("Normalized blend has no Export collection")
    export_objects = list(export_collection.objects)
    expected_names = {
        f"{args.skeletal_base}_{index:02d}{suffix}"
        for index in range(1, prototype_count + 1)
        for suffix in ("", "_Armature", "_Mesh")
    }
    actual_names = {obj.name for obj in export_objects}
    if actual_names != expected_names:
        raise RuntimeError(
            "Delivery Export collection is not isolated to the expected prototypes: "
            + ", ".join(sorted(actual_names))
        )
    plans = [
        bpy.data.objects.get(f"{args.plan_base}_{index:02d}")
        for index in range(1, args.variant_count + 1)
    ]
    if any(plan is None or plan.type != "MESH" for plan in plans):
        raise RuntimeError("Delivery blend is missing one or more normalized plans")
    camera_delivery = validate_camera_uv_delivery(
        bpy.context.scene,
        args.plan_collection,
        args.generated_material,
        plan_base=args.plan_base,
        expected_export_names=expected_names,
        expected_camera_spm=camera_spm,
        expected_camera_name=args.camera_name,
        expected_tree_spm=target_spm,
        expected_albedo_path=albedo,
        expected_material_id=args.source_material_id,
        expected_card_count=args.variant_count,
        expected_prototype_count=prototype_count,
    )

    save_to.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_to), check_existing=False)

    from atlas_leaf_mesh_builder.integration_api import configure_external_plan_target
    from atlas_leaf_mesh_builder.target_registry import registry_path_for_blend

    cluster = bpy.context.scene.speedtree_cluster_normalizer
    source_object = bpy.data.objects.get(args.source_object)
    if source_object is None or source_object.type != "MESH":
        raise RuntimeError(f"Delivery source mesh is missing: {args.source_object}")
    cluster.source_object = source_object
    cluster.plan_base_name = args.plan_base
    cluster.skeletal_base_name = args.skeletal_base
    cluster.plan_collection = args.plan_collection
    cluster.plan_material_name = args.generated_material
    cluster.source_material_name = args.source_material
    cluster.source_material_id = args.source_material_id or 0
    cluster.isolate_send2ue_export = True
    cluster.source_reference_collection = args.source_reference_collection
    cluster.prepare_atlas_handoff = True
    cluster.atlas_albedo_path = str(albedo)
    cluster.atlas_camera_spm = str(camera_spm)
    cluster.atlas_camera_name = args.camera_name
    cluster.atlas_target_spm = str(target_spm)
    cluster.atlas_only_target = True
    cluster.atlas_mesh_scale = 1.0

    configured = configure_external_plan_target(
        bpy.context.scene.atlas_leaf_builder,
        collection_name=args.plan_collection,
        generated_material_name=args.generated_material,
        source_material_name=args.source_material,
        albedo_path=str(albedo),
        target_spm=str(target_spm),
        source_material_id=args.source_material_id,
        adopt_source_material=True,
        only_target=True,
        mesh_geometry_scale=1.0,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(save_to), check_existing=False)
    registry_path = registry_path_for_blend(save_to)
    if not registry_path.is_file():
        raise RuntimeError(f"Atlas target sidecar was not written: {registry_path}")

    payload = {
        "delivered_blend": str(save_to),
        "delivered_sha256": sha256(save_to),
        "export_objects": sorted(actual_names),
        "plans": [plan.name for plan in plans],
        "card_count": args.variant_count,
        "prototype_count": prototype_count,
        "camera_uv_delivery": camera_delivery,
        "explicit_camera_spm": str(camera_spm),
        "explicit_camera_name": args.camera_name,
        "atlas_configuration": configured,
        "atlas_target_registry": str(registry_path),
        "target_spm_sha256_before_build": sha256(target_spm),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STCLUSTER_FINAL_DELIVERY=" + str(report_path))


if __name__ == "__main__":
    main()
