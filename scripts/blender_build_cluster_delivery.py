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
    parser.add_argument("--blend", required=True)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--plan-base", required=True)
    parser.add_argument("--skeletal-base", required=True)
    parser.add_argument("--plan-collection", default="Atlas_Cluster_Cards")
    parser.add_argument("--generated-material", required=True)
    parser.add_argument("--source-material", required=True)
    parser.add_argument("--source-material-id", required=True, type=int)
    parser.add_argument("--atlas-albedo", required=True)
    parser.add_argument("--camera-spm", required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--target-spm", required=True)
    parser.add_argument(
        "--source-partition-mode",
        choices=(
            "AUTO",
            "PER_DEFORM_ROOT",
            "PER_CONNECTED_DEFORM_CLUSTER",
            "WHOLE_MESH",
            "COMPOSITE_PER_DEFORM_ROOT",
        ),
        default="AUTO",
    )
    parser.add_argument("--whole-mesh-pivot-object")
    parser.add_argument("--source-reference-collection", default="Cluster_Source_Reference")
    parser.add_argument("--plan-margin-ratio", type=float, default=0.015)
    parser.add_argument("--configure-send2ue", action="store_true")
    return parser.parse_args(values)


def fingerprint(path):
    candidate = Path(path).resolve()
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def required_file(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def main():
    args = parse_args()
    opened = required_file(args.blend, "Input blend")
    save_to = Path(args.save_to).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    albedo = required_file(args.atlas_albedo, "Atlas albedo")
    camera_spm = required_file(args.camera_spm, "Camera SPM")
    target_spm = required_file(args.target_spm, "Target tree SPM")
    if Path(bpy.data.filepath).resolve() != opened:
        raise RuntimeError("Blender did not open the requested input blend")
    before = fingerprint(opened)

    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    if args.configure_send2ue:
        addon_utils.enable("send2ue", default_set=False)
    source = bpy.data.objects.get(args.source_object)
    if source is None or source.type != "MESH":
        raise RuntimeError(f"Source mesh is missing: {args.source_object}")
    pivot = None
    if args.whole_mesh_pivot_object:
        pivot = bpy.data.objects.get(args.whole_mesh_pivot_object)
        if pivot is None:
            raise RuntimeError(
                f"Whole-mesh pivot object is missing: {args.whole_mesh_pivot_object}"
            )

    props = bpy.context.scene.speedtree_cluster_normalizer
    props.source_object = source
    props.source_partition_mode = args.source_partition_mode
    props.whole_mesh_pivot_object = pivot
    props.plan_base_name = args.plan_base
    props.skeletal_base_name = args.skeletal_base
    props.plan_collection = args.plan_collection
    props.plan_material_name = args.generated_material
    props.source_material_name = args.source_material
    props.source_material_id = args.source_material_id
    props.plan_margin_ratio = args.plan_margin_ratio
    props.replace_generated = True
    props.configure_send2ue = args.configure_send2ue
    props.isolate_send2ue_export = True
    props.source_reference_collection = args.source_reference_collection
    props.prepare_atlas_handoff = True
    props.atlas_albedo_path = str(albedo)
    props.atlas_camera_spm = str(camera_spm)
    props.atlas_camera_name = args.camera_name
    props.atlas_target_spm = str(target_spm)
    props.atlas_only_target = True
    props.atlas_mesh_scale = 1.0

    bpy.context.view_layer.objects.active = source
    source.select_set(True)
    result = bpy.ops.speedtree_cluster.build_normalized_assets()
    if set(result) != {"FINISHED"}:
        raise RuntimeError(f"Cluster normalizer did not finish: {sorted(result)}")
    raw_report = str(
        bpy.context.scene.get("speedtree_cluster_normalizer_last_report") or ""
    )
    if not raw_report:
        raise RuntimeError("Cluster normalizer did not persist its result contract")
    build = json.loads(raw_report)
    if not build.get("geometry_committed"):
        raise RuntimeError("Cluster normalizer did not commit normalized geometry")
    atlas = build.get("atlas_handoff") or {}
    if not atlas.get("prepared"):
        raise RuntimeError(f"Atlas handoff was not prepared: {atlas}")

    save_to.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_to), check_existing=False)
    payload = {
        "schema_version": 1,
        "kind": "speedtree_cluster_normalized_blend_build",
        "status": "complete",
        "input_blend": before,
        "output_blend": fingerprint(save_to),
        "source_object": args.source_object,
        "source_partition_mode_requested": args.source_partition_mode,
        "whole_mesh_pivot_object": args.whole_mesh_pivot_object,
        "camera_spm": fingerprint(camera_spm),
        "target_spm": fingerprint(target_spm),
        "atlas_albedo": fingerprint(albedo),
        "build": build,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("STCLUSTER_BUILD_DELIVERY=" + str(report_path))


if __name__ == "__main__":
    main()
