from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix


WORKFLOW = "PHYSICAL_DIRECT_CAPTURE"


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stage", "production"), required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--role", choices=("branch", "leaf", "leaf_side"), required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-xml", required=True)
    parser.add_argument("--plan-base", required=True)
    parser.add_argument("--skeletal-base", required=True)
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--material-id", required=True, type=int)
    parser.add_argument("--capture-output-dir")
    parser.add_argument("--capture-prefix")
    parser.add_argument("--capture-resolution", type=int, default=1024)
    parser.add_argument("--capture-source-collection", default="SpeedTree_Source")
    parser.add_argument("--capture-padding-ratio", type=float, default=0.04)
    parser.add_argument("--capture-plane", choices=("XY", "YZ"), required=True)
    parser.add_argument("--capture-target-meters", type=float, default=0.1)
    parser.add_argument("--target-spm")
    parser.add_argument("--unit-probe")
    parser.add_argument(
        "--source-partition-mode",
        choices=(
            "PER_CONNECTED_DEFORM_CLUSTER",
            "PER_DEFORM_ROOT",
            "WHOLE_MESH",
            "COMPOSITE_PER_DEFORM_ROOT",
        ),
        default="PER_CONNECTED_DEFORM_CLUSTER",
    )
    parser.add_argument("--plan-margin-ratio", type=float, default=0.01)
    parser.add_argument("--plan-refinement-levels", type=int, default=1)
    parser.add_argument("--source-reference-collection", default="Cluster_Source_Reference")
    parser.add_argument("--configure-send2ue", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(values)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(candidate),
    }


def required_file(value, label):
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def identity_object_names(names):
    failed = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.matrix_world != Matrix.Identity(4):
            failed.append(name)
    if failed:
        raise RuntimeError(
            "Normalized output transforms are not Identity: " + ", ".join(failed)
        )


def main():
    args = parse_args()
    if not args.apply:
        raise RuntimeError("Refusing to mutate a blend without explicit --apply")
    opened = required_file(args.blend, "Input blend")
    source_xml = required_file(args.source_xml, "Source XML")
    save_to = Path(args.save_to).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if Path(bpy.data.filepath).resolve() != opened:
        raise RuntimeError("Blender did not open the requested input blend")
    input_before = fingerprint(opened)
    if args.capture_target_meters <= 0.0 or not math.isfinite(
        args.capture_target_meters
    ):
        raise RuntimeError("Physical capture target must be finite and positive")
    if args.role == "leaf_side" and args.capture_plane != "YZ":
        raise RuntimeError("Leaf Side production requires the explicit YZ 90-degree capture")
    if args.role != "leaf_side" and args.capture_plane != "XY":
        raise RuntimeError("Branch/Leaf production requires the XY front capture")

    production = args.mode == "production"
    capture_output = None
    target_spm = None
    unit_probe = None
    if production:
        capture_output = Path(str(args.capture_output_dir or "")).expanduser().resolve()
        if not args.capture_prefix:
            raise RuntimeError("Production requires a capture prefix")
        target_spm = required_file(args.target_spm, "Target SPM")
        unit_probe = required_file(args.unit_probe, "Verified unit probe")
    elif any(
        value
        for value in (
            args.capture_output_dir,
            args.capture_prefix,
            args.target_spm,
            args.unit_probe,
        )
    ):
        raise RuntimeError(
            "Stage mode does not bake maps or mutate SpeedTree; remove production arguments"
        )

    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    if production:
        addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    if args.configure_send2ue:
        addon_utils.enable("send2ue", default_set=False)
    if not hasattr(bpy.types.Scene, "speedtree_cluster_normalizer"):
        raise RuntimeError("SpeedTree Cluster Normalizer did not register")
    if production and not hasattr(bpy.types.Scene, "atlas_leaf_builder"):
        raise RuntimeError("Atlas Leaf Mesh Builder did not register")

    scene = bpy.context.scene
    units = scene.unit_settings
    if units.system != "METRIC" or abs(float(units.scale_length) - 1.0) > 1.0e-12:
        raise RuntimeError(
            "Physical production requires METRIC scale_length=1.0 "
            f"(got {units.system}, {units.scale_length})"
        )
    source = bpy.data.objects.get(args.source_object)
    if source is None or source.type != "MESH":
        raise RuntimeError(f"Source mesh is missing: {args.source_object}")

    props = scene.speedtree_cluster_normalizer
    props.workflow_mode = WORKFLOW
    props.source_object = source
    props.source_xml_path = str(source_xml)
    props.source_partition_mode = args.source_partition_mode
    props.plan_base_name = args.plan_base
    props.skeletal_base_name = args.skeletal_base
    props.plan_collection = args.plan_collection
    props.plan_material_name = args.material
    props.source_material_name = args.material
    props.source_material_id = args.material_id
    props.plan_margin_ratio = args.plan_margin_ratio
    props.plan_refinement_levels = args.plan_refinement_levels
    props.replace_generated = True
    props.configure_send2ue = bool(args.configure_send2ue)
    props.isolate_send2ue_export = True
    props.source_reference_collection = args.source_reference_collection
    props.capture_source_collection = args.capture_source_collection
    props.capture_padding_ratio = args.capture_padding_ratio
    props.capture_plane = args.capture_plane
    props.capture_target_meters = args.capture_target_meters
    props.prepare_atlas_handoff = production

    capture_manifest = None
    target_before = None
    if production:
        capture_output.mkdir(parents=True, exist_ok=True)
        props.capture_output_dir = str(capture_output)
        props.capture_prefix = args.capture_prefix
        props.capture_resolution = args.capture_resolution
        capture_result = bpy.ops.speedtree_cluster.bake_capture_maps()
        if set(capture_result) != {"FINISHED"}:
            raise RuntimeError(
                f"Physical capture operator did not finish: {sorted(capture_result)}"
            )
        capture_manifest = required_file(
            props.capture_manifest_path,
            "Physical capture manifest",
        )
        props.atlas_albedo_path = str(capture_output / f"{args.capture_prefix}.tga")
        props.atlas_target_spm = str(target_spm)
        props.atlas_only_target = True
        props.unit_probe_contract_path = str(unit_probe)
        target_before = fingerprint(target_spm)

    bpy.context.view_layer.objects.active = source
    source.select_set(True)
    build_result = bpy.ops.speedtree_cluster.build_normalized_assets()
    if set(build_result) != {"FINISHED"}:
        raise RuntimeError(
            f"Physical normalizer operator did not finish: {sorted(build_result)}"
        )
    raw_report = str(scene.get("speedtree_cluster_normalizer_last_report") or "")
    if not raw_report:
        raise RuntimeError("Physical normalizer did not persist its report")
    build = json.loads(raw_report)
    if (
        build.get("workflow_mode") != WORKFLOW
        or build.get("camera_dependency") != "none"
        or build.get("direct_uv_source")
        != "same_blender_physical_capture_projection"
        or build.get("generator_size_policy")
        != "preserve_user_authored_leaf_and_frond_dimensions"
        or not build.get("geometry_committed")
    ):
        raise RuntimeError("Physical production report is incomplete or legacy-dependent")
    frame = (build.get("physical_capture_contract") or {}).get("frame") or {}
    if (
        abs(float(frame.get("target_meters", [0.0])[0]) - 0.1) > 1.0e-6
        or max(
            float(frame.get("content_width", math.inf)),
            float(frame.get("content_height", math.inf)),
        )
        > float(frame.get("width", 0.0)) + 1.0e-8
    ):
        raise RuntimeError(
            "Normalized source does not fit the 0.1m capture frame: "
            + json.dumps(
                {
                    "target_meters": frame.get("target_meters"),
                    "target_blender_units": frame.get("target_blender_units"),
                    "width": frame.get("width"),
                    "height": frame.get("height"),
                    "content_width": frame.get("content_width"),
                    "content_height": frame.get("content_height"),
                },
                sort_keys=True,
            )
        )

    object_names = []
    for row in build.get("variants") or []:
        object_names.extend((row["pivot"], row["armature"], row["mesh"], row["plan"]))
        if row.get("object_transforms_identity") is not True:
            raise RuntimeError(f"Variant transforms are not baked: {row['plan']}")
        if row.get("plan_covers_projection") is not True:
            raise RuntimeError(f"Plan does not cover its source projection: {row['plan']}")
        transfer = row.get("plan_uv_transfer") or {}
        if transfer.get("policy") != "direct_physical_capture_projection":
            raise RuntimeError(f"Plan UV is not direct capture UV: {row['plan']}")
    identity_object_names(object_names)

    atlas_result = None
    atlas_manifest = None
    if production:
        handoff = build.get("atlas_handoff") or {}
        if not handoff.get("prepared"):
            raise RuntimeError(f"Atlas handoff was not prepared: {handoff}")
        atlas_result = bpy.ops.atlas_leaf.build_speedtree_spm()
        if set(atlas_result) != {"FINISHED"}:
            raise RuntimeError(f"Atlas SPM build did not finish: {sorted(atlas_result)}")
        atlas_manifest = (
            target_spm.parent
            / ".atlas_leaf_speedtree_targets"
            / f"{target_spm.stem}.json"
        )
        atlas_manifest = required_file(atlas_manifest, "Atlas target manifest")

    save_to.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_to), check_existing=False)
    if Path(bpy.data.filepath).resolve() != save_to:
        raise RuntimeError("Blender did not save the requested output blend")

    payload = {
        "kind": "speedtree_cluster_physical_capture_delivery",
        "version": 1,
        "status": "ready",
        "mode": args.mode,
        "role": args.role,
        "workflow_mode": WORKFLOW,
        "input_blend": input_before,
        "output_blend": fingerprint(save_to),
        "source_xml": fingerprint(source_xml),
        "source_object": args.source_object,
        "material": {"name": args.material, "id": args.material_id},
        "capture_plane": args.capture_plane,
        "physical_target_meters": args.capture_target_meters,
        "build": build,
        "capture_manifest": (
            fingerprint(capture_manifest) if capture_manifest is not None else None
        ),
        "unit_probe_contract": (
            json.loads(unit_probe.read_text(encoding="utf-8"))
            if unit_probe is not None
            else None
        ),
        "target_spm_before": target_before,
        "target_spm_after": (
            fingerprint(target_spm) if target_spm is not None else None
        ),
        "atlas_operator_result": (
            sorted(atlas_result) if atlas_result is not None else None
        ),
        "atlas_manifest": (
            fingerprint(atlas_manifest) if atlas_manifest is not None else None
        ),
        "manual_art_direction_adjustment_required": production,
        "generator_dimensions_automatically_modified": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STCLUSTER_PHYSICAL_DELIVERY=" + str(report_path))


if __name__ == "__main__":
    main()
