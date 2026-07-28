"""Rebuild and validate physical cluster normalization without writing any SPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import addon_utils
import bpy


WORKFLOW = "PHYSICAL_DIRECT_CAPTURE"


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-xml", required=True)
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--plan-base", required=True)
    parser.add_argument("--skeletal-base", required=True)
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--capture-source-collection", default="SpeedTree_Source")
    parser.add_argument("--capture-plane", choices=("XY", "YZ"), required=True)
    parser.add_argument("--capture-target-meters", type=float, default=0.1)
    parser.add_argument("--capture-padding-ratio", type=float, default=0.04)
    parser.add_argument(
        "--source-partition-mode",
        choices=("PER_CONNECTED_DEFORM_CLUSTER", "PER_DEFORM_ROOT"),
        default="PER_CONNECTED_DEFORM_CLUSTER",
    )
    parser.add_argument("--plan-margin-ratio", type=float, default=0.01)
    parser.add_argument("--plan-refinement-levels", type=int, default=1)
    parser.add_argument(
        "--source-reference-collection",
        default="Cluster_Source_Reference",
    )
    parser.add_argument("--guard-spm", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(values)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_file(value, label):
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def fingerprint(path):
    path = required_file(path, "Fingerprint input")
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def main():
    args = parse_args()
    if not args.apply:
        raise RuntimeError("Refusing to rebuild without explicit --apply")
    opened = required_file(args.blend, "Input blend")
    source_xml = required_file(args.source_xml, "Source XML")
    capture_manifest = required_file(
        args.capture_manifest,
        "Physical capture manifest",
    )
    save_to = Path(args.save_to).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if Path(bpy.data.filepath).resolve() != opened:
        raise RuntimeError("Blender did not open the requested input blend")
    if (
        not math.isfinite(args.capture_target_meters)
        or args.capture_target_meters <= 0.0
    ):
        raise RuntimeError("Physical target must be finite and positive")

    input_before = fingerprint(opened)
    capture_manifest_before = fingerprint(capture_manifest)
    guarded_before = {
        str(required_file(path, "Guarded SPM")): fingerprint(path)
        for path in args.guard_spm
    }
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    from speedtree_cluster_normalizer.capture_bake import (
        finalize_physical_capture_manifest,
        load_physical_capture_manifest,
        physical_capture_contract,
    )
    from speedtree_cluster_normalizer.delivery_validation import (
        validate_cluster_delivery,
    )
    from speedtree_cluster_normalizer.normalization import (
        build_normalized_cluster_assets,
    )

    scene = bpy.context.scene
    if (
        scene.unit_settings.system != "METRIC"
        or abs(float(scene.unit_settings.scale_length) - 1.0) > 1.0e-12
    ):
        raise RuntimeError("Physical normalization requires METRIC scale_length=1")
    source = bpy.data.objects.get(args.source_object)
    if source is None or source.type != "MESH":
        raise RuntimeError(f"Source mesh is missing: {args.source_object}")

    manifest_payload = json.loads(
        capture_manifest.read_text(encoding="utf-8")
    )
    captured_frame = (
        manifest_payload.get("physical_capture_contract", {}).get("frame")
        or {}
    )
    captured_plane = str(captured_frame.get("plane") or "")
    captured_padding = float(
        captured_frame.get("padding_ratio", math.nan)
    )
    captured_targets = captured_frame.get("target_meters") or []
    captured_target = (
        float(captured_targets[0]) if captured_targets else math.nan
    )
    if (
        captured_plane != args.capture_plane
        or not math.isfinite(captured_padding)
        or not math.isfinite(captured_target)
        or abs(captured_padding - args.capture_padding_ratio) > 1.0e-6
        or abs(captured_target - args.capture_target_meters) > 1.0e-6
    ):
        raise RuntimeError(
            "Requested physical capture settings differ from the existing "
            "capture manifest."
        )
    base_contract = physical_capture_contract(
        source_collection=args.capture_source_collection,
        scene=scene,
        plane=captured_plane,
        padding_ratio=captured_padding,
        target_meters=captured_target,
    )
    capture = load_physical_capture_manifest(
        str(capture_manifest),
        base_contract,
    )
    report = build_normalized_cluster_assets(
        bpy.context,
        source,
        args.plan_base,
        args.skeletal_base,
        args.plan_collection,
        args.material,
        plan_margin_ratio=args.plan_margin_ratio,
        replace_generated=True,
        configure_send2ue=False,
        isolate_send2ue_export=True,
        source_reference_collection_name=args.source_reference_collection,
        camera_uv_bundle=None,
        source_partition_mode=args.source_partition_mode,
        plan_refinement_levels=args.plan_refinement_levels,
        source_xml_path=str(source_xml),
        workflow_mode=WORKFLOW,
        physical_capture_contract=capture["contract"],
    )
    finalized = finalize_physical_capture_manifest(
        str(capture_manifest),
        report["physical_capture_contract"],
    )
    report["capture_manifest_sha256"] = finalized["manifest_sha256"]
    report["capture_maps"] = finalized["maps"]
    report["geometry_committed"] = True
    report["cluster_handoff"] = {
        "available": False,
        "prepared": False,
        "reason": "normalization_only_no_spm_write",
    }
    scene["speedtree_cluster_normalizer_last_report"] = json.dumps(
        report,
        ensure_ascii=False,
    )

    expected_export_names = {
        f"{args.skeletal_base}_{index:02d}{suffix}"
        for index in range(1, int(report["prototype_count"]) + 1)
        for suffix in ("", "_Armature", "_Mesh")
    }
    validation = validate_cluster_delivery(
        scene,
        args.plan_collection,
        args.material,
        plan_base=args.plan_base,
        expected_export_names=expected_export_names,
        expected_card_count=report["card_count"],
        expected_prototype_count=report["prototype_count"],
    )
    guarded_after = {
        path: fingerprint(path) for path in guarded_before
    }
    changed_spm = [
        path
        for path in guarded_before
        if guarded_before[path] != guarded_after[path]
    ]
    if changed_spm:
        raise RuntimeError(
            "Normalization-only rebuild changed guarded SPM files: "
            + ", ".join(changed_spm)
        )

    save_to.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_to), check_existing=False)
    payload = {
        "kind": "speedtree_cluster_physical_normalization_only",
        "version": 1,
        "status": "ready",
        "input_blend": input_before,
        "output_blend": fingerprint(save_to),
        "source_xml": fingerprint(source_xml),
        "capture_manifest_before": capture_manifest_before,
        "capture_manifest_after": fingerprint(capture_manifest),
        "guarded_spm_before": guarded_before,
        "guarded_spm_after": guarded_after,
        "spm_unchanged": True,
        "build": report,
        "validation": validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STCLUSTER_NORMALIZATION_ONLY=" + str(report_path))


if __name__ == "__main__":
    main()
