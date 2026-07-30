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
    parser.add_argument("--target-spm", required=True)
    parser.add_argument("--albedo", required=True)
    parser.add_argument("--camera-spm", required=True)
    parser.add_argument(
        "--camera-name",
        default="Dropped XY plane camera 2",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--generated-material", required=True)
    parser.add_argument("--source-material", required=True)
    parser.add_argument("--source-material-id", required=True, type=int)
    parser.add_argument("--card-count", type=int)
    parser.add_argument("--prototype-count", type=int)
    parser.add_argument("--mesh-geometry-scale", type=float, default=0.01)
    parser.add_argument(
        "--mesh-asset-scale",
        "--mesh-scale",
        dest="mesh_asset_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(values)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    blend = Path(args.blend).expanduser().resolve()
    target_spm = Path(args.target_spm).expanduser().resolve()
    albedo = Path(args.albedo).expanduser().resolve()
    camera_spm = Path(args.camera_spm).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if not args.apply:
        raise RuntimeError("Refusing to update the target without explicit --apply")
    if Path(bpy.data.filepath).resolve() != blend:
        raise RuntimeError("Blender did not open the requested delivery blend")
    if not target_spm.is_file() or not albedo.is_file() or not camera_spm.is_file():
        raise RuntimeError("Target SPM, atlas albedo, and explicit camera SPM must already exist")
    if not args.camera_name.strip():
        raise RuntimeError("Explicit camera name cannot be empty")
    if args.generated_material != args.source_material:
        raise RuntimeError(
            "Atlas update must adopt the existing source material name in place"
        )

    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    from speedtree_cluster_normalizer.delivery_validation import (
        validate_cluster_delivery,
    )
    from speedtree_cluster_normalizer.atlas_handoff import (
        GENERATOR_VARIANT_POLICY,
        _generator_mesh_coverage,
    )
    from speedtree_cluster_normalizer.generator_delivery_contract import (
        DELIVERY_MODE_RENDER_CONNECTED,
        classify_generator_delivery,
        live_export_generator_bindings,
    )
    from atlas_leaf_mesh_builder.integration_api import configure_external_plan_target
    from atlas_leaf_mesh_builder.speedtree import (
        positive_int,
        read_spm_xml,
        spm_material_mesh_ids,
    )

    plans = bpy.data.collections.get(args.plan_collection)
    if plans is None:
        raise RuntimeError(f"Plan collection is missing: {args.plan_collection}")
    plan_objects = [obj for obj in plans.objects if obj.type == "MESH"]
    if not plan_objects:
        raise RuntimeError("Plan collection contains no mesh objects")
    camera_delivery = validate_cluster_delivery(
        bpy.context.scene,
        args.plan_collection,
        args.generated_material,
        expected_camera_spm=camera_spm,
        expected_camera_name=args.camera_name,
        expected_tree_spm=target_spm,
        expected_albedo_path=albedo,
        expected_material_id=args.source_material_id,
        expected_card_count=args.card_count,
        expected_prototype_count=args.prototype_count,
    )
    before_root = read_spm_xml(target_spm)
    before_assets = before_root.find("Assets")
    before_source = [
        node
        for node in before_assets.findall("Material_v8")
        if node.attrib.get("Name") == args.source_material
    ]
    if len(before_source) != 1:
        raise RuntimeError("Source material must exist exactly once before Atlas build")
    if positive_int(before_source[0].attrib.get("ID")) != args.source_material_id:
        raise RuntimeError("Source material ID differs from the explicit contract ID")
    original_cutout_mesh_ids = sorted(spm_material_mesh_ids(before_source[0]))
    manifest_path = (
        target_spm.parent / ".atlas_leaf_speedtree_targets" / f"{target_spm.stem}.json"
    )
    previously_managed_mesh_ids = set()
    scope_dir = target_spm.parent / ".atlas_leaf_speedtree_scopes"
    prior_manifests = [manifest_path]
    if scope_dir.is_dir():
        prior_manifests.extend(scope_dir.glob(f"*__{target_spm.stem}.json"))
    for previous_manifest_path in prior_manifests:
        if not previous_manifest_path.is_file():
            continue
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
        previous_blend_text = str(previous_manifest.get("blend_file") or "").strip()
        previous_blend = Path(previous_blend_text).expanduser()
        same_delivery_scope = (
            previous_manifest.get("source_collection") == args.plan_collection
            and previous_manifest.get("material") == args.source_material
            and positive_int(previous_manifest.get("material_id")) == args.source_material_id
            and bool(previous_blend_text)
            and previous_blend.resolve() == blend
        )
        if same_delivery_scope:
            previously_managed_mesh_ids.update({
                mesh_id
                for value in previous_manifest.get("mesh_ids") or ()
                if (mesh_id := positive_int(value)) is not None
            })
    before_hash = sha256(target_spm)
    backup_dir = target_spm.parent / "_spm_backups"
    backups_before = set(backup_dir.glob("*.spm")) if backup_dir.is_dir() else set()
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
        mesh_geometry_scale=args.mesh_geometry_scale,
        mesh_asset_scale=args.mesh_asset_scale,
        generator_variant_policy=GENERATOR_VARIANT_POLICY,
    )
    cluster = bpy.context.scene.speedtree_cluster_normalizer
    cluster.atlas_camera_spm = str(camera_spm)
    cluster.atlas_camera_name = args.camera_name
    result = bpy.ops.atlas_leaf.build_speedtree_spm()
    if set(result) != {"FINISHED"}:
        raise RuntimeError(f"Atlas build operator did not finish: {result}")

    root = read_spm_xml(target_spm)
    assets = root.find("Assets")
    generated = [
        node
        for node in assets.findall("Material_v8")
        if node.attrib.get("Name") == args.generated_material
    ]
    source = [
        node
        for node in assets.findall("Material_v8")
        if node.attrib.get("Name") == args.source_material
    ]
    if len(generated) != 1 or len(source) != 1:
        raise RuntimeError("Generated/source material validation failed after Atlas build")
    if generated[0] is not source[0]:
        raise RuntimeError("Generated and source material were not adopted in place")
    if (
        args.source_material_id is not None
        and positive_int(source[0].attrib.get("ID")) != args.source_material_id
    ):
        raise RuntimeError("Source material ID changed during Atlas build")
    generated_mesh_ids = spm_material_mesh_ids(generated[0])
    if args.source_material_id is not None and positive_int(generated[0].attrib.get("ID")) != args.source_material_id:
        raise RuntimeError("Generated material did not retain the adopted source material ID")
    legacy_name = args.source_material + "_plan"
    if any(
        node.attrib.get("Name") == legacy_name
        for node in assets.findall("Material_v8")
    ):
        raise RuntimeError(f"Legacy generated material still exists: {legacy_name}")
    remaining_mesh_ids = {
        positive_int(node.attrib.get("ID"))
        for node in assets.findall("Mesh")
    }
    stale_cutout_ids = sorted(
        (set(original_cutout_mesh_ids) - previously_managed_mesh_ids)
        & remaining_mesh_ids
    )
    if stale_cutout_ids:
        raise RuntimeError(f"Old source cutout mesh IDs were not deleted: {stale_cutout_ids}")
    if len(generated_mesh_ids) != len(plan_objects):
        raise RuntimeError(
            f"Generated plan mesh count mismatch: {generated_mesh_ids} vs {len(plan_objects)}"
        )
    generator_slots = _generator_mesh_coverage(root, args.source_material_id)
    covered_mesh_ids = {
        int(row["mesh_id"])
        for row in generator_slots
        if int(row["mesh_id"]) in set(generated_mesh_ids)
    }
    missing_generator_mesh_ids = sorted(
        set(generated_mesh_ids).difference(covered_mesh_ids)
    )
    if missing_generator_mesh_ids:
        raise RuntimeError(
            "Atlas build left normalized variations unreferenced by generators: "
            + repr(missing_generator_mesh_ids)
        )

    backups_after = set(backup_dir.glob("*.spm")) if backup_dir.is_dir() else set()
    created_backups = sorted(str(path) for path in backups_after - backups_before)
    if not manifest_path.is_file():
        raise RuntimeError("Atlas target manifest was not written")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generator_delivery = classify_generator_delivery(
        spm=target_spm,
        connection=manifest.get("generator_connection"),
        target_material_id=args.source_material_id,
        normalized_target_mesh_ids=generated_mesh_ids,
        live_bindings=live_export_generator_bindings(root),
    )
    if (
        generator_delivery["delivery_mode"]
        != DELIVERY_MODE_RENDER_CONNECTED
    ):
        raise RuntimeError(
            "Atlas target Generator delivery is not render-connected: "
            + ", ".join(generator_delivery["errors"])
        )
    payload = {
        "blend": str(blend),
        "target_spm": str(target_spm),
        "target_sha256_before": before_hash,
        "target_sha256_after": sha256(target_spm),
        "operator_result": sorted(result),
        "atlas_configuration": configured,
        "camera_uv_delivery": camera_delivery,
        "card_count": camera_delivery["card_count"],
        "prototype_count": camera_delivery["prototype_count"],
        "explicit_camera_spm": str(camera_spm),
        "explicit_camera_name": args.camera_name,
        "source_material": {
            "name": args.source_material,
            "id": positive_int(source[0].attrib.get("ID")),
            "mesh_ids": sorted(spm_material_mesh_ids(source[0])),
        },
        "generated_material": {
            "name": args.generated_material,
            "id": positive_int(generated[0].attrib.get("ID")),
            "mesh_ids": sorted(generated_mesh_ids),
            "adopted_in_place": True,
            "legacy_material_absent": True,
            "deleted_original_cutout_mesh_ids": original_cutout_mesh_ids,
        },
        "plan_objects": sorted(obj.name for obj in plan_objects),
        "manifest": str(manifest_path),
        "generator_connection": manifest.get("generator_connection"),
        "generator_delivery": generator_delivery,
        "generator_variant_policy": GENERATOR_VARIANT_POLICY,
        "generator_slots": generator_slots,
        "generator_covered_mesh_ids": sorted(covered_mesh_ids),
        "atlas_backups_created": created_backups,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STCLUSTER_ATLAS_TARGET_UPDATE=" + str(report_path))


if __name__ == "__main__":
    main()
