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
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--material-id", required=True, type=int)
    parser.add_argument("--albedo", required=True)
    parser.add_argument("--mesh-geometry-scale", required=True, type=float)
    parser.add_argument("--mesh-asset-scale", required=True, type=float)
    parser.add_argument("--report", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(values)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if not args.apply:
        raise RuntimeError("Refusing to mutate a scratch SPM without explicit --apply")
    blend = Path(args.blend).expanduser().resolve()
    target = Path(args.target_spm).expanduser().resolve()
    albedo = Path(args.albedo).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if Path(bpy.data.filepath).resolve() != blend:
        raise RuntimeError("Blender did not open the requested unit-probe blend")
    if not target.is_file() or not albedo.is_file():
        raise RuntimeError("Unit-probe target SPM and albedo must already exist")
    if args.mesh_geometry_scale <= 0.0 or args.mesh_asset_scale <= 0.0:
        raise RuntimeError("Unit-probe scales must be positive")

    addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    if not hasattr(bpy.types.Scene, "atlas_leaf_builder"):
        raise RuntimeError("Atlas Leaf Mesh Builder did not register")
    from atlas_leaf_mesh_builder.integration_api import configure_external_plan_target

    props = bpy.context.scene.atlas_leaf_builder
    configured = configure_external_plan_target(
        props,
        collection_name=args.plan_collection,
        generated_material_name=args.material,
        source_material_name=args.material,
        albedo_path=str(albedo),
        target_spm=str(target),
        source_material_id=args.material_id,
        adopt_source_material=True,
        only_target=True,
        mesh_geometry_scale=args.mesh_geometry_scale,
        mesh_asset_scale=args.mesh_asset_scale,
        generator_variant_policy="ensure_all_material_cutouts",
        unit_probe_contract=None,
    )
    result = bpy.ops.atlas_leaf.build_speedtree_spm()
    if set(result) != {"FINISHED"}:
        raise RuntimeError(f"Atlas unit-probe build did not finish: {sorted(result)}")
    manifest = (
        target.parent
        / ".atlas_leaf_speedtree_targets"
        / f"{target.stem}.json"
    )
    if not manifest.is_file():
        raise RuntimeError("Atlas unit-probe target manifest was not written")
    payload = {
        "kind": "speedtree_unit_probe_scratch_spm_build",
        "version": 1,
        "status": "ready",
        "blend": str(blend),
        "target_spm": str(target),
        "target_spm_sha256": sha256(target),
        "plan_collection": args.plan_collection,
        "material": args.material,
        "material_id": args.material_id,
        "mesh_geometry_scale": args.mesh_geometry_scale,
        "mesh_asset_scale": args.mesh_asset_scale,
        "generator_scale": 1.0,
        "configuration": configured,
        "atlas_manifest": str(manifest),
        "atlas_manifest_sha256": sha256(manifest),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STCLUSTER_UNIT_PROBE_SPM=" + str(report_path))


if __name__ == "__main__":
    main()
