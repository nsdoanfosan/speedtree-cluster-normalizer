import argparse
import json
import sys
from pathlib import Path

import bpy

from speedtree_cluster_normalizer.atlas_handoff import GENERATOR_VARIANT_POLICY


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-spm", required=True)
    parser.add_argument("--plan-collection", default="Atlas_Cluster_Cards")
    parser.add_argument("--material-name", required=True)
    parser.add_argument("--source-material-id", type=int, required=True)
    parser.add_argument("--albedo", required=True)
    parser.add_argument("--unit-probe-contract", required=True)
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    target = Path(args.target_spm).expanduser().resolve()
    albedo = Path(args.albedo).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    unit_probe_path = Path(args.unit_probe_contract).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target SPM does not exist: {target}")
    if not albedo.is_file():
        raise FileNotFoundError(f"Atlas Color map does not exist: {albedo}")
    if not unit_probe_path.is_file():
        raise FileNotFoundError(
            f"Verified unit-probe receipt does not exist: {unit_probe_path}"
        )
    unit_probe_contract = json.loads(unit_probe_path.read_text(encoding="utf-8"))
    props = getattr(bpy.context.scene, "atlas_leaf_builder", None)
    if props is None:
        raise RuntimeError("Atlas Leaf Mesh Builder is not registered")
    from atlas_leaf_mesh_builder.integration_api import configure_external_plan_target

    configuration = configure_external_plan_target(
        props,
        args.plan_collection,
        args.material_name,
        args.material_name,
        albedo_path=str(albedo),
        target_spm=str(target),
        source_material_id=args.source_material_id,
        adopt_source_material=True,
        only_target=True,
        generator_variant_policy=GENERATOR_VARIANT_POLICY,
        unit_probe_contract=unit_probe_contract,
    )
    result = bpy.ops.atlas_leaf.build_speedtree_spm()
    if "FINISHED" not in result:
        raise RuntimeError(f"Atlas SPM build did not finish: {sorted(result)}")
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    payload = {
        "status": "ready",
        "blend": bpy.data.filepath,
        "target_spm": str(target),
        "configuration": configuration,
        "operator_result": sorted(result),
        "atlas_last_report": str(props.last_report),
        "saved_blend": bool(args.save_blend),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STCLUSTER_AUTO_SPM=" + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
