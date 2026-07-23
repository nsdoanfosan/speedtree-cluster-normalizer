from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import addon_utils
import bpy


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--camera-spm", required=True)
    parser.add_argument("--camera-name", default="Dropped XY plane camera 2")
    parser.add_argument("--tree-spm", required=True)
    parser.add_argument("--albedo", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--material-id", required=True, type=int)
    parser.add_argument("--plan-collection", required=True)
    parser.add_argument("--plan-base", required=True)
    return parser.parse_args(values)


def main():
    args = parse_args()
    blend = Path(args.blend).expanduser().resolve()
    if Path(bpy.data.filepath).resolve() != blend:
        raise RuntimeError("Blender did not open the requested real delivery blend")
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    from speedtree_cluster_normalizer.delivery_validation import (
        validate_camera_uv_delivery,
    )

    result = validate_camera_uv_delivery(
        bpy.context.scene,
        args.plan_collection,
        args.material,
        plan_base=args.plan_base,
        expected_camera_spm=Path(args.camera_spm).expanduser().resolve(),
        expected_camera_name=args.camera_name,
        expected_tree_spm=Path(args.tree_spm).expanduser().resolve(),
        expected_albedo_path=Path(args.albedo).expanduser().resolve(),
        expected_material_id=args.material_id,
    )
    report = Path(args.report).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"status": "passed", "delivery": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("STCLUSTER_REAL_DELIVERY_VALIDATION=" + str(report))


if __name__ == "__main__":
    main()
