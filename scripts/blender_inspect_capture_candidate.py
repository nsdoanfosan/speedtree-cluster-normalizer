from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-collection", default="SpeedTree_Source")
    parser.add_argument("--plan-collection", default="Atlas_Cluster_Cards")
    parser.add_argument("--plane", choices=("AUTO", "XY", "XZ", "YZ"), default="AUTO")
    parser.add_argument("--report")
    return parser.parse_args(argv)


def scalar_properties(data_block):
    return {
        key: data_block[key]
        for key in data_block.keys()
        if isinstance(data_block[key], (str, int, float, bool, type(None)))
    }


def uv_bounds(obj):
    layer = obj.data.uv_layers.active
    if layer is None or not layer.data:
        return None
    values = [tuple(loop.uv) for loop in layer.data]
    return {
        "minimum": [min(value[0] for value in values), min(value[1] for value in values)],
        "maximum": [max(value[0] for value in values), max(value[1] for value in values)],
    }


def main():
    args = parse_args()
    addon_root = Path(__file__).resolve().parents[1] / "addons"
    if str(addon_root) not in sys.path:
        sys.path.insert(0, str(addon_root))
    from speedtree_cluster_normalizer.capture_bake import (
        auto_capture_frame,
        renderable_source_selection,
    )

    sources, duplicates = renderable_source_selection(args.source_collection)
    frame = auto_capture_frame(sources, plane=args.plane)
    plan_collection = bpy.data.collections.get(args.plan_collection)
    plans = []
    if plan_collection is not None:
        for obj in sorted(plan_collection.all_objects, key=lambda item: item.name.casefold()):
            if obj.type != "MESH":
                continue
            plans.append(
                {
                    "name": obj.name,
                    "vertices": len(obj.data.vertices),
                    "polygons": len(obj.data.polygons),
                    "materials": [item.name if item else None for item in obj.data.materials],
                    "uv_bounds": uv_bounds(obj),
                    "properties": scalar_properties(obj),
                }
            )
    payload = {
        "blend": bpy.data.filepath,
        "collections": [
            {
                "name": collection.name,
                "objects": len(collection.all_objects),
                "properties": scalar_properties(collection),
            }
            for collection in sorted(
                bpy.data.collections,
                key=lambda item: item.name.casefold(),
            )
        ],
        "source_collection": args.source_collection,
        "source_objects": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "materials": [item.name if item else None for item in obj.data.materials],
            }
            for obj in sources
        ],
        "excluded_exact_duplicates": duplicates,
        "frame": frame,
        "plan_collection": args.plan_collection,
        "plan_collection_properties": (
            scalar_properties(plan_collection) if plan_collection is not None else None
        ),
        "plans": plans,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        report = Path(args.report).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8")
    print("STCLUSTER_CAPTURE_CANDIDATE=" + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
