import argparse
import json
import sys
from pathlib import Path


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--source-collection", default="SpeedTree_Source")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--padding-ratio", type=float, default=0.04)
    parser.add_argument("--plane", choices=("XY", "XZ", "YZ"), default="XY")
    parser.add_argument(
        "--workflow-mode",
        choices=("LEGACY_CAMERA_UV", "PHYSICAL_DIRECT_CAPTURE"),
        default="PHYSICAL_DIRECT_CAPTURE",
    )
    parser.add_argument("--target-meters", type=float, default=0.1)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    addon_root = Path(__file__).resolve().parents[1] / "addons"
    sys.path.insert(0, str(addon_root))
    from speedtree_cluster_normalizer.capture_bake import bake_capture_maps

    result = bake_capture_maps(
        output_dir=args.output_dir,
        prefix=args.prefix,
        source_collection=args.source_collection,
        resolution=args.resolution,
        padding_ratio=args.padding_ratio,
        plane=args.plane,
        workflow_mode=args.workflow_mode,
        target_meters=args.target_meters,
    )
    print("STCLUSTER_AUTO_CAPTURE=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
