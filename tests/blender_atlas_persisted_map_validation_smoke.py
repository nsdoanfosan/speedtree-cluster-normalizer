from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import addon_utils


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args(values)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_map(name, filename, enabled, size):
    node = ET.Element("Map", {"Name": name})
    ET.SubElement(node, "TexFilename").text = filename
    ET.SubElement(node, "TexEnabled").text = enabled
    ET.SubElement(node, "TexSizeX").text = str(size[0])
    ET.SubElement(node, "TexSizeY").text = str(size[1])
    return node


def expect_failure(label, callback, phrase):
    try:
        callback()
    except ValueError as exc:
        if phrase not in str(exc):
            raise RuntimeError(f"{label} failed for the wrong reason: {exc}") from exc
        return str(exc)
    raise RuntimeError(f"Persisted map guard accepted {label}")


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    asset_dir = output.parent / (output.stem + "_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    tree_spm = asset_dir / "synthetic_tree.spm"
    tree_spm.write_text("synthetic", encoding="utf-8")
    color = asset_dir / "Color.tga"
    opacity = asset_dir / "Opacity.tga"
    normal = asset_dir / "Normal.tga"
    color.write_bytes(b"color-contract")
    opacity.write_bytes(b"opacity-contract")
    normal.write_bytes(b"normal-contract")

    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    from speedtree_cluster_normalizer.atlas_handoff import (
        _validate_adopted_material_map,
    )

    persisted = {
        "albedo_sha256": sha256(color),
        "opacity_sha256": sha256(opacity),
    }

    def validate(map_name, texture, enabled, current_size, original_enabled=None):
        original_enabled = enabled if original_enabled is None else original_enabled
        row = {
            "stored": texture.name,
            "path": str(texture),
            "size": [2048, 2048],
            "sha256": sha256(texture),
        }
        return _validate_adopted_material_map(
            tree_spm,
            persisted,
            map_name,
            row,
            make_map(map_name, texture.name, enabled, current_size),
            make_map(map_name, texture.name, original_enabled, (2048, 2048)),
        )

    passed = {
        "active_color_exact": validate("Color", color, "true", (2048, 2048)),
        "active_opacity_exact": validate("Opacity", opacity, "true", (2048, 2048)),
        "disabled_exact": validate("Normal", normal, "false", (2048, 2048)),
        "disabled_zero_normalized": validate("Normal", normal, "false", (0, 0)),
    }
    failures = {
        "active_color_zero_rejected": expect_failure(
            "active Color zero-size bypass",
            lambda: validate("Color", color, "true", (0, 0)),
            "contract mismatch",
        ),
        "active_opacity_zero_rejected": expect_failure(
            "active Opacity zero-size bypass",
            lambda: validate("Opacity", opacity, "true", (0, 0)),
            "contract mismatch",
        ),
        "disabled_nonzero_mismatch_rejected": expect_failure(
            "disabled nonzero size drift",
            lambda: validate("Normal", normal, "false", (1, 0)),
            "contract mismatch",
        ),
        "enabled_state_drift_rejected": expect_failure(
            "TexEnabled state drift",
            lambda: validate("Normal", normal, "false", (0, 0), "true"),
            "contract mismatch",
        ),
    }

    wrong_path = asset_dir / "Other.tga"
    wrong_path.write_bytes(normal.read_bytes())
    failures["texture_path_drift_rejected"] = expect_failure(
        "texture filename/path drift",
        lambda: _validate_adopted_material_map(
            tree_spm,
            persisted,
            "Normal",
            {
                "stored": normal.name,
                "path": str(normal),
                "size": [2048, 2048],
                "sha256": sha256(normal),
            },
            make_map("Normal", wrong_path.name, "false", (0, 0)),
            make_map("Normal", normal.name, "false", (2048, 2048)),
        ),
        "contract mismatch",
    )

    failures["equivalent_filename_drift_rejected"] = expect_failure(
        "equivalent but non-exact TexFilename drift",
        lambda: _validate_adopted_material_map(
            tree_spm,
            persisted,
            "Normal",
            {
                "stored": normal.name,
                "path": str(normal),
                "size": [2048, 2048],
                "sha256": sha256(normal),
            },
            make_map("Normal", ".\\" + normal.name, "false", (0, 0)),
            make_map("Normal", normal.name, "false", (2048, 2048)),
        ),
        "contract mismatch",
    )

    original_color = color.read_bytes()
    color.write_bytes(b"mutated-color")
    failures["active_color_hash_drift_rejected"] = expect_failure(
        "active Color hash drift",
        lambda: validate("Color", color, "true", (2048, 2048)),
        "content hash mismatch",
    )
    color.write_bytes(original_color)

    report = {
        "status": "passed",
        "passed": passed,
        "failures": failures,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ATLAS_PERSISTED_MAP_VALIDATION_SMOKE=" + str(output))


if __name__ == "__main__":
    main()
