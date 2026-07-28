from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--material-name", required=True)
    parser.add_argument("--material-id", required=True, type=int)
    parser.add_argument("--expected-mesh-count", type=int, default=1)
    parser.add_argument("--report")
    return parser.parse_args()


def read_root(path):
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return ET.fromstring(data)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_node(node):
    return {
        "tag": node.tag,
        "attributes": sorted(node.attrib.items()),
        "text": (node.text or "").strip(),
        "children": [canonical_node(child) for child in list(node)],
    }


def node_hash(node):
    payload = json.dumps(
        canonical_node(node),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def asset_map(assets, tag):
    return {
        int(node.get("ID")): node
        for node in assets.findall(tag)
        if str(node.get("ID", "")).isdigit()
    }


def material_mesh_ids(material):
    result = []
    primary = material.findtext("CutoutMeshID")
    try:
        value = int((primary or "").strip())
    except ValueError:
        value = -1
    if value >= 0:
        result.append(value)
    supplemental = material.find("SupplementalCutoutMeshIDs")
    for node in supplemental.findall("CutoutMesh") if supplemental is not None else []:
        try:
            value = int(node.get("ID", "").strip())
        except ValueError:
            continue
        if value >= 0 and value not in result:
            result.append(value)
    return result


def filename_fields(node):
    values = {}
    for child in node.iter():
        if child is node or "filename" not in child.tag.lower():
            continue
        text = (child.text or "").strip()
        if text:
            values.setdefault(child.tag, []).append(text)
    return values


def direct_fields(node):
    return {
        child.tag: (child.text or "").strip()
        for child in list(node)
        if not list(child)
    }


def describe_material(node):
    return {
        "id": int(node.get("ID")),
        "name": node.get("Name"),
        "mesh_ids": material_mesh_ids(node),
        "filenames": filename_fields(node),
        "direct_fields": direct_fields(node),
        "hash": node_hash(node),
    }


def describe_mesh(node, spm_path):
    fields = direct_fields(node)
    filename = fields.get("Filename", "")
    mesh_path = Path(filename.replace("\\", "/"))
    if not mesh_path.is_absolute():
        mesh_path = Path(spm_path).resolve().parent / mesh_path
    return {
        "id": int(node.get("ID")),
        "name": node.get("Name"),
        "filename": filename,
        "scale": fields.get("Scale"),
        "resolved_file": str(mesh_path.resolve()),
        "file_exists": mesh_path.is_file(),
        "file_size": mesh_path.stat().st_size if mesh_path.is_file() else None,
        "hash": node_hash(node),
    }


def compare_maps(before_map, after_map, ignored_ids):
    shared = sorted(set(before_map) & set(after_map) - set(ignored_ids))
    return {
        "added_ids": sorted(set(after_map) - set(before_map)),
        "removed_ids": sorted(set(before_map) - set(after_map)),
        "changed_untargeted_ids": [
            asset_id
            for asset_id in shared
            if node_hash(before_map[asset_id]) != node_hash(after_map[asset_id])
        ],
    }


def generator_material_mesh_slots(root, material_id):
    slots = []
    for generator_index, generator in enumerate(root.iter("Generator")):
        properties = generator.find("Properties")
        if properties is None:
            continue
        by_name = {
            str(node.findtext("Name") or ""): node
            for node in properties.findall("Property")
        }
        for name, material_node in by_name.items():
            if not name.endswith(":Material"):
                continue
            slot_prefix = name[: -len(":Material")]
            mesh_node = by_name.get(slot_prefix + ":Mesh")
            if mesh_node is None:
                continue
            try:
                slot_material_id = int(material_node.findtext("Value"))
                slot_mesh_id = int(mesh_node.findtext("Value"))
            except (TypeError, ValueError):
                continue
            if slot_material_id != int(material_id):
                continue
            parent_prefix, _, ordinal_text = slot_prefix.rpartition(":")
            parent = by_name.get(parent_prefix)
            try:
                child_count = int(parent.findtext("MultiPropertyChildren"))
            except (AttributeError, TypeError, ValueError):
                child_count = None
            slots.append(
                {
                    "generator_index": generator_index,
                    "generator_name": str(generator.findtext("Name") or ""),
                    "generator_type": str(generator.get("Type") or ""),
                    "slot_prefix": slot_prefix,
                    "parent_prefix": parent_prefix,
                    "ordinal": int(ordinal_text) if ordinal_text.isdigit() else None,
                    "parent_child_count": child_count,
                    "material_id": slot_material_id,
                    "mesh_id": slot_mesh_id,
                }
            )
    return slots


def main():
    args = parse_args()
    before_path = Path(args.before).expanduser().resolve()
    after_path = Path(args.after).expanduser().resolve()
    before_root = read_root(before_path)
    after_root = read_root(after_path)
    before_assets = before_root.find("Assets")
    after_assets = after_root.find("Assets")
    if before_assets is None or after_assets is None:
        raise RuntimeError("Both SPM files must contain Assets")

    before_materials = asset_map(before_assets, "Material_v8")
    after_materials = asset_map(after_assets, "Material_v8")
    before_meshes = asset_map(before_assets, "Mesh")
    after_meshes = asset_map(after_assets, "Mesh")
    if args.material_id not in before_materials or args.material_id not in after_materials:
        raise RuntimeError("Target Material_v8 ID is missing")
    before_material = before_materials[args.material_id]
    after_material = after_materials[args.material_id]
    if (
        before_material.get("Name") != args.material_name
        or after_material.get("Name") != args.material_name
    ):
        raise RuntimeError("Target Material_v8 name/ID contract does not match")

    before_target_meshes = material_mesh_ids(before_material)
    after_target_meshes = material_mesh_ids(after_material)
    before_generator_slots = generator_material_mesh_slots(
        before_root, args.material_id
    )
    after_generator_slots = generator_material_mesh_slots(
        after_root, args.material_id
    )
    after_generator_mesh_ids = sorted(
        {
            int(row["mesh_id"])
            for row in after_generator_slots
            if int(row["mesh_id"]) in set(after_target_meshes)
        }
    )
    missing_generator_mesh_ids = sorted(
        set(after_target_meshes).difference(after_generator_mesh_ids)
    )
    ignored_mesh_ids = set(before_target_meshes) | set(after_target_meshes)
    payload = {
        "before": str(before_path),
        "after": str(after_path),
        "target_material": {
            "before": describe_material(before_material),
            "after": describe_material(after_material),
            "id_preserved": int(after_material.get("ID")) == args.material_id,
            "name_preserved": after_material.get("Name") == args.material_name,
        },
        "target_meshes": {
            "before": [
                describe_mesh(before_meshes[mesh_id], before_path)
                for mesh_id in before_target_meshes
                if mesh_id in before_meshes
            ],
            "after": [
                describe_mesh(after_meshes[mesh_id], after_path)
                for mesh_id in after_target_meshes
                if mesh_id in after_meshes
            ],
        },
        "generator_coverage": {
            "before_slots": before_generator_slots,
            "after_slots": after_generator_slots,
            "covered_target_mesh_ids": after_generator_mesh_ids,
            "missing_target_mesh_ids": missing_generator_mesh_ids,
            "all_target_meshes_referenced": not missing_generator_mesh_ids,
        },
        "materials": {
            "before_count": len(before_materials),
            "after_count": len(after_materials),
            **compare_maps(before_materials, after_materials, {args.material_id}),
        },
        "meshes": {
            "before_count": len(before_meshes),
            "after_count": len(after_meshes),
            **compare_maps(before_meshes, after_meshes, ignored_mesh_ids),
        },
    }
    payload["valid"] = (
        payload["target_material"]["id_preserved"]
        and payload["target_material"]["name_preserved"]
        and not payload["materials"]["added_ids"]
        and not payload["materials"]["removed_ids"]
        and not payload["materials"]["changed_untargeted_ids"]
        and not payload["meshes"]["added_ids"]
        and not payload["meshes"]["removed_ids"]
        and not payload["meshes"]["changed_untargeted_ids"]
        and len(after_target_meshes) == args.expected_mesh_count
        and payload["generator_coverage"]["all_target_meshes_referenced"]
        and all(
            item["file_exists"] and item["scale"] in {"1", "1.0", "1.000000"}
            for item in payload["target_meshes"]["after"]
        )
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        report = Path(args.report).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8")
    print(text)
    if not payload["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
