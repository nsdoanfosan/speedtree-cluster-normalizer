from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix


PRODUCTION_SPM_ENV = "SPEEDTREE_CLUSTER_PRODUCTION_SPM"
PLAN_COLLECTION = "Atlas_Branch_Plans"
GENERATED_MATERIAL = "M_branch_elm_01"
LEGACY_GENERATED_MATERIAL = "M_branch_elm_01_plan"
SOURCE_MATERIAL = "M_branch_elm_01"
SOURCE_MATERIAL_ID = 8
SOURCE_CUTOUT_IDS = [1, 2, 9]


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-blend", required=True)
    parser.add_argument("--target-spm", required=True)
    parser.add_argument("--albedo", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--production-spm",
        default=os.environ.get(PRODUCTION_SPM_ENV),
        help=(
            "Optional production SPM path that the smoke test must never mutate. "
            f"Defaults to the {PRODUCTION_SPM_ENV} environment variable."
        ),
    )
    return parser.parse_args(values)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_spm(path):
    return ET.fromstring(gzip.decompress(path.read_bytes()))


def positive_int(value):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def material_mesh_ids(material):
    mesh_ids = []
    primary = material.findtext("CutoutMeshID")
    if primary and primary != "-1":
        mesh_ids.append(int(primary))
    supplemental = material.find("SupplementalCutoutMeshIDs")
    if supplemental is not None:
        for child in supplemental.findall("CutoutMesh"):
            mesh_id = child.attrib.get("ID")
            if mesh_id and mesh_id != "-1":
                mesh_ids.append(int(mesh_id))
    return mesh_ids


def exactly_one_material(root, name):
    assets = root.find("Assets")
    if assets is None:
        raise RuntimeError("Target SPM has no Assets node")
    matches = [
        node for node in assets.findall("Material_v8") if node.attrib.get("Name") == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Material_v8 named {name!r}; found {len(matches)}")
    return matches[0]


def mesh_nodes_by_id(root):
    assets = root.find("Assets")
    return {
        int(node.attrib["ID"]): node
        for node in assets.findall("Mesh")
        if positive_int(node.attrib.get("ID")) is not None
    }


def generator_slot(root, generator_name, slot_prefix):
    matches = []
    for generator in root.iter("Generator"):
        if str(generator.findtext("Name") or "") != generator_name:
            continue
        properties = generator.find("Properties")
        if properties is None:
            continue
        by_name = {
            str(node.findtext("Name") or ""): node
            for node in properties.findall("Property")
        }
        material = by_name.get(f"{slot_prefix}:Material")
        mesh = by_name.get(f"{slot_prefix}:Mesh")
        if material is not None and mesh is not None:
            matches.append((int(material.findtext("Value")), int(mesh.findtext("Value"))))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Generator slot {generator_name!r}/{slot_prefix!r}; found {len(matches)}"
        )
    return matches[0]


def validate_source_contract(root, phase):
    material = exactly_one_material(root, SOURCE_MATERIAL)
    material_id = int(material.attrib.get("ID", "-1"))
    cutout_ids = material_mesh_ids(material)
    if material_id != SOURCE_MATERIAL_ID or cutout_ids != SOURCE_CUTOUT_IDS:
        raise RuntimeError(
            f"{phase}: source material contract changed: ID={material_id}, cutouts={cutout_ids}"
        )
    meshes = mesh_nodes_by_id(root)
    for mesh_id in SOURCE_CUTOUT_IDS:
        mesh = meshes.get(mesh_id)
        if mesh is None or str(mesh.findtext("Embedded") or "").casefold() != "true":
            raise RuntimeError(f"{phase}: embedded source mesh {mesh_id} was not preserved")
    return material


def validate_plans():
    collection = bpy.data.collections.get(PLAN_COLLECTION)
    if collection is None:
        raise RuntimeError(f"Normalized plan collection is missing: {PLAN_COLLECTION}")
    plans = sorted(
        (obj for obj in collection.objects if obj.type == "MESH"),
        key=lambda obj: obj.name.casefold(),
    )
    expected_names = [f"branch_elm_01_{index:02d}" for index in range(1, 4)]
    if [obj.name for obj in plans] != expected_names:
        raise RuntimeError(f"Normalized plan names mismatch: {[obj.name for obj in plans]}")
    for index, plan in enumerate(plans, 1):
        ordinal = positive_int(plan.get("speedtree_cluster_variant_index"))
        if ordinal != index:
            raise RuntimeError(f"Plan {plan.name} has no explicit ordinal {index}: {ordinal}")
        if plan.matrix_world != Matrix.Identity(4):
            raise RuntimeError(f"Plan {plan.name} is not normalized to an identity transform")
    return plans


def main():
    args = parse_args()
    normalized_blend = Path(args.normalized_blend).expanduser().resolve()
    target_spm = Path(args.target_spm).expanduser().resolve()
    albedo = Path(args.albedo).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    production_spm = (
        Path(args.production_spm).expanduser().resolve()
        if str(args.production_spm or "").strip()
        else None
    )

    if Path(bpy.data.filepath).resolve() != normalized_blend:
        raise RuntimeError("Blender did not open the requested normalized blend")
    if production_spm is not None and target_spm == production_spm:
        raise RuntimeError("Refusing to run the smoke test against the production SPM")
    if not target_spm.is_file():
        raise RuntimeError(f"Isolated target SPM does not exist: {target_spm}")
    if not albedo.is_file():
        raise RuntimeError(f"Atlas albedo does not exist: {albedo}")

    plans = validate_plans()
    before_hash = sha256(target_spm)
    before_root = read_spm(target_spm)
    validate_source_contract(before_root, "before build")
    legacy = exactly_one_material(before_root, LEGACY_GENERATED_MATERIAL)
    legacy_material_id = int(legacy.attrib["ID"])
    legacy_mesh_ids = material_mesh_ids(legacy)
    if generator_slot(before_root, "Frond 36", "Material:Frond:0") != (
        legacy_material_id,
        legacy_mesh_ids[0],
    ):
        raise RuntimeError("Isolated target does not contain the previous managed plan binding")

    addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    if not hasattr(bpy.types.Scene, "atlas_leaf_builder"):
        raise RuntimeError("Atlas Leaf Mesh Builder could not be enabled")
    from atlas_leaf_mesh_builder.integration_api import configure_external_plan_target

    atlas = bpy.context.scene.atlas_leaf_builder
    configured = configure_external_plan_target(
        atlas,
        collection_name=PLAN_COLLECTION,
        generated_material_name=GENERATED_MATERIAL,
        source_material_name=SOURCE_MATERIAL,
        albedo_path=str(albedo),
        target_spm=str(target_spm),
        source_material_id=SOURCE_MATERIAL_ID,
        adopt_source_material=True,
        mesh_geometry_scale=1.0,
        only_target=True,
    )
    if configured.get("target_count") != 1 or configured.get("mesh_geometry_scale") != 1.0:
        raise RuntimeError(f"Atlas public configuration contract mismatch: {configured}")
    print(
        "ATLAS_CLUSTER_SPM_CONFIG="
        + json.dumps(
            {
                "configured": configured,
                "source_mapping": json.loads(atlas.speedtree_source_materials_json),
            },
            ensure_ascii=False,
        )
    )

    first_operator_result = bpy.ops.atlas_leaf.build_speedtree_spm()
    if set(first_operator_result) != {"FINISHED"}:
        raise RuntimeError(f"Atlas first build operator did not finish: {first_operator_result}")

    after_root = read_spm(target_spm)
    generated = exactly_one_material(after_root, GENERATED_MATERIAL)
    generated_material_id = int(generated.attrib["ID"])
    if generated_material_id != SOURCE_MATERIAL_ID:
        raise RuntimeError("In-place adoption did not preserve Material_v8 ID 8")
    if any(
        node.attrib.get("Name") == LEGACY_GENERATED_MATERIAL
        for node in after_root.find("Assets").findall("Material_v8")
    ):
        raise RuntimeError("Previous separate generated material remains after adoption")
    generated_mesh_ids = material_mesh_ids(generated)
    if len(generated_mesh_ids) != 3 or len(set(generated_mesh_ids)) != 3:
        raise RuntimeError(f"Expected three generated plan meshes: {generated_mesh_ids}")
    meshes = mesh_nodes_by_id(after_root)
    if any(mesh_id in meshes for mesh_id in SOURCE_CUTOUT_IDS):
        raise RuntimeError("Replaced embedded source Mesh IDs 1, 2, or 9 remain in the SPM")
    for mesh_id in generated_mesh_ids:
        mesh = meshes.get(mesh_id)
        contract = (
            str(mesh.findtext("Embedded") or "").casefold() if mesh is not None else None,
            str(mesh.findtext("PivotStyle") or "") if mesh is not None else None,
            str(mesh.findtext("Scale") or "") if mesh is not None else None,
        )
        if contract != ("false", "0", "1") or not str(mesh.findtext("Filename") or ""):
            raise RuntimeError(f"Generated Mesh {mesh_id} has an invalid external-mesh contract: {contract}")

    manifest_path = (
        target_spm.parent / ".atlas_leaf_speedtree_targets" / f"{target_spm.stem}.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"Atlas target manifest was not written: {manifest_path}")
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adoption = first_manifest.get("source_material_adoption") or {}
    migration = adoption.get("migrated_previous_scope") or {}
    if (
        adoption.get("material_id") != 8
        or adoption.get("original_mesh_ids") != SOURCE_CUTOUT_IDS
        or adoption.get("removed_original_mesh_ids") != SOURCE_CUTOUT_IDS
        or migration.get("legacy_material_id") != legacy_material_id
        or migration.get("reusable_mesh_ids") != legacy_mesh_ids
    ):
        raise RuntimeError(f"Source-material adoption manifest mismatch: {adoption}")

    second_operator_result = bpy.ops.atlas_leaf.build_speedtree_spm()
    if set(second_operator_result) != {"FINISHED"}:
        raise RuntimeError(f"Atlas second build operator did not finish: {second_operator_result}")
    second_root = read_spm(target_spm)
    second_material = exactly_one_material(second_root, SOURCE_MATERIAL)
    if material_mesh_ids(second_material) != generated_mesh_ids:
        raise RuntimeError("Second build changed the adopted plan Mesh IDs")
    if any(mesh_id in mesh_nodes_by_id(second_root) for mesh_id in SOURCE_CUTOUT_IDS):
        raise RuntimeError("Second build resurrected replaced embedded source Mesh IDs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = [
        group
        for group in manifest.get("material_groups") or []
        if group.get("material") == GENERATED_MATERIAL
    ]
    if len(groups) != 1:
        raise RuntimeError(f"Generated material group mismatch: {groups}")
    exported_plans = [
        (item.get("source_object"), positive_int(item.get("source_ordinal")))
        for item in groups[0].get("meshes") or []
    ]
    expected_plans = [(plan.name, index) for index, plan in enumerate(plans, 1)]
    if exported_plans != expected_plans:
        raise RuntimeError(
            f"Manifest did not preserve the three explicit plan ordinals: {exported_plans}"
        )

    connection = manifest.get("generator_connection") or {}
    bindings = [
        binding
        for binding in connection.get("bindings") or []
        if binding.get("generator_name") == "Frond 36"
        and binding.get("slot_prefix") == "Material:Frond:0"
    ]
    if len(bindings) != 1:
        raise RuntimeError(f"Frond 36 manifest binding mismatch: {bindings}")
    binding = bindings[0]
    expected_binding = {
        "source_material_id": 8,
        "source_material_name": SOURCE_MATERIAL,
        "source_mesh_id": 1,
        "leaf_ordinal": 1,
        "target_material_id": generated_material_id,
        "target_mesh_id": generated_mesh_ids[0],
    }
    for key, value in expected_binding.items():
        if binding.get(key) != value:
            raise RuntimeError(f"Manifest source binding {key} mismatch: {binding}")
    expected_slot = (generated_material_id, generated_mesh_ids[0])
    if generator_slot(after_root, "Frond 36", "Material:Frond:0") != expected_slot:
        raise RuntimeError(f"Frond 36 was not connected to generated plan 01: {expected_slot}")
    if generator_slot(second_root, "Frond 36", "Material:Frond:0") != expected_slot:
        raise RuntimeError("Second build changed the adopted Frond 36 binding")
    if connection.get("changed_slot_pairs") != 0:
        raise RuntimeError(f"Second build was not idempotent: {connection}")

    report = {
        "normalized_blend": str(normalized_blend),
        "isolated_target_spm": str(target_spm),
        "target_sha256_before": before_hash,
        "target_sha256_after": sha256(target_spm),
        "atlas_configuration": configured,
        "source_material": {
            "name": SOURCE_MATERIAL,
            "id": SOURCE_MATERIAL_ID,
            "original_cutout_mesh_ids": SOURCE_CUTOUT_IDS,
            "adopted_cutout_mesh_ids": generated_mesh_ids,
        },
        "generated_material": {
            "name": GENERATED_MATERIAL,
            "id": generated_material_id,
            "mesh_ids": generated_mesh_ids,
        },
        "plans": [
            {"name": name, "source_ordinal": ordinal}
            for name, ordinal in exported_plans
        ],
        "frond_36": {
            "slot": "Material:Frond:0",
            "material_id": expected_slot[0],
            "mesh_id": expected_slot[1],
        },
        "manifest": str(manifest_path),
        "source_binding": binding,
        "source_material_adoption": manifest.get("source_material_adoption"),
        "first_spm_action": first_manifest.get("spm_action"),
        "second_spm_action": manifest.get("spm_action"),
        "first_operator_result": sorted(first_operator_result),
        "second_operator_result": sorted(second_operator_result),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ATLAS_CLUSTER_SPM_HANDOFF_SMOKE=" + str(report_path))


if __name__ == "__main__":
    main()
