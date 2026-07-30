import hashlib
import json
import math
import os
from pathlib import Path

import addon_utils
import bpy

from .generator_delivery_contract import (
    DELIVERY_MODE_RENDER_CONNECTED,
    classify_generator_delivery,
    live_export_generator_bindings,
)


CAMERA_REFERENCE_COLLECTION = "Atlas_Camera_Reference"
CAMERA_CONTRACT_KEY = "speedtree_cluster_camera_uv_contract"
CAMERA_CONTRACT_HASH_KEY = "speedtree_cluster_camera_uv_contract_sha256"
CAMERA_BUNDLE_KEY = "speedtree_cluster_camera_uv_bundle"
CANONICAL_SPEEDTREE_EFFECTIVE_MESH_SCALE = 0.01
GENERATOR_VARIANT_POLICY = "ensure_all_material_cutouts"


def _read_external_camera_contract(reader, *args, camera_name, **kwargs):
    """Use Batch Tools' authoritative, side-camera-aware public reader unchanged."""
    return reader(*args, camera_name=camera_name, **kwargs)


def _enable_atlas_addon():
    if hasattr(bpy.types.Scene, "atlas_leaf_builder"):
        return True
    try:
        addon_utils.enable("atlas_leaf_mesh_builder", default_set=False)
    except Exception:
        return False
    return hasattr(bpy.types.Scene, "atlas_leaf_builder")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_path(first, second):
    return os.path.normcase(os.path.realpath(str(first))) == os.path.normcase(
        os.path.realpath(str(second))
    )


def _contract_contains(reference, current):
    if isinstance(reference, dict):
        return isinstance(current, dict) and all(
            key in current and _contract_contains(value, current[key])
            for key, value in reference.items()
        )
    if isinstance(reference, list):
        return isinstance(current, list) and len(reference) == len(current) and all(
            _contract_contains(first, second)
            for first, second in zip(reference, current)
        )
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(current, (int, float))
        and not isinstance(current, bool)
        and (isinstance(reference, float) or isinstance(current, float))
    ):
        return math.isclose(
            float(reference),
            float(current),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    return reference == current


def _required_file(value, label, suffix=None):
    path = Path(bpy.path.abspath(str(value or ""))).expanduser().absolute()
    if suffix and path.suffix.casefold() != suffix.casefold():
        raise ValueError(f"{label} must end with {suffix}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _validate_reference_artifacts(contract, reference_blend, manifest_path, validation_path):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Camera reference metadata is unreadable: {exc}") from exc

    if manifest.get("kind") != "speedtree_cluster_card_camera_projection":
        raise ValueError("Camera normalization manifest kind is not authoritative.")
    for key in ("camera", "material", "planes"):
        if not _contract_contains(manifest.get(key), contract.get(key)):
            raise ValueError(f"Camera normalization manifest {key} contract is stale.")
    if manifest.get("camera_spm") != contract.get("camera_spm"):
        raise ValueError("Camera normalization manifest camera SPM fingerprint is stale.")
    tree_path_changed = not _same_path(
        manifest.get("tree_spm", {}).get("path", ""),
        contract.get("tree_spm", {}).get("path", ""),
    )

    if validation.get("status") != "ready":
        raise ValueError("Camera Blender validation report is not ready.")
    blend_row = validation.get("blend") or {}
    if not _same_path(blend_row.get("path", ""), reference_blend):
        raise ValueError("Camera Blender validation targets another reference blend.")
    actual_blend_hash = _sha256(reference_blend)
    if str(blend_row.get("sha256") or "").casefold() != actual_blend_hash:
        raise ValueError("Camera reference blend hash does not match its validation report.")
    if int(blend_row.get("size") or -1) != reference_blend.stat().st_size:
        raise ValueError("Camera reference blend size does not match its validation report.")
    if not all(
        validation.get(key) is True
        for key in (
            "all_transform_identity",
            "all_planar",
            "all_uv_preserved",
            "all_fbx_roundtrip_preserved",
        )
    ):
        raise ValueError("Camera reference Blender validation contains a failed invariant.")
    validation_planes = {
        row.get("name"): row for row in validation.get("planes") or []
    }
    for plane in contract["planes"]:
        row = validation_planes.get(plane["name"])
        if row is None:
            raise ValueError(f"Camera validation is missing plane {plane['name']}.")
        if (
            int(row.get("source_mesh_id", -1)) != int(plane["source_mesh_id"])
            or int(row.get("vertex_count", -1)) != len(plane["vertices"])
            or int(row.get("triangle_count", -1)) != len(plane["faces"])
            or row.get("uv_corner_preserved") is not True
        ):
            raise ValueError(f"Camera validation plane contract drifted: {plane['name']}")
    return {
        "manifest": manifest,
        "validation": validation,
        "reference_blend_sha256": actual_blend_hash,
        "manifest_sha256": _sha256(manifest_path),
        "validation_sha256": _sha256(validation_path),
        "tree_file_changed_since_reference_build": (
            tree_path_changed
            or
            manifest.get("tree_spm", {}).get("sha256")
            != contract.get("tree_spm", {}).get("sha256")
        ),
    }


def _ensure_capture_receipt(contract, camera_spm, manifest_path):
    from atlas_leaf_mesh_builder.integration_api import (
        ensure_external_camera_capture_refresh,
    )

    try:
        return ensure_external_camera_capture_refresh(
            contract,
            str(camera_spm),
            str(manifest_path),
        )
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def _persisted_contract_bundle(
    scene,
    *,
    camera_spm,
    tree_spm,
    albedo,
    material_name,
    material_id,
    output_prefix,
    camera_name,
):
    contract_text = scene.get(CAMERA_CONTRACT_KEY)
    bundle_text = scene.get(CAMERA_BUNDLE_KEY)
    expected_hash = str(scene.get(CAMERA_CONTRACT_HASH_KEY) or "")
    if not contract_text or not bundle_text or not expected_hash:
        return None
    try:
        contract = json.loads(contract_text)
        persisted = json.loads(bundle_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Persisted camera UV contract is invalid JSON: {exc}") from exc
    if _canonical_sha256(contract) != expected_hash:
        raise ValueError("Persisted camera UV contract hash is stale.")
    if persisted.get("contract_sha256") != expected_hash:
        raise ValueError("Persisted camera UV bundle references another contract.")
    if not _same_path(contract.get("camera_spm", {}).get("path", ""), camera_spm):
        raise ValueError("Persisted camera UV contract targets another camera SPM.")
    if not _same_path(contract.get("tree_spm", {}).get("path", ""), tree_spm):
        raise ValueError("Persisted camera UV contract targets another tree SPM.")
    if _sha256(camera_spm) != contract.get("camera_spm", {}).get("sha256"):
        raise ValueError("Camera SPM changed since the persisted UV contract was captured.")
    if contract.get("camera", {}).get("name") != camera_name:
        raise ValueError("Persisted camera UV contract camera name mismatch.")
    material = contract.get("material") or {}
    if material.get("name") != material_name:
        raise ValueError("Persisted camera UV contract material name mismatch.")
    if material_id is not None and int(material.get("id", -1)) != material_id:
        raise ValueError("Persisted camera UV contract material ID mismatch.")
    planes = contract.get("planes") or []
    expected_names = [f"{output_prefix}_{index:02d}" for index in range(1, len(planes) + 1)]
    if [plane.get("name") for plane in planes] != expected_names:
        raise ValueError("Persisted camera UV contract output prefix mismatch.")
    color_path = _required_file(
        material.get("maps", {}).get("Color", {}).get("path"),
        "Persisted contract Color map",
    )
    opacity_path = _required_file(
        material.get("maps", {}).get("Opacity", {}).get("path"),
        "Persisted contract Opacity map",
    )
    if not _same_path(color_path, albedo):
        raise ValueError("Persisted camera UV contract Color map selection mismatch.")
    reference_blend = _required_file(
        persisted.get("reference_blend"), "Persisted camera reference blend", ".blend"
    )
    manifest_path = _required_file(
        persisted.get("manifest_path"), "Persisted camera manifest", ".json"
    )
    validation_path = _required_file(
        persisted.get("validation_path"), "Persisted camera validation", ".json"
    )
    capture = _ensure_capture_receipt(contract, camera_spm, manifest_path)
    contract = capture["contract"]
    material = contract.get("material") or {}
    color_path = _required_file(
        material.get("maps", {}).get("Color", {}).get("path"),
        "Receipt-backed contract Color map",
    )
    opacity_path = _required_file(
        material.get("maps", {}).get("Opacity", {}).get("path"),
        "Receipt-backed contract Opacity map",
    )
    if not _same_path(color_path, albedo):
        raise ValueError("Receipt-backed camera contract Color map selection mismatch.")
    artifacts = _validate_reference_artifacts(
        contract, reference_blend, manifest_path, validation_path
    )
    expected_artifact_hashes = {
        "reference_blend_sha256": artifacts["reference_blend_sha256"],
        "validation_sha256": artifacts["validation_sha256"],
    }
    for key, actual in expected_artifact_hashes.items():
        if str(persisted.get(key) or "").casefold() != actual:
            raise ValueError(f"Persisted camera artifact hash is stale: {key}")
    expected_hash = _canonical_sha256(contract)
    persisted.update(
        {
            "contract_sha256": expected_hash,
            "albedo_sha256": _sha256(color_path),
            "opacity_sha256": _sha256(opacity_path),
            "manifest_sha256": artifacts["manifest_sha256"],
            "camera_capture_receipt_path": capture["receipt_path"],
            "camera_capture_receipt_sha256": capture["receipt"][
                "receipt_sha256"
            ],
        }
    )
    scene[CAMERA_CONTRACT_KEY] = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
    )
    scene[CAMERA_CONTRACT_HASH_KEY] = expected_hash
    scene[CAMERA_BUNDLE_KEY] = json.dumps(
        persisted,
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        **persisted,
        "contract": contract,
        "contract_sha256": expected_hash,
        "camera_spm": str(camera_spm),
        "tree_spm": str(tree_spm),
        "albedo_path": str(color_path),
        "opacity_path": str(opacity_path),
        "reference_collection": CAMERA_REFERENCE_COLLECTION,
        "contract_source": "persisted_scene_after_source_material_adoption",
        "tree_file_changed_since_reference_build": True,
    }


def _live_contract_matches_persisted(live_contract, persisted_contract):
    return all(
        _contract_contains(persisted_contract.get(key), live_contract.get(key))
        for key in ("camera", "material", "planes")
    )


def _validate_adopted_material_map(
    tree_spm,
    persisted,
    map_name,
    row,
    current_map,
    original_map,
):
    """Validate one persisted map while accepting SpeedTree's disabled-size rewrite."""
    expected_size = [str(value) for value in row.get("size") or []]
    if current_map is None or original_map is None or len(expected_size) != 2:
        raise ValueError(f"Adopted target material map is missing: {map_name}")

    filename = str(current_map.findtext("TexFilename") or "")
    original_filename = str(original_map.findtext("TexFilename") or "")
    expected_filename = str(row.get("stored") or "")
    resolved_texture = (tree_spm.parent / filename).resolve() if filename else None
    current_enabled = str(current_map.findtext("TexEnabled") or "").casefold()
    original_enabled = str(original_map.findtext("TexEnabled") or "").casefold()
    current_size = [
        str(current_map.findtext("TexSizeX") or ""),
        str(current_map.findtext("TexSizeY") or ""),
    ]
    size_policy = "exact_contract_size"
    if current_size != expected_size:
        if (
            current_enabled == "false"
            and original_enabled == "false"
            and current_size == ["0", "0"]
        ):
            size_policy = "disabled_speedtree_zero_size_normalization"
        else:
            raise ValueError(f"Adopted target material map contract mismatch: {map_name}")

    if (
        resolved_texture is None
        or not resolved_texture.is_file()
        or filename != expected_filename
        or filename != original_filename
        or not _same_path(resolved_texture, row.get("path", ""))
        or current_enabled != original_enabled
    ):
        raise ValueError(f"Adopted target material map contract mismatch: {map_name}")

    expected_hash = row.get("sha256")
    if map_name == "Color":
        expected_hash = persisted.get("albedo_sha256")
    elif map_name == "Opacity":
        expected_hash = persisted.get("opacity_sha256")
    if expected_hash and _sha256(resolved_texture) != expected_hash:
        raise ValueError(f"Adopted target material map content hash mismatch: {map_name}")
    return {
        "map_name": map_name,
        "tex_filename": filename,
        "tex_enabled": current_enabled,
        "tex_size": current_size,
        "size_policy": size_policy,
        "resolved_texture": str(resolved_texture),
    }


def _generator_mesh_coverage(root, material_id):
    """Return generator Material/Mesh slots that actively use one material."""
    material_id = int(material_id)
    return [
        row
        for row in live_export_generator_bindings(root)
        if row["export_participates"]
        and row["material_id"] == material_id
    ]


def _validate_adopted_target_spm(
    tree_spm,
    persisted,
    material_name,
    material_id,
    scene,
    plan_collection_name,
):
    from atlas_leaf_mesh_builder.speedtree import (
        ATLAS_LEAF_COLLECTION_SCOPE_KEY,
        decode_spm_node_snapshot,
        positive_int,
        read_json_file,
        read_spm_xml,
        scope_manifest_path,
        spm_material_mesh_ids,
        target_manifest_path,
    )

    plan_collection = bpy.data.collections.get(plan_collection_name)
    collection_scope = str(
        plan_collection.get(ATLAS_LEAF_COLLECTION_SCOPE_KEY) or ""
    ) if plan_collection is not None else ""
    persisted_scope = str(persisted.get("atlas_export_scope_id") or "")
    if collection_scope and persisted_scope and collection_scope != persisted_scope:
        raise ValueError(
            "Persisted camera contract cannot be used: Atlas collection scope changed."
        )
    scope_id = collection_scope or persisted_scope
    scoped_manifest_path = scope_manifest_path(
        tree_spm.parent,
        {"export_scope_id": scope_id},
        tree_spm,
    )
    manifest_path = (
        scoped_manifest_path
        if scope_id and scoped_manifest_path.is_file()
        else target_manifest_path(tree_spm)
    )
    manifest = read_json_file(manifest_path, {})
    adoption = manifest.get("source_material_adoption") or {}
    material_id = (
        int(material_id)
        if material_id is not None
        else int(persisted["contract"]["material"]["id"])
    )
    expected_original_ids = [
        int(value)
        for value in persisted["contract"]["material"]["ordered_cutout_mesh_ids"]
    ]
    generated_ids = [
        positive_int(value) for value in adoption.get("generated_mesh_ids") or []
    ]
    generated_ids = [value for value in generated_ids if value is not None]
    snapshot_ids = sorted(
        positive_int(row.get("mesh_id"))
        for row in adoption.get("original_mesh_snapshots") or []
    )
    if (
        not manifest
        or not _same_path(manifest.get("spm", ""), tree_spm)
        or adoption.get("version") != 1
        or adoption.get("material_name") != material_name
        or positive_int(adoption.get("material_id")) != material_id
        or [positive_int(value) for value in adoption.get("original_mesh_ids") or []]
        != expected_original_ids
        or [positive_int(value) for value in adoption.get("removed_original_mesh_ids") or []]
        != expected_original_ids
        or not adoption.get("original_material_snapshot")
        or len(adoption.get("original_mesh_snapshots") or []) != len(expected_original_ids)
        or snapshot_ids != sorted(expected_original_ids)
        or not generated_ids
        or generated_ids != [positive_int(value) for value in manifest.get("mesh_ids") or []]
        or manifest.get("material_name") != material_name
        or positive_int(manifest.get("material_id")) != material_id
        or manifest.get("source_collection") != plan_collection_name
        or not manifest.get("export_scope_id")
        or str(persisted.get("atlas_export_scope_id") or "")
        != str(manifest.get("export_scope_id"))
        or adoption.get("scope") != manifest.get("export_scope_id")
        or collection_scope != str(manifest.get("export_scope_id"))
        or not manifest.get("blend_file")
        or not bpy.data.filepath
        or not _same_path(manifest.get("blend_file"), bpy.data.filepath)
    ):
        raise ValueError(
            "Persisted camera contract cannot be used: current target SPM has no matching "
            "Atlas source-material adoption proof."
        )
    textures = manifest.get("textures") or {}
    if not (
        _same_path(textures.get("albedo", ""), persisted["albedo_path"])
        and _same_path(textures.get("alpha", ""), persisted["opacity_path"])
    ):
        raise ValueError("Adopted target manifest texture lineage does not match the camera contract.")

    root = read_spm_xml(tree_spm)
    assets = root.find("Assets")
    if assets is None:
        raise ValueError("Adopted target SPM has no Assets node.")
    materials = [
        node
        for node in assets.findall("Material_v8")
        if node.attrib.get("Name") == material_name
        and positive_int(node.attrib.get("ID")) == material_id
    ]
    if len(materials) != 1 or spm_material_mesh_ids(materials[0]) != generated_ids:
        raise ValueError("Adopted target material/mesh lineage does not match its manifest.")
    generator_slots = _generator_mesh_coverage(root, material_id)
    covered_generated_ids = sorted(
        {
            int(row["mesh_id"])
            for row in generator_slots
            if int(row["mesh_id"]) in set(generated_ids)
        }
    )
    if covered_generated_ids != sorted(generated_ids):
        missing = sorted(set(generated_ids).difference(covered_generated_ids))
        raise ValueError(
            "Adopted target generators do not actively reference every normalized "
            f"variation: missing Mesh IDs {missing}."
        )
    connection = manifest.get("generator_connection") or {}
    if connection.get("generator_variant_policy") != GENERATOR_VARIANT_POLICY:
        raise ValueError(
            "Adopted target manifest does not prove the normalized generator-variation policy."
        )
    generator_delivery = classify_generator_delivery(
        spm=tree_spm,
        connection=connection,
        target_material_id=material_id,
        normalized_target_mesh_ids=generated_ids,
        live_bindings=live_export_generator_bindings(root),
    )
    if (
        generator_delivery["delivery_mode"]
        != DELIVERY_MODE_RENDER_CONNECTED
    ):
        raise ValueError(
            "Adopted target Generator delivery is not render-connected: "
            + ", ".join(generator_delivery["errors"])
        )
    current_material = materials[0]
    original_material = decode_spm_node_snapshot(adoption["original_material_snapshot"])
    contract_material = persisted["contract"]["material"]
    if (
        str(current_material.findtext("Width") or "") != str(contract_material["width"])
        or str(current_material.findtext("Height") or "") != str(contract_material["height"])
        or current_material.findtext("Width") != original_material.findtext("Width")
        or current_material.findtext("Height") != original_material.findtext("Height")
    ):
        raise ValueError("Adopted target material dimensions changed from the camera contract.")
    current_maps = {node.attrib.get("Name"): node for node in current_material.findall("Map")}
    original_maps = {node.attrib.get("Name"): node for node in original_material.findall("Map")}
    validated_maps = []
    for map_name, row in contract_material.get("maps", {}).items():
        validated_maps.append(
            _validate_adopted_material_map(
                tree_spm,
                persisted,
                map_name,
                row,
                current_maps.get(map_name),
                original_maps.get(map_name),
            )
        )
    current_mesh_ids = {
        positive_int(node.attrib.get("ID")) for node in assets.findall("Mesh")
    }
    if set(expected_original_ids).intersection(current_mesh_ids):
        raise ValueError("Adopted target SPM still contains stale original cutout mesh IDs.")
    legacy_name = material_name + "_plan"
    if any(
        node.attrib.get("Name") == legacy_name for node in assets.findall("Material_v8")
    ):
        raise ValueError("Adopted target SPM still contains the legacy _plan material.")
    mesh_nodes = {
        positive_int(node.attrib.get("ID")): node for node in assets.findall("Mesh")
    }
    manifest_meshes = sorted(
        manifest.get("meshes") or [],
        key=lambda row: int(row.get("source_ordinal") or 0),
    )
    try:
        manifest_mesh_geometry_scale = float(manifest.get("mesh_geometry_scale"))
        manifest_mesh_asset_scale = float(manifest.get("mesh_asset_scale"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Adopted target manifest has no complete SpeedTree mesh scale contract."
        ) from exc
    manifest_effective_mesh_scale = (
        manifest_mesh_geometry_scale * manifest_mesh_asset_scale
    )
    if not math.isclose(
        manifest_effective_mesh_scale,
        CANONICAL_SPEEDTREE_EFFECTIVE_MESH_SCALE,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "Adopted target manifest effective SpeedTree mesh scale is not "
            "the canonical 0.01."
        )
    if len(manifest_meshes) != len(generated_ids):
        raise ValueError("Adopted target manifest mesh lineage count mismatch.")
    for mesh_id, row in zip(generated_ids, manifest_meshes):
        node = mesh_nodes.get(mesh_id)
        filename = str(node.findtext("Filename") or "") if node is not None else ""
        resolved_asset = (tree_spm.parent / filename).resolve() if filename else None
        try:
            node_mesh_asset_scale = float(node.findtext("Scale")) if node is not None else None
        except (TypeError, ValueError):
            node_mesh_asset_scale = None
        if (
            node is None
            or str(node.findtext("Embedded") or "").casefold() != "false"
            or str(node.findtext("PivotStyle") or "") != "0"
            or node_mesh_asset_scale is None
            or not math.isclose(
                node_mesh_asset_scale,
                manifest_mesh_asset_scale,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or resolved_asset is None
            or not _same_path(resolved_asset, row.get("asset", ""))
            or not resolved_asset.is_file()
        ):
            raise ValueError(f"Adopted target external mesh lineage mismatch: Mesh {mesh_id}")
    return {
        "target_manifest": str(manifest_path),
        "target_manifest_sha256": _sha256(manifest_path),
        "adopted_material_id": material_id,
        "adopted_generated_mesh_ids": generated_ids,
        "adopted_original_mesh_ids": expected_original_ids,
        "adopted_material_maps": validated_maps,
        "adopted_generator_variant_policy": GENERATOR_VARIANT_POLICY,
        "adopted_generator_slots": generator_slots,
        "generator_delivery": generator_delivery,
    }


def resolve_camera_uv_contract(props, output_prefix):
    """Fail-closed preflight for the exact camera-SPM template and reference blend."""
    if not _enable_atlas_addon():
        raise RuntimeError(
            "atlas_leaf_mesh_builder is required for the camera SPM UV contract."
        )
    albedo = _required_file(props.atlas_albedo_path, "Atlas Color map")
    tree_spm = _required_file(props.atlas_target_spm, "Target tree SPM", ".spm")
    camera_spm = _required_file(props.atlas_camera_spm, "Camera SPM", ".spm")
    camera_name = str(props.atlas_camera_name or "").strip()
    if not camera_name:
        raise ValueError("Explicit camera name cannot be empty.")
    source_material_name = str(props.source_material_name or "").strip()
    source_material_id = int(props.source_material_id) if props.source_material_id else None
    if not source_material_name:
        raise ValueError("Source SpeedTree material name cannot be empty.")

    from atlas_leaf_mesh_builder.integration_api import read_external_plan_uv_contract

    persisted = _persisted_contract_bundle(
        props.id_data,
        camera_spm=camera_spm,
        tree_spm=tree_spm,
        albedo=albedo,
        material_name=source_material_name,
        material_id=source_material_id,
        output_prefix=output_prefix,
        camera_name=camera_name,
    )
    if persisted is not None:
        try:
            live_contract = _read_external_camera_contract(
                read_external_plan_uv_contract,
                str(camera_spm),
                str(tree_spm),
                material_name=source_material_name,
                material_id=source_material_id,
                output_prefix=output_prefix,
                camera_name=camera_name,
            )
        except Exception:
            live_contract = None
        if live_contract is not None and _live_contract_matches_persisted(
            live_contract, persisted["contract"]
        ):
            persisted["contract_source"] = "persisted_scene_live_tree_contract_revalidated"
            return persisted
        adoption = _validate_adopted_target_spm(
            tree_spm,
            persisted,
            source_material_name,
            source_material_id,
            props.id_data,
            props.plan_collection,
        )
        persisted["contract_source"] = "persisted_scene_validated_adoption_lineage"
        persisted["target_adoption"] = adoption
        return persisted

    contract = _read_external_camera_contract(
        read_external_plan_uv_contract,
        str(camera_spm),
        str(tree_spm),
        material_name=source_material_name,
        material_id=source_material_id,
        output_prefix=output_prefix,
        camera_name=camera_name,
    )
    if contract.get("validation", {}).get("status") != "ready":
        raise ValueError("Camera SPM UV template contract is not ready.")
    if contract.get("validation", {}).get("strict_vertex_uv_topology") is not True:
        raise ValueError("Camera SPM contract does not have strict vertex/UV topology.")
    color_path = _required_file(
        contract.get("material", {}).get("maps", {}).get("Color", {}).get("path"),
        "Contract Color map",
    )
    opacity_path = _required_file(
        contract.get("material", {}).get("maps", {}).get("Opacity", {}).get("path"),
        "Contract Opacity map",
    )
    if not _same_path(color_path, albedo):
        raise ValueError(
            "Selected Atlas Color map does not match the camera/tree SPM material contract."
        )
    if contract.get("material", {}).get("name") != source_material_name:
        raise ValueError("Camera SPM contract resolved an unexpected material name.")
    if source_material_id is not None and int(contract["material"]["id"]) != source_material_id:
        raise ValueError("Camera SPM contract resolved an unexpected material ID.")

    output_dir = camera_spm.parent / "card_pipeline_outputs" / camera_spm.stem
    reference_blend = _required_file(
        output_dir / f"{camera_spm.stem}.blend",
        "Camera reference blend",
        ".blend",
    )
    manifest_path = _required_file(
        output_dir / f"{camera_spm.stem}_normalization_manifest.json",
        "Camera normalization manifest",
        ".json",
    )
    validation_path = _required_file(
        output_dir / f"{camera_spm.stem}_blender_validation.json",
        "Camera Blender validation",
        ".json",
    )
    capture = _ensure_capture_receipt(contract, camera_spm, manifest_path)
    contract = capture["contract"]
    color_path = _required_file(
        contract.get("material", {}).get("maps", {}).get("Color", {}).get("path"),
        "Receipt-backed contract Color map",
    )
    opacity_path = _required_file(
        contract.get("material", {}).get("maps", {}).get("Opacity", {}).get("path"),
        "Receipt-backed contract Opacity map",
    )
    artifacts = _validate_reference_artifacts(
        contract,
        reference_blend,
        manifest_path,
        validation_path,
    )
    contract_sha256 = _canonical_sha256(contract)
    return {
        "contract": contract,
        "contract_sha256": contract_sha256,
        "camera_spm": str(camera_spm),
        "tree_spm": str(tree_spm),
        "albedo_path": str(color_path),
        "opacity_path": str(opacity_path),
        "albedo_sha256": _sha256(color_path),
        "opacity_sha256": _sha256(opacity_path),
        "reference_blend": str(reference_blend),
        "reference_blend_sha256": artifacts["reference_blend_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": artifacts["manifest_sha256"],
        "validation_path": str(validation_path),
        "validation_sha256": artifacts["validation_sha256"],
        "camera_capture_receipt_path": capture["receipt_path"],
        "camera_capture_receipt_sha256": capture["receipt"]["receipt_sha256"],
        "tree_file_changed_since_reference_build": artifacts[
            "tree_file_changed_since_reference_build"
        ],
        "reference_collection": CAMERA_REFERENCE_COLLECTION,
        "contract_source": "live_public_camera_spm_contract",
    }


def _assign_preview_material(collection_names, material):
    assigned = []
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            continue
        for obj in collection.objects:
            if obj.type != "MESH":
                continue
            obj.data.materials.clear()
            obj.data.materials.append(material)
            assigned.append(obj.name)
    return sorted(set(assigned))


def load_verified_unit_probe_contract(path, target_meters):
    raw_path = str(path or "").strip()
    if not raw_path:
        raise ValueError(
            "Physical Direct Capture requires a verified Blender-to-SpeedTree "
            "unit-probe receipt before production handoff."
        )
    resolved = Path(bpy.path.abspath(raw_path)).expanduser().absolute()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Verified unit-probe receipt does not exist: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Verified unit-probe receipt is not valid JSON: {exc}"
        ) from exc
    if not _enable_atlas_addon():
        raise RuntimeError(
            "Atlas Leaf Mesh Builder must be installed to validate the "
            "production unit-probe receipt."
        )
    from atlas_leaf_mesh_builder.unit_contract import validate_unit_probe_contract

    verified = validate_unit_probe_contract(value)
    if abs(float(verified["physical_target_meters"]) - float(target_meters)) > max(
        abs(float(target_meters)) * 1.0e-6,
        1.0e-9,
    ):
        raise ValueError(
            "Verified unit-probe target does not match the Blender physical "
            f"capture target ({verified['physical_target_meters']} m vs "
            f"{float(target_meters)} m)."
        )
    return {
        **verified,
        "receipt_path": str(resolved),
        "receipt_file_sha256": _sha256(resolved),
    }


def prepare_atlas_handoff(
    context,
    props,
    plan_collection,
    uv_bundle=None,
    unit_probe_contract=None,
):
    if not _enable_atlas_addon():
        return {
            "available": False,
            "prepared": False,
            "reason": "atlas_leaf_mesh_builder is not installed or could not be enabled",
        }
    from atlas_leaf_mesh_builder.integration_api import (
        configure_external_plan_target,
        ensure_external_plan_preview_material,
    )

    atlas = context.scene.atlas_leaf_builder
    configured = configure_external_plan_target(
        atlas,
        collection_name=plan_collection,
        generated_material_name=props.plan_material_name,
        source_material_name=props.source_material_name,
        albedo_path=props.atlas_albedo_path,
        target_spm=props.atlas_target_spm,
        source_material_id=(props.source_material_id or None),
        adopt_source_material=(
            str(props.plan_material_name or "").strip()
            == str(props.source_material_name or "").strip()
        ),
        only_target=props.atlas_only_target,
        mesh_geometry_scale=props.atlas_mesh_scale,
        mesh_asset_scale=props.atlas_mesh_asset_scale,
        generator_variant_policy=GENERATOR_VARIANT_POLICY,
        unit_probe_contract=unit_probe_contract,
    )
    if uv_bundle is not None:
        uv_bundle["atlas_export_scope_id"] = configured["export_scope_id"]
        persisted_bundle = {
            key: value for key, value in uv_bundle.items() if key != "contract"
        }
        context.scene[CAMERA_BUNDLE_KEY] = json.dumps(
            persisted_bundle,
            ensure_ascii=False,
            sort_keys=True,
        )
    preview = None
    if uv_bundle is not None:
        preview = ensure_external_plan_preview_material(
            props.plan_material_name,
            uv_bundle["albedo_path"],
            uv_bundle["opacity_path"],
        )
        material = bpy.data.materials.get(preview["material_name"])
        preview["assigned_objects"] = _assign_preview_material(
            (plan_collection, uv_bundle["reference_collection"]),
            material,
        )
    return {
        "available": True,
        "prepared": True,
        **configured,
        "camera_uv_contract_sha256": (
            uv_bundle["contract_sha256"] if uv_bundle is not None else None
        ),
        "unit_probe_contract_sha256": configured.get(
            "unit_probe_contract_sha256"
        ),
        "unit_scale_location": configured.get("unit_scale_location"),
        "preview_material": preview,
    }
