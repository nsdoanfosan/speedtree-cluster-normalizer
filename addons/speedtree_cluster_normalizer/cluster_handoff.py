"""Cluster-owned SPM handoff metadata and Blender preview configuration."""

import hashlib
import json
from pathlib import Path

import bpy

from .unit_contract import validate_unit_probe_contract


HANDOFF_SCENE_KEY = "speedtree_cluster_spm_handoff"
GENERATOR_VARIANT_POLICY = "ensure_all_material_cutouts"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _preview_material(name, color_path, opacity_path):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    color = nodes.new("ShaderNodeTexImage")
    opacity = nodes.new("ShaderNodeTexImage")
    color.name = "Cluster Color"
    opacity.name = "Cluster Opacity"
    color.image = bpy.data.images.load(str(color_path), check_existing=True)
    opacity.image = bpy.data.images.load(str(opacity_path), check_existing=True)
    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    links.new(opacity.outputs["Color"], shader.inputs["Alpha"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material.surface_render_method = "DITHERED"
    return material


def capture_maps_by_role(capture_maps):
    """Normalize the finalized physical-capture map inventory.

    Physical capture manifests persist maps as an ordered list of records so
    their eight-role order is stable.  Older handoff code treated that list as
    a role-keyed dictionary and crashed after normalization had already built
    the assets.  Accept both representations at this boundary and keep the
    original payload unchanged in the persisted handoff.
    """
    if isinstance(capture_maps, dict):
        return dict(capture_maps)
    if not isinstance(capture_maps, list):
        raise ValueError(
            "Cluster capture maps must be a role-keyed object or a list of "
            "role records."
        )
    mapped = {}
    for index, row in enumerate(capture_maps):
        if not isinstance(row, dict):
            raise ValueError(
                f"Cluster capture map row {index} is not an object."
            )
        role = str(row.get("role") or "").strip()
        if not role:
            raise ValueError(
                f"Cluster capture map row {index} has no role."
            )
        if role in mapped:
            raise ValueError(
                f"Cluster capture maps contain duplicate role {role!r}."
            )
        mapped[role] = row
    return mapped


def prepare_cluster_handoff(
    context,
    props,
    plan_collection,
    unit_probe_contract,
    capture_maps,
):
    collection = bpy.data.collections.get(str(plan_collection or ""))
    if collection is None:
        raise ValueError(f"Normalized plan collection is missing: {plan_collection}")
    target = Path(bpy.path.abspath(props.atlas_target_spm)).expanduser().absolute()
    if not target.is_file():
        raise FileNotFoundError(f"Target tree SPM does not exist: {target}")
    material_name = str(props.plan_material_name or "").strip()
    source_name = str(props.source_material_name or "").strip()
    if not material_name:
        raise ValueError("Cluster output material name cannot be empty.")
    if not source_name:
        raise ValueError("Cluster source material name cannot be empty.")
    if not isinstance(unit_probe_contract, dict):
        raise ValueError("Cluster handoff has no verified unit contract.")
    verified = validate_unit_probe_contract(unit_probe_contract)

    maps_by_role = capture_maps_by_role(capture_maps)
    color = Path((maps_by_role.get("Color") or {}).get("path") or "")
    opacity = Path((maps_by_role.get("Opacity") or {}).get("path") or "")
    if not color.is_file() or not opacity.is_file():
        raise FileNotFoundError(
            "Cluster capture Color/Opacity maps are missing from the handoff."
        )
    if not hasattr(context.scene, "atlas_leaf_builder"):
        raise RuntimeError(
            "Atlas Leaf Mesh Builder is not registered for Cluster SPM handoff."
        )
    from atlas_leaf_mesh_builder.integration_api import (
        configure_external_plan_target,
    )

    configured = configure_external_plan_target(
        context.scene.atlas_leaf_builder,
        collection_name=collection.name,
        generated_material_name=material_name,
        source_material_name=source_name,
        albedo_path=str(color),
        target_spm=str(target),
        source_material_id=(int(props.source_material_id or 0) or None),
        # Same-name physical delivery refreshes existing Material_v8/Mesh
        # assets without rewiring generators.  When a prior Atlas pass replaced
        # the authored cluster slots with another source material, an explicit
        # different source name restores only those live slots to the current
        # normalized cluster outputs.
        adopt_source_material=False,
        connect_generators=(material_name != source_name),
        only_target=bool(props.atlas_only_target),
        mesh_geometry_scale=verified["mesh_geometry_scale"],
        mesh_asset_scale=verified["mesh_asset_scale"],
        generator_variant_policy=GENERATOR_VARIANT_POLICY,
        unit_probe_contract=unit_probe_contract,
    )
    # The capture manifest is authoritative; do not depend on filename
    # inference when a role-specific Opacity map was already verified.
    context.scene.atlas_leaf_builder.alpha_path = str(opacity)
    preview = _preview_material(material_name, color, opacity)
    assigned = []
    for obj in collection.objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(preview)
        assigned.append(obj.name)

    payload = {
        "kind": "speedtree_cluster_spm_handoff",
        "version": 1,
        "prepared": True,
        "blend": str(Path(bpy.data.filepath).expanduser().absolute()),
        "plan_collection": collection.name,
        "generated_material_name": material_name,
        "source_material_name": source_name,
        "source_material_id": int(props.source_material_id or 0),
        "target_spm": str(target),
        "mesh_geometry_scale": verified["mesh_geometry_scale"],
        "mesh_asset_scale": verified["mesh_asset_scale"],
        "unit_probe_contract_sha256": verified["contract_sha256"],
        "unit_scale_location": verified["scale_location"],
        "capture_maps": capture_maps,
        "assigned_objects": sorted(assigned),
        "atlas_configuration": configured,
    }
    context.scene[HANDOFF_SCENE_KEY] = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "available": True,
        **payload,
        "preview_material": preview.name,
    }
