"""Cluster-owned validation for the shared Blender -> SpeedTree unit receipt."""

import hashlib
import json
import math


UNIT_PROBE_KIND = "speedtree_fbx_spm_unit_probe"
UNIT_PROBE_VERSION = 1
SCALE_LOCATIONS = {"IDENTITY", "FBX_GEOMETRY", "SPM_MESH_ASSET"}
REQUIRED_GENERATOR_TYPES = {"Frond", "Leaf Mesh"}


def canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and greater than zero.")
    return result


def _is_identity(value, tolerance=1.0e-9):
    return abs(float(value) - 1.0) <= tolerance


def validate_unit_probe_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("SpeedTree unit probe contract must be a JSON object.")
    if contract.get("kind") != UNIT_PROBE_KIND:
        raise ValueError("SpeedTree unit probe contract kind is invalid.")
    if int(contract.get("version", 0)) != UNIT_PROBE_VERSION:
        raise ValueError("SpeedTree unit probe contract version is unsupported.")
    if str(contract.get("status") or "").casefold() != "verified":
        raise ValueError("SpeedTree unit probe contract is not verified.")

    target_meters = _positive_float(
        contract.get("physical_target_meters"),
        "Physical target meters",
    )
    blender_units = contract.get("blender_units")
    if not isinstance(blender_units, dict):
        raise ValueError("SpeedTree unit probe contract has no Blender unit evidence.")
    if str(blender_units.get("system") or "").upper() != "METRIC":
        raise ValueError("Production SpeedTree unit probe requires Blender METRIC units.")
    scale_length = _positive_float(
        blender_units.get("scale_length"),
        "Blender scale_length",
    )
    target_bu = _positive_float(
        blender_units.get("target_blender_units"),
        "Physical target Blender Units",
    )
    if abs(target_bu * scale_length - target_meters) > max(
        target_meters * 1.0e-6,
        1.0e-9,
    ):
        raise ValueError(
            "SpeedTree unit probe target does not match Blender scale_length."
        )

    selected = contract.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("SpeedTree unit probe has no selected scale contract.")
    geometry_scale = _positive_float(
        selected.get("mesh_geometry_scale"),
        "FBX geometry scale",
    )
    mesh_asset_scale = _positive_float(
        selected.get("mesh_asset_scale"),
        "SpeedTree Mesh asset scale",
    )
    generator_scale = _positive_float(
        selected.get("generator_scale", 1.0),
        "Generator scale",
    )
    if not _is_identity(generator_scale):
        raise ValueError(
            "Production unit contract forbids generator or Frond role-specific scale."
        )
    non_identity = [
        name
        for name, value in (
            ("FBX_GEOMETRY", geometry_scale),
            ("SPM_MESH_ASSET", mesh_asset_scale),
        )
        if not _is_identity(value)
    ]
    if len(non_identity) > 1:
        raise ValueError(
            "SpeedTree unit conversion is duplicated across FBX geometry and Mesh Scale."
        )
    expected_location = non_identity[0] if non_identity else "IDENTITY"
    selected_location = str(selected.get("scale_location") or "").upper()
    if selected_location not in SCALE_LOCATIONS:
        raise ValueError("SpeedTree unit probe scale location is invalid.")
    if selected_location != expected_location:
        raise ValueError(
            "SpeedTree unit probe scale location disagrees with its numeric scales."
        )
    effective_scale = _positive_float(
        selected.get("effective_scale"),
        "Effective SpeedTree scale",
    )
    if abs(effective_scale - geometry_scale * mesh_asset_scale) > max(
        effective_scale * 1.0e-9,
        1.0e-12,
    ):
        raise ValueError("SpeedTree effective scale is internally inconsistent.")

    generator_results = contract.get("generator_results")
    if not isinstance(generator_results, list):
        raise ValueError("SpeedTree unit probe has no generator measurements.")
    measured_types = {
        str(row.get("generator_type") or "")
        for row in generator_results
        if isinstance(row, dict)
        and str(row.get("status") or "").casefold() == "verified"
        and bool(row.get("same_unit_contract"))
    }
    missing = REQUIRED_GENERATOR_TYPES - measured_types
    if missing:
        raise ValueError(
            "SpeedTree unit probe did not verify the common contract for: "
            + ", ".join(sorted(missing))
        )

    normalized = json.loads(
        json.dumps(contract, ensure_ascii=False, sort_keys=True)
    )
    normalized["physical_target_meters"] = target_meters
    normalized["blender_units"]["scale_length"] = scale_length
    normalized["blender_units"]["target_blender_units"] = target_bu
    normalized["selected"].update(
        {
            "mesh_geometry_scale": geometry_scale,
            "mesh_asset_scale": mesh_asset_scale,
            "generator_scale": generator_scale,
            "scale_location": selected_location,
            "effective_scale": effective_scale,
        }
    )
    return {
        "contract": normalized,
        "contract_sha256": canonical_sha256(normalized),
        "mesh_geometry_scale": geometry_scale,
        "mesh_asset_scale": mesh_asset_scale,
        "generator_scale": generator_scale,
        "scale_location": selected_location,
        "effective_scale": effective_scale,
        "physical_target_meters": target_meters,
        "target_blender_units": target_bu,
        "verified_generator_types": sorted(measured_types),
    }
