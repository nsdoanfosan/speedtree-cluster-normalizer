"""Read-only, fail-closed validation for cluster delivery entry points.

This module deliberately does not repair scene data.  Both delivery scripts call it
before saving a delivery blend or invoking Atlas' SPM build operator.
"""

import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from .atlas_handoff import (
    CAMERA_BUNDLE_KEY,
    CAMERA_CONTRACT_HASH_KEY,
    CAMERA_CONTRACT_KEY,
    CAMERA_REFERENCE_COLLECTION,
    _sha256,
    _same_path,
    _validate_reference_artifacts,
)
from .normalization import (
    ASSET_ROLE_KEY,
    CARD_PROTOTYPE_MAP_HASH_KEY,
    CARD_PROTOTYPE_MAP_KEY,
    CAMERA_REFERENCE_KEY,
    COMPOSITE_PARTS_KEY,
    COUNTERPART_KEY,
    PROJECTION_BASIS_KEY,
    PROTOTYPE_ASSET_KEY,
    PROTOTYPE_INDEX_KEY,
    SOURCE_PARTITION_MODE_KEY,
    UV_TRANSFER_ATTACHMENT_POLICY,
    UV_TRANSFER_CANDIDATE_SELECTION_POLICY,
    UV_TRANSFER_KEY,
    UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR,
    UV_TRANSFER_MAX_NORMALIZED_RMS,
    _canonical_sha256,
    _ordered_boundary_indices,
    point_in_convex_polygon,
)


EXPECTED_TRANSFER_POLICY = "closed_loop_similarity_to_exact_camera_reference_boundary"
_TOLERANCE = 1.0e-6
_UV_TOLERANCE = 1.0e-7


def _json_property(owner, key, label):
    value = owner.get(key)
    if not value:
        raise ValueError(f"Camera delivery is missing {label}.")
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Camera delivery {label} is invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Camera delivery {label} must be a JSON object.")
    return result


def _validate_composite_parts(value, label):
    if not value:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")
    checked = []
    for expected_index, row in enumerate(value, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-object subpart.")
        index = int(row.get("subpart_index") or 0)
        asset = str(row.get("skeletal_asset_name") or "")
        raw_matrix = row.get("subpart_to_card_matrix")
        if index != expected_index or not asset.casefold().startswith("sk_"):
            raise ValueError(f"{label} has invalid subpart indices/assets.")
        try:
            matrix = Matrix(raw_matrix)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid 4x4 matrix: {asset}") from exc
        if (
            len(matrix) != 4
            or any(len(matrix_row) != 4 for matrix_row in matrix)
            or max(
                abs(float(matrix[3][axis]) - (1.0 if axis == 3 else 0.0))
                for axis in range(4)
            ) > _TOLERANCE
            or float(matrix.to_3x3().determinant()) <= 0.0
        ):
            raise ValueError(f"{label} contains a mirrored/non-affine matrix: {asset}")
        columns = [Vector(matrix.to_3x3().col[axis]) for axis in range(3)]
        lengths = [column.length for column in columns]
        if (
            min(lengths) <= _TOLERANCE
            or max(lengths) - min(lengths) > _TOLERANCE
            or max(
                abs((columns[first] / lengths[first]).dot(columns[second] / lengths[second]))
                for first, second in ((0, 1), (0, 2), (1, 2))
            ) > _TOLERANCE
        ):
            raise ValueError(f"{label} contains scale/shear: {asset}")
        checked.append({
            "subpart_index": index,
            "skeletal_asset_name": asset,
            "source_bone": str(row.get("source_bone") or ""),
            "endpoint_bone": str(row.get("endpoint_bone") or ""),
            "subpart_to_card_matrix": [
                [float(item) for item in matrix_row] for matrix_row in matrix
            ],
            "pivot_contract": str(row.get("pivot_contract") or ""),
        })
    if len({row["skeletal_asset_name"] for row in checked}) != len(checked):
        raise ValueError(f"{label} contains duplicate skeletal assets.")
    return checked


def _plan_composite_parts(plan):
    raw = plan.get(COMPOSITE_PARTS_KEY)
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Plan composite parts are invalid JSON: {plan.name}: {exc}"
        ) from exc
    return _validate_composite_parts(value, f"Plan composite parts on {plan.name}")


def _identity_object(obj):
    identity = Matrix.Identity(4)
    return obj.parent is None and all(
        abs(float(matrix[row][column] - identity[row][column])) <= _TOLERANCE
        for matrix in (obj.matrix_world, obj.matrix_basis)
        for row in range(4)
        for column in range(4)
    )


def _finite_vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != int(length):
        raise ValueError(f"Plan UV transfer {label} has the wrong shape.")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"Plan UV transfer {label} contains a non-finite value.")
    return result


def _validate_attachment_transfer(transfer, plane, camera, plan_name):
    if (
        transfer.get("candidate_selection_policy")
        != UV_TRANSFER_CANDIDATE_SELECTION_POLICY
        or transfer.get("attachment_policy") != UV_TRANSFER_ATTACHMENT_POLICY
    ):
        raise ValueError(f"Plan UV transfer attachment policy mismatch: {plan_name}")
    plan_attachment = _finite_vector(
        transfer.get("plan_attachment_xy"), 2, "plan attachment"
    )
    if max(abs(value) for value in plan_attachment) > _TOLERANCE:
        raise ValueError(f"Plan UV transfer attachment is not the normalized origin: {plan_name}")
    expected_reference_attachment = _finite_vector(
        (plane.get("attachment") or {}).get("source_plane_xy"),
        2,
        "camera contract attachment",
    )
    stored_reference_attachment = _finite_vector(
        transfer.get("reference_attachment_xy"), 2, "reference attachment"
    )
    if max(
        abs(stored_reference_attachment[axis] - expected_reference_attachment[axis])
        for axis in range(2)
    ) > _TOLERANCE:
        raise ValueError(f"Plan UV transfer reference attachment mismatch: {plan_name}")

    right = Vector(_finite_vector(camera.get("right"), 3, "camera right basis"))
    up = Vector(_finite_vector(camera.get("up"), 3, "camera up basis"))
    projected = [
        (float(Vector(vertex).dot(right)), float(Vector(vertex).dot(up)))
        for vertex in plane.get("vertices") or []
    ]
    if not projected:
        raise ValueError(f"Plan UV transfer reference has no vertices: {plan_name}")
    span_x = max(point[0] for point in projected) - min(point[0] for point in projected)
    span_y = max(point[1] for point in projected) - min(point[1] for point in projected)
    reference_extent = math.hypot(span_x, span_y)
    stored_extent = float(transfer.get("reference_extent_diagonal", math.nan))
    if (
        not math.isfinite(reference_extent)
        or reference_extent <= _TOLERANCE
        or not math.isfinite(stored_extent)
        or abs(stored_extent - reference_extent) > _TOLERANCE
    ):
        raise ValueError(f"Plan UV transfer reference extent mismatch: {plan_name}")

    scale = float(transfer.get("scale", math.nan))
    rotation = transfer.get("rotation")
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not isinstance(rotation, (list, tuple))
        or len(rotation) != 2
    ):
        raise ValueError(f"Plan UV transfer attachment fit is invalid: {plan_name}")
    rotation_rows = [
        _finite_vector(row, 2, "attachment rotation") for row in rotation
    ]
    translation = _finite_vector(
        transfer.get("translation"), 2, "attachment translation"
    )
    mapped_attachment = [
        scale * sum(
            plan_attachment[source_axis] * rotation_rows[source_axis][target_axis]
            for source_axis in range(2)
        )
        + translation[target_axis]
        for target_axis in range(2)
    ]
    stored_mapped_attachment = _finite_vector(
        transfer.get("mapped_attachment_xy"), 2, "mapped attachment"
    )
    if max(
        abs(stored_mapped_attachment[axis] - mapped_attachment[axis])
        for axis in range(2)
    ) > _TOLERANCE:
        raise ValueError(f"Plan UV transfer mapped attachment mismatch: {plan_name}")
    attachment_error = math.dist(mapped_attachment, expected_reference_attachment)
    attachment_error_normalized = attachment_error / reference_extent
    stored_error = float(transfer.get("attachment_origin_error", math.nan))
    stored_normalized = float(
        transfer.get("attachment_origin_error_normalized", math.nan)
    )
    stored_maximum = float(
        transfer.get("max_attachment_origin_error_normalized", math.nan)
    )
    if (
        not all(math.isfinite(value) for value in (
            stored_error,
            stored_normalized,
            stored_maximum,
        ))
        or abs(stored_error - attachment_error) > _TOLERANCE
        or abs(stored_normalized - attachment_error_normalized) > _TOLERANCE
        or abs(stored_maximum - UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR)
        > _TOLERANCE
        or attachment_error_normalized
        > UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR + _TOLERANCE
    ):
        raise ValueError(f"Plan UV transfer attachment origin is invalid: {plan_name}")


def _collection_objects_recursive(collection):
    result = set(collection.objects)
    for child in collection.children:
        result.update(_collection_objects_recursive(child))
    return result


def _vertex_uvs(mesh, *, expected_uvs=None, label=None):
    layer = mesh.uv_layers.get("UVMap")
    if layer is None:
        raise ValueError(f"Mesh has no UVMap: {label or mesh.name}")
    values = [None] * len(mesh.vertices)
    expected = expected_uvs if expected_uvs is not None else None
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv = [float(value) for value in layer.data[loop_index].uv]
            if expected is not None:
                if vertex_index >= len(expected) or max(
                    abs(uv[axis] - float(expected[vertex_index][axis]))
                    for axis in range(2)
                ) > _TOLERANCE:
                    raise ValueError(
                        f"Mesh loop UV payload differs from its exact reference: "
                        f"{label or mesh.name}"
                    )
            previous = values[vertex_index]
            if previous is not None and max(
                abs(uv[axis] - previous[axis]) for axis in range(2)
            ) > _UV_TOLERANCE:
                raise ValueError(
                    f"Mesh contains an unexpected UV seam: {label or mesh.name}"
                )
            values[vertex_index] = uv
    if any(value is None for value in values):
        raise ValueError(f"Mesh has an unreferenced UV vertex: {label or mesh.name}")
    return values


def _validate_reference_payload(reference, plane):
    if reference.type != "MESH" or not _identity_object(reference):
        raise ValueError(f"Camera reference is not an identity mesh: {reference.name}")
    mesh = reference.data
    expected_vertices = plane.get("vertices") or []
    expected_faces = plane.get("faces") or []
    expected_uvs = plane.get("uvs") or []
    if (
        len(mesh.vertices) != len(expected_vertices)
        or len(mesh.polygons) != len(expected_faces)
        or len(expected_uvs) != len(expected_vertices)
    ):
        raise ValueError(f"Camera reference topology count mismatch: {reference.name}")
    if any(
        (Vector(vertex.co) - Vector(expected)).length > _TOLERANCE
        for vertex, expected in zip(mesh.vertices, expected_vertices)
    ):
        raise ValueError(f"Camera reference vertex payload mismatch: {reference.name}")
    actual_faces = [list(int(value) for value in polygon.vertices) for polygon in mesh.polygons]
    normalized_expected_faces = [list(int(value) for value in face) for face in expected_faces]
    if actual_faces != normalized_expected_faces:
        raise ValueError(f"Camera reference face/loop order mismatch: {reference.name}")
    actual_uvs = _vertex_uvs(
        mesh,
        expected_uvs=expected_uvs,
        label=reference.name,
    )
    return actual_uvs


def _contract_file(contract, section, *, require_hash=True):
    row = contract.get(section) or {}
    path = Path(str(row.get("path") or "")).expanduser().absolute()
    if not path.is_file():
        raise ValueError(f"Camera delivery contract file is missing: {section}")
    if require_hash and _sha256(path) != str(row.get("sha256") or ""):
        raise ValueError(f"Camera delivery contract file hash is stale: {section}")
    return path


def _validate_material(
    material_name,
    contract,
    bundle,
    expected_albedo_path=None,
    expected_material_id=None,
):
    contract_material = contract.get("material") or {}
    if contract_material.get("name") != material_name:
        raise ValueError("Camera delivery material name differs from the camera contract.")
    try:
        contract_material_id = int(contract_material.get("id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Camera delivery contract material ID is invalid.") from exc
    if expected_material_id is not None and contract_material_id != int(expected_material_id):
        raise ValueError("Explicit source material ID differs from the camera contract.")
    maps = contract_material.get("maps") or {}
    color_path = Path(str((maps.get("Color") or {}).get("path") or "")).absolute()
    opacity_path = Path(str((maps.get("Opacity") or {}).get("path") or "")).absolute()
    if not color_path.is_file() or not opacity_path.is_file():
        raise ValueError("Camera contract Color + Opacity files must both exist.")
    if expected_albedo_path is not None and not _same_path(color_path, expected_albedo_path):
        raise ValueError("Explicit Atlas Color map differs from the camera contract.")
    if not _same_path(color_path, bundle.get("albedo_path", "")):
        raise ValueError("Persisted camera bundle Color path differs from the contract.")
    if not _same_path(opacity_path, bundle.get("opacity_path", "")):
        raise ValueError("Persisted camera bundle Opacity path differs from the contract.")

    material = bpy.data.materials.get(material_name)
    if material is None or not material.use_nodes:
        raise ValueError(f"Camera delivery material is missing: {material_name}")
    color = material.node_tree.nodes.get("SpeedTree Color")
    opacity = material.node_tree.nodes.get("SpeedTree Opacity")
    if (
        color is None
        or opacity is None
        or color.type != "TEX_IMAGE"
        or opacity.type != "TEX_IMAGE"
        or color.image is None
        or opacity.image is None
    ):
        raise ValueError("Camera delivery material does not contain real Color + Opacity nodes.")
    if color.extension != "CLIP" or opacity.extension != "CLIP":
        raise ValueError("Camera delivery texture nodes must use CLIP for overscan UVs.")
    for node, expected_path, label in (
        (color, color_path, "Color"),
        (opacity, opacity_path, "Opacity"),
    ):
        actual_path = Path(bpy.path.abspath(node.image.filepath)).expanduser().absolute()
        if node.image.source != "FILE" or not actual_path.is_file():
            raise ValueError(f"Camera delivery {label} node is not backed by a real file.")
        if not _same_path(actual_path, expected_path):
            raise ValueError(f"Camera delivery {label} node uses another image.")
    return material


def _expected_export_from_plans(plans):
    skeletal_bases = []
    for plan in plans:
        composite_parts = _plan_composite_parts(plan)
        if composite_parts:
            skeletal_bases.extend(
                row["skeletal_asset_name"] for row in composite_parts
            )
            continue
        skeletal_name = str(plan.get(COUNTERPART_KEY) or "")
        if not skeletal_name:
            raise ValueError(f"Plan has no Send to Unreal counterpart: {plan.name}")
        skeletal_bases.append(skeletal_name)
    return {
        name + suffix
        for name in set(skeletal_bases)
        for suffix in ("", "_Armature", "_Mesh")
    }


def _validate_export(scene, plans, references, expected_export_names=None):
    export_collection = bpy.data.collections.get("Export")
    if export_collection is None:
        raise ValueError("Camera delivery has no Send to Unreal Export collection.")
    derived_names = _expected_export_from_plans(plans)
    if expected_export_names is not None and set(expected_export_names) != derived_names:
        raise ValueError("Caller Export expectation differs from plan counterpart lineage.")
    recursive_objects = _collection_objects_recursive(export_collection)
    actual_names = {obj.name for obj in recursive_objects}
    if actual_names != derived_names:
        raise ValueError(
            "Send to Unreal Export is not isolated to the expected normalized assets: "
            + ", ".join(sorted(actual_names))
        )
    if export_collection.children:
        raise ValueError("Send to Unreal Export must not contain nested collections.")
    leaked = {obj.name for obj in (*plans, *references) if obj in recursive_objects}
    if leaked:
        raise ValueError(
            "Plan/reference leaked into Send to Unreal Export: "
            + ", ".join(sorted(leaked))
        )
    validated_assets = set()
    for plan in plans:
        composite_parts = _plan_composite_parts(plan)
        names = (
            [row["skeletal_asset_name"] for row in composite_parts]
            if composite_parts
            else [str(plan[COUNTERPART_KEY])]
        )
        for name in names:
            if name in validated_assets:
                continue
            validated_assets.add(name)
            pivot = bpy.data.objects.get(name)
            armature = bpy.data.objects.get(name + "_Armature")
            mesh = bpy.data.objects.get(name + "_Mesh")
            if (
                pivot is None
                or pivot.type != "EMPTY"
                or armature is None
                or armature.type != "ARMATURE"
                or mesh is None
                or mesh.type != "MESH"
                or armature.parent is not pivot
                or mesh.parent is not armature
                or any(obj.matrix_world != Matrix.Identity(4) for obj in (pivot, armature, mesh))
                or pivot.get(ASSET_ROLE_KEY) != "send2ue_pivot"
                or armature.get(ASSET_ROLE_KEY) != "skeletal_armature"
                or mesh.get(ASSET_ROLE_KEY) != "skeletal_mesh"
            ):
                raise ValueError(f"Send to Unreal hierarchy/role mismatch: {name}")
    return sorted(derived_names)


def validate_camera_uv_delivery(
    scene,
    plan_collection_name,
    material_name,
    *,
    plan_base=None,
    expected_export_names=None,
    expected_camera_spm=None,
    expected_camera_name=None,
    expected_tree_spm=None,
    expected_albedo_path=None,
    expected_material_id=None,
    expected_card_count=None,
    expected_prototype_count=None,
):
    """Assert the exact persisted camera-UV delivery without mutating Blender data."""
    contract = _json_property(scene, CAMERA_CONTRACT_KEY, "Scene camera contract")
    bundle = _json_property(scene, CAMERA_BUNDLE_KEY, "Scene camera bundle")
    contract_hash = str(scene.get(CAMERA_CONTRACT_HASH_KEY) or "")
    if not contract_hash or _canonical_sha256(contract) != contract_hash:
        raise ValueError("Scene camera contract hash is missing or stale.")
    if bundle.get("contract_sha256") != contract_hash:
        raise ValueError("Scene camera bundle references another contract.")
    if bundle.get("reference_collection") != CAMERA_REFERENCE_COLLECTION:
        raise ValueError("Scene camera bundle references another reference collection.")
    validation = contract.get("validation") or {}
    if (
        validation.get("status") != "ready"
        or validation.get("strict_vertex_uv_topology") is not True
    ):
        raise ValueError("Scene camera contract is not strict and ready.")

    camera_spm = _contract_file(contract, "camera_spm", require_hash=True)
    tree_spm = _contract_file(contract, "tree_spm", require_hash=False)
    if not _same_path(bundle.get("camera_spm", ""), camera_spm):
        raise ValueError("Persisted camera bundle targets another camera SPM.")
    if not _same_path(bundle.get("tree_spm", ""), tree_spm):
        raise ValueError("Persisted camera bundle targets another tree SPM.")
    if expected_camera_spm is not None and not _same_path(camera_spm, expected_camera_spm):
        raise ValueError("Explicit Camera SPM differs from the persisted camera contract.")
    if expected_tree_spm is not None and not _same_path(tree_spm, expected_tree_spm):
        raise ValueError("Explicit target SPM differs from the persisted camera contract.")
    if expected_camera_name is not None and (
        contract.get("camera", {}).get("name") != expected_camera_name
    ):
        raise ValueError("Explicit camera name differs from the persisted camera contract.")

    for path_key, hash_key, label in (
        ("albedo_path", "albedo_sha256", "Color map"),
        ("opacity_path", "opacity_sha256", "Opacity map"),
        ("reference_blend", "reference_blend_sha256", "reference blend"),
        ("manifest_path", "manifest_sha256", "normalization manifest"),
        ("validation_path", "validation_sha256", "Blender validation"),
    ):
        path = Path(str(bundle.get(path_key) or ""))
        if not path.is_file() or _sha256(path) != str(bundle.get(hash_key) or ""):
            raise ValueError(f"Camera delivery {label} hash is missing or stale.")
    _validate_reference_artifacts(
        contract,
        Path(bundle["reference_blend"]),
        Path(bundle["manifest_path"]),
        Path(bundle["validation_path"]),
    )

    planes = contract.get("planes") or []
    plane_count = len(planes)
    if plane_count < 1:
        raise ValueError("Camera delivery contract contains no card planes.")
    if expected_card_count is not None and plane_count != int(expected_card_count):
        raise ValueError(
            f"Camera delivery card count differs from the explicit expectation: "
            f"{plane_count} vs {int(expected_card_count)}."
        )
    plane_names = [str(plane.get("name") or "") for plane in planes]
    if not all(plane_names) or len(set(plane_names)) != plane_count:
        raise ValueError("Camera delivery contract plane names are missing or duplicated.")
    if plan_base:
        expected_names = [
            f"{plan_base}_{index:02d}"
            for index in range(1, plane_count + 1)
        ]
        if plane_names != expected_names:
            raise ValueError("Camera delivery plan base does not match the contract.")

    plan_collection = bpy.data.collections.get(plan_collection_name)
    reference_collection = bpy.data.collections.get(CAMERA_REFERENCE_COLLECTION)
    if plan_collection is None or reference_collection is None:
        raise ValueError("Camera delivery plan/reference collection is missing.")
    expected_reference_names = {"AtlasCameraRef_" + name for name in plane_names}
    if plan_collection.children or {obj.name for obj in plan_collection.objects} != set(plane_names):
        raise ValueError("Camera delivery plan collection does not exactly match contract planes.")
    if (
        reference_collection.children
        or {obj.name for obj in reference_collection.objects} != expected_reference_names
    ):
        raise ValueError("Camera delivery reference collection does not exactly match contract planes.")

    material = _validate_material(
        material_name,
        contract,
        bundle,
        expected_albedo_path=expected_albedo_path,
        expected_material_id=expected_material_id,
    )
    plan_objects = {obj.name: obj for obj in plan_collection.objects}
    prototype_map = _json_property(
        scene, CARD_PROTOTYPE_MAP_KEY, "card/prototype mapping contract"
    )
    prototype_map_hash = str(scene.get(CARD_PROTOTYPE_MAP_HASH_KEY) or "")
    if not prototype_map_hash or _canonical_sha256(prototype_map) != prototype_map_hash:
        raise ValueError("Card/prototype mapping contract hash is missing or stale.")
    mapping_rows = prototype_map.get("cards") or []
    if (
        int(prototype_map.get("card_count", -1)) != plane_count
        or len(mapping_rows) != plane_count
    ):
        raise ValueError("Card/prototype mapping count differs from camera planes.")
    mapping_by_plan = {str(row.get("plan") or ""): row for row in mapping_rows}
    if set(mapping_by_plan) != set(plane_names):
        raise ValueError("Card/prototype mapping plan names differ from camera planes.")
    partition_mode = str(prototype_map.get("source_partition_mode") or "")
    composite_parts = _validate_composite_parts(
        prototype_map.get("composite_parts") or [],
        "Card/prototype composite contract",
    )
    composite_set_id = str(prototype_map.get("composite_set_id") or "")
    if partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
        if (
            int(prototype_map.get("version", 0)) != 2
            or not composite_parts
            or composite_set_id != _canonical_sha256(composite_parts)
        ):
            raise ValueError("Card/prototype composite set is missing or stale.")
        for mapping in mapping_rows:
            row_parts = _validate_composite_parts(
                mapping.get("composite_parts") or [],
                "Card composite mapping",
            )
            if (
                row_parts != composite_parts
                or str(mapping.get("composite_set_id") or "") != composite_set_id
            ):
                raise ValueError("Cards do not reference one identical composite set.")
        prototype_assets = {
            row["skeletal_asset_name"] for row in composite_parts
        }
    else:
        if composite_parts or composite_set_id:
            raise ValueError("Non-composite card mapping contains composite data.")
        prototype_assets = {
            str(row.get("prototype_asset") or "") for row in mapping_rows
        }
    if "" in prototype_assets:
        raise ValueError("Card/prototype mapping contains an empty prototype asset.")
    prototype_count = len(prototype_assets)
    if int(prototype_map.get("prototype_count", -1)) != prototype_count:
        raise ValueError("Card/prototype mapping prototype count is stale.")
    if (
        expected_prototype_count is not None
        and prototype_count != int(expected_prototype_count)
    ):
        raise ValueError(
            "Camera delivery prototype count differs from the explicit expectation."
        )
    references = []
    rows = []
    for plane in planes:
        plan = plan_objects[plane["name"]]
        mapping = mapping_by_plan[plane["name"]]
        reference_name = "AtlasCameraRef_" + plane["name"]
        reference = bpy.data.objects.get(reference_name)
        if plan.type != "MESH" or reference is None:
            raise ValueError(f"Camera delivery plan/reference mesh is missing: {plane['name']}")
        if plan.get(ASSET_ROLE_KEY) != "speedtree_plan":
            raise ValueError(f"Camera delivery plan role mismatch: {plan.name}")
        references.append(reference)
        _validate_reference_payload(reference, plane)
        if reference.get(CAMERA_REFERENCE_KEY) != plane["name"]:
            raise ValueError(f"Camera reference plane pointer mismatch: {reference_name}")
        if reference.get(CAMERA_CONTRACT_HASH_KEY) != contract_hash:
            raise ValueError(f"Camera reference contract hash mismatch: {reference_name}")
        if plan.get(CAMERA_CONTRACT_HASH_KEY) != contract_hash:
            raise ValueError(f"Plan contract hash mismatch: {plan.name}")
        if plan.data.get(CAMERA_CONTRACT_HASH_KEY) != contract_hash:
            raise ValueError(f"Plan mesh contract hash mismatch: {plan.name}")
        if plan.get(CAMERA_REFERENCE_KEY) != reference_name:
            raise ValueError(f"Plan reference pointer mismatch: {plan.name}")
        if not _identity_object(plan):
            raise ValueError(f"Camera delivery plan is not an identity object: {plan.name}")

        transfer = _json_property(plan, UV_TRANSFER_KEY, f"UV transfer on {plan.name}")
        projection_basis = _json_property(
            plan, PROJECTION_BASIS_KEY, f"projection basis on {plan.name}"
        )
        plan_composite_parts = _plan_composite_parts(plan)
        mapping_composite_parts = _validate_composite_parts(
            mapping.get("composite_parts") or [],
            f"Card composite mapping on {plan.name}",
        )
        if partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
            if (
                plan_composite_parts != composite_parts
                or mapping_composite_parts != composite_parts
                or str(plan.get("speedtree_cluster_composite_set_id") or "")
                != composite_set_id
            ):
                raise ValueError(f"Plan composite lineage mismatch: {plan.name}")
        elif plan_composite_parts or mapping_composite_parts:
            raise ValueError(f"Non-composite plan contains composite data: {plan.name}")
        if (
            transfer.get("policy") != EXPECTED_TRANSFER_POLICY
            or transfer.get("contract_sha256") != contract_hash
            or transfer.get("reference_plane") != plane["name"]
            or transfer.get("reference_object") != reference_name
            or int(transfer.get("source_mesh_id", -1)) != int(plane["source_mesh_id"])
            or transfer.get("reference_topology_sha256") != plane["topology_sha256"]
            or transfer.get("reference_uv_sha256") != plane["uv_sha256"]
            or not _same_path(transfer.get("reference_blend", ""), bundle["reference_blend"])
            or transfer.get("reference_blend_sha256") != bundle["reference_blend_sha256"]
            or transfer.get("projection_basis") != projection_basis
            or int(transfer.get("prototype_index", -1))
            != int(mapping.get("prototype_index", -2))
            or transfer.get("prototype_asset") != mapping.get("prototype_asset")
            or transfer.get("source_partition_mode")
            != prototype_map.get("source_partition_mode")
            or plan.get(PROTOTYPE_INDEX_KEY) != int(mapping.get("prototype_index", -1))
            or plan.get(PROTOTYPE_ASSET_KEY) != mapping.get("prototype_asset")
            or plan.get(SOURCE_PARTITION_MODE_KEY)
            != prototype_map.get("source_partition_mode")
            or plan.get(COUNTERPART_KEY) != mapping.get("prototype_asset")
            or int(mapping.get("source_mesh_id", -1)) != int(plane["source_mesh_id"])
        ):
            raise ValueError(f"Plan UV transfer lineage mismatch: {plan.name}")
        normalized_rms = float(transfer.get("normalized_rms", math.inf))
        maximum_rms = float(transfer.get("max_normalized_rms", -math.inf))
        if (
            not math.isfinite(normalized_rms)
            or normalized_rms < 0.0
            or abs(maximum_rms - UV_TRANSFER_MAX_NORMALIZED_RMS) > _TOLERANCE
            or normalized_rms > maximum_rms
        ):
            raise ValueError(f"Plan UV transfer fit is invalid: {plan.name}")
        determinant = float(transfer.get("determinant", math.nan))
        if (
            transfer.get("orientation_preserving") is not True
            or not math.isfinite(determinant)
            or determinant <= 0.0
            or abs(determinant - 1.0) > _TOLERANCE
        ):
            raise ValueError(f"Plan UV transfer is mirrored: {plan.name}")
        _validate_attachment_transfer(
            transfer,
            plane,
            contract.get("camera") or {},
            plan.name,
        )
        expected_uvs = transfer.get("result_uvs") or []
        if len(expected_uvs) != len(plan.data.vertices):
            raise ValueError(f"Plan stored UV result count mismatch: {plan.name}")
        actual_uvs = _vertex_uvs(
            plan.data,
            expected_uvs=expected_uvs,
            label=plan.name,
        )
        stored_hash = _canonical_sha256(expected_uvs)
        actual_hash = _canonical_sha256(actual_uvs)
        if stored_hash != transfer.get("result_uv_sha256") or actual_hash != stored_hash:
            raise ValueError(f"Plan actual UV hash does not match the stored result: {plan.name}")
        if [slot for slot in plan.data.materials] != [material]:
            raise ValueError(f"Plan does not use exactly {material_name}: {plan.name}")
        if [slot for slot in reference.data.materials] != [material]:
            raise ValueError(f"Reference does not use exactly {material_name}: {reference.name}")
        if reference.name in plan_collection.objects or plan.name in reference_collection.objects:
            raise ValueError(f"Plan/reference collection roles are mixed: {plan.name}")

        if plan_composite_parts:
            coverage_points = []
            for subpart in plan_composite_parts:
                part_name = subpart["skeletal_asset_name"] + "_Mesh"
                part = bpy.data.objects.get(part_name)
                if part is None or part.type != "MESH":
                    raise ValueError(
                        f"Plan composite counterpart mesh is missing: {part_name}"
                    )
                relative = Matrix(subpart["subpart_to_card_matrix"])
                coverage_points.extend(
                    relative @ vertex.co for vertex in part.data.vertices
                )
        else:
            part_name = str(plan.get(COUNTERPART_KEY) or "") + "_Mesh"
            part = bpy.data.objects.get(part_name)
            if part is None or part.type != "MESH":
                raise ValueError(f"Plan counterpart mesh is missing: {plan.name}")
            coverage_points = [vertex.co.copy() for vertex in part.data.vertices]
        try:
            right = Vector(projection_basis["right"])
            up = Vector(projection_basis["up"])
            normal = Vector(projection_basis["normal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Plan projection basis is invalid: {plan.name}") from exc
        if min(right.length, up.length, normal.length) <= _TOLERANCE:
            raise ValueError(f"Plan projection basis is degenerate: {plan.name}")
        right.normalize()
        up.normalize()
        normal.normalize()
        if (
            abs(right.dot(up)) > _TOLERANCE
            or abs(right.dot(normal)) > _TOLERANCE
            or abs(up.dot(normal)) > _TOLERANCE
            or right.cross(up).dot(normal) < 1.0 - _TOLERANCE
        ):
            raise ValueError(f"Plan projection basis is not right-handed: {plan.name}")
        boundary_indices = _ordered_boundary_indices(
            [tuple(int(value) for value in polygon.vertices) for polygon in plan.data.polygons]
        )
        plan_boundary = [
            (
                float(plan.data.vertices[index].co.dot(right)),
                float(plan.data.vertices[index].co.dot(up)),
            )
            for index in boundary_indices
        ]
        if any(
            abs(float(vertex.co.dot(normal))) > _TOLERANCE
            for vertex in plan.data.vertices
        ):
            raise ValueError(f"Camera delivery plan is not planar: {plan.name}")
        spans = [
            max(point[axis] for point in plan_boundary)
            - min(point[axis] for point in plan_boundary)
            for axis in range(2)
        ]
        tolerance = max(max(spans), 1.0) ** 2 * 1.0e-7
        if any(
            not point_in_convex_polygon(
                (
                    float(vertex.dot(right)),
                    float(vertex.dot(up)),
                ),
                plan_boundary,
                tolerance=tolerance,
            )
            for vertex in coverage_points
        ):
            raise ValueError(f"Plan no longer covers its normalized part: {plan.name}")
        rows.append(
            {
                "plan": plan.name,
                "reference": reference.name,
                "source_mesh_id": int(plane["source_mesh_id"]),
                "reference_uv_sha256": plane["uv_sha256"],
                "result_uv_sha256": actual_hash,
                "normalized_rms": normalized_rms,
                "composite_subpart_count": len(plan_composite_parts),
            }
        )

    export_names = _validate_export(
        scene,
        list(plan_objects.values()),
        references,
        expected_export_names=expected_export_names,
    )
    return {
        "contract_sha256": contract_hash,
        "bundle_sha256": _canonical_sha256(bundle),
        "camera_spm": str(camera_spm),
        "camera_name": contract.get("camera", {}).get("name"),
        "reference_blend_sha256": bundle["reference_blend_sha256"],
        "material": material_name,
        "plan_collection": plan_collection_name,
        "card_count": plane_count,
        "prototype_count": prototype_count,
        "source_partition_mode": prototype_map.get("source_partition_mode"),
        "card_prototype_map_sha256": prototype_map_hash,
        "export_objects": export_names,
        "planes": rows,
    }
