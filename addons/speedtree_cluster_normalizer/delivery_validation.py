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
from .attachment_contract import (
    GEOMETRY_SUPPORTED_ATTACHMENT_POLICY,
    _parse_speedtree_xml,
    attachment_endpoint_world,
    attachment_origin_world,
)
from .normalization import (
    ASSET_ROLE_KEY,
    CARD_PROTOTYPE_MAP_HASH_KEY,
    CARD_PROTOTYPE_MAP_KEY,
    CAMERA_REFERENCE_KEY,
    COMPOSITE_PARTS_KEY,
    COUNTERPART_KEY,
    PROJECTION_BASIS_KEY,
    PROJECTION_COVERAGE_KEY,
    PROTOTYPE_ASSET_KEY,
    PROTOTYPE_INDEX_KEY,
    DIRECT_CAPTURE_UV_KEY,
    DIRECT_CAPTURE_UV_SOURCE,
    PHYSICAL_CAPTURE_ALIGNED_FRAME_POLICY,
    PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK,
    PHYSICAL_CAPTURE_DIRECTION_PROJECTED_XML,
    PHYSICAL_CAPTURE_CONTRACT_HASH_KEY,
    PHYSICAL_CAPTURE_CONTRACT_KEY,
    physical_capture_projection_tolerance,
    ROOT_BRIDGE_MAX_GAP_RATIO,
    SOURCE_PARTITION_MODE_KEY,
    SOURCE_3D_CONTRACT_HASH_KEY,
    SOURCE_3D_CONTRACT_KEY,
    XML_ATTACHMENT_KEY,
    PLAN_ROOT_LOCK_KEY,
    UV_TRANSFER_ATTACHMENT_POLICY,
    UV_TRANSFER_CANDIDATE_SELECTION_POLICY,
    UV_TRANSFER_KEY,
    UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR,
    UV_TRANSFER_MAX_NORMALIZED_RMS,
    _canonical_sha256,
    _validate_physical_capture_contract,
    _transfer_camera_boundary_uvs,
    _uniform_plan_triangulation,
    _ordered_boundary_indices,
    projection_coverage_2d,
)


EXPECTED_TRANSFER_POLICY = "closed_loop_similarity_to_exact_camera_reference_boundary"
AUTO_CAPTURE_CONTRACT_KEY = "speedtree_cluster_auto_capture_contract"
AUTO_CAPTURE_CONTRACT_HASH_KEY = "speedtree_cluster_auto_capture_contract_sha256"
AUTO_CAPTURE_COLLECTION = "Atlas_Auto_Capture"
AUTO_CAPTURE_KIND = "speedtree_cluster_blender_auto_capture_contract"
AUTO_CAPTURE_MANIFEST_KIND = "speedtree_cluster_blender_auto_capture"
AUTO_CAPTURE_UV_POLICY = "direct_world_axis_capture_projection"
AUTO_CAPTURE_MAP_ROLES = (
    "Color",
    "Opacity",
    "Normal",
    "Gloss",
    "SubsurfaceColor",
    "SubsurfaceAmount",
    "AO",
    "Height",
)
AUTO_CAPTURE_BASES = {
    "XY": {
        "right": (1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "normal": (0.0, 0.0, 1.0),
        "view_direction": (0.0, 0.0, -1.0),
        "rotation_degrees": 0.0,
    },
    "XZ": {
        "right": (1.0, 0.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "normal": (0.0, -1.0, 0.0),
        "view_direction": (0.0, 1.0, 0.0),
        "rotation_degrees": 90.0,
    },
    "YZ": {
        "right": (0.0, 1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "normal": (1.0, 0.0, 0.0),
        "view_direction": (-1.0, 0.0, 0.0),
        "rotation_degrees": 90.0,
    },
}
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


def _validate_source_3d_contract(scene):
    contract = _json_property(scene, SOURCE_3D_CONTRACT_KEY, "source 3D contract")
    expected_hash = str(scene.get(SOURCE_3D_CONTRACT_HASH_KEY) or "")
    if not expected_hash or _canonical_sha256(contract) != expected_hash:
        raise ValueError("Source 3D contract hash is missing or stale.")
    for path_key, hash_key, label in (
        ("xml_path", "xml_sha256", "Source 3D XML"),
        ("source_spm", "source_spm_sha256", "Source 3D SPM"),
        ("source_fbx", "source_fbx_sha256", "Source 3D FBX"),
    ):
        path = Path(str(contract.get(path_key) or ""))
        expected = str(contract.get(hash_key) or "")
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise ValueError(f"{label} hash is missing or stale.")
    scale = float(contract.get("scale", math.nan))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Source 3D contract scale is invalid.")
    root_ids = [int(value) for value in contract.get("root_ids") or []]
    if not root_ids or len(root_ids) != len(set(root_ids)):
        raise ValueError("Source 3D contract root IDs are missing or duplicated.")
    xml_path = Path(contract["xml_path"]).resolve()
    parsed_source_spm_text, parsed_bones = _parse_speedtree_xml(xml_path)
    parsed_source_spm = Path(parsed_source_spm_text).expanduser()
    if not parsed_source_spm.is_absolute():
        parsed_source_spm = xml_path.parent / parsed_source_spm
    if not _same_path(parsed_source_spm, contract["source_spm"]):
        raise ValueError("Source 3D XML points to a different source SPM.")
    parsed_roots = [bone for bone in parsed_bones if int(bone["parent_id"]) == -1]
    if [int(root["id"]) for root in parsed_roots] != root_ids:
        raise ValueError("Source 3D XML structural roots differ from its contract.")
    authoritative_roots = [
        {
            "xml_bone_id": int(root["id"]),
            "xml_parent_id": int(root["parent_id"]),
            "xml_start_world": [float(value) for value in root["start_raw"] / scale],
            "xml_end_world": [float(value) for value in root["end_raw"] / scale],
            "xml_radius_world": float(root["radius_raw"] / scale),
            "xml_generator": str(root.get("generator") or ""),
        }
        for root in parsed_roots
    ]
    xml_mtime_ns = int(contract.get("xml_mtime_ns", -1))
    if xml_mtime_ns != xml_path.stat().st_mtime_ns:
        raise ValueError("Source 3D XML timestamp differs from its contract.")
    for source_key in ("source_spm", "source_fbx"):
        if Path(str(contract[source_key])).stat().st_mtime_ns > xml_mtime_ns:
            raise ValueError("Source 3D XML is older than its source export.")
    return contract, expected_hash, root_ids, authoritative_roots


def _validate_xml_attachment(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    start = _finite_vector(value.get("xml_start_world"), 3, f"{label} start")
    end = _finite_vector(value.get("xml_end_world"), 3, f"{label} end")
    if (Vector(end) - Vector(start)).length <= _TOLERANCE:
        raise ValueError(f"{label} segment is degenerate.")
    root_id = int(value.get("xml_bone_id", -1))
    if root_id < 0 or int(value.get("xml_parent_id", 0)) >= 0:
        raise ValueError(f"{label} is not an XML structural root.")
    tolerance = float(value.get("match_tolerance", math.nan))
    start_error = float(value.get("start_match_error", math.nan))
    end_error = float(value.get("end_match_error", math.nan))
    if (
        not all(math.isfinite(item) for item in (tolerance, start_error, end_error))
        or tolerance < 0.0
        or start_error < 0.0
        or end_error < 0.0
        or start_error > tolerance + _TOLERANCE
        or end_error > tolerance + _TOLERANCE
    ):
        raise ValueError(f"{label} match error exceeds its contract.")
    effective_keys = {
        "effective_attachment_world",
        "effective_endpoint_world",
        "effective_support_distance_world",
        "effective_direction_length_world",
        "effective_geometry_scale_world",
        "effective_support_min_projection_world",
        "effective_support_max_projection_world",
        "effective_support_tolerance_world",
        "effective_attachment_policy",
    }
    present_effective_keys = effective_keys.intersection(value)
    if present_effective_keys and present_effective_keys != effective_keys:
        raise ValueError(f"{label} geometry-supported attachment is incomplete.")
    if present_effective_keys:
        effective_start = Vector(
            _finite_vector(
                value["effective_attachment_world"],
                3,
                f"{label} effective start",
            )
        )
        effective_end = Vector(
            _finite_vector(
                value["effective_endpoint_world"],
                3,
                f"{label} effective end",
            )
        )
        xml_start = Vector(start)
        xml_end = Vector(end)
        xml_direction = xml_end - xml_start
        xml_length = xml_direction.length
        axis = xml_direction / xml_length
        support_distance = float(value["effective_support_distance_world"])
        direction_length = float(value["effective_direction_length_world"])
        geometry_scale = float(value["effective_geometry_scale_world"])
        minimum_projection = float(
            value["effective_support_min_projection_world"]
        )
        maximum_projection = float(
            value["effective_support_max_projection_world"]
        )
        support_tolerance = float(value["effective_support_tolerance_world"])
        if (
            value["effective_attachment_policy"]
            != GEOMETRY_SUPPORTED_ATTACHMENT_POLICY
            or not all(
                math.isfinite(item)
                for item in (
                    support_distance,
                    direction_length,
                    geometry_scale,
                    minimum_projection,
                    maximum_projection,
                    support_tolerance,
                )
            )
            or support_distance < 0.0
            or support_distance > xml_length + _TOLERANCE
            or direction_length <= 0.0
            or direction_length
            > min(xml_length - support_distance, geometry_scale) + _TOLERANCE
            or geometry_scale <= 0.0
            or minimum_projection > maximum_projection
            or support_tolerance <= 0.0
            or (
                effective_start
                - (xml_start + axis * support_distance)
            ).length
            > _TOLERANCE
            or (
                effective_end
                - (effective_start + axis * direction_length)
            ).length
            > _TOLERANCE
        ):
            raise ValueError(
                f"{label} geometry-supported attachment evidence is invalid."
            )
    return value


def _attachment_matches_contract(attachment, root):
    if (
        int(attachment.get("xml_bone_id", -1)) != int(root.get("xml_bone_id", -2))
        or int(attachment.get("xml_parent_id", 0)) != int(root.get("xml_parent_id", 1))
        or str(attachment.get("xml_generator") or "")
        != str(root.get("xml_generator") or "")
    ):
        return False
    for key in ("xml_start_world", "xml_end_world"):
        if (Vector(attachment[key]) - Vector(root[key])).length > _TOLERANCE:
            return False
    return abs(
        float(attachment.get("xml_radius_world", math.nan))
        - float(root.get("xml_radius_world", math.nan))
    ) <= _TOLERANCE


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
            or max(abs(length - 1.0) for length in lengths) > _TOLERANCE
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


def _validate_triangle_disk(mesh, label):
    faces = [tuple(int(value) for value in polygon.vertices) for polygon in mesh.polygons]
    if not faces or any(len(face) != 3 for face in faces):
        raise ValueError(f"Plan CDT topology is not all triangles: {label}")
    edge_counts = {}
    referenced = set()
    for face in faces:
        if len(set(face)) != 3:
            raise ValueError(f"Plan CDT topology contains a degenerate triangle: {label}")
        referenced.update(face)
        for offset, first in enumerate(face):
            edge = tuple(sorted((first, face[(offset + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if referenced != set(range(len(mesh.vertices))):
        raise ValueError(f"Plan CDT topology contains an unreferenced vertex: {label}")
    if any(count not in {1, 2} for count in edge_counts.values()):
        raise ValueError(f"Plan CDT topology is non-manifold: {label}")
    boundary_indices = _ordered_boundary_indices(faces)
    if len(mesh.vertices) - len(edge_counts) + len(faces) != 1:
        raise ValueError(f"Plan CDT topology is not one manifold disk: {label}")
    return faces, boundary_indices


def _validate_external_camera_uv(plan, plane, camera, actual_uvs, transfer):
    faces, boundary_indices = _validate_triangle_disk(plan.data, plan.name)
    boundary = [
        (
            float(plan.data.vertices[index].co.x),
            float(plan.data.vertices[index].co.y),
        )
        for index in boundary_indices
    ]
    external_boundary_uvs, _external_transfer = _transfer_camera_boundary_uvs(
        boundary,
        plane,
        camera,
        plan_attachment_xy=(0.0, 0.0),
    )
    if any(
        max(
            abs(actual_uvs[vertex_index][axis] - external_boundary_uvs[offset][axis])
            for axis in range(2)
        )
        > _TOLERANCE
        for offset, vertex_index in enumerate(boundary_indices)
    ):
        raise ValueError(
            f"Plan boundary UV differs from the external camera contract: {plan.name}"
        )
    try:
        refinement_levels = int(transfer.get("plan_refinement_levels", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Plan refinement contract is invalid: {plan.name}") from exc
    pivot_uv = _finite_vector(
        (plane.get("attachment") or {}).get("pivot_uv"),
        2,
        "camera contract pivot UV",
    )
    root_lock = transfer.get("plan_root_lock") or {}
    try:
        containment_tolerance = float(root_lock["tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Plan root-lock containment tolerance is invalid: {plan.name}"
        ) from exc
    (
        expected_vertices,
        expected_uvs,
        expected_faces,
        expected_attachment_index,
    ) = _uniform_plan_triangulation(
        boundary,
        external_boundary_uvs.tolist(),
        refinement_levels,
        attachment_point=(0.0, 0.0),
        attachment_uv=pivot_uv,
        containment_tolerance=containment_tolerance,
    )
    if len(expected_vertices) != len(plan.data.vertices):
        raise ValueError(f"Plan CDT vertex count differs from canonical rebuild: {plan.name}")
    maximum_vertex_error = max(
        math.dist(
            (float(vertex.co.x), float(vertex.co.y)),
            expected_vertices[index],
        )
        for index, vertex in enumerate(plan.data.vertices)
    )
    if maximum_vertex_error > _TOLERANCE:
        raise ValueError(f"Plan CDT vertices differ from canonical rebuild: {plan.name}")
    canonical_faces = sorted(tuple(sorted(face)) for face in faces)
    rebuilt_faces = sorted(tuple(sorted(int(value) for value in face)) for face in expected_faces)
    if canonical_faces != rebuilt_faces:
        raise ValueError(f"Plan CDT faces differ from canonical rebuild: {plan.name}")
    if any(
        max(
            abs(actual_uvs[index][axis] - expected_uvs[index][axis])
            for axis in range(2)
        )
        > _TOLERANCE
        for index in range(len(actual_uvs))
        if index != expected_attachment_index
    ):
        raise ValueError(
            f"Plan interior UV differs from canonical camera-contract interpolation: {plan.name}"
        )
    return {
        "boundary_vertex_count": len(boundary_indices),
        "triangle_count": len(faces),
        "attachment_vertex_index": int(expected_attachment_index),
        "maximum_vertex_error": float(maximum_vertex_error),
    }


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
    expected_pivot_uv = _finite_vector(
        (plane.get("attachment") or {}).get("pivot_uv"),
        2,
        "camera contract pivot UV",
    )
    stored_pivot_uv = _finite_vector(
        transfer.get("reference_pivot_uv"),
        2,
        "stored camera pivot UV",
    )
    if max(
        abs(stored_pivot_uv[axis] - expected_pivot_uv[axis])
        for axis in range(2)
    ) > _UV_TOLERANCE:
        raise ValueError(f"Plan UV transfer pivot UV mismatch: {plan_name}")

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
        or abs(stored_maximum - 0.0) > _TOLERANCE
        or attachment_error > _TOLERANCE
        or attachment_error_normalized > _TOLERANCE
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


def _vectors_close(actual, expected, tolerance=_TOLERANCE):
    try:
        left = Vector(actual)
        right = Vector(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError("Auto capture basis contains an invalid vector.") from exc
    return len(left) == len(right) and (left - right).length <= tolerance


def _validate_auto_capture_frame(contract, manifest):
    frame = contract.get("frame") or {}
    manifest_frame = manifest.get("frame") or {}
    if frame != manifest_frame:
        raise ValueError("Auto capture contract frame differs from its manifest.")
    if frame.get("policy") != "world_axis_locked_auto_bounds":
        raise ValueError("Auto capture frame is not world-axis locked.")
    plane = str(frame.get("plane") or "").upper()
    expected = AUTO_CAPTURE_BASES.get(plane)
    if expected is None:
        raise ValueError(f"Auto capture uses an unsupported plane: {plane!r}")
    for key in ("right", "up", "normal", "view_direction"):
        if not _vectors_close(frame.get(key), expected[key], 1.0e-9):
            raise ValueError(
                f"Auto capture {plane} {key} is tilted or points along another axis."
            )
    rotation = float(frame.get("rotation_degrees", math.nan))
    if (
        not math.isfinite(rotation)
        or abs(rotation - float(expected["rotation_degrees"])) > 1.0e-9
    ):
        raise ValueError(
            f"Auto capture {plane} rotation must be exactly "
            f"{expected['rotation_degrees']:.1f} degrees."
        )
    right = Vector(frame["right"])
    up = Vector(frame["up"])
    normal = Vector(frame["normal"])
    view = Vector(frame["view_direction"])
    orthogonality = max(
        abs(right.dot(up)),
        abs(right.dot(normal)),
        abs(up.dot(normal)),
    )
    handedness = right.cross(up).dot(normal)
    if (
        orthogonality > 1.0e-9
        or abs(handedness - 1.0) > 1.0e-9
        or (view + normal).length > 1.0e-9
    ):
        raise ValueError("Auto capture frame is not an orthogonal right-handed basis.")
    stored_orthogonality = float(
        frame.get("orthogonality_error", math.nan)
    )
    if (
        not math.isfinite(stored_orthogonality)
        or abs(stored_orthogonality - orthogonality) > 1.0e-9
    ):
        raise ValueError("Auto capture orthogonality evidence is stale.")
    if "handedness" in frame and (
        abs(float(frame.get("handedness", math.nan)) - handedness) > 1.0e-9
    ):
        raise ValueError("Auto capture handedness evidence is stale.")
    width = float(frame.get("width", math.nan))
    height = float(frame.get("height", math.nan))
    content_width = float(frame.get("content_width", math.nan))
    content_height = float(frame.get("content_height", math.nan))
    if (
        not all(
            math.isfinite(value)
            for value in (width, height, content_width, content_height)
        )
        or min(width, height, content_width, content_height) <= 0.0
        or abs(width - height) > _TOLERANCE
        or content_width > width + _TOLERANCE
        or content_height > height + _TOLERANCE
    ):
        raise ValueError("Auto capture square area dimensions are invalid.")
    center = Vector(_finite_vector(frame.get("center"), 3, "capture center"))
    camera_location = Vector(
        _finite_vector(frame.get("camera_location"), 3, "capture camera location")
    )
    camera_delta = camera_location - center
    if (
        camera_delta.dot(normal) <= 0.0
        or (camera_delta - normal * camera_delta.dot(normal)).length > _TOLERANCE
    ):
        raise ValueError("Auto capture camera is not centered on its locked normal.")
    resolution = contract.get("resolution")
    if (
        resolution != manifest.get("resolution")
        or not isinstance(resolution, list)
        or len(resolution) != 2
        or any(int(value) <= 0 for value in resolution)
        or int(resolution[0]) != int(resolution[1])
    ):
        raise ValueError("Auto capture resolution is not a valid square contract.")
    return {
        "plane": plane,
        "right": right,
        "up": up,
        "normal": normal,
        "view": view,
        "center": center,
        "camera_location": camera_location,
        "width": width,
        "height": height,
        "rotation_degrees": rotation,
        "orthogonality_error": orthogonality,
        "handedness": handedness,
    }


def _validate_auto_capture_maps(contract, manifest, expected_albedo_path=None):
    map_rows = manifest.get("maps") or []
    if len(map_rows) != len(AUTO_CAPTURE_MAP_ROLES):
        raise ValueError("Auto capture manifest does not contain exactly eight maps.")
    by_role = {}
    for row in map_rows:
        role = str(row.get("role") or "")
        if role in by_role:
            raise ValueError(f"Auto capture manifest duplicates map role: {role}")
        path = Path(str(row.get("path") or "")).expanduser().absolute()
        expected_hash = str(row.get("sha256") or "")
        expected_size = int(row.get("size", -1))
        if (
            role not in AUTO_CAPTURE_MAP_ROLES
            or not path.is_file()
            or not expected_hash
            or _sha256(path) != expected_hash
            or path.stat().st_size != expected_size
        ):
            raise ValueError(f"Auto capture map fingerprint is stale: {role or '<empty>'}")
        by_role[role] = {
            "path": path,
            "sha256": expected_hash,
            "size": expected_size,
        }
    if set(by_role) != set(AUTO_CAPTURE_MAP_ROLES):
        raise ValueError("Auto capture manifest map roles are incomplete.")
    color_path = Path(str(contract.get("color") or "")).expanduser().absolute()
    opacity_path = Path(str(contract.get("opacity") or "")).expanduser().absolute()
    for role, path in (("Color", color_path), ("Opacity", opacity_path)):
        if (
            not path.is_file()
            or _sha256(path) != by_role[role]["sha256"]
        ):
            raise ValueError(
                f"Auto capture promoted {role} map differs from the captured map."
            )
    if (
        expected_albedo_path is not None
        and not _same_path(color_path, expected_albedo_path)
    ):
        raise ValueError("Explicit Atlas Color map differs from auto capture.")
    return color_path, opacity_path, by_role


def _validate_auto_capture_material(material_name, color_path, opacity_path):
    material = bpy.data.materials.get(material_name)
    if material is None or not material.use_nodes or material.node_tree is None:
        raise ValueError(f"Auto capture preview material is missing: {material_name}")
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
        raise ValueError(
            "Auto capture material must contain real Color and Opacity image nodes."
        )
    if color.extension != "CLIP" or opacity.extension != "CLIP":
        raise ValueError("Auto capture texture nodes must use CLIP.")
    for node, expected_path, label in (
        (color, color_path, "Color"),
        (opacity, opacity_path, "Opacity"),
    ):
        actual_path = Path(
            bpy.path.abspath(node.image.filepath)
        ).expanduser().absolute()
        if (
            node.image.source != "FILE"
            or not actual_path.is_file()
            or not _same_path(actual_path, expected_path)
        ):
            raise ValueError(f"Auto capture {label} node uses another image.")
    alpha_links = [
        link
        for link in material.node_tree.links
        if link.from_node.name == opacity.name and link.to_socket.name == "Alpha"
    ]
    if not alpha_links:
        raise ValueError("Auto capture Opacity node is not connected to material Alpha.")
    return material


def _curve_world_points(obj):
    return [
        obj.matrix_world @ Vector(point.co[:3])
        for spline in obj.data.splines
        for point in spline.points
    ]


def _validate_auto_capture_rig(scene, contract_hash, frame):
    collection = bpy.data.collections.get(AUTO_CAPTURE_COLLECTION)
    if collection is None or collection.children:
        raise ValueError("Auto capture area collection is missing or nested.")
    expected_names = {
        "STAutoCapture_Camera",
        "STAutoCapture_Area",
        "STAutoCapture_AttachmentOrigin",
    }
    if {obj.name for obj in collection.objects} != expected_names:
        raise ValueError("Auto capture area collection contains unexpected objects.")
    camera = bpy.data.objects.get("STAutoCapture_Camera")
    area = bpy.data.objects.get("STAutoCapture_Area")
    origin = bpy.data.objects.get("STAutoCapture_AttachmentOrigin")
    for obj, role in (
        (camera, "camera"),
        (area, "area"),
        (origin, "attachment_origin"),
    ):
        if (
            obj is None
            or obj.get("speedtree_cluster_auto_capture_role") != role
            or obj.get(AUTO_CAPTURE_CONTRACT_HASH_KEY) != contract_hash
        ):
            raise ValueError(f"Auto capture {role} object is missing or stale.")
    if (
        camera.type != "CAMERA"
        or camera.data.type != "ORTHO"
        or scene.camera is not camera
        or abs(float(camera.data.ortho_scale) - frame["height"]) > _TOLERANCE
    ):
        raise ValueError("Auto capture orthographic camera contract is stale.")
    camera_axes = (
        Vector(camera.matrix_world.col[0][:3]),
        Vector(camera.matrix_world.col[1][:3]),
        Vector(camera.matrix_world.col[2][:3]),
    )
    if (
        (camera_axes[0] - frame["right"]).length > _TOLERANCE
        or (camera_axes[1] - frame["up"]).length > _TOLERANCE
        or (camera_axes[2] - frame["normal"]).length > _TOLERANCE
        or (camera.matrix_world.translation - frame["camera_location"]).length
        > _TOLERANCE
    ):
        raise ValueError("Auto capture camera is tilted or displaced.")
    if (
        area.type != "CURVE"
        or len(area.data.splines) != 1
        or not area.data.splines[0].use_cyclic_u
        or len(area.data.splines[0].points) != 4
        or not _identity_object(area)
    ):
        raise ValueError("Auto capture area frame topology is invalid.")
    half_width = frame["width"] * 0.5
    half_height = frame["height"] * 0.5
    expected_corners = [
        frame["center"] - frame["right"] * half_width - frame["up"] * half_height,
        frame["center"] + frame["right"] * half_width - frame["up"] * half_height,
        frame["center"] + frame["right"] * half_width + frame["up"] * half_height,
        frame["center"] - frame["right"] * half_width + frame["up"] * half_height,
    ]
    actual_corners = _curve_world_points(area)
    if any(
        (actual - expected).length > _TOLERANCE
        for actual, expected in zip(actual_corners, expected_corners)
    ):
        raise ValueError("Auto capture area corners differ from the locked frame.")
    if (
        origin.type != "CURVE"
        or len(origin.data.splines) != 2
        or any(len(spline.points) != 2 for spline in origin.data.splines)
        or not _identity_object(origin)
    ):
        raise ValueError("Auto capture attachment-origin marker is invalid.")
    origin_points = _curve_world_points(origin)
    if (
        len(origin_points) != 4
        or (origin_points[0] + origin_points[1]).length > _TOLERANCE
        or (origin_points[2] + origin_points[3]).length > _TOLERANCE
    ):
        raise ValueError("Auto capture attachment-origin marker is not centered at 0,0,0.")
    return {
        "collection": collection.name,
        "camera": camera.name,
        "area": area.name,
        "attachment_origin": origin.name,
    }


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


def validate_auto_capture_delivery(
    scene,
    plan_collection_name,
    material_name,
    *,
    plan_base=None,
    expected_export_names=None,
    expected_tree_spm=None,
    expected_albedo_path=None,
    expected_card_count=None,
    expected_prototype_count=None,
):
    """Validate a Blender world-axis capture delivery without legacy camera data."""
    contract = _json_property(
        scene,
        AUTO_CAPTURE_CONTRACT_KEY,
        "Scene auto capture contract",
    )
    contract_hash = str(scene.get(AUTO_CAPTURE_CONTRACT_HASH_KEY) or "")
    if (
        contract.get("kind") != AUTO_CAPTURE_KIND
        or int(contract.get("version", 0)) != 1
        or not contract_hash
        or _canonical_sha256(contract) != contract_hash
    ):
        raise ValueError("Scene auto capture contract is missing, unsupported, or stale.")
    for legacy_key in (
        CAMERA_CONTRACT_KEY,
        CAMERA_CONTRACT_HASH_KEY,
        CAMERA_BUNDLE_KEY,
    ):
        if scene.get(legacy_key):
            raise ValueError("Auto capture delivery contains an active legacy camera contract.")
    for collection_name in ("Atlas_Capture_Sync", CAMERA_REFERENCE_COLLECTION):
        if bpy.data.collections.get(collection_name) is not None:
            raise ValueError(
                f"Auto capture delivery contains a legacy camera collection: "
                f"{collection_name}"
            )

    manifest_path = Path(
        str(contract.get("capture_manifest") or "")
    ).expanduser().absolute()
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path)
        != str(contract.get("capture_manifest_sha256") or "")
    ):
        raise ValueError("Auto capture manifest is missing or stale.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Auto capture manifest is invalid: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != AUTO_CAPTURE_MANIFEST_KIND
        or int(manifest.get("version", 0)) != 1
    ):
        raise ValueError("Auto capture manifest kind/version is unsupported.")
    blend_path = Path(bpy.data.filepath).expanduser().absolute()
    if (
        not blend_path.is_file()
        or not _same_path(contract.get("source_blend", ""), blend_path)
        or not _same_path(manifest.get("blend", ""), blend_path)
    ):
        raise ValueError("Auto capture contract targets another Blender file.")
    source_collection_name = str(contract.get("source_collection") or "")
    source_collection = bpy.data.collections.get(source_collection_name)
    if (
        not source_collection_name
        or source_collection is None
        or manifest.get("source_collection") != source_collection_name
    ):
        raise ValueError("Auto capture source collection is missing or changed.")
    source_objects = {
        obj.name: obj
        for obj in _collection_objects_recursive(source_collection)
        if obj.type == "MESH"
    }
    manifest_sources = manifest.get("source_objects") or []
    if not manifest_sources:
        raise ValueError("Auto capture manifest contains no source meshes.")
    for row in manifest_sources:
        source = source_objects.get(str(row.get("name") or ""))
        if (
            source is None
            or len(source.data.vertices) != int(row.get("vertices", -1))
            or len(source.data.polygons) != int(row.get("polygons", -1))
        ):
            raise ValueError(
                f"Auto capture source mesh changed: {row.get('name') or '<empty>'}"
            )

    frame = _validate_auto_capture_frame(contract, manifest)
    color_path, opacity_path, map_rows = _validate_auto_capture_maps(
        contract,
        manifest,
        expected_albedo_path=expected_albedo_path,
    )
    material = _validate_auto_capture_material(
        material_name,
        color_path,
        opacity_path,
    )
    rig = _validate_auto_capture_rig(scene, contract_hash, frame)
    (
        source_3d_contract,
        source_3d_contract_hash,
        source_3d_root_ids,
        authoritative_source_3d_roots,
    ) = _validate_source_3d_contract(scene)

    plan_collection = bpy.data.collections.get(plan_collection_name)
    if plan_collection is None or plan_collection.children:
        raise ValueError("Auto capture plan collection is missing or nested.")
    plans = [
        obj
        for obj in plan_collection.objects
        if obj.type == "MESH"
        and obj.get(ASSET_ROLE_KEY) == "speedtree_plan"
    ]
    if len(plans) != len(plan_collection.objects) or not plans:
        raise ValueError("Auto capture plan collection contains unexpected objects.")
    plans = sorted(plans, key=lambda obj: obj.name.casefold())
    card_count = len(plans)
    if (
        expected_card_count is not None
        and card_count != int(expected_card_count)
    ):
        raise ValueError("Auto capture plan count differs from the explicit expectation.")
    if plan_base:
        expected_plan_names = [
            f"{plan_base}_{index:02d}"
            for index in range(1, card_count + 1)
        ]
        if [plan.name for plan in plans] != expected_plan_names:
            raise ValueError("Auto capture plan base/ordinal sequence is invalid.")

    prototype_map = _json_property(
        scene,
        CARD_PROTOTYPE_MAP_KEY,
        "card/prototype mapping contract",
    )
    prototype_map_hash = str(scene.get(CARD_PROTOTYPE_MAP_HASH_KEY) or "")
    if (
        not prototype_map_hash
        or _canonical_sha256(prototype_map) != prototype_map_hash
        or int(prototype_map.get("card_count", -1)) != card_count
        or str(prototype_map.get("source_3d_contract_sha256") or "")
        != source_3d_contract_hash
    ):
        raise ValueError("Auto capture card/prototype mapping is stale.")
    mapping_rows = prototype_map.get("cards") or []
    mapping_by_plan = {
        str(row.get("plan") or ""): row
        for row in mapping_rows
    }
    plan_names = {plan.name for plan in plans}
    if (
        len(mapping_rows) != card_count
        or set(mapping_by_plan) != plan_names
        or prototype_map.get("composite_parts")
        or prototype_map.get("composite_set_id")
    ):
        raise ValueError("Auto capture requires one physical prototype per plan.")
    prototype_assets = {
        str(row.get("prototype_asset") or "")
        for row in mapping_rows
    }
    if "" in prototype_assets:
        raise ValueError("Auto capture mapping contains an empty prototype asset.")
    prototype_count = len(prototype_assets)
    if (
        int(prototype_map.get("prototype_count", -1)) != prototype_count
        or prototype_count != len(source_3d_root_ids)
        or (
            expected_prototype_count is not None
            and prototype_count != int(expected_prototype_count)
        )
    ):
        raise ValueError("Auto capture prototype count is stale.")

    rows = []
    seen_xml_root_ids = set()
    for plan in plans:
        mapping = mapping_by_plan[plan.name]
        if (
            not _identity_object(plan)
            or plan.get(AUTO_CAPTURE_CONTRACT_HASH_KEY) != contract_hash
            or plan.data.get(AUTO_CAPTURE_CONTRACT_HASH_KEY)
            or plan.get("speedtree_cluster_uv_policy") != AUTO_CAPTURE_UV_POLICY
            or plan.get(CAMERA_CONTRACT_HASH_KEY)
            or plan.get(CAMERA_REFERENCE_KEY)
            or plan.get(UV_TRANSFER_KEY)
        ):
            raise ValueError(f"Auto capture plan role/contract is stale: {plan.name}")
        if [slot for slot in plan.data.materials] != [material]:
            raise ValueError(
                f"Auto capture plan does not use exactly {material_name}: {plan.name}"
            )
        plan_source_3d_contract = _json_property(
            plan,
            SOURCE_3D_CONTRACT_KEY,
            f"source 3D contract on {plan.name}",
        )
        plan_attachment = _validate_xml_attachment(
            _json_property(
                plan,
                XML_ATTACHMENT_KEY,
                f"XML attachment on {plan.name}",
            ),
            f"XML attachment on {plan.name}",
        )
        root_id = int(plan_attachment["xml_bone_id"])
        contract_root = next(
            (
                root
                for root in authoritative_source_3d_roots
                if int(root["xml_bone_id"]) == root_id
            ),
            None,
        )
        if (
            plan_source_3d_contract != source_3d_contract
            or contract_root is None
            or not _attachment_matches_contract(plan_attachment, contract_root)
            or root_id in seen_xml_root_ids
            or root_id not in source_3d_root_ids
            or int(mapping.get("xml_bone_id", -1)) != root_id
            or int(mapping.get("prototype_index", -1))
            != int(plan.get(PROTOTYPE_INDEX_KEY, -2))
            or mapping.get("prototype_asset") != plan.get(PROTOTYPE_ASSET_KEY)
            or mapping.get("prototype_asset") != plan.get(COUNTERPART_KEY)
            or prototype_map.get("source_partition_mode")
            != plan.get(SOURCE_PARTITION_MODE_KEY)
        ):
            raise ValueError(f"Auto capture XML/prototype lineage mismatch: {plan.name}")
        seen_xml_root_ids.add(root_id)

        try:
            plan_frame = Matrix(json.loads(plan["speedtree_cluster_frame_world"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Auto capture plan frame is invalid: {plan.name}") from exc
        plan_axes = (
            Vector(plan_frame.col[0][:3]),
            Vector(plan_frame.col[1][:3]),
            Vector(plan_frame.col[2][:3]),
        )
        if (
            (plan_axes[0] - frame["right"]).length > _TOLERANCE
            or (plan_axes[1] - frame["up"]).length > _TOLERANCE
            or (plan_axes[2] - frame["normal"]).length > _TOLERANCE
            or (
                plan_frame.translation
                - attachment_origin_world(plan_attachment)
            ).length
            > _TOLERANCE
        ):
            raise ValueError(
                f"Auto capture plan is tilted or detached from XML root: {plan.name}"
            )
        attachment_index = int(
            plan.get("speedtree_cluster_attachment_vertex_index", -1)
        )
        if (
            attachment_index < 0
            or attachment_index >= len(plan.data.vertices)
            or plan.data.vertices[attachment_index].co.length > _TOLERANCE
        ):
            raise ValueError(f"Auto capture plan pivot is not local 0,0,0: {plan.name}")

        expected_uvs = []
        for vertex in plan.data.vertices:
            world = plan_frame @ vertex.co
            relative = world - frame["center"]
            expected_uvs.append(
                [
                    0.5 + float(relative.dot(frame["right"])) / frame["width"],
                    0.5 + float(relative.dot(frame["up"])) / frame["height"],
                ]
            )
        actual_uvs = _vertex_uvs(
            plan.data,
            expected_uvs=expected_uvs,
            label=plan.name,
        )
        actual_minimum = [
            min(uv[axis] for uv in actual_uvs)
            for axis in range(2)
        ]
        actual_maximum = [
            max(uv[axis] for uv in actual_uvs)
            for axis in range(2)
        ]
        stored_bounds = _json_property(
            plan,
            "speedtree_cluster_auto_capture_uv_bounds",
            f"auto capture UV bounds on {plan.name}",
        )
        stored_minimum = _finite_vector(
            stored_bounds.get("minimum"),
            2,
            "stored auto capture UV minimum",
        )
        stored_maximum = _finite_vector(
            stored_bounds.get("maximum"),
            2,
            "stored auto capture UV maximum",
        )
        if (
            min(actual_minimum) < -_UV_TOLERANCE
            or max(actual_maximum) > 1.0 + _UV_TOLERANCE
            or max(
                abs(actual_minimum[axis] - stored_minimum[axis])
                for axis in range(2)
            )
            > _TOLERANCE
            or max(
                abs(actual_maximum[axis] - stored_maximum[axis])
                for axis in range(2)
            )
            > _TOLERANCE
        ):
            raise ValueError(f"Auto capture plan UV bounds are stale: {plan.name}")

        counterpart_name = str(plan.get(COUNTERPART_KEY) or "")
        part = bpy.data.objects.get(counterpart_name + "_Mesh")
        if part is None or part.type != "MESH":
            raise ValueError(f"Auto capture counterpart mesh is missing: {plan.name}")
        part_source_3d_contract = _json_property(
            part,
            SOURCE_3D_CONTRACT_KEY,
            f"source 3D contract on {part.name}",
        )
        part_attachment = _validate_xml_attachment(
            _json_property(
                part,
                XML_ATTACHMENT_KEY,
                f"XML attachment on {part.name}",
            ),
            f"XML attachment on {part.name}",
        )
        try:
            part_frame = Matrix(
                json.loads(part["speedtree_cluster_frame_world"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Auto capture 3D frame is invalid: {part.name}") from exc
        if (
            part_source_3d_contract != source_3d_contract
            or part_attachment != plan_attachment
            or any(
                abs(float(part_frame[row][column] - plan_frame[row][column]))
                > _TOLERANCE
                for row in range(4)
                for column in range(4)
            )
        ):
            raise ValueError(f"Auto capture plan/3D frame mismatch: {plan.name}")

        projection_basis = _json_property(
            plan,
            PROJECTION_BASIS_KEY,
            f"projection basis on {plan.name}",
        )
        stored_coverage = _json_property(
            plan,
            PROJECTION_COVERAGE_KEY,
            f"projection coverage on {plan.name}",
        )
        if (
            projection_basis.get("policy")
            != "camera_aligned_canonical_local_xy"
            or not _vectors_close(projection_basis.get("right"), (1.0, 0.0, 0.0))
            or not _vectors_close(projection_basis.get("up"), (0.0, 1.0, 0.0))
            or not _vectors_close(projection_basis.get("normal"), (0.0, 0.0, 1.0))
            or any(abs(float(vertex.co.z)) > _TOLERANCE for vertex in plan.data.vertices)
        ):
            raise ValueError(f"Auto capture plan local plane is not canonical XY: {plan.name}")
        boundary_indices = _ordered_boundary_indices(
            [
                tuple(int(value) for value in polygon.vertices)
                for polygon in plan.data.polygons
            ]
        )
        plan_boundary = [
            (
                float(plan.data.vertices[index].co.x),
                float(plan.data.vertices[index].co.y),
            )
            for index in boundary_indices
        ]
        projected_part = [
            (float(vertex.co.x), float(vertex.co.y))
            for vertex in part.data.vertices
        ]
        actual_coverage = projection_coverage_2d(
            projected_part,
            plan_boundary,
        )
        if (
            not actual_coverage["covers_projection"]
            or stored_coverage != actual_coverage
        ):
            raise ValueError(f"Auto capture plan coverage is stale: {plan.name}")
        rows.append(
            {
                "plan": plan.name,
                "prototype_asset": counterpart_name,
                "prototype_index": int(mapping.get("prototype_index", -1)),
                "xml_bone_id": root_id,
                "uv_minimum": actual_minimum,
                "uv_maximum": actual_maximum,
                "projection_coverage": actual_coverage,
            }
        )

    if seen_xml_root_ids != set(source_3d_root_ids):
        raise ValueError("Auto capture plans do not map every XML root exactly once.")
    export_names = _validate_export(
        scene,
        plans,
        [],
        expected_export_names=expected_export_names,
    )
    target_spm = None
    if expected_tree_spm is not None:
        target_spm = Path(expected_tree_spm).expanduser().absolute()
        if not target_spm.is_file():
            raise ValueError("Explicit target SPM does not exist.")
    return {
        "delivery_mode": "blender_world_axis_auto_capture",
        "contract_sha256": contract_hash,
        "capture_manifest": str(manifest_path),
        "capture_manifest_sha256": contract["capture_manifest_sha256"],
        "capture_plane": frame["plane"],
        "capture_rotation_degrees": frame["rotation_degrees"],
        "capture_axes": {
            "right": list(frame["right"]),
            "up": list(frame["up"]),
            "normal": list(frame["normal"]),
            "view_direction": list(frame["view"]),
        },
        "orthogonality_error": frame["orthogonality_error"],
        "handedness": frame["handedness"],
        "rig": rig,
        "material": material_name,
        "color": str(color_path),
        "opacity": str(opacity_path),
        "map_fingerprints": {
            role: {
                "path": str(row["path"]),
                "sha256": row["sha256"],
                "size": row["size"],
            }
            for role, row in map_rows.items()
        },
        "plan_collection": plan_collection_name,
        "card_count": card_count,
        "prototype_count": prototype_count,
        "source_partition_mode": prototype_map.get("source_partition_mode"),
        "card_prototype_map_sha256": prototype_map_hash,
        "source_3d_contract_sha256": source_3d_contract_hash,
        "target_spm": str(target_spm) if target_spm is not None else None,
        "export_objects": export_names,
        "planes": rows,
    }


def validate_physical_direct_capture_delivery(
    scene,
    plan_collection_name,
    material_name,
    *,
    plan_base=None,
    expected_export_names=None,
    expected_tree_spm=None,
    expected_card_count=None,
    expected_prototype_count=None,
):
    """Validate maps, plans, prototypes, and attachments against one final SHA."""
    contract = _json_property(
        scene,
        PHYSICAL_CAPTURE_CONTRACT_KEY,
        "Scene physical capture contract",
    )
    contract, frame, contract_hash = _validate_physical_capture_contract(contract)
    if str(scene.get(PHYSICAL_CAPTURE_CONTRACT_HASH_KEY) or "") != contract_hash:
        raise ValueError("Scene physical capture contract hash is stale.")
    for legacy_key in (
        CAMERA_CONTRACT_KEY,
        CAMERA_CONTRACT_HASH_KEY,
        CAMERA_BUNDLE_KEY,
    ):
        if scene.get(legacy_key):
            raise ValueError(
                "Physical direct capture contains an active legacy camera contract."
            )
    if bpy.data.collections.get(CAMERA_REFERENCE_COLLECTION) is not None:
        raise ValueError(
            "Physical direct capture contains a legacy camera reference collection."
        )

    manifest_path = Path(
        str(contract.get("capture_manifest") or "")
    ).expanduser().absolute()
    if not manifest_path.is_file():
        raise ValueError("Physical direct capture manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Physical direct capture manifest is invalid: {exc}") from exc
    if (
        manifest.get("normalization_status") != "finalized"
        or manifest.get("physical_capture_contract_sha256") != contract_hash
        or manifest.get("physical_capture_contract") != contract
        or manifest.get("frame") != frame
    ):
        raise ValueError(
            "Physical capture maps and normalized plans do not share one final contract."
        )
    contract_maps = {
        str(row.get("role") or ""): row
        for row in contract.get("capture_maps") or []
    }
    manifest_maps = {
        str(row.get("role") or ""): row
        for row in manifest.get("maps") or []
    }
    if set(contract_maps) != set(AUTO_CAPTURE_MAP_ROLES) or set(
        manifest_maps
    ) != set(AUTO_CAPTURE_MAP_ROLES):
        raise ValueError("Physical direct capture does not contain all eight maps.")
    for role in AUTO_CAPTURE_MAP_ROLES:
        contract_row = contract_maps[role]
        manifest_row = manifest_maps[role]
        map_path = Path(str(contract_row.get("path") or "")).expanduser().absolute()
        if (
            not map_path.is_file()
            or int(contract_row.get("size", -1)) != map_path.stat().st_size
            or str(contract_row.get("sha256") or "") != _sha256(map_path)
            or str(manifest_row.get("path") or "")
            != str(contract_row.get("path") or "")
            or int(manifest_row.get("size", -1))
            != int(contract_row.get("size", -2))
            or str(manifest_row.get("sha256") or "")
            != str(contract_row.get("sha256") or "")
            or manifest_row.get("physical_capture_contract_sha256")
            != contract_hash
        ):
            raise ValueError(f"Physical direct capture map lineage is stale: {role}")

    (
        source_3d_contract,
        source_3d_contract_hash,
        source_3d_root_ids,
        authoritative_source_3d_roots,
    ) = _validate_source_3d_contract(scene)
    plan_collection = bpy.data.collections.get(plan_collection_name)
    if plan_collection is None or plan_collection.children:
        raise ValueError("Physical direct capture plan collection is missing or nested.")
    plans = sorted(
        (
            obj
            for obj in plan_collection.objects
            if obj.type == "MESH"
            and obj.get(ASSET_ROLE_KEY) == "speedtree_plan"
        ),
        key=lambda obj: obj.name.casefold(),
    )
    if len(plans) != len(plan_collection.objects) or not plans:
        raise ValueError(
            "Physical direct capture plan collection contains unexpected objects."
        )
    if expected_card_count is not None and len(plans) != int(expected_card_count):
        raise ValueError("Physical direct capture plan count is unexpected.")
    if plan_base:
        expected_names = [
            f"{plan_base}_{index:02d}"
            for index in range(1, len(plans) + 1)
        ]
        if [plan.name for plan in plans] != expected_names:
            raise ValueError("Physical direct capture plan names are not ordinal.")

    prototype_map = _json_property(
        scene,
        CARD_PROTOTYPE_MAP_KEY,
        "card/prototype mapping contract",
    )
    prototype_map_hash = str(scene.get(CARD_PROTOTYPE_MAP_HASH_KEY) or "")
    mapping_rows = prototype_map.get("cards") or []
    mapping_by_plan = {
        str(row.get("plan") or ""): row for row in mapping_rows
    }
    prototype_assets = {
        str(row.get("prototype_asset") or "") for row in mapping_rows
    }
    if (
        not prototype_map_hash
        or _canonical_sha256(prototype_map) != prototype_map_hash
        or int(prototype_map.get("card_count", -1)) != len(plans)
        or str(prototype_map.get("source_3d_contract_sha256") or "")
        != source_3d_contract_hash
        or set(mapping_by_plan) != {plan.name for plan in plans}
        or any(
            row.get("uv_source") != DIRECT_CAPTURE_UV_SOURCE
            for row in mapping_rows
        )
        or "" in prototype_assets
        or (
            expected_prototype_count is not None
            and len(prototype_assets) != int(expected_prototype_count)
        )
    ):
        raise ValueError("Physical direct card/prototype mapping is stale.")

    material = bpy.data.materials.get(material_name)
    if material is None:
        raise ValueError(f"Physical direct material is missing: {material_name}")
    capture_center = Vector(frame["center"])
    capture_right = Vector(frame["right"])
    capture_up = Vector(frame["up"])
    capture_normal = Vector(frame["normal"])
    fit_scale = float(frame["fit_scale"])
    attachment_by_asset = {
        str(row.get("prototype_asset") or ""): row
        for row in contract.get("attachment_pivots") or []
    }
    rows = []
    seen_roots = set()
    for plan in plans:
        mapping = mapping_by_plan[plan.name]
        counterpart = str(plan.get(COUNTERPART_KEY) or "")
        part = bpy.data.objects.get(counterpart + "_Mesh")
        if (
            not _identity_object(plan)
            or part is None
            or part.type != "MESH"
            or any(
                abs(
                    float(
                        matrix[row][column]
                        - Matrix.Identity(4)[row][column]
                    )
                )
                > _TOLERANCE
                for matrix in (part.matrix_world, part.matrix_basis)
                for row in range(4)
                for column in range(4)
            )
            or plan.get(PHYSICAL_CAPTURE_CONTRACT_HASH_KEY) != contract_hash
            or plan.data.get(PHYSICAL_CAPTURE_CONTRACT_HASH_KEY) != contract_hash
            or part.get(PHYSICAL_CAPTURE_CONTRACT_HASH_KEY) != contract_hash
            or plan.get(CAMERA_CONTRACT_HASH_KEY)
            or plan.get(CAMERA_REFERENCE_KEY)
            or plan.get(UV_TRANSFER_KEY)
            or plan.get("speedtree_cluster_uv_policy")
            != "direct_physical_capture_projection"
            or [slot for slot in plan.data.materials] != [material]
        ):
            raise ValueError(f"Physical direct plan/prototype is stale: {plan.name}")
        transfer = _json_property(
            plan,
            DIRECT_CAPTURE_UV_KEY,
            f"direct capture UV contract on {plan.name}",
        )
        plan_attachment = _validate_xml_attachment(
            _json_property(
                plan,
                XML_ATTACHMENT_KEY,
                f"XML attachment on {plan.name}",
            ),
            f"XML attachment on {plan.name}",
        )
        root_id = int(plan_attachment["xml_bone_id"])
        authoritative = next(
            (
                row
                for row in authoritative_source_3d_roots
                if int(row["xml_bone_id"]) == root_id
            ),
            None,
        )
        if (
            transfer.get("policy") != "direct_physical_capture_projection"
            or transfer.get("direct_uv_source") != DIRECT_CAPTURE_UV_SOURCE
            or transfer.get("capture_contract_sha256") != contract_hash
            or mapping.get("uv_source") != DIRECT_CAPTURE_UV_SOURCE
            or int(mapping.get("xml_bone_id", -1)) != root_id
            or mapping.get("prototype_asset") != counterpart
            or authoritative is None
            or not _attachment_matches_contract(plan_attachment, authoritative)
            or root_id in seen_roots
        ):
            raise ValueError(f"Physical direct attachment lineage is stale: {plan.name}")
        seen_roots.add(root_id)

        try:
            plan_frame = Matrix(json.loads(plan["speedtree_cluster_frame_world"]))
            part_frame = Matrix(json.loads(part["speedtree_cluster_frame_world"]))
            plan_source_frame = Matrix(
                json.loads(plan["speedtree_cluster_source_frame_world"])
            )
            part_source_frame = Matrix(
                json.loads(part["speedtree_cluster_source_frame_world"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Physical direct frame is invalid: {plan.name}") from exc
        projection_basis = _json_property(
            plan,
            PROJECTION_BASIS_KEY,
            f"projection basis on {plan.name}",
        )
        try:
            local_right = Vector(projection_basis["right"]).normalized()
            local_up = Vector(projection_basis["up"]).normalized()
            local_normal = Vector(projection_basis["normal"]).normalized()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Physical direct projection basis is invalid: {plan.name}"
            ) from exc
        attachment = attachment_by_asset.get(counterpart)
        if attachment is None:
            raise ValueError(f"Physical capture attachment is missing: {counterpart}")
        source_world = Vector(attachment["source_world"])
        fitted_world = Vector(attachment["fitted_capture_world"])
        expected_fitted = capture_center + (
            source_world - capture_center
        ) * fit_scale
        plan_rotation = plan_frame.to_3x3()
        world_right = (plan_rotation @ local_right).normalized()
        world_up = (plan_rotation @ local_up).normalized()
        world_normal = (plan_rotation @ local_normal).normalized()
        raw_xml_direction_world = (
            attachment_endpoint_world(plan_attachment)
            - attachment_origin_world(plan_attachment)
        )
        raw_projected_xml_direction_world = (
            raw_xml_direction_world
            - capture_normal
            * raw_xml_direction_world.dot(capture_normal)
        )
        tangent = transfer.get("attachment_tangent_projection") or {}
        direction_policy = str(tangent.get("direction_policy") or "")
        projection_tolerance = float(
            tangent.get("projection_tolerance") or 1.0e-9
        )
        expected_projection_tolerance = (
            physical_capture_projection_tolerance(
                raw_xml_direction_world,
                plan_attachment,
            )
        )
        recorded_projection = Vector(
            tangent.get("capture_plane_world") or (math.nan,) * 3
        )
        aligned_xml_direction_world = Vector(
            tangent.get("aligned_capture_plane_world") or (math.nan,) * 3
        )
        if (
            not all(math.isfinite(float(value)) for value in recorded_projection)
            or not all(
                math.isfinite(float(value))
                for value in aligned_xml_direction_world
            )
            or projection_tolerance <= 0.0
            or abs(
                projection_tolerance - expected_projection_tolerance
            )
            > max(expected_projection_tolerance * 1.0e-9, 1.0e-12)
            or (
                recorded_projection - raw_projected_xml_direction_world
            ).length
            > _TOLERANCE
            or aligned_xml_direction_world.length <= projection_tolerance
        ):
            raise ValueError(
                f"Physical capture attachment direction evidence is invalid: {plan.name}"
            )
        if direction_policy == PHYSICAL_CAPTURE_DIRECTION_PROJECTED_XML:
            if (
                raw_projected_xml_direction_world.length <= projection_tolerance
                or (
                    aligned_xml_direction_world
                    - raw_projected_xml_direction_world
                ).length
                > _TOLERANCE
            ):
                raise ValueError(
                    f"Projected XML attachment direction is stale: {plan.name}"
                )
        elif (
            direction_policy
            == PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK
        ):
            if (
                raw_projected_xml_direction_world.length > projection_tolerance
                or (
                    aligned_xml_direction_world.normalized()
                    - capture_up
                ).length
                > _TOLERANCE
            ):
                raise ValueError(
                    f"Capture-normal attachment fallback is stale: {plan.name}"
                )
        else:
            raise ValueError(
                f"Physical capture attachment direction policy is invalid: {plan.name}"
            )
        aligned_xml_direction_world.normalize()
        xml_direction_local = (
            plan_rotation.inverted_safe() @ aligned_xml_direction_world
        ).normalized()
        frame_axis_y_world = (
            plan_rotation @ Vector((0.0, 1.0, 0.0))
        ).normalized()
        frame_axis_z_world = (
            plan_rotation @ Vector((0.0, 0.0, 1.0))
        ).normalized()
        if (
            plan.get("speedtree_cluster_frame_policy")
            != PHYSICAL_CAPTURE_ALIGNED_FRAME_POLICY
            or (world_right - capture_right).length > _TOLERANCE
            or (world_up - capture_up).length > _TOLERANCE
            or (world_normal - capture_normal).length > _TOLERANCE
            or (xml_direction_local - Vector((0.0, 1.0, 0.0))).length
            > _TOLERANCE
            or (
                frame_axis_y_world - aligned_xml_direction_world
            ).length
            > _TOLERANCE
            or (frame_axis_z_world - capture_normal).length > _TOLERANCE
            or (plan_frame.translation - fitted_world).length > _TOLERANCE
            or (expected_fitted - fitted_world).length > _TOLERANCE
            or any(
                abs(float(plan_frame[row][column] - part_frame[row][column]))
                > _TOLERANCE
                for row in range(4)
                for column in range(4)
            )
            or any(
                abs(
                    float(
                        part_source_frame[row][column]
                        - plan_source_frame[row][column]
                    )
                )
                > _TOLERANCE
                for row in range(4)
                for column in range(4)
            )
        ):
            raise ValueError(f"Physical direct pair frame drifted: {plan.name}")
        attachment_index = int(transfer.get("attachment_vertex_index", -1))
        if (
            attachment_index < 0
            or attachment_index >= len(plan.data.vertices)
            or plan.data.vertices[attachment_index].co.length > _TOLERANCE
        ):
            raise ValueError(f"Physical direct pivot is not local zero: {plan.name}")

        expected_uvs = []
        for vertex in plan.data.vertices:
            world = plan_frame @ vertex.co
            relative = world - capture_center
            expected_uvs.append(
                [
                    0.5 + float(relative.dot(capture_right)) / float(frame["width"]),
                    0.5 + float(relative.dot(capture_up)) / float(frame["height"]),
                ]
            )
        actual_uvs = _vertex_uvs(
            plan.data,
            expected_uvs=transfer.get("result_uvs"),
            label=plan.name,
        )
        if max(
            abs(actual_uvs[index][axis] - expected_uvs[index][axis])
            for index in range(len(actual_uvs))
            for axis in range(2)
        ) > 5.0e-6:
            raise ValueError(
                f"Physical direct UV differs from its capture projection: {plan.name}"
            )
        if min(value for uv in actual_uvs for value in uv) < -_UV_TOLERANCE or max(
            value for uv in actual_uvs for value in uv
        ) > 1.0 + _UV_TOLERANCE:
            raise ValueError(f"Physical direct UV escaped its frame: {plan.name}")
        boundary = _ordered_boundary_indices(
            [
                tuple(int(value) for value in polygon.vertices)
                for polygon in plan.data.polygons
            ]
        )
        if max(
            abs(float(vertex.co.dot(local_normal)))
            for vertex in plan.data.vertices
        ) > _TOLERANCE:
            raise ValueError(
                f"Physical direct plan left its captured projection plane: {plan.name}"
            )
        coverage = projection_coverage_2d(
            [
                (
                    float(vertex.co.dot(local_right)),
                    float(vertex.co.dot(local_up)),
                )
                for vertex in part.data.vertices
            ],
            [
                (
                    float(plan.data.vertices[index].co.dot(local_right)),
                    float(plan.data.vertices[index].co.dot(local_up)),
                )
                for index in boundary
            ],
        )
        stored_coverage = _json_property(
            plan,
            PROJECTION_COVERAGE_KEY,
            f"projection coverage on {plan.name}",
        )
        coverage_integer_keys = (
            "projected_point_count",
            "outside_point_count",
            "outside_point_indices",
            "boundary_vertex_count",
        )
        if (
            not coverage["covers_projection"]
            or stored_coverage.get("covers_projection") is not True
            or any(
                coverage.get(key) != stored_coverage.get(key)
                for key in coverage_integer_keys
            )
            or abs(
                float(coverage["cross_product_tolerance"])
                - float(stored_coverage.get("cross_product_tolerance", math.nan))
            )
            > max(
                float(coverage["cross_product_tolerance"]),
                float(stored_coverage.get("cross_product_tolerance", 0.0)),
                1.0e-12,
            )
            * 1.0e-5
        ):
            raise ValueError(f"Physical direct plan coverage is stale: {plan.name}")
        rows.append(
            {
                "plan": plan.name,
                "prototype_asset": counterpart,
                "xml_bone_id": root_id,
                "uv_minimum": [
                    min(uv[axis] for uv in actual_uvs) for axis in range(2)
                ],
                "uv_maximum": [
                    max(uv[axis] for uv in actual_uvs) for axis in range(2)
                ],
                "fitted_attachment_world": list(fitted_world),
            }
        )
    if seen_roots != set(source_3d_root_ids):
        raise ValueError("Physical direct plans do not map every XML root exactly once.")
    export_names = _validate_export(
        scene,
        plans,
        [],
        expected_export_names=expected_export_names,
    )
    target_spm = None
    if expected_tree_spm is not None:
        target_spm = Path(expected_tree_spm).expanduser().absolute()
        if not target_spm.is_file():
            raise ValueError("Explicit target SPM does not exist.")
    return {
        "delivery_mode": "physical_direct_capture",
        "physical_capture_contract_sha256": contract_hash,
        "capture_manifest": str(manifest_path),
        "capture_manifest_sha256": _sha256(manifest_path),
        "capture_plane": frame["plane"],
        "capture_rotation_degrees": frame["rotation_degrees"],
        "physical_target_meters": list(frame["target_meters"]),
        "physical_fit_scale": fit_scale,
        "direct_uv_source": DIRECT_CAPTURE_UV_SOURCE,
        "card_count": len(plans),
        "prototype_count": len(prototype_assets),
        "card_prototype_map_sha256": prototype_map_hash,
        "source_3d_contract_sha256": source_3d_contract_hash,
        "target_spm": str(target_spm) if target_spm is not None else None,
        "export_objects": export_names,
        "planes": rows,
    }


def validate_cluster_delivery(
    scene,
    plan_collection_name,
    material_name,
    **kwargs,
):
    """Dispatch to the active persisted delivery contract without repairing data."""
    if scene.get(PHYSICAL_CAPTURE_CONTRACT_KEY):
        physical_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "plan_base",
                "expected_export_names",
                "expected_tree_spm",
                "expected_card_count",
                "expected_prototype_count",
            }
        }
        return validate_physical_direct_capture_delivery(
            scene,
            plan_collection_name,
            material_name,
            **physical_kwargs,
        )
    if scene.get(AUTO_CAPTURE_CONTRACT_KEY):
        auto_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "plan_base",
                "expected_export_names",
                "expected_tree_spm",
                "expected_albedo_path",
                "expected_card_count",
                "expected_prototype_count",
            }
        }
        return validate_auto_capture_delivery(
            scene,
            plan_collection_name,
            material_name,
            **auto_kwargs,
        )
    return validate_camera_uv_delivery(
        scene,
        plan_collection_name,
        material_name,
        **kwargs,
    )


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
    (
        source_3d_contract,
        source_3d_contract_hash,
        source_3d_root_ids,
        authoritative_source_3d_roots,
    ) = _validate_source_3d_contract(scene)
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

    if contract.get("kind") == "speedtree_cluster_card_uv_template":
        receipt_row = contract.get("camera_capture_receipt") or {}
        receipt_path = Path(str(receipt_row.get("path") or ""))
        receipt_sha256 = str(receipt_row.get("sha256") or "")
        if (
            not receipt_path.is_file()
            or not receipt_sha256
            or bundle.get("camera_capture_receipt_path") != str(receipt_path.resolve())
            or bundle.get("camera_capture_receipt_sha256") != receipt_sha256
        ):
            raise ValueError("Camera delivery capture receipt is missing or stale.")
        from atlas_leaf_mesh_builder.integration_api import (
            validate_external_camera_capture_receipt,
        )

        validate_external_camera_capture_receipt(
            receipt_path,
            contract,
            camera_spm,
        )

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
        str(prototype_map.get("source_3d_contract_sha256") or "")
        != source_3d_contract_hash
        or prototype_count != len(source_3d_root_ids)
    ):
        raise ValueError("Card/prototype XML root contract is missing or stale.")
    if (
        expected_prototype_count is not None
        and prototype_count != int(expected_prototype_count)
    ):
        raise ValueError(
            "Camera delivery prototype count differs from the explicit expectation."
        )
    references = []
    rows = []
    seen_xml_root_ids = set()
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

        plan_source_3d_contract = _json_property(
            plan, SOURCE_3D_CONTRACT_KEY, f"source 3D contract on {plan.name}"
        )
        plan_attachment = _validate_xml_attachment(
            _json_property(plan, XML_ATTACHMENT_KEY, f"XML attachment on {plan.name}"),
            f"XML attachment on {plan.name}",
        )
        plan_root_lock = _json_property(
            plan, PLAN_ROOT_LOCK_KEY, f"root lock on {plan.name}"
        )
        contract_root = next(
            (
                root
                for root in authoritative_source_3d_roots
                if int(root["xml_bone_id"])
                == int(plan_attachment["xml_bone_id"])
            ),
            None,
        )
        if (
            plan_source_3d_contract != source_3d_contract
            or contract_root is None
            or not _attachment_matches_contract(plan_attachment, contract_root)
            or int(mapping.get("xml_bone_id", -1))
            != int(plan_attachment["xml_bone_id"])
            or int(plan_attachment["xml_bone_id"]) not in source_3d_root_ids
        ):
            raise ValueError(f"Plan XML attachment lineage mismatch: {plan.name}")
        xml_root_id = int(plan_attachment["xml_bone_id"])
        if xml_root_id in seen_xml_root_ids:
            raise ValueError(f"Duplicate XML structural root mapping: {xml_root_id}")
        seen_xml_root_ids.add(xml_root_id)

        transfer = _json_property(plan, UV_TRANSFER_KEY, f"UV transfer on {plan.name}")
        projection_basis = _json_property(
            plan, PROJECTION_BASIS_KEY, f"projection basis on {plan.name}"
        )
        stored_projection_coverage = _json_property(
            plan,
            PROJECTION_COVERAGE_KEY,
            f"projection coverage on {plan.name}",
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
            or transfer.get("source_3d_contract") != source_3d_contract
            or transfer.get("xml_attachment") != plan_attachment
            or transfer.get("plan_root_lock") != plan_root_lock
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
        external_uv_validation = _validate_external_camera_uv(
            plan,
            plane,
            contract.get("camera") or {},
            actual_uvs,
            transfer,
        )
        attachment_vertex_index = int(transfer.get("attachment_vertex_index", -1))
        stored_attachment_index = int(
            plan.get("speedtree_cluster_attachment_vertex_index", -1)
        )
        if (
            attachment_vertex_index < 0
            or attachment_vertex_index >= len(plan.data.vertices)
            or stored_attachment_index != attachment_vertex_index
            or external_uv_validation["attachment_vertex_index"]
            != attachment_vertex_index
        ):
            raise ValueError(f"Plan attachment vertex index is missing or stale: {plan.name}")
        attachment_vertex = plan.data.vertices[attachment_vertex_index].co
        if attachment_vertex.length > _TOLERANCE:
            raise ValueError(f"Plan attachment vertex is not local origin: {plan.name}")
        pivot_uv = _finite_vector(
            transfer.get("reference_pivot_uv"), 2, "plan pinned pivot UV"
        )
        stored_attachment_uv = _finite_vector(
            transfer.get("attachment_vertex_uv"),
            2,
            "stored attachment vertex UV",
        )
        if max(
            max(
                abs(actual_uvs[attachment_vertex_index][axis] - pivot_uv[axis]),
                abs(stored_attachment_uv[axis] - pivot_uv[axis]),
                abs(
                    stored_attachment_uv[axis]
                    - actual_uvs[attachment_vertex_index][axis]
                ),
            )
            for axis in range(2)
        ) > _UV_TOLERANCE:
            raise ValueError(f"Plan attachment vertex UV is not pinned: {plan.name}")
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
            part_source_3d_contract = _json_property(
                part, SOURCE_3D_CONTRACT_KEY, f"source 3D contract on {part.name}"
            )
            part_attachment = _validate_xml_attachment(
                _json_property(part, XML_ATTACHMENT_KEY, f"XML attachment on {part.name}"),
                f"XML attachment on {part.name}",
            )
            if (
                part_source_3d_contract != source_3d_contract
                or part_attachment != plan_attachment
            ):
                raise ValueError(f"Plan/3D XML attachment mismatch: {plan.name}")
            try:
                part_frame = Matrix(
                    json.loads(part["speedtree_cluster_frame_world"])
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"3D part frame is invalid: {part.name}") from exc
            if (
                part_frame.translation - attachment_origin_world(part_attachment)
            ).length > _TOLERANCE:
                raise ValueError(f"3D part frame does not start at XML root: {part.name}")
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
        expected_axes = (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        )
        if (
            projection_basis.get("policy") != "camera_aligned_canonical_local_xy"
            or any(
                (actual - expected).length > _TOLERANCE
                for actual, expected in zip((right, up, normal), expected_axes)
            )
        ):
            raise ValueError(f"Plan projection basis is not canonical local XY: {plan.name}")
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
        if any(abs(float(vertex.co.z)) > _TOLERANCE for vertex in plan.data.vertices):
            raise ValueError(f"Camera delivery plan is not on canonical local XY: {plan.name}")
        projected_coverage_points = [
            (
                float(vertex.dot(right)),
                float(vertex.dot(up)),
            )
            for vertex in coverage_points
        ]
        actual_projection_coverage = projection_coverage_2d(
            projected_coverage_points,
            plan_boundary,
        )
        if not actual_projection_coverage["covers_projection"]:
            raise ValueError(f"Plan no longer covers its normalized part: {plan.name}")
        for key in (
            "covers_projection",
            "projected_point_count",
            "outside_point_count",
            "outside_point_indices",
            "boundary_vertex_count",
        ):
            if stored_projection_coverage.get(key) != actual_projection_coverage[key]:
                raise ValueError(
                    f"Plan stored projection coverage is stale: {plan.name}"
                )
        stored_coverage_tolerance = float(
            stored_projection_coverage.get("cross_product_tolerance", math.nan)
        )
        if (
            not math.isfinite(stored_coverage_tolerance)
            or abs(
                stored_coverage_tolerance
                - actual_projection_coverage["cross_product_tolerance"]
            )
            > _TOLERANCE
        ):
            raise ValueError(f"Plan stored projection coverage is stale: {plan.name}")
        if plan_composite_parts:
            raise ValueError(
                "XML physical-root delivery does not accept a composite card frame."
            )
        root_axis = Vector(
            _finite_vector(plan_root_lock.get("root_axis_xy"), 2, "plan root axis")
        )
        source_camera_right = Vector(
            _finite_vector(
                projection_basis.get("source_camera_right"),
                3,
                "source camera right",
            )
        )
        source_camera_up = Vector(
            _finite_vector(
                projection_basis.get("source_camera_up"),
                3,
                "source camera up",
            )
        )
        xml_direction = (
            attachment_endpoint_world(plan_attachment)
            - attachment_origin_world(plan_attachment)
        )
        expected_root_axis = Vector(
            (xml_direction.dot(source_camera_right), xml_direction.dot(source_camera_up))
        )
        if expected_root_axis.length <= _TOLERANCE:
            raise ValueError(f"XML root collapses in plan projection: {plan.name}")
        expected_root_axis.normalize()
        if (
            abs(root_axis.length - 1.0) > _TOLERANCE
            or (root_axis - expected_root_axis).length > _TOLERANCE
        ):
            raise ValueError(f"Plan root axis is not normalized: {plan.name}")
        part_root_support = min(
            Vector(point).dot(root_axis) for point in projected_coverage_points
        )
        plan_root_support = min(
            Vector(point).dot(root_axis) for point in plan_boundary
        )
        stored_unexpanded_support = float(
            plan_root_lock.get("unexpanded_root_support", math.nan)
        )
        stored_locked_support = float(
            plan_root_lock.get("locked_root_support", math.nan)
        )
        root_tolerance = float(plan_root_lock.get("tolerance", math.nan))
        spans = [
            max(point[index] for point in projected_coverage_points)
            - min(point[index] for point in projected_coverage_points)
            for index in range(2)
        ]
        expected_root_tolerance = max(math.hypot(*spans) * 1.0e-7, 1.0e-9)
        maximum_trim = float(
            plan_root_lock.get("maximum_root_margin_trim", math.nan)
        )
        root_policy = str(plan_root_lock.get("policy") or "")
        common_root_lock_invalid = (
            not all(
                math.isfinite(value)
                for value in (
                    part_root_support,
                    plan_root_support,
                    stored_unexpanded_support,
                    stored_locked_support,
                    root_tolerance,
                    maximum_trim,
                )
            )
            or root_tolerance <= 0.0
            or abs(root_tolerance - expected_root_tolerance) > _TOLERANCE * 1.0e-3
            or maximum_trim < 0.0
            or plan_root_lock.get("attachment_xy") != [0.0, 0.0]
            or abs(part_root_support - stored_unexpanded_support) > root_tolerance
            or abs(plan_root_support - stored_locked_support) > root_tolerance
        )
        if root_policy == "xml_root_tangent_preserve_unexpanded_projection_support":
            policy_root_lock_invalid = (
                plan_root_lock.get("attachment_inside_unexpanded_projection")
                is not True
                or abs(part_root_support - plan_root_support) > root_tolerance
            )
        elif root_policy == "xml_root_forward_ray_bridge_to_projection_support":
            bridge_entry = float(
                plan_root_lock.get("attachment_forward_ray_entry", math.nan)
            )
            bridge_exit = float(
                plan_root_lock.get("attachment_forward_ray_exit", math.nan)
            )
            bridge_ratio = float(
                plan_root_lock.get("attachment_gap_ratio", math.nan)
            )
            maximum_bridge_ratio = float(
                plan_root_lock.get(
                    "maximum_attachment_gap_ratio",
                    math.nan,
                )
            )
            policy_root_lock_invalid = (
                plan_root_lock.get("attachment_inside_unexpanded_projection")
                is not False
                or not all(
                    math.isfinite(value)
                    for value in (
                        bridge_entry,
                        bridge_exit,
                        bridge_ratio,
                        maximum_bridge_ratio,
                    )
                )
                or bridge_entry <= root_tolerance
                or bridge_exit <= bridge_entry
                or bridge_ratio < 0.0
                or bridge_ratio > ROOT_BRIDGE_MAX_GAP_RATIO + _TOLERANCE
                or abs(
                    maximum_bridge_ratio - ROOT_BRIDGE_MAX_GAP_RATIO
                )
                > _TOLERANCE
                or abs(plan_root_support) > root_tolerance
                or part_root_support < plan_root_support - root_tolerance
            )
        else:
            policy_root_lock_invalid = True
        if common_root_lock_invalid or policy_root_lock_invalid:
            raise ValueError(f"Plan XML root support lock is stale: {plan.name}")
        rows.append(
            {
                "plan": plan.name,
                "reference": reference.name,
                "source_mesh_id": int(plane["source_mesh_id"]),
                "reference_uv_sha256": plane["uv_sha256"],
                "result_uv_sha256": actual_hash,
                "normalized_rms": normalized_rms,
                "composite_subpart_count": len(plan_composite_parts),
                "projection_coverage": actual_projection_coverage,
                "external_uv_validation": external_uv_validation,
                "xml_bone_id": int(plan_attachment["xml_bone_id"]),
                "plan_root_lock": plan_root_lock,
            }
        )

    if seen_xml_root_ids != set(source_3d_root_ids):
        raise ValueError("Delivery plans do not map every XML structural root exactly once.")
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
        "source_3d_contract_sha256": source_3d_contract_hash,
        "export_objects": export_names,
        "planes": rows,
    }
