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
from .attachment_contract import _parse_speedtree_xml
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
    _transfer_camera_boundary_uvs,
    _uniform_plan_triangulation,
    _ordered_boundary_indices,
    projection_coverage_2d,
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
                part_frame.translation - Vector(part_attachment["xml_start_world"])
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
        xml_direction = Vector(plan_attachment["xml_end_world"]) - Vector(
            plan_attachment["xml_start_world"]
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
        if (
            plan_root_lock.get("policy")
            != "xml_root_tangent_preserve_unexpanded_projection_support"
            or plan_root_lock.get("attachment_inside_unexpanded_projection") is not True
            or not all(
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
            or abs(part_root_support - plan_root_support) > root_tolerance
        ):
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
