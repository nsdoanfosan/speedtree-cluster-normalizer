import json
import hashlib
import math
import re
import uuid
from collections import Counter, defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.geometry import delaunay_2d_cdt

from .attachment_contract import (
    attachment_endpoint_world,
    attachment_origin_world,
    fit_attachment_to_geometry,
    load_attachment_contract,
    match_root_attachment,
    serialized_attachment,
    serialized_contract_source,
)


GENERATED_FLAG = "speedtree_cluster_generated"
LEGACY_GENERATED_FLAG = "atlas_leaf_cluster_generated"
SOURCE_OBJECT_KEY = "speedtree_cluster_source_object"
SOURCE_BONE_KEY = "speedtree_cluster_source_bone"
ENDPOINT_BONE_KEY = "speedtree_cluster_endpoint_bone"
ASSET_ROLE_KEY = "speedtree_cluster_asset_role"
VARIANT_INDEX_KEY = "speedtree_cluster_variant_index"
COUNTERPART_KEY = "speedtree_cluster_counterpart"
PROTOTYPE_INDEX_KEY = "speedtree_cluster_prototype_index"
PROTOTYPE_ASSET_KEY = "speedtree_cluster_prototype_asset"
SOURCE_PARTITION_MODE_KEY = "speedtree_cluster_source_partition_mode"
PROJECTION_BASIS_KEY = "speedtree_cluster_projection_basis"
PROJECTION_COVERAGE_KEY = "speedtree_cluster_projection_coverage"
CARD_PROTOTYPE_MAP_KEY = "speedtree_cluster_card_prototype_map"
CARD_PROTOTYPE_MAP_HASH_KEY = "speedtree_cluster_card_prototype_map_sha256"
COMPOSITE_PARTS_KEY = "speedtree_cluster_composite_parts"
CAMERA_CONTRACT_KEY = "speedtree_cluster_camera_uv_contract"
CAMERA_CONTRACT_HASH_KEY = "speedtree_cluster_camera_uv_contract_sha256"
CAMERA_REFERENCE_KEY = "speedtree_cluster_camera_reference"
UV_TRANSFER_KEY = "speedtree_cluster_uv_transfer"
SOURCE_3D_CONTRACT_KEY = "speedtree_cluster_source_3d_contract"
SOURCE_3D_CONTRACT_HASH_KEY = "speedtree_cluster_source_3d_contract_sha256"
XML_ATTACHMENT_KEY = "speedtree_cluster_xml_attachment"
PLAN_ROOT_LOCK_KEY = "speedtree_cluster_plan_root_lock"
# Camera cutouts and generated covering cards need not share the same outline.
# Keep this scale-independent guard broad enough for intentionally different
# side captures while the separate planarity, coverage, orientation, and UV
# invariants continue to fail closed.
UV_TRANSFER_MAX_NORMALIZED_RMS = 0.5
UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR = 0.0
UV_TRANSFER_ATTACHMENT_POLICY = (
    "exact_normalized_plan_origin_to_camera_contract_source_plane_xy"
)
UV_TRANSFER_CANDIDATE_SELECTION_POLICY = (
    "exact_attachment_constrained_then_outline_normalized_rms"
)
CAMERA_ALIGNED_FRAME_POLICY = "shared_camera_right_up_normal_rigid_frame"
PHYSICAL_CAPTURE_FRAME_POLICY = "physical_target_uniform_whole_source_fit"
PHYSICAL_CAPTURE_ALIGNED_FRAME_POLICY = (
    "physical_capture_projected_attachment_axis_xy_uniform_fit"
)
PHYSICAL_CAPTURE_DIRECTION_PROJECTED_XML = (
    "project_xml_attachment_into_capture_plane"
)
PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK = (
    "capture_up_for_capture_normal_attachment"
)
WORKFLOW_LEGACY_CAMERA_UV = "LEGACY_CAMERA_UV"
WORKFLOW_PHYSICAL_DIRECT_CAPTURE = "PHYSICAL_DIRECT_CAPTURE"
DIRECT_CAPTURE_UV_SOURCE = "same_blender_physical_capture_projection"
PHYSICAL_CAPTURE_CONTRACT_KEY = "speedtree_cluster_physical_capture_contract"
PHYSICAL_CAPTURE_CONTRACT_HASH_KEY = (
    "speedtree_cluster_physical_capture_contract_sha256"
)
DIRECT_CAPTURE_UV_KEY = "speedtree_cluster_direct_capture_uv"
ROOT_BRIDGE_MAX_GAP_RATIO = 0.25


def physical_capture_projection_tolerance(raw_direction, attachment=None):
    """Return the shared scale-aware capture-plane direction tolerance."""
    direction = Vector(raw_direction)
    if not all(math.isfinite(float(value)) for value in direction):
        raise ValueError("Physical capture attachment direction is not finite.")
    attachment = attachment if isinstance(attachment, dict) else {}
    evidence = [
        float(direction.length) * 1.0e-8,
        1.0e-9,
    ]
    for key in (
        "match_tolerance",
        "effective_support_tolerance_world",
    ):
        value = float(attachment.get(key) or 0.0)
        if math.isfinite(value) and value > 0.0:
            evidence.append(value)
    geometry_scale = float(
        attachment.get("effective_geometry_scale_world") or 0.0
    )
    if math.isfinite(geometry_scale) and geometry_scale > 0.0:
        evidence.append(geometry_scale * 1.0e-8)
    return max(evidence)


def _natural_key(value):
    return [int(token) if token.isdigit() else token.casefold() for token in re.split(r"(\d+)", value)]


def _matrix_rows(matrix):
    return [[float(value) for value in row] for row in matrix]


def _identity_matrix(matrix, tolerance=1.0e-6):
    identity = Matrix.Identity(4)
    return all(
        abs(float(matrix[row][column] - identity[row][column])) <= tolerance
        for row in range(4)
        for column in range(4)
    )


def _bounds(points):
    if not points:
        return None
    minimum = [min(float(point[axis]) for point in points) for axis in range(3)]
    maximum = [max(float(point[axis]) for point in points) for axis in range(3)]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
        "center": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
    }


def find_source_armature(source):
    if source is None or source.type != "MESH":
        raise ValueError("Cluster source must be a mesh object.")
    modifier_armatures = [
        modifier.object
        for modifier in source.modifiers
        if modifier.type == "ARMATURE" and modifier.object is not None
    ]
    if len(modifier_armatures) == 1:
        return modifier_armatures[0]
    if len(modifier_armatures) > 1:
        names = ", ".join(obj.name for obj in modifier_armatures)
        raise ValueError(f"Cluster source has multiple armature modifiers: {names}")
    if source.parent is not None and source.parent.type == "ARMATURE":
        return source.parent
    raise ValueError(f"Cluster source has no armature: {source.name}")


def _vertex_bone_weights(source, armature):
    bone_names = {bone.name for bone in armature.data.bones}
    canonical_bone_names = {}
    for bone_name in bone_names:
        match = re.search(r"_(\d+)_End$", bone_name, flags=re.IGNORECASE)
        if match is None:
            canonical_bone_names[bone_name] = bone_name
            continue
        expected = re.sub(r"_End$", "_Start", bone_name, flags=re.IGNORECASE)
        canonical_bone_names[bone_name] = expected if expected in bone_names else bone_name
    group_names = {
        group.index: canonical_bone_names[group.name]
        for group in source.vertex_groups
        if group.name in bone_names
    }
    weights = {}
    for vertex in source.data.vertices:
        row = defaultdict(float)
        for element in vertex.groups:
            if element.group not in group_names or element.weight <= 0.0:
                continue
            row[group_names[element.group]] += float(element.weight)
        row = dict(row)
        weights[vertex.index] = row
    return weights


def _face_group_assignments(source, weights):
    assignments = defaultdict(list)
    vertex_assignments = defaultdict(list)
    unweighted_vertices = []
    tied_vertices = []
    dominant_by_vertex = {}
    for vertex in source.data.vertices:
        row = sorted(
            ((weight, name) for name, weight in weights[vertex.index].items()),
            key=lambda item: (-item[0], _natural_key(item[1])),
        )
        if not row:
            unweighted_vertices.append(vertex.index)
            continue
        if len(row) > 1 and abs(row[0][0] - row[1][0]) <= 1.0e-7:
            tied_vertices.append(vertex.index)
        dominant_by_vertex[vertex.index] = row[0][1]
        vertex_assignments[row[0][1]].append(vertex.index)

    mixed_faces = []
    unweighted_faces = []
    for polygon in source.data.polygons:
        scores = Counter()
        dominant_names = set()
        for vertex_index in polygon.vertices:
            dominant_name = dominant_by_vertex.get(vertex_index)
            if dominant_name:
                dominant_names.add(dominant_name)
            for name, weight in weights[vertex_index].items():
                scores[name] += weight
        if not scores:
            unweighted_faces.append(polygon.index)
            continue
        ordered = sorted(scores.items(), key=lambda item: (-item[1], _natural_key(item[0])))
        assignments[ordered[0][0]].append(polygon.index)
        if len(dominant_names) > 1:
            mixed_faces.append(polygon.index)

    return {
        "faces": assignments,
        "vertices": vertex_assignments,
        "unweighted_vertices": unweighted_vertices,
        "tied_vertices": tied_vertices,
        "mixed_faces": mixed_faces,
        "unweighted_faces": unweighted_faces,
    }


def _connected_deform_clusters(source, assignments, valid_per_deform_rows):
    """Merge deform roots that participate in one topological mesh component.

    A SpeedTree cluster can use several deform roots inside one complete 3D
    prototype.  Treating every root as an export part tears that prototype
    apart.  Faces that meet through a shared source vertex are in the same
    topology component, so all of their assigned deform roots must belong to
    one exported cluster.
    """
    rows_by_name = {bone.name: (ordinal, bone) for ordinal, bone in valid_per_deform_rows}
    parent = {name: name for name in rows_by_name}

    def find(name):
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != name:
            following = parent[name]
            parent[name] = root
            name = following
        return root

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        first_ordinal = rows_by_name[first_root][0]
        second_ordinal = rows_by_name[second_root][0]
        if first_ordinal <= second_ordinal:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    face_bones = {}
    for bone_name, face_indices in assignments["faces"].items():
        if bone_name not in rows_by_name:
            continue
        for face_index in face_indices:
            if face_index in face_bones:
                raise ValueError(f"Face {face_index} has more than one deform assignment.")
            face_bones[face_index] = bone_name
    if len(face_bones) != len(source.data.polygons):
        raise ValueError(
            "Connected cluster partition requires every source face to resolve to a valid "
            f"deform root: {len(face_bones)} of {len(source.data.polygons)} faces."
        )

    bones_by_vertex = defaultdict(set)
    for polygon in source.data.polygons:
        bone_name = face_bones[polygon.index]
        for vertex_index in polygon.vertices:
            bones_by_vertex[int(vertex_index)].add(bone_name)
    for bone_names in bones_by_vertex.values():
        ordered = sorted(bone_names, key=lambda name: rows_by_name[name][0])
        for bone_name in ordered[1:]:
            union(ordered[0], bone_name)

    grouped_names = defaultdict(list)
    for name in rows_by_name:
        grouped_names[find(name)].append(name)
    groups = []
    for names in grouped_names.values():
        names.sort(key=lambda name: rows_by_name[name][0])
        representative_name = names[0]
        face_indices = sorted(
            face_index
            for name in names
            for face_index in assignments["faces"][name]
        )
        groups.append(
            {
                "ordinal": rows_by_name[representative_name][0],
                "representative_bone": rows_by_name[representative_name][1],
                "bone_names": names,
                "face_indices": face_indices,
            }
        )
    groups.sort(key=lambda item: item["ordinal"])
    covered_faces = [face_index for group in groups for face_index in group["face_indices"]]
    if sorted(covered_faces) != list(range(len(source.data.polygons))):
        raise ValueError("Connected cluster partition lost or duplicated source faces.")
    return groups


def _preferred_endpoint_bone(bone, populated_bones):
    if bone.name.casefold().endswith("_start"):
        expected = bone.name[:-6] + "_End"
        for child in bone.children:
            if child.name.casefold() == expected.casefold():
                return child, "matching_end_child"
        return None, "start_bone_tail_axis_endpoint"
    if bone.name.casefold().endswith("_end") and bone.parent is None:
        return None, "orphan_end_uses_validated_asset_root_pivot"
    raise ValueError(
        f"Populated deform bone '{bone.name}' must be a *_Start axis bone or an "
        "orphan root *_End handled by a validated asset pivot."
    )


def _explicit_start_bone_ordinal(bone_name):
    match = re.search(r"_(\d+)_(?:Start|End)$", str(bone_name), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Populated deform bone '{bone_name}' needs an explicit *_N_Start or orphan *_N_End ordinal."
        )
    ordinal = int(match.group(1))
    if ordinal < 1:
        raise ValueError(f"Bone ordinal must be greater than zero: {bone_name}")
    return ordinal


def _stable_perpendicular(axis, armature):
    rotation = armature.matrix_world.to_quaternion().to_matrix()
    candidates = [
        rotation @ Vector((1.0, 0.0, 0.0)),
        rotation @ Vector((0.0, 1.0, 0.0)),
        rotation @ Vector((0.0, 0.0, 1.0)),
    ]
    projected = []
    for candidate in candidates:
        value = candidate - axis * candidate.dot(axis)
        projected.append((value.length_squared, value))
    length_squared, value = max(projected, key=lambda item: item[0])
    if length_squared <= 1.0e-20:
        raise ValueError("Cannot derive a stable perpendicular axis.")
    return value.normalized()


def canonical_frame(
    source,
    armature,
    bone,
    endpoint_bone,
    vertex_indices,
    attachment=None,
):
    armature_world = armature.matrix_world
    if attachment is None:
        origin = armature_world @ bone.head_local
        endpoint = (
            armature_world @ endpoint_bone.head_local
            if endpoint_bone is not None
            else armature_world @ bone.tail_local
        )
    else:
        origin = attachment_origin_world(attachment)
        endpoint = attachment_endpoint_world(attachment)
    world_points = [
        source.matrix_world @ source.data.vertices[index].co
        for index in vertex_indices
    ]
    if len(world_points) < 3:
        raise ValueError(f"Bone group has fewer than three vertices: {bone.name}")
    world_bounds = _bounds(world_points)
    geometry_scale = max(world_bounds["size"])
    if geometry_scale <= 0.0:
        raise ValueError(f"Bone group is geometrically degenerate: {bone.name}")
    relative_epsilon = geometry_scale * 1.0e-12
    axis_y = endpoint - origin
    if axis_y.length <= relative_epsilon:
        raise ValueError(f"Bone endpoint is degenerate relative to its geometry: {bone.name}")
    length = axis_y.length
    axis_y.normalize()

    basis_u = _stable_perpendicular(axis_y, armature)
    basis_v = axis_y.cross(basis_u).normalized()
    coordinates = [
        ((point - origin).dot(basis_u), (point - origin).dot(basis_v))
        for point in world_points
    ]
    mean_u = sum(value[0] for value in coordinates) / len(coordinates)
    mean_v = sum(value[1] for value in coordinates) / len(coordinates)
    covariance_uu = sum((value[0] - mean_u) ** 2 for value in coordinates) / len(coordinates)
    covariance_uv = sum(
        (value[0] - mean_u) * (value[1] - mean_v) for value in coordinates
    ) / len(coordinates)
    covariance_vv = sum((value[1] - mean_v) ** 2 for value in coordinates) / len(coordinates)
    eigengap = math.sqrt(
        (covariance_uu - covariance_vv) ** 2 + 4.0 * covariance_uv ** 2
    )
    covariance_scale = covariance_uu + covariance_vv
    if covariance_scale <= 0.0:
        raise ValueError(f"Bone group has no measurable cross-section: {bone.name}")
    if eigengap <= covariance_scale * 1.0e-8:
        axis_x = basis_u
    else:
        angle = 0.5 * math.atan2(
            2.0 * covariance_uv,
            covariance_uu - covariance_vv,
        )
        axis_x = (math.cos(angle) * basis_u + math.sin(angle) * basis_v).normalized()
    if axis_x.dot(basis_u) < 0.0:
        axis_x.negate()
    axis_z = axis_x.cross(axis_y).normalized()
    axis_x = axis_y.cross(axis_z).normalized()

    frame = Matrix.Identity(4)
    for row in range(3):
        frame[row][0] = axis_x[row]
        frame[row][1] = axis_y[row]
        frame[row][2] = axis_z[row]
        frame[row][3] = origin[row]
    local_points = [frame.inverted_safe() @ point for point in world_points]
    result = {
        "matrix_world": frame,
        "origin_world": [float(value) for value in origin],
        "endpoint_world": [float(value) for value in endpoint],
        "endpoint_length": float(length),
        "source_world_bounds": world_bounds,
        "normalized_bounds": _bounds(local_points),
    }
    if attachment is not None:
        result["xml_attachment"] = serialized_attachment(attachment)
    return result


def _object_is_ancestor(ancestor, obj):
    current = obj.parent
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False


def _whole_mesh_pivot(source, explicit_pivot):
    pivot = explicit_pivot
    if pivot is None:
        pivot = source.parent
        if pivot is None or pivot.type != "EMPTY":
            raise ValueError(
                "WHOLE_MESH requires an explicit ancestor pivot or an immediate EMPTY parent."
            )
    if pivot is source or not _object_is_ancestor(pivot, source):
        raise ValueError("WHOLE_MESH pivot must be an ancestor of the source mesh.")
    matrix = pivot.matrix_world.copy()
    linear = matrix.to_3x3()
    determinant = float(linear.determinant())
    if not math.isfinite(determinant) or determinant <= 1.0e-12:
        raise ValueError("WHOLE_MESH pivot has a mirrored or singular world transform.")
    columns = [Vector(linear.col[index]) for index in range(3)]
    lengths = [column.length for column in columns]
    if min(lengths) <= 1.0e-12:
        raise ValueError("WHOLE_MESH pivot has a zero-length transform axis.")
    normalized = [column / length for column, length in zip(columns, lengths)]
    orthogonality = max(
        abs(normalized[first].dot(normalized[second]))
        for first, second in ((0, 1), (0, 2), (1, 2))
    )
    scale_ratio = max(lengths) / min(lengths)
    if orthogonality > 1.0e-6 or scale_ratio > 1.0 + 1.0e-6:
        raise ValueError(
            "WHOLE_MESH pivot must have an orthogonal uniform-scale world transform."
        )
    return pivot


def whole_mesh_frame(source, pivot):
    matrix_world = pivot.matrix_world.copy()
    transform = matrix_world.inverted_safe() @ source.matrix_world
    local_points = [transform @ vertex.co for vertex in source.data.vertices]
    bounds = _bounds(local_points)
    if bounds is None or max(bounds["size"]) <= 0.0:
        raise ValueError("WHOLE_MESH source has no measurable geometry.")
    endpoint_length = max(float(bounds["size"][1]), max(bounds["size"]) * 0.05)
    origin = matrix_world.translation
    endpoint = matrix_world @ Vector((0.0, endpoint_length, 0.0))
    return {
        "matrix_world": matrix_world,
        "origin_world": [float(value) for value in origin],
        "endpoint_world": [float(value) for value in endpoint],
        "endpoint_length": float(endpoint_length),
        "source_world_bounds": _bounds(
            [source.matrix_world @ vertex.co for vertex in source.data.vertices]
        ),
        "normalized_bounds": bounds,
        "pivot_object": pivot.name,
    }


def pivot_subset_frame(source, pivot, vertex_indices):
    matrix_world = pivot.matrix_world.copy()
    transform = matrix_world.inverted_safe() @ source.matrix_world
    local_points = [transform @ source.data.vertices[index].co for index in vertex_indices]
    bounds = _bounds(local_points)
    if bounds is None or max(bounds["size"]) <= 0.0:
        raise ValueError("Pivot-normalized cluster subset has no measurable geometry.")
    endpoint_length = max(float(bounds["size"][1]), max(bounds["size"]) * 0.05)
    origin = matrix_world.translation
    endpoint = matrix_world @ Vector((0.0, endpoint_length, 0.0))
    return {
        "matrix_world": matrix_world,
        "origin_world": [float(value) for value in origin],
        "endpoint_world": [float(value) for value in endpoint],
        "endpoint_length": float(endpoint_length),
        "source_world_bounds": _bounds(
            [source.matrix_world @ source.data.vertices[index].co for index in vertex_indices]
        ),
        "normalized_bounds": bounds,
        "pivot_object": pivot.name,
    }


def _camera_world_axes(camera):
    try:
        right = Vector(camera["right"])
        up = Vector(camera["up"])
        expected_normal = Vector(camera["plane_normal"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Camera contract lacks a numeric right/up/normal basis.") from exc
    if min(right.length, up.length, expected_normal.length) <= 1.0e-12:
        raise ValueError("Camera basis contains a zero-length axis.")
    right.normalize()
    up = up - right * up.dot(right)
    if up.length <= 1.0e-12:
        raise ValueError("Camera right/up axes collapse during rigid normalization.")
    up.normalize()
    normal = right.cross(up)
    normal.normalize()
    expected_normal.normalize()
    if normal.dot(expected_normal) < 0.999999:
        raise ValueError("Camera right/up/normal basis is not consistently right-handed.")
    return right, up, normal


def camera_aligned_frame(source, source_frame, vertex_indices, camera):
    """Keep the validated attachment origin while replacing orientation with the camera basis."""
    indices = sorted({int(index) for index in vertex_indices})
    if len(indices) < 3:
        raise ValueError("Camera-aligned frame requires at least three source vertices.")
    right, up, normal = _camera_world_axes(camera)
    origin = source_frame["matrix_world"].translation.copy()
    matrix_world = Matrix.Identity(4)
    for row in range(3):
        matrix_world[row][0] = right[row]
        matrix_world[row][1] = up[row]
        matrix_world[row][2] = normal[row]
        matrix_world[row][3] = origin[row]
    transform = matrix_world.inverted() @ source.matrix_world
    local_points = [transform @ source.data.vertices[index].co for index in indices]
    bounds = _bounds(local_points)
    if bounds is None or max(bounds["size"]) <= 0.0:
        raise ValueError("Camera-aligned source subset has no measurable geometry.")
    endpoint_length = max(float(bounds["size"][1]), max(bounds["size"]) * 0.05)
    endpoint = matrix_world @ Vector((0.0, endpoint_length, 0.0))
    result = {
        "matrix_world": matrix_world,
        "origin_world": [float(value) for value in origin],
        "endpoint_world": [float(value) for value in endpoint],
        "endpoint_length": float(endpoint_length),
        "source_world_bounds": _bounds(
            [source.matrix_world @ source.data.vertices[index].co for index in indices]
        ),
        "normalized_bounds": bounds,
        "orientation_policy": CAMERA_ALIGNED_FRAME_POLICY,
        "source_frame_world": source_frame["matrix_world"].copy(),
        "source_endpoint_world": list(source_frame.get("endpoint_world") or []),
        "pivot_object": source_frame.get("pivot_object"),
    }
    if source_frame.get("xml_attachment") is not None:
        result["xml_attachment"] = dict(source_frame["xml_attachment"])
    return result


def _physical_capture_camera(frame):
    return {
        "right": list(frame["right"]),
        "up": list(frame["up"]),
        "plane_normal": list(frame["normal"]),
        "view_direction": list(frame["view_direction"]),
    }


def _validate_physical_capture_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("Physical direct capture requires an explicit capture contract.")
    if (
        contract.get("kind") != "speedtree_cluster_physical_capture_fit"
        or int(contract.get("version", 0)) != 1
        or contract.get("workflow_mode") != WORKFLOW_PHYSICAL_DIRECT_CAPTURE
        or contract.get("direct_uv_source") != DIRECT_CAPTURE_UV_SOURCE
    ):
        raise ValueError("Physical capture contract kind/workflow is unsupported.")
    recorded_hash = str(contract.get("contract_sha256") or "")
    hash_payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if not recorded_hash or recorded_hash != _canonical_sha256(hash_payload):
        raise ValueError("Physical capture contract hash is missing or stale.")
    frame = contract.get("frame") or {}
    if (
        frame.get("policy") != PHYSICAL_CAPTURE_FRAME_POLICY
        or frame.get("workflow_mode") != WORKFLOW_PHYSICAL_DIRECT_CAPTURE
        or frame.get("direct_uv_source") != DIRECT_CAPTURE_UV_SOURCE
    ):
        raise ValueError("Physical capture frame is not the direct-capture policy.")
    if str(frame.get("unit_system") or "") != "METRIC":
        raise ValueError("Physical capture frame is not in a METRIC scene.")
    scale_length = float(frame.get("scale_length", math.nan))
    fit_scale = float(frame.get("fit_scale", math.nan))
    width = float(frame.get("width", math.nan))
    height = float(frame.get("height", math.nan))
    target_meters = frame.get("target_meters") or []
    target_units = frame.get("target_blender_units") or []
    if (
        not all(math.isfinite(value) and value > 0.0 for value in (
            scale_length,
            fit_scale,
            width,
            height,
        ))
        or len(target_meters) != 2
        or len(target_units) != 2
        or abs(width - height) > 1.0e-9
        or max(abs(float(value) - width) for value in target_units) > 1.0e-9
        or max(
            abs(float(target_meters[index]) / scale_length - float(target_units[index]))
            for index in range(2)
        )
        > 1.0e-9
    ):
        raise ValueError("Physical capture unit/target evidence is invalid.")
    _camera_world_axes(_physical_capture_camera(frame))
    return contract, frame, recorded_hash


def physical_capture_aligned_frame(
    source,
    source_frame,
    vertex_indices,
    capture_frame,
):
    """Fit a pair while preserving its captured attachment direction in local XY."""
    indices = sorted({int(index) for index in vertex_indices})
    if len(indices) < 3:
        raise ValueError("Physical capture frame requires at least three source vertices.")
    source_matrix = source_frame["matrix_world"].copy()
    source_axes = [
        Vector(source_matrix.to_3x3().col[index]).normalized()
        for index in range(3)
    ]
    if source_axes[0].cross(source_axes[1]).dot(source_axes[2]) < 0.999999:
        raise ValueError(
            "Physical capture source frame is not a right-handed authored axis frame."
        )
    _capture_right, capture_up, capture_normal = _camera_world_axes(
        _physical_capture_camera(capture_frame)
    )
    raw_attachment = source_matrix.translation.copy()
    raw_endpoint = Vector(source_frame["endpoint_world"])
    raw_direction = raw_endpoint - raw_attachment
    raw_length = float(raw_direction.length)
    source_size = (
        (source_frame.get("source_world_bounds") or {}).get("size") or []
    )
    finite_source_size = [
        abs(float(value))
        for value in source_size
        if math.isfinite(float(value))
    ]
    geometry_scale = max(finite_source_size, default=raw_length)
    numeric_direction_tolerance = max(
        geometry_scale * 1.0e-12,
        1.0e-12,
    )
    if (
        not math.isfinite(raw_length)
        or raw_length <= numeric_direction_tolerance
    ):
        raise ValueError(
            "Physical capture XML attachment has no finite 3D direction."
        )
    projected_direction = (
        raw_direction - capture_normal * raw_direction.dot(capture_normal)
    )
    direction_tolerance = physical_capture_projection_tolerance(
        raw_direction,
        source_frame.get("xml_attachment"),
    )
    if projected_direction.length <= direction_tolerance:
        aligned_direction = capture_up * max(
            raw_length,
            float(source_frame.get("endpoint_length") or 0.0),
            direction_tolerance,
        )
        direction_policy = PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK
    else:
        aligned_direction = projected_direction.copy()
        direction_policy = PHYSICAL_CAPTURE_DIRECTION_PROJECTED_XML
    axis_y = aligned_direction.normalized()
    axis_z = capture_normal.normalized()
    axis_x = axis_y.cross(axis_z).normalized()
    alignment_matrix = Matrix.Identity(4)
    for row in range(3):
        alignment_matrix[row][0] = axis_x[row]
        alignment_matrix[row][1] = axis_y[row]
        alignment_matrix[row][2] = axis_z[row]
        alignment_matrix[row][3] = raw_attachment[row]
    fit_scale = float(capture_frame["fit_scale"])
    capture_center = Vector(capture_frame["center"])
    fitted_attachment = (
        capture_center + (raw_attachment - capture_center) * fit_scale
    )
    matrix_world = alignment_matrix.copy()
    matrix_world.translation = fitted_attachment
    source_to_normalized = (
        Matrix.Diagonal((fit_scale, fit_scale, fit_scale, 1.0))
        @ alignment_matrix.inverted_safe()
        @ source.matrix_world
    )
    local_points = [
        source_to_normalized @ source.data.vertices[index].co
        for index in indices
    ]
    bounds = _bounds(local_points)
    if bounds is None or max(bounds["size"]) <= 0.0:
        raise ValueError("Physical capture source subset has no measurable geometry.")
    projected_endpoint = raw_attachment + aligned_direction
    fitted_endpoint = (
        capture_center
        + (projected_endpoint - capture_center) * fit_scale
    )
    endpoint_local = (
        alignment_matrix.to_3x3().inverted_safe()
        @ (aligned_direction * fit_scale)
    )
    endpoint_numeric_tolerance = max(
        max(bounds["size"]) * 1.0e-12,
        1.0e-12,
    )
    endpoint_axis = (
        endpoint_local / endpoint_local.length
        if endpoint_local.length > endpoint_numeric_tolerance
        else Vector((math.nan, math.nan, math.nan))
    )
    if (
        not all(math.isfinite(float(value)) for value in endpoint_local)
        or endpoint_local.length <= endpoint_numeric_tolerance
        or (endpoint_axis - Vector((0.0, 1.0, 0.0))).length > 1.0e-6
    ):
        raise ValueError(
            "Physical capture source attachment axis is not the authored local +Y axis: "
            f"endpoint_local={tuple(float(value) for value in endpoint_local)}, "
            f"numeric_tolerance={endpoint_numeric_tolerance:.9g}."
        )
    endpoint_length = max(
        float(endpoint_local.length),
        max(bounds["size"]) * 0.05,
    )
    result = dict(source_frame)
    result.update(
        {
            "matrix_world": matrix_world,
            "origin_world": [float(value) for value in fitted_attachment],
            "endpoint_world": [float(value) for value in fitted_endpoint],
            "endpoint_length": float(endpoint_length),
            "normalized_bounds": bounds,
            "orientation_policy": PHYSICAL_CAPTURE_ALIGNED_FRAME_POLICY,
            "source_frame_world": source_matrix.copy(),
            "source_alignment_frame_world": alignment_matrix.copy(),
            "source_endpoint_world": [
                float(value) for value in raw_endpoint
            ],
            "source_to_normalized_matrix": source_to_normalized,
            "physical_fit_scale": fit_scale,
            "attachment_tangent_projection": {
                "source_world": [float(value) for value in raw_direction],
                "capture_plane_world": [
                    float(value) for value in projected_direction
                ],
                "aligned_capture_plane_world": [
                    float(value) for value in aligned_direction
                ],
                "direction_policy": direction_policy,
                "projection_tolerance": float(direction_tolerance),
                "discarded_capture_normal_component": float(
                    raw_direction.dot(capture_normal)
                ),
                "normalized_local_xy": [0.0, 1.0],
            },
            "capture_attachment": {
                "source_world": [float(value) for value in raw_attachment],
                "fitted_capture_world": [
                    float(value) for value in fitted_attachment
                ],
                "normalized_local": [0.0, 0.0, 0.0],
            },
        }
    )
    return result


def camera_projection_basis_in_part(frame, camera):
    camera_right, camera_up, camera_normal = _camera_world_axes(camera)
    if frame.get("orientation_policy") == CAMERA_ALIGNED_FRAME_POLICY:
        frame_axes = [
            Vector(frame["matrix_world"].to_3x3().col[index]).normalized()
            for index in range(3)
        ]
        camera_axes = (camera_right, camera_up, camera_normal)
        if any(
            frame_axis.dot(camera_axis) < 0.999999
            for frame_axis, camera_axis in zip(frame_axes, camera_axes)
        ):
            raise ValueError("Camera-aligned frame drifted from its source camera basis.")
        return {
            "policy": "camera_aligned_canonical_local_xy",
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "source_camera_right": [float(value) for value in camera_right],
            "source_camera_up": [float(value) for value in camera_up],
            "source_camera_normal": [float(value) for value in camera_normal],
        }
    world_to_part = frame["matrix_world"].inverted_safe().to_3x3()
    right = world_to_part @ camera_right
    up = world_to_part @ camera_up
    expected_normal = world_to_part @ camera_normal
    if min(right.length, up.length, expected_normal.length) <= 1.0e-12:
        raise ValueError("Camera basis degenerates in the canonical part frame.")
    right.normalize()
    up = up - right * up.dot(right)
    if up.length <= 1.0e-12:
        raise ValueError("Camera right/up axes collapse in the canonical part frame.")
    up.normalize()
    normal = right.cross(up)
    normal.normalize()
    expected_normal.normalize()
    if normal.dot(expected_normal) < 0.999999:
        raise ValueError("Camera basis changes handedness in the canonical part frame.")
    return {
        "policy": "camera_basis_transformed_world_to_canonical_part_local",
        "right": [float(value) for value in right],
        "up": [float(value) for value in up],
        "normal": [float(value) for value in normal],
        "source_camera_right": [float(value) for value in camera_right],
        "source_camera_up": [float(value) for value in camera_up],
        "source_camera_normal": [float(value) for value in camera_normal],
    }


def convex_hull_2d(points):
    raw = [(float(point[0]), float(point[1])) for point in points]
    if len(raw) < 3:
        raise ValueError("Projected mesh needs at least three points.")
    minimum_x = min(point[0] for point in raw)
    maximum_x = max(point[0] for point in raw)
    minimum_y = min(point[1] for point in raw)
    maximum_y = max(point[1] for point in raw)
    scale = max(maximum_x - minimum_x, maximum_y - minimum_y)
    if scale <= 0.0:
        raise ValueError("Projected mesh has no measurable extent.")
    center = ((minimum_x + maximum_x) * 0.5, (minimum_y + maximum_y) * 0.5)
    unique = sorted(
        {
            (
                (point[0] - center[0]) / scale,
                (point[1] - center[1]) / scale,
            )
            for point in raw
        }
    )
    if len(unique) < 3:
        raise ValueError("Projected mesh needs at least three unique points.")

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("Projected mesh is collinear.")
    return [
        (center[0] + point[0] * scale, center[1] + point[1] * scale)
        for point in hull
    ]


def expanded_hull(points, margin_ratio):
    hull = convex_hull_2d(points)
    margin_ratio = max(float(margin_ratio), 0.0)
    if margin_ratio == 0.0:
        return hull
    center = (
        sum(point[0] for point in hull) / len(hull),
        sum(point[1] for point in hull) / len(hull),
    )
    scale = 1.0 + margin_ratio
    return [
        (
            center[0] + (point[0] - center[0]) * scale,
            center[1] + (point[1] - center[1]) * scale,
        )
        for point in hull
    ]


def root_locked_expanded_hull(points, margin_ratio, root_axis):
    base_hull = convex_hull_2d(points)
    axis = Vector((float(root_axis[0]), float(root_axis[1])))
    if axis.length <= 1.0e-12:
        raise ValueError("XML root segment collapses in the camera plan plane.")
    axis.normalize()
    attachment = (0.0, 0.0)
    spans = [
        max(point[index] for point in base_hull)
        - min(point[index] for point in base_hull)
        for index in range(2)
    ]
    diagonal = math.hypot(*spans)
    tolerance = max(diagonal * 1.0e-7, 1.0e-9)
    # Containment decides whether the authored attachment must become part of
    # the plan hull.  A broad tolerance here can classify a point just outside
    # the source silhouette as inside, leaving the subsequently-added origin
    # as a loose CDT vertex that FBX drops.  Use exact half-plane signs for the
    # classification; the scale-aware tolerance is still used by the later
    # validation and ray-intersection guards.
    attachment_inside = point_in_convex_polygon(
        attachment,
        base_hull,
        tolerance=0.0,
    )
    ray_hits = []
    if not attachment_inside:
        ray_hits = []
        for index, first in enumerate(base_hull):
            second = base_hull[(index + 1) % len(base_hull)]
            edge = Vector((second[0] - first[0], second[1] - first[1]))
            denominator = axis.x * edge.y - axis.y * edge.x
            if abs(float(denominator)) <= tolerance:
                continue
            point = Vector(first)
            distance = (point.x * edge.y - point.y * edge.x) / denominator
            fraction = (point.x * axis.y - point.y * axis.x) / denominator
            if distance >= -tolerance and -tolerance <= fraction <= 1.0 + tolerance:
                ray_hits.append(max(float(distance), 0.0))
        ray_hits.sort()
        unique_hits = []
        for distance in ray_hits:
            if not unique_hits or abs(distance - unique_hits[-1]) > tolerance:
                unique_hits.append(distance)
        ray_hits = unique_hits

    coverage_root_support = min(
        Vector(point).dot(axis)
        for point in base_hull
    )
    bridge_entry = None
    bridge_exit = None
    bridge_gap_ratio = None
    if attachment_inside:
        support_hull = base_hull
        root_support = coverage_root_support
        policy = "xml_root_tangent_preserve_unexpanded_projection_support"
    else:
        if len(ray_hits) < 2:
            raise ValueError(
                "XML attachment origin is outside the unexpanded 3D projection "
                "hull and its forward root axis does not enter the projection; "
                "the XML/root mapping or camera contract is inconsistent."
            )
        bridge_entry = float(ray_hits[0])
        bridge_exit = float(ray_hits[-1])
        bridge_gap_ratio = bridge_entry / diagonal
        if (
            coverage_root_support < -tolerance
            or bridge_gap_ratio > ROOT_BRIDGE_MAX_GAP_RATIO
        ):
            raise ValueError(
                "XML attachment origin is outside the unexpanded 3D projection "
                "hull and its forward root gap exceeds the validated bridge "
                "contract; "
                f"entry={bridge_entry:.9g}, diagonal={diagonal:.9g}, "
                f"ratio={bridge_gap_ratio:.9g}, "
                f"maximum={ROOT_BRIDGE_MAX_GAP_RATIO:.9g}."
            )
        # The XML axis is valid and enters the source silhouette shortly after
        # the physical attachment.  Keep that authored attachment as the plan
        # root, bridge only the verified forward gap, and never manufacture a
        # root-side margin behind it.
        support_hull = convex_hull_2d([*base_hull, attachment])
        root_support = 0.0
        policy = "xml_root_forward_ray_bridge_to_projection_support"
    distal_support = max(Vector(point).dot(axis) for point in base_hull)
    expanded = expanded_hull(support_hull, margin_ratio)
    locked_points = []
    maximum_trim = 0.0
    for point in expanded:
        value = Vector(point)
        signed = value.dot(axis)
        if signed < root_support:
            maximum_trim = max(maximum_trim, float(root_support - signed))
            value += axis * (root_support - signed)
        locked_points.append((float(value.x), float(value.y)))
    # Clipping only the expanded vertices can move an adjacent support edge
    # across a sharp root corner.  Retain the complete unexpanded hull in the
    # final convex set so root locking can never sacrifice source coverage.
    locked_points.extend(support_hull)
    hull = convex_hull_2d(locked_points)
    locked_support = min(Vector(point).dot(axis) for point in hull)
    if abs(float(locked_support - root_support)) > tolerance:
        raise ValueError("Plan root support drifted while applying the XML root lock.")
    if not point_in_convex_polygon(attachment, hull, tolerance=tolerance):
        raise ValueError("Root-locked plan no longer contains the XML attachment origin.")
    return hull, {
        "policy": policy,
        "root_axis_xy": [float(axis.x), float(axis.y)],
        "unexpanded_root_support": float(coverage_root_support),
        "unexpanded_distal_support": float(distal_support),
        "locked_root_support": float(locked_support),
        "maximum_root_margin_trim": float(maximum_trim),
        "attachment_xy": [0.0, 0.0],
        "attachment_inside_unexpanded_projection": attachment_inside,
        "attachment_forward_ray_entry": bridge_entry,
        "attachment_forward_ray_exit": bridge_exit,
        "attachment_gap_ratio": bridge_gap_ratio,
        "maximum_attachment_gap_ratio": ROOT_BRIDGE_MAX_GAP_RATIO,
        "tolerance": float(tolerance),
    }


def point_in_convex_polygon(point, polygon, tolerance=1.0e-7):
    sign = 0
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        cross = (
            (second[0] - first[0]) * (point[1] - first[1])
            - (second[1] - first[1]) * (point[0] - first[0])
        )
        if abs(cross) <= tolerance:
            continue
        current = 1 if cross > 0.0 else -1
        if sign and current != sign:
            return False
        sign = current
    return True


def projection_coverage_2d(points, polygon):
    projected = [tuple(float(value) for value in point[:2]) for point in points]
    boundary = [tuple(float(value) for value in point[:2]) for point in polygon]
    if not projected or len(boundary) < 3:
        raise ValueError("Projection coverage needs source points and a polygon boundary.")
    spans = [
        max(point[axis] for point in boundary)
        - min(point[axis] for point in boundary)
        for axis in range(2)
    ]
    tolerance = max(max(spans), 1.0e-6) ** 2 * 1.0e-7
    outside_indices = [
        index
        for index, point in enumerate(projected)
        if not point_in_convex_polygon(point, boundary, tolerance=tolerance)
    ]
    return {
        "covers_projection": not outside_indices,
        "projected_point_count": len(projected),
        "outside_point_count": len(outside_indices),
        "outside_point_indices": outside_indices,
        "boundary_vertex_count": len(boundary),
        "cross_product_tolerance": float(tolerance),
    }


def _is_generated(data_block):
    return bool(
        data_block.get(GENERATED_FLAG)
        or data_block.get(LEGACY_GENERATED_FLAG)
    )


def _stage_named_generated_objects(names):
    existing = [bpy.data.objects.get(name) for name in names]
    existing = [obj for obj in existing if obj is not None]
    unmanaged = [obj.name for obj in existing if not _is_generated(obj)]
    if unmanaged:
        raise ValueError(
            "Refusing to replace non-generated objects: " + ", ".join(sorted(unmanaged))
        )
    token = uuid.uuid4().hex
    backups = []
    for index, obj in enumerate(existing):
        original_name = obj.name
        obj.name = f"__STCLUSTER_BACKUP_{token}_{index:03d}"
        backups.append((obj, original_name))
    return backups


def _delete_object_and_orphan_data(obj):
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Armature):
            bpy.data.armatures.remove(data)


def _commit_staged_objects(backups):
    for obj, _original_name in backups:
        if obj.name in bpy.data.objects:
            _delete_object_and_orphan_data(obj)


def _restore_staged_objects(backups):
    for obj, original_name in backups:
        if obj.name in bpy.data.objects:
            obj.name = original_name


def _cleanup_journal(journal):
    for obj in reversed(journal["objects"]):
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in reversed(journal["meshes"]):
        if mesh.name in bpy.data.meshes and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for armature in reversed(journal["armatures"]):
        if armature.name in bpy.data.armatures and armature.users == 0:
            bpy.data.armatures.remove(armature)
    for material in reversed(journal["materials"]):
        if material.name in bpy.data.materials and material.users == 0:
            bpy.data.materials.remove(material)
    for collection in reversed(journal["collections"]):
        if collection.name in bpy.data.collections and not collection.objects and not collection.children:
            bpy.data.collections.remove(collection)


def _isolate_existing_export_objects(export_collection, reference_collection, journal):
    moved = []
    for obj in list(export_collection.objects):
        linked_reference = obj.name not in reference_collection.objects
        if linked_reference:
            reference_collection.objects.link(obj)
        export_collection.objects.unlink(obj)
        item = {
            "object": obj,
            "original_name": obj.name,
            "export_collection": export_collection,
            "reference_collection": reference_collection,
            "linked_reference": linked_reference,
        }
        journal["collection_moves"].append(item)
        moved.append(obj.name)
    return moved


def _hide_source_reference_collection(reference_collection):
    """Keep rebuild evidence in the blend without exposing it as delivery data."""
    reference_collection.hide_viewport = True
    reference_collection.hide_render = True
    bpy.context.view_layer.update()
    for obj in tuple(reference_collection.all_objects):
        if obj is None:
            continue
        obj.hide_render = True
        try:
            obj.hide_set(True)
        except RuntimeError:
            pass


def _validate_single_bone_export_contract(export_collection, built_prototypes):
    expected_armatures = {
        row["armature"].name for row in built_prototypes.values()
    }
    exported_armatures = {
        obj.name for obj in export_collection.all_objects
        if obj.type == "ARMATURE"
        and not obj.name.startswith("__STCLUSTER_BACKUP_")
    }
    if exported_armatures != expected_armatures:
        raise ValueError(
            "Normalized Export contains an unexpected armature set: "
            f"expected {sorted(expected_armatures)}, got "
            f"{sorted(exported_armatures)}."
        )

    rows = []
    for row in built_prototypes.values():
        armature = row["armature"]
        part = row["part"]
        bone_names = [bone.name for bone in armature.data.bones]
        group_names = [group.name for group in part.vertex_groups]
        if bone_names != ["part_root"]:
            raise ValueError(
                f"Normalized export armature '{armature.name}' must contain "
                f"exactly one part_root bone; got {bone_names}."
            )
        if group_names != ["part_root"]:
            raise ValueError(
                f"Normalized export mesh '{part.name}' must contain exactly "
                f"one part_root vertex group; got {group_names}."
            )
        rows.append(
            {
                "armature": armature.name,
                "mesh": part.name,
                "bone_count": 1,
                "bone": "part_root",
                "vertex_groups": ["part_root"],
            }
        )
    return rows


def _restore_collection_moves(journal):
    for item in reversed(journal.get("collection_moves", [])):
        obj = item["object"]
        if obj.name not in bpy.data.objects:
            continue
        export_collection = item["export_collection"]
        reference_collection = item["reference_collection"]
        if obj.name not in export_collection.objects:
            export_collection.objects.link(obj)
        if item["linked_reference"] and obj.name in reference_collection.objects:
            reference_collection.objects.unlink(obj)


def _ensure_collection(scene, name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
        return collection, True
    scene_collections = {child.name for child in scene.collection.children_recursive}
    if collection.name not in scene_collections:
        scene.collection.children.link(collection)
    return collection, False


def _tag(obj, source, bone_name, endpoint_name, role, index, counterpart):
    obj[GENERATED_FLAG] = True
    obj[SOURCE_OBJECT_KEY] = source.name
    obj[SOURCE_BONE_KEY] = bone_name
    obj[ENDPOINT_BONE_KEY] = endpoint_name or ""
    obj[ASSET_ROLE_KEY] = role
    obj[VARIANT_INDEX_KEY] = int(index)
    obj[COUNTERPART_KEY] = counterpart


def _compact_material_slots(mesh):
    used_indices = sorted({polygon.material_index for polygon in mesh.polygons})
    if not used_indices:
        mesh.materials.clear()
        return
    original = list(mesh.materials)
    original_polygon_indices = [polygon.material_index for polygon in mesh.polygons]
    remap = {old: new for new, old in enumerate(used_indices)}
    mesh.materials.clear()
    for index in used_indices:
        if index >= len(original) or original[index] is None:
            raise ValueError(f"Used material slot {index} is empty or missing.")
        mesh.materials.append(original[index])
    for polygon, original_index in zip(mesh.polygons, original_polygon_indices):
        polygon.material_index = remap[original_index]


def _split_source_mesh(source, face_indices, transform, mesh_name, journal):
    determinant = transform.to_3x3().determinant()
    if determinant <= 0.0:
        raise ValueError(
            "Mirrored or singular source transforms are not supported; apply a positive transform first."
        )
    mesh = source.data.copy()
    journal["meshes"].append(mesh)
    mesh.name = mesh_name
    keep = set(face_indices)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        delete_faces = [face for face in bm.faces if face.index not in keep]
        if delete_faces:
            bmesh.ops.delete(bm, geom=delete_faces, context="FACES")
        orphan_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
        if orphan_vertices:
            bmesh.ops.delete(bm, geom=orphan_vertices, context="VERTS")
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.transform(transform, shape_keys=True)
    _compact_material_slots(mesh)
    mesh.update()
    if not mesh.polygons:
        raise ValueError(f"Split produced no faces: {mesh_name}")
    return mesh


def _build_part_hierarchy(
    source,
    export_collection,
    asset_name,
    bone_name,
    endpoint_name,
    face_indices,
    frame,
    index,
    plan_name,
    journal,
):
    mesh_name = asset_name + "_MeshData"
    transform = frame.get("source_to_normalized_matrix")
    if transform is None:
        transform = frame["matrix_world"].inverted_safe() @ source.matrix_world
    else:
        transform = transform.copy()
    mesh = _split_source_mesh(source, face_indices, transform, mesh_name, journal)
    part = bpy.data.objects.new(asset_name + "_Mesh", mesh)
    journal["objects"].append(part)
    while part.vertex_groups:
        part.vertex_groups.remove(part.vertex_groups[0])
    part.matrix_world = Matrix.Identity(4)
    part.hide_viewport = False
    part.hide_render = False
    part.hide_set(False)
    export_collection.objects.link(part)

    armature_data = bpy.data.armatures.new(asset_name + "_ArmatureData")
    journal["armatures"].append(armature_data)
    armature = bpy.data.objects.new(asset_name + "_Armature", armature_data)
    journal["objects"].append(armature)
    export_collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root = armature_data.edit_bones.new("part_root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, frame["endpoint_length"], 0.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    pivot = bpy.data.objects.new(asset_name, None)
    journal["objects"].append(pivot)
    pivot.empty_display_type = "ARROWS"
    normalized_size = frame["normalized_bounds"]["size"]
    display_size = max(normalized_size) * 0.05
    if display_size <= 0.0:
        raise ValueError(f"Normalized part has no measurable size: {asset_name}")
    pivot.empty_display_size = display_size
    export_collection.objects.link(pivot)
    pivot.matrix_world = Matrix.Identity(4)
    armature.parent = pivot
    armature.matrix_parent_inverse = Matrix.Identity(4)
    armature.matrix_basis = Matrix.Identity(4)
    part.parent = armature
    part.matrix_parent_inverse = Matrix.Identity(4)
    part.matrix_basis = Matrix.Identity(4)
    modifier = part.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    group = part.vertex_groups.new(name="part_root")
    group.add(list(range(len(mesh.vertices))), 1.0, "REPLACE")

    _tag(pivot, source, bone_name, endpoint_name, "send2ue_pivot", index, plan_name)
    _tag(armature, source, bone_name, endpoint_name, "skeletal_armature", index, plan_name)
    _tag(part, source, bone_name, endpoint_name, "skeletal_mesh", index, plan_name)
    mesh[GENERATED_FLAG] = True
    armature_data[GENERATED_FLAG] = True
    part["speedtree_cluster_frame_world"] = json.dumps(_matrix_rows(frame["matrix_world"]))
    part["speedtree_cluster_frame_policy"] = str(
        frame.get("orientation_policy") or ""
    )
    if frame.get("source_to_normalized_matrix") is not None:
        part["speedtree_cluster_source_to_normalized_matrix"] = json.dumps(
            _matrix_rows(frame["source_to_normalized_matrix"])
        )
        part["speedtree_cluster_physical_fit_scale"] = float(
            frame["physical_fit_scale"]
        )
        part["speedtree_cluster_capture_attachment"] = json.dumps(
            frame["capture_attachment"],
            ensure_ascii=False,
            sort_keys=True,
        )
        part["speedtree_cluster_attachment_tangent_projection"] = json.dumps(
            frame["attachment_tangent_projection"],
            ensure_ascii=False,
            sort_keys=True,
        )
    if frame.get("source_frame_world") is not None:
        part["speedtree_cluster_source_frame_world"] = json.dumps(
            _matrix_rows(frame["source_frame_world"])
        )
    return pivot, armature, part


def _plan_material(name, journal):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        journal["materials"].append(material)
    return material


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_boundary_indices(faces):
    edge_counts = Counter()
    for face in faces:
        for offset, first in enumerate(face):
            second = face[(offset + 1) % len(face)]
            edge_counts[tuple(sorted((int(first), int(second))))] += 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency = defaultdict(list)
    for first, second in boundary_edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Camera reference does not have one closed manifold boundary.")
    start = min(adjacency)
    loop = [start]
    previous = None
    current = start
    while True:
        candidates = [value for value in adjacency[current] if value != previous]
        if not candidates:
            raise ValueError("Camera reference boundary traversal stopped early.")
        following = min(candidates) if previous is None else candidates[0]
        if following == start:
            break
        if following in loop:
            raise ValueError("Camera reference boundary is self-connected or ambiguous.")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacency):
        raise ValueError("Camera reference contains multiple boundary loops.")
    return loop


def _closed_loop_parameters(points):
    points = np.asarray(points, dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    total = float(lengths.sum())
    if not np.isfinite(total) or total <= 1.0e-12:
        raise ValueError("Closed boundary has no measurable perimeter.")
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / total, lengths / total


def _resample_closed_loop(points, count):
    points = np.asarray(points, dtype=np.float64)
    parameters, fractions = _closed_loop_parameters(points)
    extended_parameters = np.concatenate((parameters, [1.0]))
    extended_points = np.vstack((points, points[0]))
    samples = np.arange(int(count), dtype=np.float64) / float(count)
    result = np.empty((int(count), points.shape[1]), dtype=np.float64)
    for axis in range(points.shape[1]):
        result[:, axis] = np.interp(
            samples,
            extended_parameters,
            extended_points[:, axis],
        )
    return result, parameters, fractions


def _similarity_fit(source, target, *, source_attachment=None, target_attachment=None):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source_attachment is None or target_attachment is None:
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
    else:
        source_center = np.asarray(source_attachment, dtype=np.float64)
        target_center = np.asarray(target_attachment, dtype=np.float64)
        if (
            source_center.shape != (2,)
            or target_center.shape != (2,)
            or not np.isfinite(source_center).all()
            or not np.isfinite(target_center).all()
        ):
            raise ValueError("Similarity attachment constraints are malformed.")
    centered_source = source - source_center
    centered_target = target - target_center
    denominator = float(np.sum(centered_source * centered_source))
    if denominator <= 1.0e-20:
        raise ValueError("Plan boundary is degenerate for camera-template alignment.")
    left, singular, right = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.identity(2, dtype=np.float64)
    if np.linalg.det(left @ right) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right
    scale = float(np.sum(singular * np.diag(correction)) / denominator)
    if not math.isfinite(scale) or scale <= 1.0e-20:
        raise ValueError("Camera-template alignment requires a mirrored transform.")
    translation = target_center - scale * (source_center @ rotation)
    fitted = scale * (source @ rotation) + translation
    residual = fitted - target
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    radius_points = target - target.mean(axis=0)
    target_radius = float(np.sqrt(np.mean(np.sum(radius_points * radius_points, axis=1))))
    if target_radius <= 1.0e-20:
        raise ValueError("Camera reference boundary is degenerate.")
    return {
        "rotation": rotation,
        "scale": scale,
        "translation": translation,
        "rms": rms,
        "normalized_rms": rms / target_radius,
        "determinant": float(np.linalg.det(rotation)),
        "orientation_preserving": True,
    }


def _closed_interp(values, source_parameters, query_parameters):
    values = np.asarray(values, dtype=np.float64)
    source_parameters = np.asarray(source_parameters, dtype=np.float64)
    query_parameters = np.mod(np.asarray(query_parameters, dtype=np.float64), 1.0)
    extended_parameters = np.concatenate((source_parameters, [1.0]))
    extended_values = np.vstack((values, values[0]))
    result = np.empty((len(query_parameters), values.shape[1]), dtype=np.float64)
    for axis in range(values.shape[1]):
        result[:, axis] = np.interp(
            query_parameters,
            extended_parameters,
            extended_values[:, axis],
        )
    return result


def _reference_plane_coordinates(reference_plane, camera):
    vertices = np.asarray(reference_plane["vertices"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Camera reference vertices are not a 3D payload.")
    if right.shape != (3,) or up.shape != (3,):
        raise ValueError("Camera right/up basis is not three-dimensional.")
    return np.column_stack((vertices @ right, vertices @ up))


def _transfer_camera_boundary_uvs(
    plan_boundary,
    reference_plane,
    camera,
    *,
    plan_attachment_xy=(0.0, 0.0),
):
    plan_boundary = np.asarray(plan_boundary, dtype=np.float64)
    reference_vertices = _reference_plane_coordinates(reference_plane, camera)
    reference_uvs = np.asarray(reference_plane["uvs"], dtype=np.float64)
    plan_attachment = np.asarray(plan_attachment_xy, dtype=np.float64)
    reference_attachment = np.asarray(
        (reference_plane.get("attachment") or {}).get("source_plane_xy"),
        dtype=np.float64,
    )
    reference_pivot_uv = np.asarray(
        (reference_plane.get("attachment") or {}).get("pivot_uv"),
        dtype=np.float64,
    )
    if not (
        np.isfinite(plan_boundary).all()
        and np.isfinite(reference_vertices).all()
        and np.isfinite(reference_uvs).all()
        and plan_attachment.shape == (2,)
        and np.isfinite(plan_attachment).all()
        and reference_attachment.shape == (2,)
        and np.isfinite(reference_attachment).all()
        and reference_pivot_uv.shape == (2,)
        and np.isfinite(reference_pivot_uv).all()
    ):
        raise ValueError(
            "Camera-template transfer requires finite plan/reference attachment coordinates."
        )
    reference_extent_diagonal = float(
        np.linalg.norm(np.ptp(reference_vertices, axis=0))
    )
    if (
        not math.isfinite(reference_extent_diagonal)
        or reference_extent_diagonal <= 1.0e-12
    ):
        raise ValueError("Camera reference has no measurable relative extent.")
    boundary_indices = _ordered_boundary_indices(reference_plane["faces"])
    reference_boundary = reference_vertices[boundary_indices]
    reference_boundary_uvs = reference_uvs[boundary_indices]
    sample_count = 512
    plan_samples, plan_parameters, _plan_fractions = _resample_closed_loop(
        plan_boundary,
        sample_count,
    )
    candidates = []
    for reversed_boundary in (False, True):
        if reversed_boundary:
            candidate_boundary = reference_boundary[::-1]
            candidate_uvs = reference_boundary_uvs[::-1]
        else:
            candidate_boundary = reference_boundary
            candidate_uvs = reference_boundary_uvs
        reference_samples, reference_parameters, _reference_fractions = _resample_closed_loop(
            candidate_boundary,
            sample_count,
        )
        for shift in range(sample_count):
            shifted = np.roll(reference_samples, shift, axis=0)
            try:
                fit = _similarity_fit(
                    plan_samples,
                    shifted,
                    source_attachment=plan_attachment,
                    target_attachment=reference_attachment,
                )
            except ValueError:
                continue
            mapped_attachment = (
                fit["scale"] * (plan_attachment @ fit["rotation"])
                + fit["translation"]
            )
            attachment_error = float(
                np.linalg.norm(mapped_attachment - reference_attachment)
            )
            attachment_error_normalized = (
                attachment_error / reference_extent_diagonal
            )
            candidates.append({
                "fit": fit,
                "shift": shift,
                "reversed": reversed_boundary,
                "reference_boundary": candidate_boundary,
                "reference_uvs": candidate_uvs,
                "reference_parameters": reference_parameters,
                "mapped_attachment": mapped_attachment,
                "attachment_error": attachment_error,
                "attachment_error_normalized": attachment_error_normalized,
            })
    finite_candidates = [
        candidate
        for candidate in candidates
        if (
            np.isfinite(candidate["fit"]["normalized_rms"])
            and math.isfinite(candidate["attachment_error_normalized"])
        )
    ]
    if not finite_candidates:
        raise ValueError("Camera-template alignment did not produce a finite solution.")
    attachment_tolerance = reference_extent_diagonal * 1.0e-12
    anchor_candidates = [
        candidate
        for candidate in finite_candidates
        if candidate["attachment_error"] <= attachment_tolerance
    ]
    if not anchor_candidates:
        nearest = min(
            finite_candidates,
            key=lambda candidate: (
                candidate["attachment_error_normalized"],
                candidate["fit"]["normalized_rms"],
                candidate["reversed"],
                candidate["shift"],
            ),
        )
        raise ValueError(
            "Camera-template attachment constraint was not exact: origin error "
            f"{nearest['attachment_error']:.12g}."
        )
    best = min(
        anchor_candidates,
        key=lambda candidate: (
            candidate["fit"]["normalized_rms"],
            candidate["attachment_error_normalized"],
            candidate["reversed"],
            candidate["shift"],
        ),
    )
    if best["fit"]["normalized_rms"] > UV_TRANSFER_MAX_NORMALIZED_RMS:
        raise ValueError(
            "Camera-template boundary alignment is ambiguous: normalized RMS "
            f"{best['fit']['normalized_rms']:.6f} exceeds "
            f"{UV_TRANSFER_MAX_NORMALIZED_RMS:.6f}."
        )
    phase = float(best["shift"]) / float(sample_count)
    reference_query = np.mod(plan_parameters - phase, 1.0)
    transferred_uvs = _closed_interp(
        best["reference_uvs"],
        best["reference_parameters"],
        reference_query,
    )
    if not np.isfinite(transferred_uvs).all():
        raise ValueError("Camera-template UV transfer produced a non-finite value.")
    fit = best["fit"]
    details = {
        "policy": "closed_loop_similarity_to_exact_camera_reference_boundary",
        "sample_count": sample_count,
        "cyclic_shift": int(best["shift"]),
        "phase": phase,
        "reference_boundary_reversed": bool(best["reversed"]),
        "reference_boundary_indices": [int(value) for value in boundary_indices],
        "scale": float(fit["scale"]),
        "rotation": [[float(value) for value in row] for row in fit["rotation"]],
        "translation": [float(value) for value in fit["translation"]],
        "determinant": float(fit["determinant"]),
        "orientation_preserving": bool(fit["orientation_preserving"]),
        "rms": float(fit["rms"]),
        "normalized_rms": float(fit["normalized_rms"]),
        "max_normalized_rms": UV_TRANSFER_MAX_NORMALIZED_RMS,
        "candidate_selection_policy": UV_TRANSFER_CANDIDATE_SELECTION_POLICY,
        "attachment_policy": UV_TRANSFER_ATTACHMENT_POLICY,
        "plan_attachment_xy": [float(value) for value in plan_attachment],
        "reference_attachment_xy": [
            float(value) for value in reference_attachment
        ],
        "reference_pivot_uv": [float(value) for value in reference_pivot_uv],
        "mapped_attachment_xy": [
            float(value) for value in best["mapped_attachment"]
        ],
        "attachment_origin_error": float(best["attachment_error"]),
        "attachment_origin_error_normalized": float(
            best["attachment_error_normalized"]
        ),
        "max_attachment_origin_error_normalized": (
            UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR
        ),
        "reference_extent_diagonal": reference_extent_diagonal,
    }
    return transferred_uvs, details


def _signed_area_2d(points):
    return 0.5 * sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )


def _strictly_inside_triangle(point, first, second, third, orientation, epsilon):
    crosses = []
    for start, end in ((first, second), (second, third), (third, first)):
        crosses.append(
            orientation
            * ((end[0] - start[0]) * (point[1] - start[1])
               - (end[1] - start[1]) * (point[0] - start[0]))
        )
    return all(value > epsilon for value in crosses)


def _triangulate_uv_boundary(uvs):
    points = [tuple(float(value) for value in row) for row in uvs]
    area = _signed_area_2d(points)
    if abs(area) <= 1.0e-14:
        raise ValueError("Transferred camera UV boundary has zero signed area.")
    orientation = 1.0 if area > 0.0 else -1.0
    remaining = list(range(len(points)))
    triangles = []
    epsilon = max(
        max(max(row) for row in points) - min(min(row) for row in points),
        1.0,
    ) ** 2 * 1.0e-12
    while len(remaining) > 3:
        ear = None
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            first, second, third = points[previous], points[current], points[following]
            cross = orientation * (
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
            if cross <= epsilon:
                continue
            if any(
                _strictly_inside_triangle(
                    points[index], first, second, third, orientation, epsilon
                )
                for index in remaining
                if index not in {previous, current, following}
            ):
                continue
            ear = (offset, (previous, current, following))
            break
        if ear is None:
            raise ValueError("Transferred camera UV boundary is not a simple polygon.")
        offset, triangle = ear
        triangles.append(triangle)
        remaining.pop(offset)
    triangles.append(tuple(remaining))
    return triangles


def _point_segment_parameter(point, first, second):
    direction = second - first
    length_squared = float(direction.length_squared)
    if length_squared <= 1.0e-20:
        return 0.0, float((point - first).length)
    parameter = max(0.0, min(1.0, float((point - first).dot(direction) / length_squared)))
    closest = first + direction * parameter
    return parameter, float((point - closest).length)


def _mean_value_boundary_uv(point, boundary, boundary_uvs, tolerance):
    point = Vector(point)
    polygon = [Vector(value) for value in boundary]
    for index, vertex in enumerate(polygon):
        if (point - vertex).length <= tolerance:
            return tuple(float(value) for value in boundary_uvs[index])
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        parameter, distance = _point_segment_parameter(point, first, second)
        if distance <= tolerance:
            first_uv = Vector(boundary_uvs[index])
            second_uv = Vector(boundary_uvs[(index + 1) % len(polygon)])
            uv = first_uv.lerp(second_uv, parameter)
            return tuple(float(value) for value in uv)

    vectors = [vertex - point for vertex in polygon]
    distances = [float(value.length) for value in vectors]
    weights = []
    for index, vector in enumerate(vectors):
        previous = vectors[index - 1]
        following = vectors[(index + 1) % len(vectors)]
        previous_angle = math.atan2(
            abs(float(previous.cross(vector))), float(previous.dot(vector))
        )
        following_angle = math.atan2(
            abs(float(vector.cross(following))), float(vector.dot(following))
        )
        weights.append(
            (math.tan(previous_angle * 0.5) + math.tan(following_angle * 0.5))
            / distances[index]
        )
    total = sum(weights)
    if not math.isfinite(total) or total <= 1.0e-20:
        raise ValueError("Cannot interpolate an interior plan UV from its boundary.")
    uv = Vector((0.0, 0.0))
    for weight, value in zip(weights, boundary_uvs):
        uv += Vector(value) * (weight / total)
    return tuple(float(value) for value in uv)


def _uniform_plan_triangulation(
    boundary,
    boundary_uvs,
    refinement_levels,
    *,
    attachment_point=(0.0, 0.0),
    attachment_uv=None,
    containment_tolerance=None,
):
    """Build a constrained, near-uniform interior instead of an ear-clipped fan."""
    levels = int(refinement_levels)
    if levels < 0 or levels > 2:
        raise ValueError("Plan refinement levels must be between 0 and 2.")
    points = [tuple(float(value) for value in row) for row in boundary]
    uvs = [tuple(float(value) for value in row) for row in boundary_uvs]
    if _signed_area_2d(points) < 0.0:
        points.reverse()
        uvs.reverse()
    area = abs(_signed_area_2d(points))
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    diagonal = math.hypot(maximum_x - minimum_x, maximum_y - minimum_y)
    if area <= 1.0e-20 or diagonal <= 1.0e-20:
        raise ValueError("Plan boundary has no measurable area.")
    attachment_point = tuple(float(value) for value in attachment_point)
    if len(attachment_point) != 2 or not all(
        math.isfinite(value) for value in attachment_point
    ):
        raise ValueError("Plan attachment point is malformed.")
    if attachment_uv is None:
        raise ValueError("Plan attachment UV is required.")
    attachment_uv = tuple(float(value) for value in attachment_uv)
    if len(attachment_uv) != 2 or not all(
        math.isfinite(value) for value in attachment_uv
    ):
        raise ValueError("Plan attachment UV is malformed.")
    if containment_tolerance is None:
        containment_tolerance = max(diagonal * 1.0e-7, 1.0e-9)
    try:
        containment_tolerance = float(containment_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("Plan attachment containment tolerance is malformed.") from exc
    if (
        not math.isfinite(containment_tolerance)
        or containment_tolerance <= 0.0
    ):
        raise ValueError("Plan attachment containment tolerance must be positive.")
    boundary_points = list(points)
    if not point_in_convex_polygon(
        attachment_point,
        boundary_points,
        tolerance=containment_tolerance,
    ):
        raise ValueError("Plan boundary does not contain its normalized attachment origin.")

    # If the attachment is only accepted by the shared scale-aware tolerance,
    # make it an explicit constrained boundary vertex.  Passing it merely as a
    # free CDT point can leave it unreferenced by every face; Blender then has
    # no loop UV for the vertex and FBX legitimately removes it.
    if not point_in_convex_polygon(
        attachment_point,
        boundary_points,
        tolerance=0.0,
    ):
        attachment_vector = Vector(attachment_point)
        nearest_edge = min(
            (
                _point_segment_parameter(
                    attachment_vector,
                    Vector(boundary_points[index]),
                    Vector(boundary_points[(index + 1) % len(boundary_points)]),
                )[1],
                index,
            )
            for index in range(len(boundary_points))
        )
        if nearest_edge[0] > containment_tolerance:
            raise ValueError(
                "Plan attachment is outside the boundary beyond its "
                "containment tolerance."
            )
        insert_at = nearest_edge[1] + 1
        boundary_points.insert(insert_at, attachment_point)
        uvs.insert(insert_at, attachment_uv)
        points = list(boundary_points)

    if levels > 0:
        target_triangles = min(
            128,
            max(24, len(points) * 2) * (2 ** (levels - 1)),
        )
        coordinates = np.asarray(boundary_points, dtype=np.float64)
        center = coordinates.mean(axis=0)
        covariance = np.cov(coordinates - center, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor = np.array((-major[1], major[0]), dtype=np.float64)
        projected_major = (coordinates - center) @ major
        projected_minor = (coordinates - center) @ minor
        major_min = float(projected_major.min())
        major_max = float(projected_major.max())
        minor_min = float(projected_minor.min())
        minor_max = float(projected_minor.max())
        major_span = max(major_max - major_min, diagonal * 1.0e-6)
        minor_span = max(minor_max - minor_min, diagonal * 1.0e-6)
        # For a triangulated disk F = 2V - B - 2.  Account for the existing
        # boundary vertices so the requested density is not effectively
        # doubled on detailed outlines.
        target_interior = max(
            4,
            int(round((target_triangles - len(boundary_points) + 2) * 0.5)),
        )
        major_steps = max(
            1,
            int(round(math.sqrt(target_interior * major_span / minor_span))),
        )
        minor_steps = max(
            1,
            int(round(target_interior / major_steps)),
        )
        major_steps = min(32, major_steps)
        minor_steps = min(16, minor_steps)
        major_spacing = major_span / float(major_steps + 1)
        minor_spacing = minor_span / float(minor_steps + 1)
        edge_clearance = min(major_spacing, minor_spacing) * 0.18
        for major_index in range(1, major_steps + 1):
            major_value = major_min + major_spacing * major_index
            minor_offset = 0.25 * minor_spacing if major_index % 2 else 0.0
            for minor_index in range(1, minor_steps + 1):
                minor_value = minor_min + minor_spacing * minor_index + minor_offset
                if minor_value >= minor_max:
                    continue
                candidate_array = center + major * major_value + minor * minor_value
                candidate = (float(candidate_array[0]), float(candidate_array[1]))
                if not point_in_convex_polygon(
                    candidate,
                    boundary_points,
                    tolerance=diagonal * diagonal * 1.0e-12,
                ):
                    continue
                point_vector = Vector(candidate)
                minimum_edge_distance = min(
                    _point_segment_parameter(
                        point_vector,
                        Vector(boundary_points[index]),
                        Vector(boundary_points[(index + 1) % len(boundary_points)]),
                    )[1]
                    for index in range(len(boundary_points))
                )
                if minimum_edge_distance >= edge_clearance:
                    points.append(candidate)
        if len(points) == len(boundary_points):
            centroid = tuple(float(value) for value in center)
            if point_in_convex_polygon(centroid, boundary_points):
                points.append(centroid)

    input_attachment_index = next(
        (
            index
            for index, point in enumerate(points)
            if math.dist(point, attachment_point) <= diagonal * 1.0e-10
        ),
        None,
    )
    if input_attachment_index is None:
        points.append(attachment_point)

    boundary_count = len(boundary_points)
    edges = [
        (index, (index + 1) % boundary_count)
        for index in range(boundary_count)
    ]
    cdt_vertices, _cdt_edges, cdt_faces, _orig_v, _orig_e, _orig_f = delaunay_2d_cdt(
        [Vector(value) for value in points],
        edges,
        [tuple(range(boundary_count))],
        1,
        diagonal * 1.0e-10,
        False,
    )
    faces = [tuple(int(value) for value in face) for face in cdt_faces]
    if not faces or any(len(face) != 3 for face in faces):
        raise ValueError("Constrained plan triangulation did not return triangles.")
    tolerance = diagonal * 1.0e-8
    result_uvs = [
        _mean_value_boundary_uv(vertex, boundary_points, uvs, tolerance)
        for vertex in cdt_vertices
    ]
    attachment_vertex_index = min(
        range(len(cdt_vertices)),
        key=lambda index: math.dist(tuple(cdt_vertices[index]), attachment_point),
    )
    if math.dist(tuple(cdt_vertices[attachment_vertex_index]), attachment_point) > tolerance:
        raise ValueError("CDT did not preserve the normalized attachment origin.")
    if not any(
        attachment_vertex_index in face
        for face in faces
    ):
        raise ValueError("CDT left the normalized attachment origin unreferenced.")
    cdt_vertices[attachment_vertex_index] = Vector(attachment_point)
    result_uvs[attachment_vertex_index] = attachment_uv
    return (
        [tuple(float(value) for value in vertex) for vertex in cdt_vertices],
        result_uvs,
        faces,
        int(attachment_vertex_index),
    )


def _refine_plan_triangles(vertices, uvs, faces, levels):
    levels = int(levels)
    if levels < 0 or levels > 2:
        raise ValueError("Plan refinement levels must be between 0 and 2.")
    refined_vertices = [tuple(float(value) for value in row) for row in vertices]
    refined_uvs = [tuple(float(value) for value in row) for row in uvs]
    refined_faces = [tuple(int(value) for value in row) for row in faces]
    for _level in range(levels):
        edge_midpoints = {}

        def midpoint(first, second):
            key = tuple(sorted((int(first), int(second))))
            existing = edge_midpoints.get(key)
            if existing is not None:
                return existing
            position = tuple(
                (refined_vertices[first][axis] + refined_vertices[second][axis]) * 0.5
                for axis in range(3)
            )
            uv = tuple(
                (refined_uvs[first][axis] + refined_uvs[second][axis]) * 0.5
                for axis in range(2)
            )
            index = len(refined_vertices)
            refined_vertices.append(position)
            refined_uvs.append(uv)
            edge_midpoints[key] = index
            return index

        next_faces = []
        for first, second, third in refined_faces:
            first_second = midpoint(first, second)
            second_third = midpoint(second, third)
            third_first = midpoint(third, first)
            next_faces.extend(
                (
                    (first, first_second, third_first),
                    (first_second, second, second_third),
                    (third_first, second_third, third),
                    (first_second, second_third, third_first),
                )
            )
        refined_faces = next_faces
    return refined_vertices, refined_uvs, refined_faces


def _validate_reference_object(obj, plane):
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Camera reference object is missing or not a mesh: {plane['name']}")
    if not _identity_matrix(obj.matrix_world):
        raise ValueError(f"Camera reference transform is not identity: {plane['name']}")
    mesh = obj.data
    if len(mesh.vertices) != len(plane["vertices"]) or len(mesh.polygons) != len(plane["faces"]):
        raise ValueError(f"Camera reference topology count mismatch: {plane['name']}")
    maximum_vertex_error = max(
        (Vector(vertex.co) - Vector(expected)).length
        for vertex, expected in zip(mesh.vertices, plane["vertices"])
    )
    if maximum_vertex_error > 1.0e-6:
        raise ValueError(f"Camera reference vertex payload mismatch: {plane['name']}")
    actual_faces = sorted(tuple(sorted(int(value) for value in polygon.vertices)) for polygon in mesh.polygons)
    expected_faces = sorted(tuple(sorted(int(value) for value in face)) for face in plane["faces"])
    if actual_faces != expected_faces:
        raise ValueError(f"Camera reference face payload mismatch: {plane['name']}")
    uv_layer = mesh.uv_layers.get("UVMap")
    if uv_layer is None:
        raise ValueError(f"Camera reference UVMap is missing: {plane['name']}")
    actual_uvs = [None] * len(mesh.vertices)
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv = tuple(float(value) for value in uv_layer.data[loop_index].uv)
            previous = actual_uvs[vertex_index]
            if previous is not None and max(abs(uv[axis] - previous[axis]) for axis in range(2)) > 1.0e-7:
                raise ValueError(f"Camera reference contains an unexpected UV seam: {plane['name']}")
            actual_uvs[vertex_index] = uv
    if any(value is None for value in actual_uvs):
        raise ValueError(f"Camera reference has an unreferenced UV vertex: {plane['name']}")
    maximum_uv_error = max(
        max(abs(actual[axis] - float(expected[axis])) for axis in range(2))
        for actual, expected in zip(actual_uvs, plane["uvs"])
    )
    if maximum_uv_error > 1.0e-6:
        raise ValueError(f"Camera reference UV payload mismatch: {plane['name']}")
    return {
        "vertex_max_error": float(maximum_vertex_error),
        "uv_max_error": float(maximum_uv_error),
        "topology_sha256": plane["topology_sha256"],
        "uv_sha256": plane["uv_sha256"],
    }


def _load_camera_reference_objects(
    scene,
    source,
    variants,
    material_name,
    uv_bundle,
    collection,
    journal,
):
    contract = uv_bundle["contract"]
    planes = contract.get("planes") or []
    if len(planes) != len(variants):
        raise ValueError(
            f"Camera reference plane count {len(planes)} does not match normalized variants {len(variants)}."
        )
    for plane, variant in zip(planes, variants):
        if plane.get("name") != variant["plan_name"]:
            raise ValueError(
                f"Camera reference order/name mismatch: {plane.get('name')} vs {variant['plan_name']}"
            )
    requested_names = [plane["name"] for plane in planes]
    existing_materials = {material.name for material in bpy.data.materials}
    with bpy.data.libraries.load(uv_bundle["reference_blend"], link=False) as (source_data, target_data):
        missing = [name for name in requested_names if name not in source_data.objects]
        if missing:
            raise ValueError(
                "Camera reference blend is missing exact plane objects: " + ", ".join(missing)
            )
        target_data.objects = requested_names
    loaded = []
    material = _plan_material(material_name, journal)
    for plane, variant, obj in zip(planes, variants, target_data.objects):
        validation = _validate_reference_object(obj, plane)
        reference_name = "AtlasCameraRef_" + plane["name"]
        obj.name = reference_name
        obj.data.name = reference_name + "_Mesh"
        collection.objects.link(obj)
        journal["objects"].append(obj)
        journal["meshes"].append(obj.data)
        obj.data.materials.clear()
        obj.data.materials.append(material)
        _tag(
            obj,
            source,
            variant["bone_name"],
            variant["endpoint_name"],
            "camera_uv_reference",
            variant["index"],
            variant["plan_name"],
        )
        obj[CAMERA_REFERENCE_KEY] = plane["name"]
        obj[CAMERA_CONTRACT_HASH_KEY] = uv_bundle["contract_sha256"]
        obj["speedtree_cluster_reference_blend"] = uv_bundle["reference_blend"]
        obj["speedtree_cluster_reference_blend_sha256"] = uv_bundle[
            "reference_blend_sha256"
        ]
        obj["speedtree_cluster_reference_validation"] = json.dumps(
            validation, ensure_ascii=False, sort_keys=True
        )
        loaded.append(obj)
    for material_block in bpy.data.materials:
        if material_block.name not in existing_materials:
            journal["materials"].append(material_block)
    return loaded


def _build_plan(
    source,
    plan_collection,
    plan_name,
    material_name,
    coverage_points,
    bone_name,
    endpoint_name,
    frame,
    margin_ratio,
    index,
    skeletal_name,
    reference_plane,
    reference_object,
    uv_bundle,
    projection_basis,
    prototype_index,
    prototype_asset,
    source_partition_mode,
    plan_refinement_levels,
    source_3d_contract,
    journal,
    physical_capture_contract=None,
):
    right = Vector(projection_basis["right"])
    up = Vector(projection_basis["up"])
    normal = Vector(projection_basis["normal"])
    axis_tolerance = 1.0e-6
    if min(right.length, up.length, normal.length) <= axis_tolerance:
        raise ValueError("Generated plan projection basis contains a zero-length axis.")
    right.normalize()
    up.normalize()
    normal.normalize()
    if (
        abs(right.dot(up)) > axis_tolerance
        or abs(right.dot(normal)) > axis_tolerance
        or abs(up.dot(normal)) > axis_tolerance
        or right.cross(up).dot(normal) < 1.0 - axis_tolerance
    ):
        raise ValueError(
            "Generated plan projection basis is not right-handed orthonormal: "
            f"right_up={right.dot(up):.9g}, "
            f"right_normal={right.dot(normal):.9g}, "
            f"up_normal={up.dot(normal):.9g}, "
            f"handedness={right.cross(up).dot(normal):.9g}."
        )
    if physical_capture_contract is None and (
        (right - Vector((1.0, 0.0, 0.0))).length > axis_tolerance
        or (up - Vector((0.0, 1.0, 0.0))).length > axis_tolerance
        or (normal - Vector((0.0, 0.0, 1.0))).length > axis_tolerance
    ):
        raise ValueError(
            "Legacy generated plans require the shared camera-aligned local XY frame."
        )
    points = [
        (float(point.dot(right)), float(point.dot(up)))
        for point in coverage_points
    ]
    attachment_point = (0.0, 0.0)
    xml_attachment = frame.get("xml_attachment")
    if not isinstance(xml_attachment, dict):
        raise ValueError("Generated plans require a validated XML physical attachment.")
    xml_direction_world = (
        attachment_endpoint_world(xml_attachment)
        - attachment_origin_world(xml_attachment)
    )
    root_direction_world = xml_direction_world
    root_direction_policy = "xml_attachment_direction"
    if physical_capture_contract is not None:
        tangent = frame.get("attachment_tangent_projection") or {}
        try:
            root_direction_world = Vector(
                tangent["aligned_capture_plane_world"]
            )
            root_direction_policy = str(tangent["direction_policy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Physical capture frame lacks its resolved in-plane attachment "
                "direction."
            ) from exc
    root_direction_local = (
        frame["matrix_world"].inverted_safe().to_3x3()
        @ root_direction_world
    )
    hull, root_lock = root_locked_expanded_hull(
        points,
        margin_ratio,
        (
            float(root_direction_local.dot(right)),
            float(root_direction_local.dot(up)),
        ),
    )
    root_lock["direction_policy"] = root_direction_policy
    if physical_capture_contract is not None:
        capture_frame = physical_capture_contract["frame"]
        capture_center = Vector(capture_frame["center"])
        capture_right = Vector(capture_frame["right"])
        capture_up = Vector(capture_frame["up"])
        capture_width = float(capture_frame["width"])
        capture_height = float(capture_frame["height"])

        def direct_uv(point):
            fitted_world = frame["matrix_world"] @ (
                right * float(point[0]) + up * float(point[1])
            )
            relative = fitted_world - capture_center
            return [
                0.5 + float(relative.dot(capture_right)) / capture_width,
                0.5 + float(relative.dot(capture_up)) / capture_height,
            ]

        boundary_uvs = [direct_uv(point) for point in hull]
        pivot_uv = direct_uv(attachment_point)
        uv_values = [
            coordinate for uv in boundary_uvs + [pivot_uv] for coordinate in uv
        ]
        if min(uv_values) < -1.0e-6 or max(uv_values) > 1.0 + 1.0e-6:
            raise ValueError(
                "Direct-capture plan exceeds the physical capture frame: "
                f"{plan_name}; uv_min={min(uv_values):.9g}, "
                f"uv_max={max(uv_values):.9g}"
            )
        transfer = {
            "policy": "direct_physical_capture_projection",
            "direct_uv_source": DIRECT_CAPTURE_UV_SOURCE,
            "capture_contract_sha256": physical_capture_contract[
                "contract_sha256"
            ],
            "capture_plane": capture_frame["plane"],
            "capture_center": list(capture_frame["center"]),
            "capture_width": capture_width,
            "capture_height": capture_height,
            "capture_attachment": dict(frame["capture_attachment"]),
            "attachment_tangent_projection": dict(
                frame["attachment_tangent_projection"]
            ),
            "plan_attachment_xy": [0.0, 0.0],
            "attachment_vertex_uv": [float(value) for value in pivot_uv],
        }
    else:
        transferred_uvs, transfer = _transfer_camera_boundary_uvs(
            hull,
            reference_plane,
            uv_bundle["contract"]["camera"],
            plan_attachment_xy=attachment_point,
        )
        boundary_uvs = transferred_uvs.tolist()
        pivot_uv = (reference_plane.get("attachment") or {}).get("pivot_uv")
    triangulated_points, refined_uvs, faces, attachment_vertex_index = (
        _uniform_plan_triangulation(
            hull,
            boundary_uvs,
            plan_refinement_levels,
            attachment_point=attachment_point,
            attachment_uv=pivot_uv,
            containment_tolerance=root_lock["tolerance"],
        )
    )
    boundary_vertices = [
        tuple(right * float(point[0]) + up * float(point[1]))
        for point in hull
    ]
    vertices = [
        tuple(right * float(point[0]) + up * float(point[1]))
        for point in triangulated_points
    ]
    mesh = bpy.data.meshes.new(plan_name + "_Mesh")
    journal["meshes"].append(mesh)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = tuple(
                float(value) for value in refined_uvs[vertex_index]
            )
    mesh.materials.append(_plan_material(material_name, journal))
    stored_uvs = [None] * len(mesh.vertices)
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            stored_uvs[vertex_index] = [
                float(value) for value in uv_layer.data[loop_index].uv
            ]
    plan = bpy.data.objects.new(plan_name, mesh)
    journal["objects"].append(plan)
    plan_collection.objects.link(plan)
    plan.matrix_world = Matrix.Identity(4)
    _tag(plan, source, bone_name, endpoint_name, "speedtree_plan", index, skeletal_name)
    mesh[GENERATED_FLAG] = True
    plan["speedtree_cluster_frame_world"] = json.dumps(_matrix_rows(frame["matrix_world"]))
    plan["speedtree_cluster_frame_policy"] = str(
        frame.get("orientation_policy") or ""
    )
    if frame.get("source_frame_world") is not None:
        plan["speedtree_cluster_source_frame_world"] = json.dumps(
            _matrix_rows(frame["source_frame_world"])
        )
    plan["speedtree_cluster_margin_ratio"] = float(margin_ratio)
    plan["speedtree_cluster_attachment_vertex_index"] = int(
        attachment_vertex_index
    )
    plan[PROTOTYPE_INDEX_KEY] = int(prototype_index)
    plan[PROTOTYPE_ASSET_KEY] = prototype_asset
    plan[SOURCE_PARTITION_MODE_KEY] = source_partition_mode
    plan[PROJECTION_BASIS_KEY] = json.dumps(
        projection_basis, ensure_ascii=False, sort_keys=True
    )
    coverage = projection_coverage_2d(points, hull)
    plan[PROJECTION_COVERAGE_KEY] = json.dumps(
        coverage, ensure_ascii=False, sort_keys=True
    )
    plan[SOURCE_3D_CONTRACT_KEY] = json.dumps(
        source_3d_contract, ensure_ascii=False, sort_keys=True
    )
    plan[XML_ATTACHMENT_KEY] = json.dumps(
        xml_attachment, ensure_ascii=False, sort_keys=True
    )
    plan[PLAN_ROOT_LOCK_KEY] = json.dumps(
        root_lock, ensure_ascii=False, sort_keys=True
    )
    transfer.update(
        {
            "result_uvs": stored_uvs,
            "projection_basis": projection_basis,
            "prototype_index": int(prototype_index),
            "prototype_asset": prototype_asset,
            "source_partition_mode": source_partition_mode,
            "plan_refinement_levels": int(plan_refinement_levels),
            "boundary_vertex_count": len(boundary_vertices),
            "interior_vertex_count": len(vertices) - len(boundary_vertices),
            "attachment_vertex_index": int(attachment_vertex_index),
            "attachment_vertex_uv": [float(value) for value in pivot_uv],
            "source_3d_contract": source_3d_contract,
            "xml_attachment": xml_attachment,
            "plan_root_lock": root_lock,
        }
    )
    if physical_capture_contract is None:
        transfer.update(
            {
                "reference_plane": reference_plane["name"],
                "reference_object": reference_object.name,
                "source_mesh_id": int(reference_plane["source_mesh_id"]),
                "reference_topology_sha256": reference_plane[
                    "topology_sha256"
                ],
                "reference_uv_sha256": reference_plane["uv_sha256"],
                "contract_sha256": uv_bundle["contract_sha256"],
                "reference_blend": uv_bundle["reference_blend"],
                "reference_blend_sha256": uv_bundle[
                    "reference_blend_sha256"
                ],
            }
        )
    transfer["result_uv_sha256"] = _canonical_sha256(transfer["result_uvs"])
    if physical_capture_contract is not None:
        capture_hash = physical_capture_contract["contract_sha256"]
        plan[PHYSICAL_CAPTURE_CONTRACT_HASH_KEY] = capture_hash
        plan[DIRECT_CAPTURE_UV_KEY] = json.dumps(
            transfer, ensure_ascii=False, sort_keys=True
        )
        plan["speedtree_cluster_uv_policy"] = (
            "direct_physical_capture_projection"
        )
        plan["speedtree_cluster_physical_fit_scale"] = float(
            frame["physical_fit_scale"]
        )
        plan["speedtree_cluster_capture_attachment"] = json.dumps(
            frame["capture_attachment"],
            ensure_ascii=False,
            sort_keys=True,
        )
        plan["speedtree_cluster_attachment_tangent_projection"] = json.dumps(
            frame["attachment_tangent_projection"],
            ensure_ascii=False,
            sort_keys=True,
        )
        mesh[PHYSICAL_CAPTURE_CONTRACT_HASH_KEY] = capture_hash
    else:
        plan[CAMERA_CONTRACT_HASH_KEY] = uv_bundle["contract_sha256"]
        plan[CAMERA_REFERENCE_KEY] = reference_object.name
        plan[UV_TRANSFER_KEY] = json.dumps(
            transfer, ensure_ascii=False, sort_keys=True
        )
        mesh[CAMERA_CONTRACT_HASH_KEY] = uv_bundle["contract_sha256"]

    maximum_plane_error = max(
        abs(float(Vector(vertex).dot(normal))) for vertex in vertices
    )
    if maximum_plane_error > 1.0e-6:
        raise ValueError(f"Generated plan left its camera projection plane: {plan_name}")
    if not coverage["covers_projection"]:
        outside_samples = [
            points[value]
            for value in coverage["outside_point_indices"][:8]
        ]
        raise ValueError(
            "Generated plan does not cover "
            f"{coverage['outside_point_count']} projected vertices: {plan_name}; "
            f"samples={outside_samples}; root_lock={root_lock}"
        )
    return plan, hull, transfer, coverage, root_lock


def configure_send2ue_handoff(scene):
    properties = getattr(scene, "send2ue", None)
    if properties is None:
        raise RuntimeError(
            "Send to Unreal is required but is not available in the current Blender session."
        )
    extensions = getattr(properties, "extensions", None)
    collections_as_folders = getattr(extensions, "use_collections_as_folders", None)
    if (
        collections_as_folders is not None
        and bool(getattr(collections_as_folders, "use_collections_as_folders", False))
    ):
        raise RuntimeError(
            "Send to Unreal Use Collections as Folders conflicts with Use Immediate Parent Name. "
            "Disable it before building cluster assets."
        )
    properties.use_object_origin = True
    extension = getattr(
        extensions,
        "use_immediate_parent_name",
        None,
    )
    if extension is None:
        raise RuntimeError("Send to Unreal Use Immediate Parent Name extension is unavailable.")
    extension.use_immediate_parent_name = True
    return {
        "available": True,
        "use_object_origin": bool(properties.use_object_origin),
        "use_immediate_parent_name": (
            bool(extension.use_immediate_parent_name) if extension is not None else None
        ),
    }


def build_normalized_cluster_assets(
    context,
    source,
    plan_base_name,
    skeletal_base_name,
    plan_collection_name,
    plan_material_name,
    plan_margin_ratio=0.01,
    replace_generated=True,
    configure_send2ue=True,
    isolate_send2ue_export=True,
    source_reference_collection_name="Cluster_Source_Reference",
    camera_uv_bundle=None,
    source_partition_mode="AUTO",
    whole_mesh_pivot_object=None,
    plan_refinement_levels=1,
    source_xml_path="",
    workflow_mode=WORKFLOW_LEGACY_CAMERA_UV,
    physical_capture_contract=None,
):
    if context.mode != "OBJECT":
        raise ValueError("Cluster normalization must start in Object Mode.")
    plan_base_name = plan_base_name.strip().rstrip("_")
    skeletal_base_name = skeletal_base_name.strip().rstrip("_")
    plan_collection_name = plan_collection_name.strip()
    plan_material_name = plan_material_name.strip()
    source_reference_collection_name = source_reference_collection_name.strip()
    source_partition_mode = str(source_partition_mode or "AUTO").strip().upper()
    workflow_mode = str(
        workflow_mode or WORKFLOW_LEGACY_CAMERA_UV
    ).strip().upper()
    plan_refinement_levels = int(plan_refinement_levels)
    if plan_refinement_levels < 0 or plan_refinement_levels > 2:
        raise ValueError("Plan refinement levels must be between 0 and 2.")
    if source_partition_mode not in {
        "AUTO",
        "PER_DEFORM_ROOT",
        "PER_CONNECTED_DEFORM_CLUSTER",
        "WHOLE_MESH",
        "COMPOSITE_PER_DEFORM_ROOT",
    }:
        raise ValueError(f"Unsupported source partition mode: {source_partition_mode}")
    if source_partition_mode in {"WHOLE_MESH", "COMPOSITE_PER_DEFORM_ROOT"}:
        raise ValueError(
            f"{source_partition_mode} is not supported by the XML physical-root "
            "delivery contract. Use PER_CONNECTED_DEFORM_CLUSTER."
        )
    if not plan_base_name or not skeletal_base_name:
        raise ValueError("Plan and skeletal base names cannot be empty.")
    if plan_base_name == skeletal_base_name:
        raise ValueError("Plan and skeletal base names must be different.")
    if not plan_collection_name or plan_collection_name.casefold() == "export":
        raise ValueError("Plan collection cannot be empty or named Export.")
    if isolate_send2ue_export and (
        not source_reference_collection_name
        or source_reference_collection_name.casefold() == "export"
        or source_reference_collection_name == plan_collection_name
    ):
        raise ValueError(
            "Source reference collection must be non-empty and differ from Export and the plan collection."
        )
    if not plan_material_name:
        raise ValueError("Plan material name cannot be empty.")
    if workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
        if not isinstance(camera_uv_bundle, dict) or not camera_uv_bundle.get(
            "contract"
        ):
            raise ValueError(
                "Exact camera SPM UV contract is required only for the explicit "
                "LEGACY_CAMERA_UV workflow."
            )
        if camera_uv_bundle.get("contract_sha256") != _canonical_sha256(
            camera_uv_bundle["contract"]
        ):
            raise ValueError(
                "Camera SPM UV contract hash mismatch before normalization."
            )
        if physical_capture_contract is not None:
            raise ValueError(
                "LEGACY_CAMERA_UV cannot consume a physical capture contract."
            )
        physical_capture_frame = None
        physical_capture_hash = None
    elif workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
        if camera_uv_bundle is not None:
            raise ValueError(
                "PHYSICAL_DIRECT_CAPTURE must not read a SpeedTree camera UV "
                "bundle or camera reference."
            )
        (
            physical_capture_contract,
            physical_capture_frame,
            physical_capture_hash,
        ) = _validate_physical_capture_contract(physical_capture_contract)
    else:
        raise ValueError(f"Unsupported workflow mode: {workflow_mode}")
    unsupported_modifiers = [
        modifier.name for modifier in source.modifiers if modifier.type != "ARMATURE"
    ]
    if unsupported_modifiers:
        raise ValueError(
            "Apply or remove non-armature source modifiers before normalization: "
            + ", ".join(unsupported_modifiers)
        )

    send2ue = (
        configure_send2ue_handoff(context.scene)
        if configure_send2ue
        else {
            "available": getattr(context.scene, "send2ue", None) is not None,
            "use_object_origin": None,
            "use_immediate_parent_name": None,
        }
    )
    armature = find_source_armature(source)
    attachment_contract = load_attachment_contract(
        context.scene,
        source,
        armature,
        explicit_xml_path=source_xml_path,
    )
    source_3d_contract = serialized_contract_source(attachment_contract)
    used_xml_root_ids = set()
    weights = _vertex_bone_weights(source, armature)
    assignments = _face_group_assignments(source, weights)
    populated = {
        name for name, faces in assignments["faces"].items() if faces
    }
    bones = [bone for bone in armature.data.bones if bone.name in populated]
    if workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
        reference_planes = camera_uv_bundle["contract"].get("planes") or []
        if not reference_planes:
            raise ValueError("Camera UV contract contains no card planes.")
        camera_contract = camera_uv_bundle["contract"].get("camera") or {}
        _camera_world_axes(camera_contract)
    else:
        reference_planes = [
            None for _root in attachment_contract.get("roots") or []
        ]
        if not reference_planes:
            raise ValueError(
                "Physical direct capture source XML contains no attachment roots."
            )
        camera_contract = _physical_capture_camera(physical_capture_frame)
        _camera_world_axes(camera_contract)

    def aligned_frame(source_frame, vertex_indices):
        if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
            return physical_capture_aligned_frame(
                source,
                source_frame,
                vertex_indices,
                physical_capture_frame,
            )
        return camera_aligned_frame(
            source,
            source_frame,
            vertex_indices,
            camera_contract,
        )

    valid_per_deform_rows = []
    per_deform_error = None
    try:
        if not populated:
            raise ValueError("Source has no populated armature deform groups.")
        if assignments["unweighted_faces"]:
            raise ValueError(
                f"Source has {len(assignments['unweighted_faces'])} faces without deform weights."
            )
        valid_per_deform_rows = [
            (_explicit_start_bone_ordinal(bone.name), bone) for bone in bones
        ]
        ordinals = [row[0] for row in valid_per_deform_rows]
        expected_ordinals = list(range(1, len(valid_per_deform_rows) + 1))
        if len(set(ordinals)) != len(ordinals) or sorted(ordinals) != expected_ordinals:
            raise ValueError(
                "Populated *_N_Start bone ordinals must be unique and consecutive 1..N; "
                f"found {sorted(ordinals)}."
            )
        for _ordinal, bone in valid_per_deform_rows:
            _preferred_endpoint_bone(bone, populated)
        valid_per_deform_rows.sort(key=lambda row: row[0])
    except ValueError as exc:
        valid_per_deform_rows = []
        per_deform_error = str(exc)

    resolved_partition_mode = source_partition_mode
    whole_pivot = None
    composite_frame = None
    if source_partition_mode == "AUTO":
        if valid_per_deform_rows and len(valid_per_deform_rows) == len(reference_planes):
            resolved_partition_mode = "PER_DEFORM_ROOT"
        else:
            raise ValueError(
                "AUTO could not prove a one-root-per-prototype contract. Use "
                "PER_CONNECTED_DEFORM_CLUSTER for production cluster data. "
                f"Per-deform: {per_deform_error or 'card/prototype count mismatch'}."
            )

    if resolved_partition_mode == "WHOLE_MESH" and len(reference_planes) != 1:
        raise ValueError(
            "WHOLE_MESH requires exactly one camera card and one XML physical root. "
            "Use PER_CONNECTED_DEFORM_CLUSTER for multi-card data."
        )

    connected_deform_groups = []
    if valid_per_deform_rows:
        connected_deform_groups = _connected_deform_clusters(
            source, assignments, valid_per_deform_rows
        )

    prototypes = []
    if resolved_partition_mode in {
        "PER_DEFORM_ROOT",
        "COMPOSITE_PER_DEFORM_ROOT",
    }:
        if not valid_per_deform_rows:
            raise ValueError(per_deform_error or "Invalid per-deform source contract.")
        if (
            resolved_partition_mode == "PER_DEFORM_ROOT"
            and len(reference_planes) != len(valid_per_deform_rows)
        ):
            raise ValueError(
                "PER_DEFORM_ROOT requires one camera card per populated deform root: "
                f"{len(reference_planes)} cards vs {len(valid_per_deform_rows)} roots."
            )
        if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
            if len(attachment_contract["roots"]) != 1:
                raise ValueError(
                    "COMPOSITE_PER_DEFORM_ROOT cannot represent multiple independent XML "
                    "attachment roots. Use PER_CONNECTED_DEFORM_CLUSTER."
                )
            whole_pivot = _whole_mesh_pivot(source, whole_mesh_pivot_object)
            composite_source_frame = whole_mesh_frame(source, whole_pivot)
            composite_frame = aligned_frame(
                composite_source_frame,
                range(len(source.data.vertices)),
            )
        for index, (_ordinal, bone) in enumerate(valid_per_deform_rows, 1):
            endpoint_bone, endpoint_policy = _preferred_endpoint_bone(bone, populated)
            face_indices = assignments["faces"][bone.name]
            vertex_indices = sorted(
                {
                    vertex_index
                    for face_index in face_indices
                    for vertex_index in source.data.polygons[face_index].vertices
                }
            )
            geometry_bounds = _bounds(
                [source.matrix_world @ source.data.vertices[value].co for value in vertex_indices]
            )
            geometry_scale = max(geometry_bounds["size"])
            attachment = match_root_attachment(
                attachment_contract,
                armature,
                bone,
                endpoint_bone,
                geometry_scale,
                used_xml_root_ids,
            )
            attachment = fit_attachment_to_geometry(
                attachment,
                [
                    source.matrix_world @ source.data.vertices[value].co
                    for value in vertex_indices
                ],
                geometry_scale,
            )
            source_frame = canonical_frame(
                source,
                armature,
                bone,
                endpoint_bone,
                vertex_indices,
                attachment=attachment,
            )
            prototypes.append(
                {
                    "index": index,
                    "asset_name": f"{skeletal_base_name}_{index:02d}",
                    "bone_name": bone.name,
                    "endpoint_name": (
                        endpoint_bone.name if endpoint_bone is not None else ""
                    ),
                    "endpoint_policy": endpoint_policy,
                    "source_bones": [bone.name],
                    "face_indices": face_indices,
                    "xml_attachment": serialized_attachment(attachment),
                    "frame": aligned_frame(
                        source_frame,
                        vertex_indices,
                    ),
                }
            )
        if composite_frame is not None:
            for prototype in prototypes:
                prototype["subpart_to_card_matrix"] = (
                    composite_frame["matrix_world"].inverted_safe()
                    @ prototype["frame"]["matrix_world"]
                )
    elif resolved_partition_mode == "PER_CONNECTED_DEFORM_CLUSTER":
        if not valid_per_deform_rows:
            raise ValueError(per_deform_error or "Invalid connected deform cluster contract.")
        if len(connected_deform_groups) != len(reference_planes):
            summary = [group["bone_names"] for group in connected_deform_groups]
            raise ValueError(
                "PER_CONNECTED_DEFORM_CLUSTER requires one complete topology cluster "
                "per camera card: "
                f"{len(reference_planes)} cards vs {len(connected_deform_groups)} clusters; "
                f"groups={summary}."
            )
        for index, group in enumerate(connected_deform_groups, 1):
            bone = group["representative_bone"]
            endpoint_bone, endpoint_policy = _preferred_endpoint_bone(bone, populated)
            face_indices = group["face_indices"]
            vertex_indices = sorted(
                {
                    vertex_index
                    for face_index in face_indices
                    for vertex_index in source.data.polygons[face_index].vertices
                }
            )
            geometry_bounds = _bounds(
                [source.matrix_world @ source.data.vertices[value].co for value in vertex_indices]
            )
            geometry_scale = max(geometry_bounds["size"])
            attachment = match_root_attachment(
                attachment_contract,
                armature,
                bone,
                endpoint_bone,
                geometry_scale,
                used_xml_root_ids,
            )
            attachment = fit_attachment_to_geometry(
                attachment,
                [
                    source.matrix_world @ source.data.vertices[value].co
                    for value in vertex_indices
                ],
                geometry_scale,
            )
            endpoint_name = endpoint_bone.name if endpoint_bone is not None else ""
            source_frame = canonical_frame(
                source,
                armature,
                bone,
                endpoint_bone,
                vertex_indices,
                attachment=attachment,
            )
            frame = aligned_frame(
                source_frame,
                vertex_indices,
            )
            prototypes.append(
                {
                    "index": index,
                    "asset_name": f"{skeletal_base_name}_{index:02d}",
                    "bone_name": bone.name,
                    "endpoint_name": endpoint_name,
                    "endpoint_policy": "connected_deform_cluster_" + endpoint_policy,
                    "source_bones": list(group["bone_names"]),
                    "face_indices": face_indices,
                    "xml_attachment": serialized_attachment(attachment),
                    "frame": frame,
                }
            )
    else:
        if len(attachment_contract["roots"]) != 1:
            raise ValueError(
                "WHOLE_MESH requires exactly one XML structural root; found "
                f"{len(attachment_contract['roots'])}."
            )
        whole_pivot = whole_pivot or _whole_mesh_pivot(
            source, whole_mesh_pivot_object
        )
        source_frame = whole_mesh_frame(source, whole_pivot)
        xml_root = attachment_contract["roots"][0]
        attachment = {
            "xml_bone_id": int(xml_root["id"]),
            "xml_parent_id": int(xml_root["parent_id"]),
            "xml_generator": xml_root["generator"],
            "xml_start_world": xml_root["start_world"].copy(),
            "xml_end_world": xml_root["end_world"].copy(),
            "xml_radius_world": float(xml_root["radius_world"]),
            "representative_bone": "",
            "endpoint_bone": "",
            "match_policy": "unique_xml_structural_root_whole_mesh",
            "start_match_error": 0.0,
            "end_match_error": 0.0,
            "match_tolerance": 0.0,
        }
        whole_world_points = [
            source.matrix_world @ vertex.co for vertex in source.data.vertices
        ]
        whole_geometry_scale = max(_bounds(whole_world_points)["size"])
        attachment = fit_attachment_to_geometry(
            attachment,
            whole_world_points,
            whole_geometry_scale,
        )
        effective_origin = attachment_origin_world(attachment)
        effective_endpoint = attachment_endpoint_world(attachment)
        source_frame["matrix_world"].translation = effective_origin
        source_frame["origin_world"] = [float(value) for value in effective_origin]
        source_frame["endpoint_world"] = [float(value) for value in effective_endpoint]
        source_frame["endpoint_length"] = float(
            (effective_endpoint - effective_origin).length
        )
        source_frame["xml_attachment"] = serialized_attachment(attachment)
        used_xml_root_ids.add(xml_root["id"])
        prototypes.append(
            {
                "index": 1,
                "asset_name": f"{skeletal_base_name}_01",
                "bone_name": "",
                "endpoint_name": "",
                "endpoint_policy": "validated_asset_root_pivot",
                "source_bones": [],
                "face_indices": [polygon.index for polygon in source.data.polygons],
                "xml_attachment": serialized_attachment(attachment),
                "frame": aligned_frame(
                    source_frame,
                    range(len(source.data.vertices)),
                ),
            }
        )

    expected_xml_root_ids = {int(root["id"]) for root in attachment_contract["roots"]}
    if used_xml_root_ids != expected_xml_root_ids:
        raise ValueError(
            "Normalized prototypes do not map 1:1 to XML structural roots: "
            f"used={sorted(used_xml_root_ids)}, expected={sorted(expected_xml_root_ids)}."
        )
    if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
        physical_capture_contract = json.loads(
            json.dumps(physical_capture_contract, ensure_ascii=False)
        )
        physical_capture_contract["attachment_pivots"] = [
            {
                "prototype_index": int(prototype["index"]),
                "prototype_asset": prototype["asset_name"],
                "xml_bone_id": int(
                    prototype["xml_attachment"]["xml_bone_id"]
                ),
                **dict(prototype["frame"]["capture_attachment"]),
                "attachment_tangent_projection": dict(
                    prototype["frame"]["attachment_tangent_projection"]
                ),
            }
            for prototype in prototypes
        ]
        physical_capture_contract.pop("contract_sha256", None)
        physical_capture_hash = _canonical_sha256(physical_capture_contract)
        physical_capture_contract["contract_sha256"] = physical_capture_hash

    composite_parts = []
    composite_set_id = None
    if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
        composite_parts = [
            {
                "subpart_index": int(prototype["index"]),
                "skeletal_asset_name": prototype["asset_name"],
                "source_bone": prototype["bone_name"],
                "endpoint_bone": prototype["endpoint_name"],
                "subpart_to_card_matrix": _matrix_rows(
                    prototype["subpart_to_card_matrix"]
                ),
                "pivot_contract": "normalized_attachment_origin_0_0_0",
            }
            for prototype in prototypes
        ]
        composite_set_id = _canonical_sha256(composite_parts)

    cards = []
    for index, plane in enumerate(reference_planes, 1):
        prototype_index = (
            index
            if resolved_partition_mode
            in {"PER_DEFORM_ROOT", "PER_CONNECTED_DEFORM_CLUSTER"}
            else 1
        )
        prototype = prototypes[prototype_index - 1]
        cards.append(
            {
                "index": index,
                "plan_name": f"{plan_base_name}_{index:02d}",
                "prototype_index": prototype_index,
                "prototype_asset": prototype["asset_name"],
                "bone_name": (
                    ""
                    if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                    else prototype["bone_name"]
                ),
                "endpoint_name": (
                    ""
                    if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                    else prototype["endpoint_name"]
                ),
                "endpoint_policy": (
                    "validated_composite_asset_root_pivot"
                    if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                    else prototype["endpoint_policy"]
                ),
                "frame": (
                    composite_frame
                    if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                    else prototype["frame"]
                ),
                "composite_parts": composite_parts,
                "composite_set_id": composite_set_id,
            }
        )

    desired_names = []
    for prototype in prototypes:
        desired_names.extend(
            [
                prototype["asset_name"],
                prototype["asset_name"] + "_Armature",
                prototype["asset_name"] + "_Mesh",
            ]
        )
    desired_names.extend(card["plan_name"] for card in cards)
    if workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
        desired_names.extend(
            "AtlasCameraRef_" + plane["name"] for plane in reference_planes
        )

    if len(desired_names) != len(set(desired_names)):
        raise ValueError("Canonical output names are not unique.")
    conflicts = [name for name in desired_names if bpy.data.objects.get(name) is not None]
    # A rebuild can legitimately reduce the prototype count (for example when
    # several deform roots are recognized as one complete topology cluster).
    # Stage prior generated outputs from the same canonical asset family as
    # stale data so they are removed at commit instead of being preserved as
    # misleading Export/reference objects.
    stale_generated = [
        obj.name
        for obj in bpy.data.objects
        if _is_generated(obj)
        and (
            str(obj.get(PROTOTYPE_ASSET_KEY) or "").startswith(
                skeletal_base_name + "_"
            )
            or obj.name.startswith(plan_base_name + "_")
            or obj.name.startswith("AtlasCameraRef_" + plan_base_name + "_")
        )
        and obj.name not in desired_names
    ]
    conflicts = list(dict.fromkeys(conflicts + stale_generated))
    if conflicts and not replace_generated:
        raise ValueError("Output objects already exist: " + ", ".join(conflicts))
    backups = _stage_named_generated_objects(conflicts) if conflicts else []
    journal = {
        "objects": [],
        "meshes": [],
        "armatures": [],
        "materials": [],
        "collections": [],
        "collection_moves": [],
    }
    previous_selection = [obj.name for obj in context.selected_objects]
    previous_active = context.view_layer.objects.active.name if context.view_layer.objects.active else None
    records = []
    try:
        export_collection, export_created = _ensure_collection(context.scene, "Export")
        plan_collection, plan_created = _ensure_collection(context.scene, plan_collection_name)
        camera_reference_collection = None
        camera_reference_created = False
        if workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
            camera_reference_collection, camera_reference_created = _ensure_collection(
                context.scene,
                camera_uv_bundle["reference_collection"],
            )
        if export_created:
            journal["collections"].append(export_collection)
        if plan_created:
            journal["collections"].append(plan_collection)
        if camera_reference_created:
            journal["collections"].append(camera_reference_collection)
        if workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
            camera_references = _load_camera_reference_objects(
                context.scene,
                source,
                cards,
                plan_material_name,
                camera_uv_bundle,
                camera_reference_collection,
                journal,
            )
        else:
            camera_references = [None for _card in cards]
        isolated_export_objects = []
        if isolate_send2ue_export and export_collection.objects:
            reference_collection, reference_created = _ensure_collection(
                context.scene,
                source_reference_collection_name,
            )
            if reference_created:
                journal["collections"].append(reference_collection)
            isolated_export_objects = _isolate_existing_export_objects(
                export_collection,
                reference_collection,
                journal,
            )
            _hide_source_reference_collection(reference_collection)
        built_prototypes = {}
        prototype_reports = []
        for prototype in prototypes:
            card_names = (
                [card["plan_name"] for card in cards]
                if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                else [
                    card["plan_name"]
                    for card in cards
                    if card["prototype_index"] == prototype["index"]
                ]
            )
            pivot, part_armature, part = _build_part_hierarchy(
                source,
                export_collection,
                prototype["asset_name"],
                prototype["bone_name"],
                prototype["endpoint_name"],
                prototype["face_indices"],
                prototype["frame"],
                prototype["index"],
                card_names[0],
                journal,
            )
            for obj in (pivot, part_armature, part):
                obj[PROTOTYPE_INDEX_KEY] = int(prototype["index"])
                obj[PROTOTYPE_ASSET_KEY] = prototype["asset_name"]
                obj[SOURCE_PARTITION_MODE_KEY] = resolved_partition_mode
                obj["speedtree_cluster_prototype_cards"] = json.dumps(card_names)
                obj[SOURCE_3D_CONTRACT_KEY] = json.dumps(
                    source_3d_contract, ensure_ascii=False, sort_keys=True
                )
                obj[XML_ATTACHMENT_KEY] = json.dumps(
                    prototype["xml_attachment"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
                    obj[PHYSICAL_CAPTURE_CONTRACT_HASH_KEY] = (
                        physical_capture_hash
                    )
                    obj["speedtree_cluster_physical_fit_scale"] = float(
                        physical_capture_frame["fit_scale"]
                    )
            built_prototypes[prototype["index"]] = {
                "pivot": pivot,
                "armature": part_armature,
                "part": part,
                "spec": prototype,
            }
            prototype_reports.append(
                {
                    "prototype_index": prototype["index"],
                    "skeletal_asset": prototype["asset_name"],
                    "pivot": pivot.name,
                    "armature": part_armature.name,
                    "mesh": part.name,
                    "cards": card_names,
                    "source_bone": prototype["bone_name"] or None,
                    "source_bones": list(prototype.get("source_bones") or []),
                    "xml_attachment": prototype["xml_attachment"],
                    "endpoint_bone": prototype["endpoint_name"] or None,
                    "endpoint_policy": prototype["endpoint_policy"],
                    "face_count": len(part.data.polygons),
                    "vertex_count": len(part.data.vertices),
                    "frame_world": _matrix_rows(prototype["frame"]["matrix_world"]),
                    "source_frame_world": (
                        _matrix_rows(prototype["frame"]["source_frame_world"])
                        if prototype["frame"].get("source_frame_world") is not None
                        else None
                    ),
                    "frame_policy": prototype["frame"].get("orientation_policy"),
                    "attachment_tangent_projection": dict(
                        prototype["frame"].get(
                            "attachment_tangent_projection"
                        )
                        or {}
                    ),
                    "subpart_to_card_matrix": (
                        _matrix_rows(prototype["subpart_to_card_matrix"])
                        if prototype.get("subpart_to_card_matrix") is not None
                        else None
                    ),
                    "normalized_bounds": _bounds(
                        [vertex.co.copy() for vertex in part.data.vertices]
                    ),
                }
            )

        for card, reference_plane, reference_object in zip(
            cards,
            reference_planes,
            camera_references,
        ):
            built = built_prototypes[card["prototype_index"]]
            prototype = built["spec"]
            pivot = built["pivot"]
            part_armature = built["armature"]
            part = built["part"]
            if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
                coverage_points = [
                    card["frame"]["matrix_world"].inverted_safe()
                    @ source.matrix_world
                    @ vertex.co
                    for vertex in source.data.vertices
                ]
            else:
                coverage_points = [
                    vertex.co.copy() for vertex in part.data.vertices
                ]
            projection_basis = camera_projection_basis_in_part(
                prototype["frame"], camera_contract
            )
            if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
                projection_basis = camera_projection_basis_in_part(
                    card["frame"], camera_contract
                )
            plan, hull, uv_transfer, projection_coverage, plan_root_lock = _build_plan(
                source,
                plan_collection,
                card["plan_name"],
                plan_material_name,
                coverage_points,
                card["bone_name"],
                card["endpoint_name"],
                card["frame"],
                plan_margin_ratio,
                card["index"],
                card["prototype_asset"],
                reference_plane,
                reference_object,
                camera_uv_bundle,
                projection_basis,
                card["prototype_index"],
                card["prototype_asset"],
                resolved_partition_mode,
                plan_refinement_levels,
                source_3d_contract,
                journal,
                physical_capture_contract=(
                    physical_capture_contract
                    if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                    else None
                ),
            )
            if card["composite_parts"]:
                plan[COMPOSITE_PARTS_KEY] = json.dumps(
                    card["composite_parts"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                plan["speedtree_cluster_composite_set_id"] = card[
                    "composite_set_id"
                ]
            records.append(
                {
                    "index": card["index"],
                    "card_index": card["index"],
                    "prototype_index": card["prototype_index"],
                    "prototype_asset": card["prototype_asset"],
                    "source_partition_mode": resolved_partition_mode,
                    "composite_set_id": card["composite_set_id"],
                    "composite_parts": card["composite_parts"],
                    "source_bone": card["bone_name"] or None,
                    "endpoint_bone": card["endpoint_name"] or None,
                    "endpoint_policy": card["endpoint_policy"],
                    "face_count": (
                        len(source.data.polygons)
                        if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                        else len(part.data.polygons)
                    ),
                    "vertex_count": len(coverage_points),
                    "skeletal_asset": card["prototype_asset"],
                    "pivot": pivot.name,
                    "armature": part_armature.name,
                    "mesh": part.name,
                    "plan": plan.name,
                    "plan_vertices": len(plan.data.vertices),
                    "plan_faces": len(plan.data.polygons),
                    "plan_boundary_vertices": len(hull),
                    "plan_interior_vertices": len(plan.data.vertices) - len(hull),
                    "plan_refinement_levels": plan_refinement_levels,
                    "plan_covers_projection": bool(
                        projection_coverage["covers_projection"]
                    ),
                    "plan_projection_coverage": projection_coverage,
                    "plan_root_lock": plan_root_lock,
                    "xml_attachment": card["frame"]["xml_attachment"],
                    "object_transforms_identity": all(
                        _identity_matrix(obj.matrix_world)
                        for obj in (
                            [plan]
                            + [
                                value
                                for built_row in built_prototypes.values()
                                for value in (
                                    built_row["pivot"],
                                    built_row["armature"],
                                    built_row["part"],
                                )
                            ]
                        )
                    ),
                    "frame_world": _matrix_rows(card["frame"]["matrix_world"]),
                    "source_frame_world": (
                        _matrix_rows(card["frame"]["source_frame_world"])
                        if card["frame"].get("source_frame_world") is not None
                        else None
                    ),
                    "frame_policy": card["frame"].get("orientation_policy"),
                    "endpoint_length": card["frame"]["endpoint_length"],
                    "source_world_bounds": card["frame"]["source_world_bounds"],
                    "normalized_bounds": _bounds(coverage_points),
                    "plan_hull": [[float(value) for value in point] for point in hull],
                    "projection_basis": projection_basis,
                    "camera_reference": (
                        reference_object.name
                        if reference_object is not None
                        else None
                    ),
                    "camera_source_mesh_id": (
                        int(reference_plane["source_mesh_id"])
                        if reference_plane is not None
                        else None
                    ),
                    "camera_reference_uv_sha256": (
                        reference_plane["uv_sha256"]
                        if reference_plane is not None
                        else None
                    ),
                    "physical_capture_contract_sha256": (
                        physical_capture_hash
                        if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                        else None
                    ),
                    "physical_fit_scale": (
                        float(card["frame"]["physical_fit_scale"])
                        if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                        else None
                    ),
                    "capture_attachment": (
                        dict(card["frame"]["capture_attachment"])
                        if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                        else None
                    ),
                    "attachment_tangent_projection": (
                        dict(card["frame"]["attachment_tangent_projection"])
                        if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                        else None
                    ),
                    "plan_uv_transfer": uv_transfer,
                    "materials": [
                        material.name if material else None for material in part.data.materials
                    ],
                    "uv_layers": [layer.name for layer in part.data.uv_layers],
                    "color_attributes": [attribute.name for attribute in part.data.color_attributes],
                }
            )
        export_rig_contract = _validate_single_bone_export_contract(
            export_collection,
            built_prototypes,
        )
        report = {
            "schema_version": 2,
            "workflow_mode": workflow_mode,
            "source_object": source.name,
            "source_armature": armature.name,
            "source_3d_contract": source_3d_contract,
            "source_preserved": True,
            "variant_count": len(records),
            "card_count": len(records),
            "prototype_count": len(prototype_reports),
            "source_partition_mode_requested": source_partition_mode,
            "source_partition_mode": resolved_partition_mode,
            "whole_mesh_pivot": whole_pivot.name if whole_pivot is not None else None,
            "composite_set_id": composite_set_id,
            "composite_frame_world": (
                _matrix_rows(composite_frame["matrix_world"])
                if composite_frame is not None
                else None
            ),
            "prototypes": prototype_reports,
            "plan_collection": plan_collection.name,
            "send2ue_export_collection": export_collection.name,
            "send2ue": send2ue,
            "send2ue_export_isolated": bool(isolate_send2ue_export),
            "isolated_export_objects": isolated_export_objects,
            "export_rig_contract": export_rig_contract,
            "source_reference_collection": (
                source_reference_collection_name if isolate_send2ue_export else None
            ),
            "mixed_face_count": len(assignments["mixed_faces"]),
            "unweighted_vertex_count": len(assignments["unweighted_vertices"]),
            "dominant_weight_tie_count": len(assignments["tied_vertices"]),
            "normalization_policy": (
                "validated_attachment_origin_with_physical_uniform_fit_and_shared_axes"
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else "validated_attachment_origin_with_shared_camera_aligned_rigid_frame"
            ),
            "size_policy": (
                "uniform_whole_source_physical_target_meters"
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else "source_relative_only_no_absolute_dimensions"
            ),
            "plan_policy": (
                "same_blender_capture_direct_projection_with_pinned_attachment"
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else "camera_aligned_local_xy_orthogonal_projection_with_pinned_attachment"
            ),
            "plan_refinement_levels": plan_refinement_levels,
            "plan_uv_policy": (
                "direct_physical_capture_projection"
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else "exact_camera_reference_closed_loop_similarity_transfer"
            ),
            "physical_capture_contract": (
                physical_capture_contract
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "physical_capture_contract_sha256": (
                physical_capture_hash
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "physical_target_meters": (
                list(physical_capture_frame["target_meters"])
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "physical_target_blender_units": (
                list(physical_capture_frame["target_blender_units"])
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "physical_fit_scale": (
                float(physical_capture_frame["fit_scale"])
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "direct_uv_source": (
                DIRECT_CAPTURE_UV_SOURCE
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "capture_manifest": (
                physical_capture_contract.get("capture_manifest")
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else None
            ),
            "capture_manifest_sha256": (
                None
            ),
            "capture_maps": (
                list(physical_capture_contract.get("capture_maps") or [])
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else []
            ),
            "camera_dependency": (
                "none"
                if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                else "explicit_legacy_camera_uv_contract"
            ),
            "generator_size_policy": (
                "preserve_user_authored_leaf_and_frond_dimensions"
            ),
            "camera_reference_collection": (
                camera_reference_collection.name
                if camera_reference_collection is not None
                else None
            ),
            "camera_uv_contract_sha256": (
                camera_uv_bundle["contract_sha256"]
                if camera_uv_bundle is not None
                else None
            ),
            "camera_reference_blend": (
                camera_uv_bundle["reference_blend"]
                if camera_uv_bundle is not None
                else None
            ),
            "camera_reference_blend_sha256": (
                camera_uv_bundle["reference_blend_sha256"]
                if camera_uv_bundle is not None
                else None
            ),
            "camera_manifest": (
                camera_uv_bundle["manifest_path"]
                if camera_uv_bundle is not None
                else None
            ),
            "camera_manifest_sha256": (
                camera_uv_bundle["manifest_sha256"]
                if camera_uv_bundle is not None
                else None
            ),
            "camera_tree_file_changed_since_reference_build": (
                camera_uv_bundle.get(
                    "tree_file_changed_since_reference_build", False
                )
                if camera_uv_bundle is not None
                else False
            ),
            "variants": records,
        }
        card_prototype_map = {
            "version": (
                2
                if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                else 1
            ),
            "source_object": source.name,
            "source_partition_mode": resolved_partition_mode,
            "card_count": len(records),
            "prototype_count": len(prototype_reports),
            "source_3d_contract_sha256": _canonical_sha256(source_3d_contract),
            "composite_set_id": composite_set_id,
            "composite_parts": composite_parts,
            "cards": [
                {
                    "card_index": row["card_index"],
                    "plan": row["plan"],
                    "source_mesh_id": row["camera_source_mesh_id"],
                    "uv_source": (
                        DIRECT_CAPTURE_UV_SOURCE
                        if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                        else "legacy_camera_uv_reference"
                    ),
                    "prototype_index": row["prototype_index"],
                    "prototype_asset": row["prototype_asset"],
                    "xml_bone_id": row["xml_attachment"]["xml_bone_id"],
                    "composite_set_id": row["composite_set_id"],
                    "composite_parts": row["composite_parts"],
                }
                for row in records
            ],
        }
        context.scene[CARD_PROTOTYPE_MAP_KEY] = json.dumps(
            card_prototype_map, ensure_ascii=False, sort_keys=True
        )
        context.scene[CARD_PROTOTYPE_MAP_HASH_KEY] = _canonical_sha256(
            card_prototype_map
        )
        context.scene[SOURCE_3D_CONTRACT_KEY] = json.dumps(
            source_3d_contract, ensure_ascii=False, sort_keys=True
        )
        context.scene[SOURCE_3D_CONTRACT_HASH_KEY] = _canonical_sha256(
            source_3d_contract
        )
        if workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
            persisted_bundle = {
                key: value
                for key, value in camera_uv_bundle.items()
                if key != "contract"
            }
            context.scene["speedtree_cluster_camera_uv_bundle"] = json.dumps(
                persisted_bundle,
                ensure_ascii=False,
                sort_keys=True,
            )
            context.scene[CAMERA_CONTRACT_KEY] = json.dumps(
                camera_uv_bundle["contract"],
                ensure_ascii=False,
                sort_keys=True,
            )
            context.scene[CAMERA_CONTRACT_HASH_KEY] = camera_uv_bundle[
                "contract_sha256"
            ]
            for key in (
                PHYSICAL_CAPTURE_CONTRACT_KEY,
                PHYSICAL_CAPTURE_CONTRACT_HASH_KEY,
            ):
                if key in context.scene:
                    del context.scene[key]
        else:
            context.scene[PHYSICAL_CAPTURE_CONTRACT_KEY] = json.dumps(
                physical_capture_contract,
                ensure_ascii=False,
                sort_keys=True,
            )
            context.scene[PHYSICAL_CAPTURE_CONTRACT_HASH_KEY] = (
                physical_capture_hash
            )
            for key in (
                "speedtree_cluster_camera_uv_bundle",
                CAMERA_CONTRACT_KEY,
                CAMERA_CONTRACT_HASH_KEY,
            ):
                if key in context.scene:
                    del context.scene[key]
        context.scene["speedtree_cluster_normalizer_last_report"] = json.dumps(
            report,
            ensure_ascii=False,
        )
        bpy.context.view_layer.update()
        _commit_staged_objects(backups)
        return report
    except Exception:
        _restore_collection_moves(journal)
        _cleanup_journal(journal)
        _restore_staged_objects(backups)
        raise
    finally:
        for obj in context.selected_objects:
            obj.select_set(False)
        for name in previous_selection:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True)
        context.view_layer.objects.active = bpy.data.objects.get(previous_active)
