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
CARD_PROTOTYPE_MAP_KEY = "speedtree_cluster_card_prototype_map"
CARD_PROTOTYPE_MAP_HASH_KEY = "speedtree_cluster_card_prototype_map_sha256"
COMPOSITE_PARTS_KEY = "speedtree_cluster_composite_parts"
CAMERA_CONTRACT_KEY = "speedtree_cluster_camera_uv_contract"
CAMERA_CONTRACT_HASH_KEY = "speedtree_cluster_camera_uv_contract_sha256"
CAMERA_REFERENCE_KEY = "speedtree_cluster_camera_reference"
UV_TRANSFER_KEY = "speedtree_cluster_uv_transfer"
# Camera cutouts and generated covering cards need not share the same outline.
# Keep this scale-independent guard broad enough for intentionally different
# side captures while the separate planarity, coverage, orientation, and UV
# invariants continue to fail closed.
UV_TRANSFER_MAX_NORMALIZED_RMS = 0.5
UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR = 0.2
UV_TRANSFER_ATTACHMENT_POLICY = (
    "normalized_plan_origin_to_camera_contract_source_plane_xy"
)
UV_TRANSFER_CANDIDATE_SELECTION_POLICY = (
    "attachment_origin_gate_then_outline_normalized_rms"
)


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
    group_names = {
        group.index: group.name
        for group in source.vertex_groups
        if group.name in bone_names
    }
    weights = {}
    for vertex in source.data.vertices:
        row = {
            group_names[element.group]: float(element.weight)
            for element in vertex.groups
            if element.group in group_names and element.weight > 0.0
        }
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


def _preferred_endpoint_bone(bone, populated_bones):
    if bone.name.casefold().endswith("_start"):
        expected = bone.name[:-6] + "_End"
        for child in bone.children:
            if child.name.casefold() == expected.casefold():
                return child, "matching_end_child"
    raise ValueError(
        f"Populated deform bone '{bone.name}' must have a direct matching *_End child."
    )


def _explicit_start_bone_ordinal(bone_name):
    match = re.search(r"_(\d+)_Start$", str(bone_name), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Populated deform bone '{bone_name}' needs an explicit *_N_Start ordinal."
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


def canonical_frame(source, armature, bone, endpoint_bone, vertex_indices):
    armature_world = armature.matrix_world
    origin = armature_world @ bone.head_local
    endpoint = (
        armature_world @ endpoint_bone.head_local
        if endpoint_bone is not None
        else armature_world @ bone.tail_local
    )
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
    return {
        "matrix_world": frame,
        "origin_world": [float(value) for value in origin],
        "endpoint_world": [float(value) for value in endpoint],
        "endpoint_length": float(length),
        "source_world_bounds": world_bounds,
        "normalized_bounds": _bounds(local_points),
    }


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


def camera_projection_basis_in_part(frame, camera):
    try:
        camera_right = Vector(camera["right"])
        camera_up = Vector(camera["up"])
        camera_normal = Vector(camera["plane_normal"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Camera contract lacks a numeric right/up/normal basis.") from exc
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
    remap = {old: new for new, old in enumerate(used_indices)}
    original_polygon_indices = [polygon.material_index for polygon in mesh.polygons]
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
    transform = frame["matrix_world"].inverted_safe() @ source.matrix_world
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


def _similarity_fit(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
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
    target_radius = float(
        np.sqrt(np.mean(np.sum(centered_target * centered_target, axis=1)))
    )
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
    if not (
        np.isfinite(plan_boundary).all()
        and np.isfinite(reference_vertices).all()
        and np.isfinite(reference_uvs).all()
        and plan_attachment.shape == (2,)
        and np.isfinite(plan_attachment).all()
        and reference_attachment.shape == (2,)
        and np.isfinite(reference_attachment).all()
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
                fit = _similarity_fit(plan_samples, shifted)
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
    anchor_candidates = [
        candidate
        for candidate in finite_candidates
        if candidate["attachment_error_normalized"]
        <= UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR
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
            "Camera-template attachment alignment is ambiguous: normalized origin error "
            f"{nearest['attachment_error_normalized']:.6f} exceeds "
            f"{UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR:.6f}."
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
    journal,
):
    right = Vector(projection_basis["right"])
    up = Vector(projection_basis["up"])
    normal = Vector(projection_basis["normal"])
    points = [
        (float(point.dot(right)), float(point.dot(up)))
        for point in coverage_points
    ]
    hull = expanded_hull(points, margin_ratio)
    transferred_uvs, transfer = _transfer_camera_boundary_uvs(
        hull,
        reference_plane,
        uv_bundle["contract"]["camera"],
        plan_attachment_xy=(0.0, 0.0),
    )
    boundary_vertices = [
        tuple(right * point[0] + up * point[1])
        for point in hull
    ]
    boundary_uvs = transferred_uvs.tolist()
    boundary_faces = _triangulate_uv_boundary(boundary_uvs)
    vertices, refined_uvs, faces = _refine_plan_triangles(
        boundary_vertices,
        boundary_uvs,
        boundary_faces,
        plan_refinement_levels,
    )
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
    plan["speedtree_cluster_margin_ratio"] = float(margin_ratio)
    plan[PROTOTYPE_INDEX_KEY] = int(prototype_index)
    plan[PROTOTYPE_ASSET_KEY] = prototype_asset
    plan[SOURCE_PARTITION_MODE_KEY] = source_partition_mode
    plan[PROJECTION_BASIS_KEY] = json.dumps(
        projection_basis, ensure_ascii=False, sort_keys=True
    )
    transfer.update(
        {
            "reference_plane": reference_plane["name"],
            "reference_object": reference_object.name,
            "source_mesh_id": int(reference_plane["source_mesh_id"]),
            "reference_topology_sha256": reference_plane["topology_sha256"],
            "reference_uv_sha256": reference_plane["uv_sha256"],
            "contract_sha256": uv_bundle["contract_sha256"],
            "reference_blend": uv_bundle["reference_blend"],
            "reference_blend_sha256": uv_bundle["reference_blend_sha256"],
            "result_uvs": stored_uvs,
            "projection_basis": projection_basis,
            "prototype_index": int(prototype_index),
            "prototype_asset": prototype_asset,
            "source_partition_mode": source_partition_mode,
            "plan_refinement_levels": int(plan_refinement_levels),
            "boundary_vertex_count": len(boundary_vertices),
            "interior_vertex_count": len(vertices) - len(boundary_vertices),
        }
    )
    transfer["result_uv_sha256"] = _canonical_sha256(transfer["result_uvs"])
    plan[CAMERA_CONTRACT_HASH_KEY] = uv_bundle["contract_sha256"]
    plan[CAMERA_REFERENCE_KEY] = reference_object.name
    plan[UV_TRANSFER_KEY] = json.dumps(transfer, ensure_ascii=False, sort_keys=True)
    mesh[CAMERA_CONTRACT_HASH_KEY] = uv_bundle["contract_sha256"]

    maximum_plane_error = max(
        abs(float(Vector(vertex).dot(normal))) for vertex in vertices
    )
    if maximum_plane_error > 1.0e-6:
        raise ValueError(f"Generated plan left its camera projection plane: {plan_name}")
    bounds_2d = _bounds([Vector((point[0], point[1], 0.0)) for point in points])
    tolerance = max(bounds_2d["size"]) ** 2 * 1.0e-7
    outside = [
        point for point in points
        if not point_in_convex_polygon(point, hull, tolerance=tolerance)
    ]
    if outside:
        raise ValueError(
            f"Generated plan does not cover {len(outside)} projected vertices: {plan_name}"
        )
    return plan, hull, transfer


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
):
    if context.mode != "OBJECT":
        raise ValueError("Cluster normalization must start in Object Mode.")
    plan_base_name = plan_base_name.strip().rstrip("_")
    skeletal_base_name = skeletal_base_name.strip().rstrip("_")
    plan_collection_name = plan_collection_name.strip()
    plan_material_name = plan_material_name.strip()
    source_reference_collection_name = source_reference_collection_name.strip()
    source_partition_mode = str(source_partition_mode or "AUTO").strip().upper()
    plan_refinement_levels = int(plan_refinement_levels)
    if plan_refinement_levels < 0 or plan_refinement_levels > 2:
        raise ValueError("Plan refinement levels must be between 0 and 2.")
    if source_partition_mode not in {
        "AUTO",
        "PER_DEFORM_ROOT",
        "WHOLE_MESH",
        "COMPOSITE_PER_DEFORM_ROOT",
    }:
        raise ValueError(f"Unsupported source partition mode: {source_partition_mode}")
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
    if not isinstance(camera_uv_bundle, dict) or not camera_uv_bundle.get("contract"):
        raise ValueError(
            "Exact camera SPM UV contract is required; run through the Cluster operator preflight."
        )
    if camera_uv_bundle.get("contract_sha256") != _canonical_sha256(
        camera_uv_bundle["contract"]
    ):
        raise ValueError("Camera SPM UV contract hash mismatch before normalization.")
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
    weights = _vertex_bone_weights(source, armature)
    assignments = _face_group_assignments(source, weights)
    populated = {
        name for name, faces in assignments["faces"].items() if faces
    }
    bones = [bone for bone in armature.data.bones if bone.name in populated]
    reference_planes = camera_uv_bundle["contract"].get("planes") or []
    if not reference_planes:
        raise ValueError("Camera UV contract contains no card planes.")

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
            try:
                whole_pivot = _whole_mesh_pivot(source, whole_mesh_pivot_object)
            except ValueError as exc:
                raise ValueError(
                    "AUTO could not prove a per-deform or whole-mesh source contract. "
                    f"Per-deform: {per_deform_error or 'card/prototype count mismatch'}. "
                    f"Whole-mesh: {exc}"
                ) from exc
            resolved_partition_mode = "WHOLE_MESH"

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
            whole_pivot = _whole_mesh_pivot(source, whole_mesh_pivot_object)
            composite_frame = whole_mesh_frame(source, whole_pivot)
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
            prototypes.append(
                {
                    "index": index,
                    "asset_name": f"{skeletal_base_name}_{index:02d}",
                    "bone_name": bone.name,
                    "endpoint_name": endpoint_bone.name,
                    "endpoint_policy": endpoint_policy,
                    "face_indices": face_indices,
                    "frame": canonical_frame(
                        source, armature, bone, endpoint_bone, vertex_indices
                    ),
                }
            )
        if composite_frame is not None:
            for prototype in prototypes:
                prototype["subpart_to_card_matrix"] = (
                    composite_frame["matrix_world"].inverted_safe()
                    @ prototype["frame"]["matrix_world"]
                )
    else:
        whole_pivot = whole_pivot or _whole_mesh_pivot(
            source, whole_mesh_pivot_object
        )
        prototypes.append(
            {
                "index": 1,
                "asset_name": f"{skeletal_base_name}_01",
                "bone_name": "",
                "endpoint_name": "",
                "endpoint_policy": "validated_asset_root_pivot",
                "face_indices": [polygon.index for polygon in source.data.polygons],
                "frame": whole_mesh_frame(source, whole_pivot),
            }
        )

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
        prototype_index = index if resolved_partition_mode == "PER_DEFORM_ROOT" else 1
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
    desired_names.extend("AtlasCameraRef_" + plane["name"] for plane in reference_planes)

    if len(desired_names) != len(set(desired_names)):
        raise ValueError("Canonical output names are not unique.")
    conflicts = [name for name in desired_names if bpy.data.objects.get(name) is not None]
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
        camera_references = _load_camera_reference_objects(
            context.scene,
            source,
            cards,
            plan_material_name,
            camera_uv_bundle,
            camera_reference_collection,
            journal,
        )
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
                    "endpoint_bone": prototype["endpoint_name"] or None,
                    "endpoint_policy": prototype["endpoint_policy"],
                    "face_count": len(part.data.polygons),
                    "vertex_count": len(part.data.vertices),
                    "frame_world": _matrix_rows(prototype["frame"]["matrix_world"]),
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
                prototype["frame"], camera_uv_bundle["contract"]["camera"]
            )
            if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT":
                projection_basis = camera_projection_basis_in_part(
                    card["frame"], camera_uv_bundle["contract"]["camera"]
                )
            plan, hull, uv_transfer = _build_plan(
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
                journal,
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
                    "plan_covers_projection": True,
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
                    "endpoint_length": card["frame"]["endpoint_length"],
                    "source_world_bounds": card["frame"]["source_world_bounds"],
                    "normalized_bounds": _bounds(coverage_points),
                    "plan_hull": [[float(value) for value in point] for point in hull],
                    "projection_basis": projection_basis,
                    "camera_reference": reference_object.name,
                    "camera_source_mesh_id": int(reference_plane["source_mesh_id"]),
                    "camera_reference_uv_sha256": reference_plane["uv_sha256"],
                    "plan_uv_transfer": uv_transfer,
                    "materials": [
                        material.name if material else None for material in part.data.materials
                    ],
                    "uv_layers": [layer.name for layer in part.data.uv_layers],
                    "color_attributes": [attribute.name for attribute in part.data.color_attributes],
                }
            )
        report = {
            "schema_version": 2,
            "source_object": source.name,
            "source_armature": armature.name,
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
            "source_reference_collection": (
                source_reference_collection_name if isolate_send2ue_export else None
            ),
            "mixed_face_count": len(assignments["mixed_faces"]),
            "unweighted_vertex_count": len(assignments["unweighted_vertices"]),
            "dominant_weight_tie_count": len(assignments["tied_vertices"]),
            "normalization_policy": (
                "dominant_deform_group_and_start_to_end_parent_child_frame"
                if resolved_partition_mode == "PER_DEFORM_ROOT"
                else (
                    "deform_subparts_with_shared_validated_composite_asset_root_frame"
                    if resolved_partition_mode == "COMPOSITE_PER_DEFORM_ROOT"
                    else "whole_merged_source_baked_to_validated_asset_root_world_frame"
                )
            ),
            "size_policy": "source_relative_only_no_absolute_dimensions",
            "plan_policy": "expanded_convex_projection_hull_with_shared_edge_midpoint_refinement",
            "plan_refinement_levels": plan_refinement_levels,
            "plan_uv_policy": "exact_camera_reference_closed_loop_similarity_transfer",
            "camera_reference_collection": camera_reference_collection.name,
            "camera_uv_contract_sha256": camera_uv_bundle["contract_sha256"],
            "camera_reference_blend": camera_uv_bundle["reference_blend"],
            "camera_reference_blend_sha256": camera_uv_bundle[
                "reference_blend_sha256"
            ],
            "camera_manifest": camera_uv_bundle["manifest_path"],
            "camera_manifest_sha256": camera_uv_bundle["manifest_sha256"],
            "camera_tree_file_changed_since_reference_build": camera_uv_bundle.get(
                "tree_file_changed_since_reference_build", False
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
            "composite_set_id": composite_set_id,
            "composite_parts": composite_parts,
            "cards": [
                {
                    "card_index": row["card_index"],
                    "plan": row["plan"],
                    "source_mesh_id": row["camera_source_mesh_id"],
                    "prototype_index": row["prototype_index"],
                    "prototype_asset": row["prototype_asset"],
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
        context.scene[CAMERA_CONTRACT_HASH_KEY] = camera_uv_bundle["contract_sha256"]
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
