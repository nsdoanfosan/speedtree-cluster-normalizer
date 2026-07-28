import hashlib
import json
import math
import re
import struct
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


AUTO_CAPTURE_COLLECTION = "Atlas_Auto_Capture"
DEFAULT_SOURCE_COLLECTION = "SpeedTree_Source"
WORKFLOW_LEGACY_CAMERA_UV = "LEGACY_CAMERA_UV"
WORKFLOW_PHYSICAL_DIRECT_CAPTURE = "PHYSICAL_DIRECT_CAPTURE"
PHYSICAL_CAPTURE_FRAME_POLICY = "physical_target_uniform_whole_source_fit"
DIRECT_CAPTURE_UV_SOURCE = "same_blender_physical_capture_projection"
MAP_SPECS = (
    ("Color", "", "SOURCE_COLOR", None),
    ("Opacity", "_Opacity", "CONSTANT", (1.0, 1.0, 1.0, 1.0)),
    ("Normal", "_Normal", "CONSTANT", (0.5, 0.5, 1.0, 1.0)),
    ("Gloss", "_Gloss", "CONSTANT", (0.5, 0.5, 0.5, 1.0)),
    ("SubsurfaceColor", "_Subsurface", "SOURCE_COLOR", None),
    (
        "SubsurfaceAmount",
        "_SubsurfaceAmount",
        "CONSTANT",
        (0.25, 0.25, 0.25, 1.0),
    ),
    ("AO", "_AO", "CONSTANT", (1.0, 1.0, 1.0, 1.0)),
    ("Height", "_Height", "CONSTANT", (0.5, 0.5, 0.5, 1.0)),
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _candidate_source_objects(collection_name=DEFAULT_SOURCE_COLLECTION):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        raise ValueError(f"Source collection does not exist: {collection_name}")
    objects = [
        obj
        for obj in collection.all_objects
        if obj.type == "MESH"
        and obj.data is not None
        and len(obj.data.vertices) > 0
        and not obj.get("atlas_leaf_cluster_generated")
    ]
    if not objects:
        raise ValueError(f"No source meshes exist in collection: {collection_name}")
    return sorted(objects, key=lambda obj: obj.name.casefold())


def _duplicate_base_name(name):
    return re.sub(r"\.\d{3}$", "", str(name or ""))


def _material_semantic_signature(material):
    if material is None:
        return None
    image_paths = []
    if material.use_nodes and material.node_tree is not None:
        for node in material.node_tree.nodes:
            image = getattr(node, "image", None)
            if image is None:
                continue
            image_paths.append(
                (
                    str(Path(bpy.path.abspath(image.filepath)).resolve()).casefold(),
                    str(getattr(image.colorspace_settings, "name", "")),
                )
            )
    base_source, alpha_source = _material_sources(material)

    def source_signature(socket):
        if socket is None:
            return None
        node = socket.node
        image = getattr(node, "image", None)
        return (
            node.bl_idname,
            socket.identifier,
            (
                str(Path(bpy.path.abspath(image.filepath)).resolve()).casefold()
                if image is not None
                else None
            ),
        )

    return (
        _duplicate_base_name(material.name_full).casefold(),
        tuple(sorted(image_paths)),
        source_signature(base_source),
        source_signature(alpha_source),
    )


def _evaluated_source_fingerprint(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    try:
        digest = hashlib.sha256()
        digest.update(struct.pack("<QQ", len(mesh.vertices), len(mesh.polygons)))
        matrix = evaluated.matrix_world
        for vertex in mesh.vertices:
            point = matrix @ vertex.co
            digest.update(struct.pack("<3d", point.x, point.y, point.z))
        for polygon in mesh.polygons:
            digest.update(struct.pack("<II", polygon.material_index, len(polygon.vertices)))
            for vertex_index in polygon.vertices:
                digest.update(struct.pack("<I", vertex_index))
        material_signature = [
            _material_semantic_signature(material)
            for material in mesh.materials
        ]
        digest.update(
            json.dumps(
                material_signature,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()
    finally:
        evaluated.to_mesh_clear()


def renderable_source_selection(
    collection_name=DEFAULT_SOURCE_COLLECTION,
    *,
    depsgraph=None,
):
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    unique = []
    duplicates = []
    seen = {}
    for obj in _candidate_source_objects(collection_name):
        fingerprint = _evaluated_source_fingerprint(obj, depsgraph)
        original = seen.get(fingerprint)
        if original is not None:
            duplicates.append(
                {
                    "name": obj.name,
                    "duplicate_of": original.name,
                    "fingerprint": fingerprint,
                }
            )
            continue
        seen[fingerprint] = obj
        unique.append(obj)
    return unique, duplicates


def _renderable_source_objects(
    collection_name=DEFAULT_SOURCE_COLLECTION,
    *,
    depsgraph=None,
):
    return renderable_source_selection(
        collection_name,
        depsgraph=depsgraph,
    )[0]


def _world_corners(objects, depsgraph=None):
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        points.extend(
            evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box
        )
    return points


def _point_bounds(points):
    if not points:
        raise ValueError("Capture source has no measurable bounds")
    minimum = Vector(
        tuple(min(float(point[axis]) for point in points) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(float(point[axis]) for point in points) for axis in range(3))
    )
    return minimum, maximum


def _matrix_rows(matrix):
    return [[float(value) for value in row] for row in matrix]


def _capture_basis(points, plane="AUTO"):
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for point in points:
        for axis in range(3):
            mins[axis] = min(mins[axis], point[axis])
            maxs[axis] = max(maxs[axis], point[axis])
    size = maxs - mins
    requested = str(plane or "AUTO").upper()
    if requested == "AUTO":
        raise ValueError(
            "Automatic capture plane is ambiguous; select XY, XZ, or YZ explicitly"
        )
    bases = {
        "XY": (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        ),
        "XZ": (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
            Vector((0.0, -1.0, 0.0)),
        ),
        "YZ": (
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
            Vector((1.0, 0.0, 0.0)),
        ),
    }
    if requested not in bases:
        raise ValueError(f"Unsupported capture plane: {requested}")
    right, up, normal = bases[requested]
    return requested, right, up, normal, mins, maxs


def auto_capture_frame(objects, *, plane="XY", padding_ratio=0.04, depsgraph=None):
    points = _world_corners(objects, depsgraph=depsgraph)
    resolved_plane, right, up, normal, world_min, world_max = _capture_basis(
        points, plane
    )
    projected_x = [point.dot(right) for point in points]
    projected_y = [point.dot(up) for point in points]
    projected_z = [point.dot(normal) for point in points]
    x0, x1 = min(projected_x), max(projected_x)
    y0, y1 = min(projected_y), max(projected_y)
    z0, z1 = min(projected_z), max(projected_z)
    content_width = x1 - x0
    content_height = y1 - y0
    if min(content_width, content_height) <= 1.0e-8:
        raise ValueError("Source projection is degenerate")
    padding = max(content_width, content_height) * max(float(padding_ratio), 0.0)
    side = max(content_width + padding * 2.0, content_height + padding * 2.0)
    center = right * ((x0 + x1) * 0.5) + up * ((y0 + y1) * 0.5)
    center += normal * ((z0 + z1) * 0.5)
    half_depth = max((z1 - z0) * 0.5, side * 0.05)
    camera_location = center + normal * (half_depth + side)
    orthogonality_error = max(
        abs(right.dot(up)),
        abs(right.dot(normal)),
        abs(up.dot(normal)),
    )
    if orthogonality_error > 1.0e-9:
        raise ValueError("Capture frame is not world-axis orthogonal")
    handedness = right.cross(up).dot(normal)
    if abs(handedness - 1.0) > 1.0e-9:
        raise ValueError("Capture frame is not right-handed")
    return {
        "policy": "world_axis_locked_auto_bounds",
        "plane": resolved_plane,
        "right": list(right),
        "up": list(up),
        "normal": list(normal),
        "view_direction": list(-normal),
        "center": list(center),
        "camera_location": list(camera_location),
        "width": side,
        "height": side,
        "content_width": content_width,
        "content_height": content_height,
        "padding_ratio": float(padding_ratio),
        "world_bounds_min": list(world_min),
        "world_bounds_max": list(world_max),
        "depth_min": z0,
        "depth_max": z1,
        "orthogonality_error": orthogonality_error,
        "handedness": handedness,
        "rotation_degrees": 0.0 if resolved_plane == "XY" else 90.0,
    }


def physical_capture_frame(
    objects,
    *,
    scene=None,
    plane="XY",
    padding_ratio=0.04,
    target_meters=0.1,
    depsgraph=None,
):
    """Fit one complete source layout into a physical square capture target.

    The fit is one uniform similarity about the complete projected bounds
    center.  Individual source objects are never centered or scaled
    independently, so multipart Branch/Side layouts retain their authored
    spacing.
    """
    scene = scene or bpy.context.scene
    units = scene.unit_settings
    unit_system = str(units.system)
    scale_length = float(units.scale_length)
    target_meters = float(target_meters)
    padding_ratio = max(float(padding_ratio), 0.0)
    if unit_system != "METRIC":
        raise ValueError(
            "Physical capture requires a METRIC Blender scene; "
            f"got {unit_system or '<empty>'}."
        )
    if not math.isfinite(scale_length) or scale_length <= 0.0:
        raise ValueError("Scene scale_length must be a finite positive value.")
    if not math.isfinite(target_meters) or target_meters <= 0.0:
        raise ValueError("Physical capture target must be greater than zero meters.")

    points = _world_corners(objects, depsgraph=depsgraph)
    resolved_plane, right, up, normal, world_min, world_max = _capture_basis(
        points, plane
    )
    projected_x = [point.dot(right) for point in points]
    projected_y = [point.dot(up) for point in points]
    projected_z = [point.dot(normal) for point in points]
    x0, x1 = min(projected_x), max(projected_x)
    y0, y1 = min(projected_y), max(projected_y)
    z0, z1 = min(projected_z), max(projected_z)
    raw_width = x1 - x0
    raw_height = y1 - y0
    if min(raw_width, raw_height) <= 1.0e-8:
        raise ValueError("Source projection is degenerate")

    target_units = target_meters / scale_length
    padded_raw_side = max(raw_width, raw_height) * (1.0 + 2.0 * padding_ratio)
    fit_scale = target_units / padded_raw_side
    if not math.isfinite(fit_scale) or fit_scale <= 0.0:
        raise ValueError("Physical capture fit scale is invalid.")

    center = (
        right * ((x0 + x1) * 0.5)
        + up * ((y0 + y1) * 0.5)
        + normal * ((z0 + z1) * 0.5)
    )
    fit_matrix = (
        Matrix.Translation(center)
        @ Matrix.Diagonal((fit_scale, fit_scale, fit_scale, 1.0))
        @ Matrix.Translation(-center)
    )
    fitted_points = [fit_matrix @ point for point in points]
    fitted_min, fitted_max = _point_bounds(fitted_points)
    fitted_width = raw_width * fit_scale
    fitted_height = raw_height * fit_scale
    fitted_depth = (z1 - z0) * fit_scale
    half_depth = max(fitted_depth * 0.5, target_units * 0.05)
    camera_location = center + normal * (half_depth + target_units)
    orthogonality_error = max(
        abs(right.dot(up)),
        abs(right.dot(normal)),
        abs(up.dot(normal)),
    )
    handedness = right.cross(up).dot(normal)
    if orthogonality_error > 1.0e-9 or abs(handedness - 1.0) > 1.0e-9:
        raise ValueError("Physical capture basis is not right-handed orthogonal.")
    if (
        fitted_width > target_units + 1.0e-9
        or fitted_height > target_units + 1.0e-9
    ):
        raise ValueError("Physical fit exceeded its target frame.")
    return {
        "policy": PHYSICAL_CAPTURE_FRAME_POLICY,
        "workflow_mode": WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
        "plane": resolved_plane,
        "right": list(right),
        "up": list(up),
        "normal": list(normal),
        "view_direction": list(-normal),
        "center": list(center),
        "camera_location": list(camera_location),
        "width": target_units,
        "height": target_units,
        "content_width": fitted_width,
        "content_height": fitted_height,
        "raw_content_width": raw_width,
        "raw_content_height": raw_height,
        "padding_ratio": padding_ratio,
        "fit_scale": fit_scale,
        "unit_system": unit_system,
        "scale_length": scale_length,
        "meters_per_blender_unit": scale_length,
        "target_meters": [target_meters, target_meters],
        "target_blender_units": [target_units, target_units],
        "raw_world_bounds_min": list(world_min),
        "raw_world_bounds_max": list(world_max),
        "fitted_world_bounds_min": list(fitted_min),
        "fitted_world_bounds_max": list(fitted_max),
        "raw_depth_min": z0,
        "raw_depth_max": z1,
        "fitted_depth": fitted_depth,
        "fit_matrix_world": _matrix_rows(fit_matrix),
        "orthogonality_error": orthogonality_error,
        "handedness": handedness,
        "rotation_degrees": 0.0 if resolved_plane == "XY" else 90.0,
        "direct_uv_source": DIRECT_CAPTURE_UV_SOURCE,
    }


def physical_capture_contract(
    *,
    source_collection=DEFAULT_SOURCE_COLLECTION,
    scene=None,
    plane="XY",
    padding_ratio=0.04,
    target_meters=0.1,
    depsgraph=None,
):
    """Return deterministic physical-fit evidence without mutating the scene."""
    scene = scene or bpy.context.scene
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    sources, excluded_duplicates = renderable_source_selection(
        source_collection,
        depsgraph=depsgraph,
    )
    frame = physical_capture_frame(
        sources,
        scene=scene,
        plane=plane,
        padding_ratio=padding_ratio,
        target_meters=target_meters,
        depsgraph=depsgraph,
    )
    source_rows = [
        {
            "name": obj.name,
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
            "evaluated_sha256": _evaluated_source_fingerprint(obj, depsgraph),
        }
        for obj in sources
    ]
    contract = {
        "kind": "speedtree_cluster_physical_capture_fit",
        "version": 1,
        "workflow_mode": WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
        "source_blend": bpy.data.filepath,
        "source_collection": source_collection,
        "source_objects": source_rows,
        "excluded_exact_duplicates": excluded_duplicates,
        "frame": frame,
        "direct_uv_source": DIRECT_CAPTURE_UV_SOURCE,
        "attachment_pivots": [],
    }
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return contract


def load_physical_capture_manifest(path, expected_contract):
    """Bind one verified map bake to the exact direct-capture fit contract."""
    raw_path = str(path or "").strip()
    if not raw_path:
        raise ValueError(
            "Physical Direct Capture production requires its Blender capture "
            "manifest before plans can be handed to SpeedTree."
        )
    manifest_path = Path(bpy.path.abspath(raw_path)).expanduser().absolute()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Physical capture manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Physical capture manifest is invalid JSON: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "speedtree_cluster_blender_auto_capture"
        or int(manifest.get("version", 0)) != 2
        or manifest.get("workflow_mode") != WORKFLOW_PHYSICAL_DIRECT_CAPTURE
        or manifest.get("direct_uv_source") != DIRECT_CAPTURE_UV_SOURCE
    ):
        raise ValueError("Physical capture manifest kind/workflow is unsupported.")
    captured_contract = manifest.get("physical_capture_contract")
    if not isinstance(captured_contract, dict):
        raise ValueError("Physical capture manifest has no embedded fit contract.")

    def validated_contract_hash(contract, label):
        recorded = str(contract.get("contract_sha256") or "")
        payload = {
            key: value for key, value in contract.items() if key != "contract_sha256"
        }
        actual = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if not recorded or recorded != actual:
            raise ValueError(f"{label} physical capture contract hash is stale.")
        return recorded

    validated_contract_hash(
        captured_contract,
        "Manifest",
    )
    validated_contract_hash(
        expected_contract,
        "Current scene",
    )

    def base_fit_payload(contract):
        payload = json.loads(
            json.dumps(contract, ensure_ascii=False, sort_keys=True)
        )
        payload.pop("contract_sha256", None)
        payload.pop("capture_manifest", None)
        payload.pop("capture_maps", None)
        payload.pop("capture_resolution", None)
        payload["attachment_pivots"] = []
        return payload

    if base_fit_payload(captured_contract) != base_fit_payload(expected_contract):
        raise ValueError(
            "Physical capture manifest was baked from different source bounds, "
            "units, plane, padding, or target size."
        )
    if manifest.get("frame") != captured_contract.get("frame"):
        raise ValueError("Physical capture manifest frame disagrees with its contract.")

    rows = manifest.get("maps")
    if not isinstance(rows, list):
        raise ValueError("Physical capture manifest has no map records.")
    maps = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Physical capture map record is invalid.")
        role = str(row.get("role") or "")
        map_path = Path(str(row.get("path") or "")).expanduser().absolute()
        if (
            role in maps
            or role not in {spec[0] for spec in MAP_SPECS}
            or not map_path.is_file()
            or int(row.get("size", -1)) != map_path.stat().st_size
            or str(row.get("sha256") or "") != _sha256(map_path)
        ):
            raise ValueError(
                f"Physical capture map is missing, duplicated, or stale: "
                f"{role or '<empty>'}"
            )
        maps[role] = {
            "role": role,
            "path": str(map_path),
            "size": map_path.stat().st_size,
            "sha256": _sha256(map_path),
        }
    expected_roles = {spec[0] for spec in MAP_SPECS}
    if set(maps) != expected_roles:
        raise ValueError("Physical capture manifest does not contain all eight maps.")

    enriched = json.loads(
        json.dumps(captured_contract, ensure_ascii=False, sort_keys=True)
    )
    enriched.pop("contract_sha256", None)
    enriched.update(
        {
            "capture_manifest": str(manifest_path),
            "capture_resolution": list(manifest.get("resolution") or []),
            "capture_maps": [maps[role] for role, *_ in MAP_SPECS],
        }
    )
    enriched["contract_sha256"] = hashlib.sha256(
        json.dumps(
            enriched,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "contract": enriched,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "maps": maps,
    }


def finalize_physical_capture_manifest(path, final_contract):
    """Rewrite a base bake receipt with the final attachment-bearing contract.

    The contract owns map fingerprints and the manifest path, but deliberately
    not the manifest file hash.  This breaks the otherwise impossible recursive
    hash dependency while allowing every map row, plan, and report to carry one
    final contract SHA.
    """
    raw_path = str(path or "").strip()
    manifest_path = Path(bpy.path.abspath(raw_path)).expanduser().absolute()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Physical capture manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Physical capture manifest is invalid JSON: {exc}") from exc
    if not isinstance(final_contract, dict):
        raise ValueError("Final physical capture contract must be a JSON object.")
    final_hash = str(final_contract.get("contract_sha256") or "")
    final_payload = {
        key: value
        for key, value in final_contract.items()
        if key != "contract_sha256"
    }
    actual_hash = hashlib.sha256(
        json.dumps(
            final_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not final_hash or final_hash != actual_hash:
        raise ValueError("Final physical capture contract hash is stale.")
    if not final_contract.get("attachment_pivots"):
        raise ValueError(
            "Final physical capture contract has no normalized attachment pivots."
        )
    if (
        final_contract.get("capture_manifest") != str(manifest_path)
        or manifest.get("frame") != final_contract.get("frame")
    ):
        raise ValueError(
            "Final physical capture contract targets another map manifest or frame."
        )
    contract_maps = {
        str(row.get("role") or ""): row
        for row in final_contract.get("capture_maps") or []
        if isinstance(row, dict)
    }
    rows = manifest.get("maps") or []
    if set(contract_maps) != {spec[0] for spec in MAP_SPECS}:
        raise ValueError("Final physical capture contract lacks all eight map hashes.")
    for row in rows:
        role = str(row.get("role") or "")
        expected = contract_maps.get(role)
        if (
            expected is None
            or str(row.get("path") or "") != str(expected.get("path") or "")
            or int(row.get("size", -1)) != int(expected.get("size", -2))
            or str(row.get("sha256") or "") != str(expected.get("sha256") or "")
        ):
            raise ValueError(f"Final physical capture map lineage drifted: {role}")
        row["physical_capture_contract_sha256"] = final_hash
    manifest["physical_capture_contract"] = json.loads(
        json.dumps(final_contract, ensure_ascii=False, sort_keys=True)
    )
    manifest["physical_capture_contract_sha256"] = final_hash
    manifest["normalization_status"] = "finalized"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "physical_capture_contract_sha256": final_hash,
        "map_count": len(rows),
        "maps": rows,
    }


def _camera_matrix(right, up, view, location):
    return Matrix(
        (
            (right.x, up.x, -view.x, location.x),
            (right.y, up.y, -view.y, location.y),
            (right.z, up.z, -view.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _material_sources(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None, None
    output = next(
        (
            node
            for node in material.node_tree.nodes
            if node.bl_idname == "ShaderNodeOutputMaterial"
            and getattr(node, "is_active_output", True)
        ),
        None,
    )
    if output is None or not output.inputs["Surface"].is_linked:
        return None, None
    shader = output.inputs["Surface"].links[0].from_node
    if shader.bl_idname != "ShaderNodeBsdfPrincipled":
        shader = next(
            (
                node
                for node in material.node_tree.nodes
                if node.bl_idname == "ShaderNodeBsdfPrincipled"
            ),
            None,
        )
    if shader is None:
        return None, None
    base = shader.inputs.get("Base Color")
    alpha = shader.inputs.get("Alpha")
    base_source = base.links[0].from_socket if base and base.is_linked else None
    alpha_source = alpha.links[0].from_socket if alpha and alpha.is_linked else None
    return base_source, alpha_source


def _capture_material(source, *, mode, constant, suffix):
    material = source.copy()
    material.name = f"{source.name}__AutoCapture_{suffix}"
    tree = material.node_tree
    if tree is None:
        material.use_nodes = True
        tree = material.node_tree
    base_source, alpha_source = _material_sources(material)
    output = next(
        (
            node
            for node in tree.nodes
            if node.bl_idname == "ShaderNodeOutputMaterial"
            and getattr(node, "is_active_output", True)
        ),
        None,
    )
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")
    for link in list(output.inputs["Surface"].links):
        tree.links.remove(link)
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    emission = tree.nodes.new("ShaderNodeEmission")
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    if mode == "SOURCE_COLOR" and base_source is not None:
        tree.links.new(base_source, emission.inputs["Color"])
    else:
        emission.inputs["Color"].default_value = constant or (0.8, 0.8, 0.8, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    if alpha_source is not None:
        tree.links.new(alpha_source, mix.inputs[0])
    else:
        mix.inputs[0].default_value = 1.0
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    return material


def _duplicate_sources(
    objects,
    collection,
    *,
    mode,
    constant,
    suffix,
    fit_matrix=None,
    depsgraph=None,
):
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    material_cache = {}
    copies = []
    for source in objects:
        evaluated = source.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        obj = bpy.data.objects.new(f"{source.name}__AutoCapture", mesh)
        obj.matrix_world = (
            fit_matrix @ evaluated.matrix_world
            if fit_matrix is not None
            else evaluated.matrix_world.copy()
        )
        obj.name = f"{source.name}__AutoCapture"
        collection.objects.link(obj)
        for index, material in enumerate(list(obj.data.materials)):
            if material is None:
                continue
            key = (material.name_full, mode, tuple(constant) if constant else None)
            capture = material_cache.get(key)
            if capture is None:
                capture = _capture_material(
                    material, mode=mode, constant=constant, suffix=suffix
                )
                material_cache[key] = capture
            obj.data.materials[index] = capture
        copies.append(obj)
    return copies, list(material_cache.values())


def _configure_scene(scene, frame, resolution):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "TARGA_RAW"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = bpy.data.worlds.new(f"{scene.name}_World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    scene.world = world
    camera_data = bpy.data.cameras.new(f"{scene.name}_CameraData")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(frame["height"])
    camera_data.clip_start = 0.0001
    distance = (
        Vector(frame["camera_location"]) - Vector(frame["center"])
    ).length
    camera_data.clip_end = max(distance * 4.0, 10.0)
    camera = bpy.data.objects.new(f"{scene.name}_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = _camera_matrix(
        Vector(frame["right"]),
        Vector(frame["up"]),
        Vector(frame["view_direction"]),
        Vector(frame["camera_location"]),
    )
    scene.camera = camera
    return world, camera


def bake_capture_maps(
    *,
    output_dir,
    prefix,
    source_collection=DEFAULT_SOURCE_COLLECTION,
    resolution=2048,
    padding_ratio=0.04,
    plane="XY",
    workflow_mode=WORKFLOW_LEGACY_CAMERA_UV,
    target_meters=0.1,
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sources, excluded_duplicates = renderable_source_selection(
        source_collection,
        depsgraph=depsgraph,
    )
    workflow_mode = str(workflow_mode or WORKFLOW_LEGACY_CAMERA_UV).upper()
    if workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
        contract = physical_capture_contract(
            source_collection=source_collection,
            scene=bpy.context.scene,
            plane=plane,
            padding_ratio=float(padding_ratio),
            target_meters=float(target_meters),
            depsgraph=depsgraph,
        )
        frame = contract["frame"]
        fit_matrix = Matrix(frame["fit_matrix_world"])
    elif workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
        contract = None
        frame = auto_capture_frame(
            sources,
            plane=plane,
            padding_ratio=float(padding_ratio),
            depsgraph=depsgraph,
        )
        fit_matrix = None
    else:
        raise ValueError(f"Unsupported capture workflow: {workflow_mode}")
    results = []
    for role, suffix, mode, constant in MAP_SPECS:
        scene = bpy.data.scenes.new(f"STCluster_AutoBake_{role}")
        collection = bpy.data.collections.new(f"STCluster_AutoBake_{role}_Sources")
        scene.collection.children.link(collection)
        copies = []
        materials = []
        world = None
        camera = None
        try:
            copies, materials = _duplicate_sources(
                sources,
                collection,
                mode=mode,
                constant=constant,
                suffix=role,
                fit_matrix=fit_matrix,
                depsgraph=depsgraph,
            )
            world, camera = _configure_scene(scene, frame, int(resolution))
            target = output_dir / f"{prefix}{suffix}.tga"
            scene.render.filepath = str(target)
            bpy.ops.render.render(scene=scene.name, write_still=True)
            if not target.is_file():
                raise RuntimeError(f"Blender did not write capture map: {target}")
            results.append(
                {
                    "role": role,
                    "path": str(target),
                    "size": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        finally:
            for obj in copies:
                mesh = obj.data
                if obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                if mesh is not None and mesh.name in bpy.data.meshes:
                    bpy.data.meshes.remove(mesh)
            for material in materials:
                if material.name in bpy.data.materials:
                    bpy.data.materials.remove(material, do_unlink=True)
            if camera is not None and camera.name in bpy.data.objects:
                camera_data = camera.data
                bpy.data.objects.remove(camera, do_unlink=True)
                if camera_data.name in bpy.data.cameras:
                    bpy.data.cameras.remove(camera_data)
            if scene.name in bpy.data.scenes:
                bpy.data.scenes.remove(scene)
            if world is not None and world.name in bpy.data.worlds:
                bpy.data.worlds.remove(world)
            if collection.name in bpy.data.collections:
                bpy.data.collections.remove(collection)
    manifest = {
        "kind": "speedtree_cluster_blender_auto_capture",
        "version": 2 if contract is not None else 1,
        "workflow_mode": workflow_mode,
        "blend": bpy.data.filepath,
        "source_collection": source_collection,
        "source_objects": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
            }
            for obj in sources
        ],
        "excluded_exact_duplicates": excluded_duplicates,
        "frame": frame,
        "physical_capture_contract": contract,
        "direct_uv_source": (
            DIRECT_CAPTURE_UV_SOURCE if contract is not None else None
        ),
        "resolution": [int(resolution), int(resolution)],
        "prefix": prefix,
        "maps": results,
    }
    manifest_path = output_dir / f"{prefix}_auto_capture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest
