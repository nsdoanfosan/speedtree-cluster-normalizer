import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


AUTO_COLLECTION = "Atlas_Auto_Capture"
LEGACY_SYNC_COLLECTION = "Atlas_Capture_Sync"
LEGACY_CAMERA_REFERENCE_COLLECTION = "Atlas_Camera_Reference"
CAPTURE_CONTRACT_KEY = "speedtree_cluster_auto_capture_contract"
CAPTURE_CONTRACT_HASH_KEY = "speedtree_cluster_auto_capture_contract_sha256"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan-collection", default="Atlas_Cluster_Cards")
    parser.add_argument("--material-name", required=True)
    parser.add_argument("--albedo", required=True)
    parser.add_argument("--opacity", required=True)
    parser.add_argument("--save-as", required=True)
    return parser.parse_args(argv)


def canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def matrix_from_rows(rows):
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Plan frame matrix is invalid")
    return Matrix(tuple(tuple(float(value) for value in row) for row in rows))


def remove_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    for obj in list(collection.all_objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)
            elif isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)
    bpy.data.collections.remove(collection)


def ensure_collection(name):
    remove_collection(name)
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    collection["speedtree_cluster_auto_capture"] = True
    return collection


def curve_object(collection, name, points, cyclic=False):
    curve = bpy.data.curves.new(name + "_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.002
    curve.bevel_resolution = 0
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for target, point in zip(spline.points, points):
        target.co = (*point, 1.0)
    spline.use_cyclic_u = bool(cyclic)
    obj = bpy.data.objects.new(name, curve)
    obj.hide_render = True
    collection.objects.link(obj)
    return obj


def cross_curve_object(collection, name, right, up, size):
    curve = bpy.data.curves.new(name + "_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.003
    curve.bevel_resolution = 0
    for points in ((-right * size, right * size), (-up * size, up * size)):
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        for target, point in zip(spline.points, points):
            target.co = (*point, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.hide_render = True
    collection.objects.link(obj)
    return obj


def camera_matrix(right, up, view, location):
    return Matrix(
        (
            (right.x, up.x, -view.x, location.x),
            (right.y, up.y, -view.y, location.y),
            (right.z, up.z, -view.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def build_capture_rig(frame, contract_hash):
    remove_collection(LEGACY_SYNC_COLLECTION)
    remove_collection(LEGACY_CAMERA_REFERENCE_COLLECTION)
    collection = ensure_collection(AUTO_COLLECTION)
    right = Vector(frame["right"])
    up = Vector(frame["up"])
    view = Vector(frame["view_direction"])
    center = Vector(frame["center"])
    location = Vector(frame["camera_location"])
    half_width = float(frame["width"]) * 0.5
    half_height = float(frame["height"]) * 0.5
    frame_points = [
        center - right * half_width - up * half_height,
        center + right * half_width - up * half_height,
        center + right * half_width + up * half_height,
        center - right * half_width + up * half_height,
    ]
    area = curve_object(
        collection,
        "STAutoCapture_Area",
        frame_points,
        cyclic=True,
    )
    origin_size = min(float(frame["width"]), float(frame["height"])) * 0.025
    origin = cross_curve_object(
        collection,
        "STAutoCapture_AttachmentOrigin",
        right,
        up,
        origin_size,
    )
    camera_data = bpy.data.cameras.new("STAutoCapture_CameraData")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(frame["height"])
    camera_data.clip_start = 0.0001
    camera_data.clip_end = max((location - center).length * 4.0, 10.0)
    camera = bpy.data.objects.new("STAutoCapture_Camera", camera_data)
    camera.matrix_world = camera_matrix(right, up, view, location)
    camera.hide_render = True
    collection.objects.link(camera)
    bpy.context.scene.camera = camera
    for obj, role in (
        (camera, "camera"),
        (area, "area"),
        (origin, "attachment_origin"),
    ):
        obj["speedtree_cluster_auto_capture_role"] = role
        obj[CAPTURE_CONTRACT_HASH_KEY] = contract_hash
    return collection, camera, area, origin


def load_atlas_preview(material_name, albedo, opacity):
    repo_parent = Path(__file__).resolve().parents[2]
    atlas_addons = repo_parent / "atlas-leaf-mesh-builder" / "addons"
    if str(atlas_addons) not in sys.path:
        sys.path.insert(0, str(atlas_addons))
    from atlas_leaf_mesh_builder.integration_api import (
        ensure_external_plan_preview_material,
    )

    result = ensure_external_plan_preview_material(
        material_name,
        str(albedo),
        str(opacity),
    )
    material = bpy.data.materials[result["material_name"]]
    return material, result


def remap_plan(plan, frame, material, contract_hash):
    stored = plan.get("speedtree_cluster_frame_world")
    if not stored:
        raise ValueError(f"Plan has no normalized frame metadata: {plan.name}")
    frame_world = matrix_from_rows(json.loads(stored))
    capture_right = Vector(frame["right"])
    capture_up = Vector(frame["up"])
    capture_center = Vector(frame["center"])
    width = float(frame["width"])
    height = float(frame["height"])
    if min(width, height) <= 0.0:
        raise ValueError("Auto capture frame has invalid dimensions")
    vertex_uvs = []
    for vertex in plan.data.vertices:
        world = frame_world @ vertex.co
        relative = world - capture_center
        vertex_uvs.append(
            (
                0.5 + float(relative.dot(capture_right)) / width,
                0.5 + float(relative.dot(capture_up)) / height,
            )
        )
    minimum = (
        min(value[0] for value in vertex_uvs),
        min(value[1] for value in vertex_uvs),
    )
    maximum = (
        max(value[0] for value in vertex_uvs),
        max(value[1] for value in vertex_uvs),
    )
    tolerance = 1.0e-5
    if (
        minimum[0] < -tolerance
        or minimum[1] < -tolerance
        or maximum[0] > 1.0 + tolerance
        or maximum[1] > 1.0 + tolerance
    ):
        raise ValueError(
            f"Plan exceeds the auto capture frame: {plan.name}; "
            f"uv_min={minimum}, uv_max={maximum}"
        )
    layer = plan.data.uv_layers.get("UVMap") or plan.data.uv_layers.new(name="UVMap")
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = plan.data.loops[loop_index].vertex_index
            layer.data[loop_index].uv = vertex_uvs[vertex_index]
    plan.data.materials.clear()
    plan.data.materials.append(material)
    plan[CAPTURE_CONTRACT_HASH_KEY] = contract_hash
    plan["speedtree_cluster_uv_policy"] = "direct_world_axis_capture_projection"
    plan["speedtree_cluster_auto_capture_uv_bounds"] = json.dumps(
        {"minimum": minimum, "maximum": maximum},
        ensure_ascii=False,
        sort_keys=True,
    )
    for key in (
        "speedtree_cluster_camera_uv_contract_sha256",
        "speedtree_cluster_camera_reference",
        "speedtree_cluster_uv_transfer",
    ):
        if key in plan:
            del plan[key]
        if key in plan.data:
            del plan.data[key]
    return {
        "name": plan.name,
        "vertices": len(plan.data.vertices),
        "polygons": len(plan.data.polygons),
        "uv_min": list(minimum),
        "uv_max": list(maximum),
    }


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    albedo = Path(args.albedo).expanduser().resolve()
    opacity = Path(args.opacity).expanduser().resolve()
    save_as = Path(args.save_as).expanduser().resolve()
    for label, path in (
        ("manifest", manifest_path),
        ("Color", albedo),
        ("Opacity", opacity),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = manifest.get("frame") or {}
    if frame.get("policy") != "world_axis_locked_auto_bounds":
        raise ValueError("Manifest is not a world-axis-locked auto capture")
    if float(frame.get("orthogonality_error", 1.0)) > 1.0e-9:
        raise ValueError("Auto capture frame is not orthogonal")
    handedness = (
        Vector(frame["right"]).cross(Vector(frame["up"])).dot(Vector(frame["normal"]))
    )
    if abs(handedness - 1.0) > 1.0e-9:
        raise ValueError("Auto capture frame is not right-handed")
    contract = {
        "kind": "speedtree_cluster_blender_auto_capture_contract",
        "version": 1,
        "capture_manifest": str(manifest_path),
        "capture_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "source_blend": manifest.get("blend"),
        "source_collection": manifest.get("source_collection"),
        "frame": frame,
        "resolution": manifest.get("resolution"),
        "color": str(albedo),
        "opacity": str(opacity),
    }
    contract_hash = canonical_sha256(contract)
    collection = bpy.data.collections.get(args.plan_collection)
    if collection is None:
        raise ValueError(f"Plan collection does not exist: {args.plan_collection}")
    plans = [
        obj
        for obj in collection.all_objects
        if obj.type == "MESH"
        and obj.get("speedtree_cluster_asset_role") == "speedtree_plan"
    ]
    if not plans:
        raise ValueError(f"No normalized plans exist in: {args.plan_collection}")
    material, preview = load_atlas_preview(
        args.material_name,
        albedo,
        opacity,
    )
    records = [
        remap_plan(plan, frame, material, contract_hash)
        for plan in sorted(plans, key=lambda obj: obj.name.casefold())
    ]
    build_capture_rig(frame, contract_hash)
    scene = bpy.context.scene
    scene[CAPTURE_CONTRACT_KEY] = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
    )
    scene[CAPTURE_CONTRACT_HASH_KEY] = contract_hash
    for key in (
        "speedtree_cluster_camera_uv_bundle",
        "speedtree_cluster_camera_uv_contract",
        "speedtree_cluster_camera_uv_contract_sha256",
    ):
        if key in scene:
            del scene[key]
    save_as.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_as))
    result = {
        "status": "ready",
        "saved_blend": str(save_as),
        "contract_sha256": contract_hash,
        "capture_plane": frame["plane"],
        "capture_rotation_degrees": frame["rotation_degrees"],
        "orthogonality_error": frame["orthogonality_error"],
        "handedness": handedness,
        "plans": records,
        "preview_material": preview,
        "capture_collection": AUTO_COLLECTION,
    }
    print("STCLUSTER_AUTO_PLAN=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
