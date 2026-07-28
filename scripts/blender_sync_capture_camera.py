from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


SYNC_COLLECTION = "Atlas_Capture_Sync"
SYNC_ROLE_KEY = "speedtree_capture_sync_role"
SYNC_SOURCE_KEY = "speedtree_capture_sync_camera_spm"
SYNC_SOURCE_HASH_KEY = "speedtree_capture_sync_camera_spm_sha256"
SYNC_CAMERA_GUID_KEY = "speedtree_capture_sync_camera_guid"


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-object")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args(values)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(values, label):
    vector = Vector(tuple(float(value) for value in values))
    if vector.length <= 1.0e-12:
        raise RuntimeError(f"{label} is zero length")
    vector.normalize()
    return vector


def ensure_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    collection["speedtree_cluster_non_export"] = True
    collection["speedtree_capture_sync"] = True
    return collection


def ensure_material(name, color):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 6.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def make_curve_object(collection, name, points, material, bevel_depth, cyclic=False):
    curve = bpy.data.curves.new(name + "_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = float(bevel_depth)
    curve.bevel_resolution = 0
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for item, point in zip(spline.points, points):
        item.co = (*point, 1.0)
    spline.use_cyclic_u = bool(cyclic)
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    return obj


def camera_matrix(right, up, view, location):
    local_z = -view
    return Matrix(
        (
            (right.x, up.x, local_z.x, location.x),
            (right.y, up.y, local_z.y, location.y),
            (right.z, up.z, local_z.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def capture_plane_center(camera, location):
    center = location.copy()
    plane = str(camera.get("resolved_plane") or camera.get("plane") or "")
    if plane == "XY":
        center.z = 0.0
    elif plane == "XZ":
        center.y = 0.0
    elif plane == "YZ":
        center.x = 0.0
    else:
        raise RuntimeError(f"Unsupported capture plane: {plane}")
    return center


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(value) for value in obj.bound_box]
    minimum = [min(point[index] for point in corners) for index in range(3)]
    maximum = [max(point[index] for point in corners) for index in range(3)]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "size": [maximum[index] - minimum[index] for index in range(3)],
        "center": [(maximum[index] + minimum[index]) * 0.5 for index in range(3)],
    }


def source_objects(explicit_name=None):
    def has_geometry(obj):
        return (
            obj is not None
            and obj.type == "MESH"
            and getattr(obj, "data", None) is not None
            and len(obj.data.vertices) > 0
        )

    if explicit_name:
        obj = bpy.data.objects.get(explicit_name)
        if obj is None:
            raise RuntimeError(f"Explicit source object does not exist: {explicit_name}")
        if has_geometry(obj):
            return [obj]
        descendants = [
            candidate
            for candidate in obj.children_recursive
            if has_geometry(candidate)
        ]
        if descendants:
            return sorted(descendants, key=lambda candidate: candidate.name.casefold())
        raise RuntimeError(f"Explicit source object has no renderable mesh: {explicit_name}")
    preferred = [
        obj
        for obj in bpy.data.objects
        if has_geometry(obj) and obj.name.casefold().endswith("_source")
    ]
    if preferred:
        return sorted(preferred, key=lambda obj: obj.name.casefold())
    collection = bpy.data.collections.get("SpeedTree_Source")
    if collection is not None:
        meshes = [obj for obj in collection.all_objects if has_geometry(obj)]
        if meshes:
            return sorted(meshes, key=lambda obj: obj.name.casefold())
    raise RuntimeError("No SpeedTree source mesh was found for the camera comparison")


def store_sync_metadata(obj, camera_spm, camera, role):
    obj[SYNC_ROLE_KEY] = role
    obj[SYNC_SOURCE_KEY] = str(camera_spm)
    obj[SYNC_SOURCE_HASH_KEY] = sha256(camera_spm)
    obj[SYNC_CAMERA_GUID_KEY] = str(camera.get("guid") or "")
    obj["speedtree_capture_camera_name"] = str(camera.get("name") or "")
    obj["speedtree_capture_width"] = float(camera["width"])
    obj["speedtree_capture_height"] = float(camera["height"])


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    camera = manifest.get("camera") or {}
    camera_spm = Path((manifest.get("camera_spm") or {}).get("path", "")).resolve()
    if not camera_spm.is_file():
        raise RuntimeError(f"Camera SPM does not exist: {camera_spm}")
    expected_spm_hash = str((manifest.get("camera_spm") or {}).get("sha256") or "")
    if sha256(camera_spm) != expected_spm_hash:
        raise RuntimeError("Camera SPM changed after the normalization manifest was written")
    if camera.get("projection") != "orthographic":
        raise RuntimeError("Capture camera is not orthographic")

    width = float(camera["width"])
    height = float(camera["height"])
    if min(width, height) <= 0.0:
        raise RuntimeError("Capture camera dimensions are invalid")
    resolution = camera.get("resolved_export_resolution_pixels") or [
        int((manifest.get("material") or {}).get("width") or 0),
        int((manifest.get("material") or {}).get("height") or 0),
    ]
    resolution_x, resolution_y = [int(value) for value in resolution]
    if min(resolution_x, resolution_y) <= 0:
        raise RuntimeError("Capture camera resolution is invalid")
    dimension_aspect = width / height
    pixel_aspect = resolution_x / resolution_y
    if not math.isclose(dimension_aspect, pixel_aspect, rel_tol=0.0, abs_tol=1.0e-5):
        raise RuntimeError(
            "SpeedTree capture dimensions and pixel aspect do not match; "
            "implicit Blender stretching is forbidden"
        )

    right = normalized(camera["right"], "Camera right")
    up = normalized(camera["up"], "Camera up")
    view = normalized(camera["view_direction"], "Camera view direction")
    if max(abs(right.dot(up)), abs(right.dot(view)), abs(up.dot(view))) > 1.0e-5:
        raise RuntimeError("Capture camera basis is not orthogonal")
    location = Vector(tuple(float(value) for value in camera["translation"]))
    center = capture_plane_center(camera, location)

    collection = ensure_collection(SYNC_COLLECTION)
    stem = camera_spm.stem
    camera_data = bpy.data.cameras.new(f"STCapture_{stem}_CameraData")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = height
    camera_data.clip_start = max(float(camera.get("near") or 0.01), 0.0001)
    camera_data.clip_end = max(float(camera.get("far") or 100.0), camera_data.clip_start + 1.0)
    camera_data.passepartout_alpha = 0.75
    camera_obj = bpy.data.objects.new(f"STCapture_{stem}_Camera", camera_data)
    collection.objects.link(camera_obj)
    camera_obj.matrix_world = camera_matrix(right, up, view, location)
    store_sync_metadata(camera_obj, camera_spm, camera, "camera")

    frame_material = ensure_material("STCapture_Frame_Yellow", (1.0, 0.45, 0.0, 1.0))
    origin_material = ensure_material("STCapture_Origin_Cyan", (0.0, 0.8, 1.0, 1.0))
    half_width = width * 0.5 * 0.995
    half_height = height * 0.5 * 0.995
    frame_points = [
        center - right * half_width - up * half_height,
        center + right * half_width - up * half_height,
        center + right * half_width + up * half_height,
        center - right * half_width + up * half_height,
    ]
    bevel = max(width, height) * 0.0015
    frame_obj = make_curve_object(
        collection,
        f"STCapture_{stem}_Area",
        frame_points,
        frame_material,
        bevel,
        cyclic=True,
    )
    store_sync_metadata(frame_obj, camera_spm, camera, "capture_area")

    origin_size = min(width, height) * 0.025
    epsilon = view * -0.001
    origin_points = [
        Vector((0.0, 0.0, 0.0)) - right * origin_size + epsilon,
        Vector((0.0, 0.0, 0.0)) + right * origin_size + epsilon,
        Vector((0.0, 0.0, 0.0)) - up * origin_size + epsilon,
        Vector((0.0, 0.0, 0.0)) + up * origin_size + epsilon,
    ]
    origin_curve = bpy.data.curves.new(f"STCapture_{stem}_Origin_Curve", "CURVE")
    origin_curve.dimensions = "3D"
    origin_curve.bevel_depth = bevel * 1.5
    origin_curve.bevel_resolution = 0
    for start in (0, 2):
        spline = origin_curve.splines.new("POLY")
        spline.points.add(1)
        for item, point in zip(spline.points, origin_points[start : start + 2]):
            item.co = (*point, 1.0)
    origin_curve.materials.append(origin_material)
    origin_obj = bpy.data.objects.new(f"STCapture_{stem}_AttachmentOrigin", origin_curve)
    collection.objects.link(origin_obj)
    store_sync_metadata(origin_obj, camera_spm, camera, "attachment_origin")

    sources = source_objects(args.source_object)
    previous_hide_render = {obj.name: bool(obj.hide_render) for obj in bpy.data.objects}
    for obj in bpy.data.objects:
        obj.hide_render = obj not in sources and obj not in {frame_obj, origin_obj}

    scene = bpy.context.scene
    previous_camera = scene.camera
    previous_engine = scene.render.engine
    previous_resolution = (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
    )
    previous_filepath = scene.render.filepath
    previous_transparent = scene.render.film_transparent
    previous_world = scene.world

    world = bpy.data.worlds.new(f"STCapture_{stem}_World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.035,
        0.035,
        0.035,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world

    light_data = bpy.data.lights.new(f"STCapture_{stem}_KeyData", "AREA")
    light_data.energy = 900.0
    light_data.shape = "DISK"
    light_data.size = max(width, height)
    light_obj = bpy.data.objects.new(f"STCapture_{stem}_Key", light_data)
    collection.objects.link(light_obj)
    light_obj.matrix_world = camera_obj.matrix_world
    light_obj.location = location - view * 0.05
    light_obj.hide_render = False

    scene.camera = camera_obj
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:
        scene.render.engine = "BLENDER_WORKBENCH"
    preview_long_edge = 1024
    if resolution_x >= resolution_y:
        render_x = preview_long_edge
        render_y = max(1, round(preview_long_edge * resolution_y / resolution_x))
    else:
        render_y = preview_long_edge
        render_x = max(1, round(preview_long_edge * resolution_x / resolution_y))
    scene.render.resolution_x = render_x
    scene.render.resolution_y = render_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    mask_output_path = (
        Path(args.mask_output).expanduser().resolve()
        if args.mask_output
        else output_path.with_name(output_path.stem + "_mask.png")
    )
    frame_obj.hide_render = True
    origin_obj.hide_render = True
    scene.render.film_transparent = True
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(mask_output_path)
    bpy.ops.render.render(write_still=True)
    frame_obj.hide_render = False
    origin_obj.hide_render = False

    bpy.data.objects.remove(light_obj, do_unlink=True)
    bpy.data.lights.remove(light_data)
    scene.world = previous_world
    bpy.data.worlds.remove(world)
    for name, value in previous_hide_render.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = value
    scene.camera = camera_obj
    scene.render.engine = previous_engine
    (
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
    ) = previous_resolution
    scene.render.filepath = previous_filepath
    scene.render.film_transparent = previous_transparent

    report = {
        "status": "ready",
        "blend": bpy.data.filepath,
        "saved": bool(args.save),
        "manifest": str(manifest_path),
        "camera_spm": str(camera_spm),
        "camera_spm_sha256": expected_spm_hash,
        "camera_name": camera.get("name"),
        "camera_guid": camera.get("guid"),
        "camera_object": camera_obj.name,
        "capture_area_object": frame_obj.name,
        "attachment_origin_object": origin_obj.name,
        "camera_location": list(location),
        "camera_right": list(right),
        "camera_up": list(up),
        "camera_view_direction": list(view),
        "capture_plane": camera.get("resolved_plane"),
        "capture_center": list(center),
        "capture_dimensions": [width, height],
        "capture_resolution": [resolution_x, resolution_y],
        "preview_resolution": [render_x, render_y],
        "preview": str(output_path),
        "mask_preview": str(mask_output_path),
        "source_objects": [
            {
                "name": obj.name,
                "bounds": world_bounds(obj),
                "materials": [material.name for material in obj.data.materials],
            }
            for obj in sources
        ],
        "previous_scene_camera": previous_camera.name if previous_camera else None,
        "sync_collection": SYNC_COLLECTION,
        "sync_collection_outside_export": all(
            parent.name != "Export"
            for parent in bpy.data.collections
            if collection.name in {child.name for child in parent.children}
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print("STCLUSTER_CAPTURE_SYNC=" + str(report_path))


if __name__ == "__main__":
    main()
