from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(values)


def arrange(objects):
    widths = [max(float(obj.dimensions.x), 1.0e-6) for obj in objects]
    spacing = max(widths) * 0.2
    cursor = 0.0
    for obj, width in zip(objects, widths):
        minimum_x = min(float(corner[0]) for corner in obj.bound_box)
        obj.location.x = cursor - minimum_x
        obj.location.y = 0.0
        cursor += width + spacing
    bpy.context.view_layer.update()


def bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def render(path, objects):
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in objects
    arrange(objects)
    minimum, maximum = bounds(objects)
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    camera_data = bpy.data.cameras.new(path.stem + "_CameraData")
    camera = bpy.data.objects.new(path.stem + "_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + Vector((0.0, 0.0, max(float(size.x), float(size.y), 1.0) * 2.0))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(float(size.y) * 1.25, float(size.x) * 0.55, 1.0)

    light_data = bpy.data.lights.new(path.stem + "_AreaData", "AREA")
    light_data.energy = 900.0
    light_data.shape = "RECTANGLE"
    light_data.size = max(float(size.x), float(size.y), 1.0) * 2.0
    light = bpy.data.objects.new(path.stem + "_Area", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = camera.location * 0.5 + center * 0.5

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.filepath = str(path)
    scene.render.resolution_x = 1536
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new(path.stem + "_World")
    scene.world.color = (0.035, 0.035, 0.035)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = [bpy.data.objects[f"branch_elm_01_{index:02d}"] for index in range(1, 4)]
    references = [
        bpy.data.objects[f"AtlasCameraRef_branch_elm_01_{index:02d}"]
        for index in range(1, 4)
    ]
    render(output_dir / "camera_uv_transferred_plans.png", plans)
    render(output_dir / "exact_camera_reference_planes.png", references)
    print("STCLUSTER_CAMERA_UV_RENDER=" + str(output_dir))


if __name__ == "__main__":
    main()
