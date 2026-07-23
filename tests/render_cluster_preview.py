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


def world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def arrange(objects, pivots=None):
    widths = [max(float(obj.dimensions.x), 1.0e-9) for obj in objects]
    spacing = max(widths) * 0.25
    cursor = 0.0
    for index, obj in enumerate(objects):
        local_min = min(float(corner[0]) for corner in obj.bound_box)
        target = pivots[index] if pivots else obj
        target.location.x = cursor - local_min
        cursor += widths[index] + spacing
    bpy.context.view_layer.update()


def camera_for(objects, direction, camera_name):
    minimum, maximum = world_bounds(objects)
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    extent = max(float(size.x), float(size.y), float(size.z))
    camera_data = bpy.data.cameras.new(camera_name + "_Data")
    camera = bpy.data.objects.new(camera_name, camera_data)
    bpy.context.scene.collection.objects.link(camera)
    direction = Vector(direction).normalized()
    camera.location = center + direction * max(extent * 3.0, 1.0)
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(float(size.x), float(size.y), float(size.z)) * 1.25
    bpy.context.scene.camera = camera
    return camera


def render(path, objects, direction, colors):
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in objects
    for obj, color in zip(objects, colors):
        obj.hide_render = False
        obj.color = color
    camera_for(objects, direction, path.stem + "_Camera")
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.curvature_ridge_factor = 1.5
    scene.display.shading.curvature_valley_factor = 1.0
    scene.render.engine = "BLENDER_WORKBENCH"
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = [bpy.data.objects[f"SK_branch_elm_01_{index:02d}_Mesh"] for index in range(1, 4)]
    pivots = [bpy.data.objects[f"SK_branch_elm_01_{index:02d}"] for index in range(1, 4)]
    plans = [bpy.data.objects[f"branch_elm_01_{index:02d}"] for index in range(1, 4)]
    arrange(parts, pivots)
    arrange(plans)
    colors = [
        (0.16, 0.55, 0.20, 1.0),
        (0.18, 0.42, 0.72, 1.0),
        (0.72, 0.34, 0.12, 1.0),
    ]
    render(output_dir / "normalized_3d_parts.png", parts, (1.1, -1.6, 1.35), colors)
    render(output_dir / "covering_plans.png", plans, (0.0, 0.0, 1.0), colors)
    print("STCLUSTER_PREVIEW=" + str(output_dir))


if __name__ == "__main__":
    main()
