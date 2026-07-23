from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-base", required=True)
    parser.add_argument("--count", required=True, type=int)
    return parser.parse_args(values)


def main():
    args = parse_args()
    plans = [bpy.data.objects[f"{args.plan_base}_{index:02d}"] for index in range(1, args.count + 1)]
    scene = bpy.context.scene
    for obj in bpy.data.objects:
        obj.hide_render = True
    audit_collection = bpy.data.collections.new("PlanTopologyAudit")
    scene.collection.children.link(audit_collection)
    spacing = 5.0
    for ordinal, plan in enumerate(plans):
        basis = json.loads(plan["speedtree_cluster_projection_basis"])
        right = Vector(basis["right"])
        up = Vector(basis["up"])
        projected = [
            (float(vertex.co.dot(right)), float(vertex.co.dot(up)))
            for vertex in plan.data.vertices
        ]
        minimum = [min(value[axis] for value in projected) for axis in range(2)]
        maximum = [max(value[axis] for value in projected) for axis in range(2)]
        center = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(2)]
        scale = 3.8 / max(maximum[0] - minimum[0], maximum[1] - minimum[1])
        vertices = [
            (
                (value[0] - center[0]) * scale + ordinal * spacing,
                (value[1] - center[1]) * scale,
                0.0,
            )
            for value in projected
        ]
        mesh = bpy.data.meshes.new(plan.name + "_TopologyAuditMesh")
        mesh.from_pydata(
            vertices,
            [],
            [tuple(int(index) for index in polygon.vertices) for polygon in plan.data.polygons],
        )
        mesh.update()
        obj = bpy.data.objects.new(plan.name + "_TopologyAudit", mesh)
        audit_collection.objects.link(obj)
        obj.hide_render = False
        obj.color = (0.46 + ordinal * 0.08, 0.70, 0.48, 1.0)
        edge_keys = set()
        for polygon in mesh.polygons:
            values = list(polygon.vertices)
            for offset, first in enumerate(values):
                second = values[(offset + 1) % len(values)]
                edge_keys.add(tuple(sorted((int(first), int(second)))))
        curve_data = bpy.data.curves.new(plan.name + "_TopologyEdgesData", "CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_depth = 0.009
        curve_data.bevel_resolution = 0
        for first, second in sorted(edge_keys):
            spline = curve_data.splines.new("POLY")
            spline.points.add(1)
            spline.points[0].co = (*vertices[first][:2], 0.02, 1.0)
            spline.points[1].co = (*vertices[second][:2], 0.02, 1.0)
        edge_object = bpy.data.objects.new(plan.name + "_TopologyEdges", curve_data)
        audit_collection.objects.link(edge_object)
        edge_object.hide_render = False
        edge_object.color = (0.025, 0.025, 0.025, 1.0)

    camera_data = bpy.data.cameras.new("PlanTopologyAuditCameraData")
    camera = bpy.data.objects.new("PlanTopologyAuditCamera", camera_data)
    audit_collection.objects.link(camera)
    camera.location = ((args.count - 1) * spacing * 0.5, 0.0, 10.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.rotation_euler[0] = 0.0
    camera.rotation_euler[1] = 0.0
    camera.rotation_euler[2] = 0.0
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Vector((0.0, 0.0, -1.0)).to_track_quat("-Z", "Y")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(5.5, args.count * spacing)
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = False
    if hasattr(scene.display.shading, "show_outline"):
        scene.display.shading.show_outline = True
    scene.display.shading.background_type = "VIEWPORT"
    scene.display.shading.background_color = (0.84, 0.84, 0.84)
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    print("STCLUSTER_PLAN_TOPOLOGY_AUDIT=" + str(output))


if __name__ == "__main__":
    main()
