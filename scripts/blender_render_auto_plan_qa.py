import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


QA_COLLECTION = "STAutoCapture_Plan_QA"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-collection", default="Atlas_Cluster_Cards")
    parser.add_argument("--resolution", type=int, default=1024)
    return parser.parse_args(argv)


def matrix_from_rows(rows):
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Plan frame matrix is invalid")
    return Matrix(tuple(tuple(float(value) for value in row) for row in rows))


def image_from_material(material, token):
    if material is None or not material.use_nodes:
        return None
    token = token.casefold()
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        identity = " ".join(
            (
                node.name,
                node.label,
                node.image.name,
                bpy.path.abspath(node.image.filepath or ""),
            )
        ).casefold()
        if token in identity:
            return node.image
    return None


def image_from_capture_contract(key):
    raw = bpy.context.scene.get("speedtree_cluster_auto_capture_contract")
    if not raw:
        return None
    try:
        path = Path(json.loads(raw).get(key, "")).expanduser().resolve()
    except (TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    if not path.is_file():
        return None
    return bpy.data.images.load(str(path), check_existing=True)


def make_emission_qa_material(source_material):
    material = bpy.data.materials.new(source_material.name + "_QA_Emission")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    nodes.clear()
    color = nodes.new("ShaderNodeTexImage")
    color.image = (
        image_from_material(source_material, "color")
        or image_from_capture_contract("color")
    )
    if color.image is None:
        raise RuntimeError(
            f"Preview material has no Color image: {source_material.name}"
        )
    opacity = nodes.new("ShaderNodeTexImage")
    opacity.image = (
        image_from_material(source_material, "opacity")
        or image_from_material(source_material, "alpha")
        or image_from_capture_contract("opacity")
    )
    if opacity.image is None:
        raise RuntimeError(
            f"Preview material has no Opacity image: {source_material.name}"
        )
    emission = nodes.new("ShaderNodeEmission")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(color.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(opacity.outputs["Color"], mix.inputs[0])
    material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def build_world_space_qa_plans(plans):
    previous = bpy.data.collections.get(QA_COLLECTION)
    if previous is not None:
        for obj in list(previous.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(previous)
    collection = bpy.data.collections.new(QA_COLLECTION)
    bpy.context.scene.collection.children.link(collection)
    duplicates = []
    qa_materials = {}
    for plan in plans:
        stored = plan.get("speedtree_cluster_frame_world")
        if not stored:
            raise RuntimeError(
                f"Plan has no normalized world frame for QA: {plan.name}"
            )
        mesh = plan.data.copy()
        mesh.materials.clear()
        source_material = plan.active_material
        if source_material is None:
            raise RuntimeError(f"Plan has no preview material: {plan.name}")
        qa_material = qa_materials.get(source_material.name)
        if qa_material is None:
            qa_material = make_emission_qa_material(source_material)
            qa_materials[source_material.name] = qa_material
        mesh.materials.append(qa_material)
        duplicate = bpy.data.objects.new(plan.name + "_QA", mesh)
        duplicate.matrix_world = matrix_from_rows(json.loads(stored))
        collection.objects.link(duplicate)
        duplicates.append(duplicate)
    return collection, duplicates, list(qa_materials.values())


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    collection = bpy.data.collections.get(args.plan_collection)
    camera = bpy.data.objects.get("STAutoCapture_Camera")
    if collection is None or camera is None:
        raise RuntimeError("Auto capture plans or camera are missing")
    plans = [
        obj
        for obj in collection.all_objects
        if obj.type == "MESH"
        and obj.get("speedtree_cluster_asset_role") == "speedtree_plan"
    ]
    if not plans:
        raise RuntimeError("No normalized plans are available for QA")
    qa_collection, qa_plans, qa_materials = build_world_space_qa_plans(plans)
    previous = {obj.name: bool(obj.hide_render) for obj in bpy.data.objects}
    for obj in bpy.data.objects:
        obj.hide_render = obj not in qa_plans
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(args.resolution)
    scene.render.resolution_y = int(args.resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("STAutoCapture_QA_World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.015, 0.015, 0.015, 1.0)
    background.inputs["Strength"].default_value = 0.3
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    for duplicate in qa_plans:
        mesh = duplicate.data
        bpy.data.objects.remove(duplicate, do_unlink=True)
        bpy.data.meshes.remove(mesh)
    for material in qa_materials:
        bpy.data.materials.remove(material)
    bpy.data.collections.remove(qa_collection)
    for name, value in previous.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = value
    print(
        "STCLUSTER_PLAN_QA="
        + json.dumps(
            {
                "output": str(output),
                "plans": [obj.name for obj in plans],
                "camera": camera.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
