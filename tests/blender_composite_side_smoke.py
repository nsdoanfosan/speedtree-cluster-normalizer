from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix, Vector


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args(values)


def create_composite_scene(box_along_segment):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    collection = bpy.data.collections.new("CompositeSource")
    bpy.context.scene.collection.children.link(collection)
    armature_data = bpy.data.armatures.new("Composite_ArmatureData")
    armature = bpy.data.objects.new("Composite_Armature", armature_data)
    collection.objects.link(armature)
    armature.matrix_world = (
        Matrix.Translation((3.0, -2.0, 5.0))
        @ Matrix.Rotation(math.radians(23.0), 4, "Z")
    )
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    specs = []
    edit_starts = {}
    for index in range(1, 13):
        angle = 2.0 * math.pi * (index - 1) / 12.0
        radius = 0.0 if index == 1 else 2.5 + 0.15 * (index % 3)
        start = Vector((radius * math.cos(angle), radius * math.sin(angle), 0.2 * (index % 2)))
        direction = Vector((0.8 * math.cos(angle), 4.0 + 0.15 * (index % 4), 0.8 * math.sin(angle)))
        end = start + direction
        name = f"Bone_{index}_Start"
        start_bone = armature_data.edit_bones.new(name)
        start_bone.head = start
        start_bone.tail = start + Vector((0.0, 0.5, 0.0))
        if index > 1:
            start_bone.parent = edit_starts[1]
        end_bone = armature_data.edit_bones.new(f"Bone_{index}_End")
        end_bone.head = end
        end_bone.tail = end + Vector((0.0, 0.5, 0.0))
        end_bone.parent = start_bone
        edit_starts[index] = start_bone
        specs.append((name, start, end))
    bpy.ops.object.mode_set(mode="OBJECT")

    vertices = []
    faces = []
    ranges = []
    for index, (_name, start, end) in enumerate(specs, 1):
        part_vertices, part_faces = box_along_segment(
            start,
            end,
            0.28 + 0.02 * (index % 3),
            0.08 + 0.01 * (index % 2),
        )
        offset = len(vertices)
        vertices.extend(part_vertices)
        faces.extend(tuple(value + offset for value in face) for face in part_faces)
        ranges.append(list(range(offset, offset + len(part_vertices))))
    mesh = bpy.data.meshes.new("Composite_ClusterMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="uv0")
    for loop_index, loop in enumerate(mesh.loops):
        point = mesh.vertices[loop.vertex_index].co
        uv_layer.data[loop_index].uv = (0.5 + point.x * 0.02, 0.5 + point.y * 0.02)
    mesh.materials.append(bpy.data.materials.new("M_composite_bark"))
    source = bpy.data.objects.new("Composite_Cluster_Source", mesh)
    collection.objects.link(source)
    source.parent = armature
    source.matrix_parent_inverse = Matrix.Identity(4)
    source.matrix_basis = Matrix.Identity(4)
    modifier = source.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    for (name, _start, _end), indices in zip(specs, ranges):
        group = source.vertex_groups.new(name=name)
        group.add(indices, 1.0, "REPLACE")
    bpy.context.view_layer.update()
    return source, armature


def point_key(point):
    return tuple(round(float(value), 3) for value in point)


def main():
    args = parse_args()
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    import speedtree_cluster_normalizer.normalization as normalization
    from speedtree_cluster_normalizer.delivery_validation import (
        _validate_external_camera_uv,
        _vertex_uvs,
    )

    test_dir = Path(__file__).resolve().parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    from blender_cluster_normalization_smoke import box_along_segment
    from blender_whole_mesh_side_smoke import side_camera_bundle

    source, armature = create_composite_scene(box_along_segment)
    pivot = bpy.data.objects.new("SK_leaf_composite_SourcePivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.matrix_world = armature.matrix_world.copy()
    armature_world = armature.matrix_world.copy()
    armature.parent = pivot
    armature.matrix_world = armature_world
    bundle = side_camera_bundle(source, pivot, normalization, args.output, 0.015)

    report = normalization.build_normalized_cluster_assets(
        bpy.context,
        source,
        "leaf_elm_side_01",
        "SK_leaf_composite_01",
        "Atlas_Composite_Side_Plans",
        "M_leaf_elm_side_01",
        plan_margin_ratio=0.015,
        replace_generated=True,
        configure_send2ue=False,
        camera_uv_bundle=bundle,
        source_partition_mode="COMPOSITE_PER_DEFORM_ROOT",
        whole_mesh_pivot_object=pivot,
    )
    if report["card_count"] != 3 or report["prototype_count"] != 12:
        raise RuntimeError("Composite side did not produce 3 cards / 12 subparts")
    if report["source_partition_mode"] != "COMPOSITE_PER_DEFORM_ROOT":
        raise RuntimeError("Composite partition mode was not retained")
    if len(report.get("composite_set_id") or "") != 64:
        raise RuntimeError("Composite set identity was not persisted")
    expected_assets = {
        f"SK_leaf_composite_01_{index:02d}" for index in range(1, 13)
    }
    actual_assets = {
        row["skeletal_asset"] for row in report["prototypes"]
    }
    if actual_assets != expected_assets:
        raise RuntimeError("Composite SK subpart names are not consecutive")
    export_names = {obj.name for obj in bpy.data.collections["Export"].objects}
    expected_export_names = {
        asset + suffix
        for asset in expected_assets
        for suffix in ("", "_Armature", "_Mesh")
    }
    if export_names != expected_export_names:
        raise RuntimeError("Send to Unreal Export is not exactly 12 subpart hierarchies")

    composite_frame = Matrix(report["composite_frame_world"])
    source_to_composite = composite_frame.inverted_safe() @ source.matrix_world
    expected_points = sorted(point_key(source_to_composite @ vertex.co) for vertex in source.data.vertices)
    reconstructed = []
    for prototype in report["prototypes"]:
        relative = Matrix(prototype["subpart_to_card_matrix"])
        mesh_obj = bpy.data.objects[prototype["mesh"]]
        reconstructed.extend(
            point_key(relative @ vertex.co) for vertex in mesh_obj.data.vertices
        )
    if sorted(reconstructed) != expected_points:
        missing = sorted(set(expected_points) - set(reconstructed))[:5]
        unexpected = sorted(set(reconstructed) - set(expected_points))[:5]
        raise RuntimeError(
            "Composite relative matrices do not reconstruct source geometry: "
            f"counts={len(expected_points)}/{len(reconstructed)} "
            f"missing={missing} unexpected={unexpected}"
        )
    if sum(row["face_count"] for row in report["prototypes"]) != len(source.data.polygons):
        raise RuntimeError("Composite partition lost or duplicated faces")

    mapping = json.loads(bpy.context.scene[normalization.CARD_PROTOTYPE_MAP_KEY])
    if (
        mapping.get("version") != 2
        or mapping.get("prototype_count") != 12
        or len(mapping.get("composite_parts") or []) != 12
    ):
        raise RuntimeError("Card/composite mapping contract is incomplete")
    for variant_index, variant in enumerate(report["variants"]):
        plan = bpy.data.objects[variant["plan"]]
        parts = json.loads(plan[normalization.COMPOSITE_PARTS_KEY])
        if len(parts) != 12 or variant["composite_set_id"] != report["composite_set_id"]:
            raise RuntimeError("Plan did not retain the shared composite set")
        if variant["plan_interior_vertices"] <= 0:
            raise RuntimeError("Composite plan has no internal fold/curl vertices")
        if variant["plan_uv_transfer"]["orientation_preserving"] is not True:
            raise RuntimeError("Composite plan UV transfer is mirrored")
        transfer = variant["plan_uv_transfer"]
        _validate_external_camera_uv(
            plan,
            bundle["contract"]["planes"][variant_index],
            bundle["contract"]["camera"],
            _vertex_uvs(
                plan.data,
                expected_uvs=transfer["result_uvs"],
                label=plan.name,
            ),
            transfer,
        )
        coverage = variant.get("plan_projection_coverage") or {}
        if (
            coverage.get("covers_projection") is not True
            or coverage.get("outside_point_count") != 0
            or coverage.get("projected_point_count") != len(source.data.vertices)
        ):
            raise RuntimeError("Composite plan coverage was not measured from all subparts")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STCLUSTER_COMPOSITE_SIDE_SMOKE=" + str(output))


if __name__ == "__main__":
    main()
