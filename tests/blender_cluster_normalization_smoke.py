from __future__ import annotations

import argparse
import hashlib
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


def stable_basis(direction):
    axis_y = direction.normalized()
    candidate = Vector((1.0, 0.0, 0.0))
    if abs(axis_y.dot(candidate)) > 0.9:
        candidate = Vector((0.0, 0.0, 1.0))
    axis_x = (candidate - axis_y * candidate.dot(axis_y)).normalized()
    axis_z = axis_x.cross(axis_y).normalized()
    return axis_x, axis_y, axis_z


def box_along_segment(start, end, width, depth):
    axis_x, axis_y, axis_z = stable_basis(end - start)
    vertices = []
    for center in (start, end):
        for x_sign, z_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append(
                center
                + axis_x * (0.5 * width * x_sign)
                + axis_z * (0.5 * depth * z_sign)
            )
    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    return vertices, faces


def snapshot_source(source):
    return {
        "parent": source.parent.name if source.parent else None,
        "matrix_world": [[float(value) for value in row] for row in source.matrix_world],
        "vertices": [[float(value) for value in vertex.co] for vertex in source.data.vertices],
        "faces": [list(polygon.vertices) for polygon in source.data.polygons],
        "materials": [material.name if material else None for material in source.data.materials],
        "uv_layers": [layer.name for layer in source.data.uv_layers],
        "color_attributes": [attribute.name for attribute in source.data.color_attributes],
    }


def create_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source_collection = bpy.data.collections.new("Source")
    bpy.context.scene.collection.children.link(source_collection)
    armature_data = bpy.data.armatures.new("Synthetic_ArmatureData")
    armature = bpy.data.objects.new("Synthetic_Armature", armature_data)
    source_collection.objects.link(armature)
    armature.matrix_world = (
        Matrix.Translation((7.0, -3.0, 4.0))
        @ Matrix.Rotation(math.radians(37.0), 4, "Z")
        @ Matrix.Diagonal((1.4, 0.8, 1.2, 1.0))
    )
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    specs = [
        ("Bone_1_Start", None, Vector((0.0, 0.0, 0.0)), Vector((2.0, 5.0, 1.0))),
        ("Bone_2_Start", "Bone_1_Start", Vector((4.0, -2.0, 1.0)), Vector((-1.0, 4.0, 5.0))),
        ("Bone_3_Start", "Bone_1_Start", Vector((-3.0, 1.0, 2.0)), Vector((5.0, 2.0, -2.0))),
    ]
    edit_bones = {}
    for name, parent_name, start, end in specs:
        start_bone = armature_data.edit_bones.new(name)
        start_bone.head = start
        start_bone.tail = start + Vector((0.0, 1.0, 0.0))
        if parent_name:
            start_bone.parent = edit_bones[parent_name]
        end_bone = armature_data.edit_bones.new(name[:-6] + "_End")
        end_bone.head = end
        end_bone.tail = end + Vector((0.0, 1.0, 0.0))
        end_bone.parent = start_bone
        edit_bones[name] = start_bone
    bpy.ops.object.mode_set(mode="OBJECT")

    vertices = []
    faces = []
    ranges = []
    dimensions = [(0.7, 0.11), (2.1, 0.38), (0.35, 0.07)]
    for (_, _, start, end), (width, depth) in zip(specs, dimensions):
        part_vertices, part_faces = box_along_segment(start, end, width, depth)
        offset = len(vertices)
        vertices.extend(part_vertices)
        faces.extend(tuple(index + offset for index in face) for face in part_faces)
        ranges.append(list(range(offset, offset + len(part_vertices))))
    mesh = bpy.data.meshes.new("Synthetic_ClusterMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="uv0")
    for loop_index, loop in enumerate(mesh.loops):
        point = mesh.vertices[loop.vertex_index].co
        uv_layer.data[loop_index].uv = (float(point.x) * 0.01 + 0.5, float(point.y) * 0.01 + 0.5)
    color = mesh.color_attributes.new(name="color", type="BYTE_COLOR", domain="CORNER")
    for item in color.data:
        item.color = (0.2, 0.4, 0.6, 1.0)
    for name in ("Synthetic_Bark_A", "Synthetic_Bark_B"):
        mesh.materials.append(bpy.data.materials.new(name))
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index % 2
    source = bpy.data.objects.new("Synthetic_Cluster_Source", mesh)
    source_collection.objects.link(source)
    source.parent = armature
    source.matrix_parent_inverse = Matrix.Identity(4)
    source.matrix_basis = Matrix.Identity(4)
    modifier = source.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    for (name, _, _, _), indices in zip(specs, ranges):
        group = source.vertex_groups.new(name=name)
        group.add(indices, 1.0, "REPLACE")
    bpy.context.view_layer.update()
    return source


def synthetic_camera_bundle(source, normalization, output_path, margin_ratio):
    armature = normalization.find_source_armature(source)
    weights = normalization._vertex_bone_weights(source, armature)
    assignments = normalization._face_group_assignments(source, weights)
    bones = sorted(
        (bone for bone in armature.data.bones if assignments["faces"].get(bone.name)),
        key=lambda bone: normalization._natural_key(bone.name),
    )
    planes = []
    objects = []
    meshes = []
    material = bpy.data.materials.new("M_branch_test")
    for index, bone in enumerate(bones, 1):
        endpoint, _policy = normalization._preferred_endpoint_bone(bone, set(assignments["faces"]))
        face_indices = assignments["faces"][bone.name]
        vertex_indices = sorted({
            vertex_index
            for face_index in face_indices
            for vertex_index in source.data.polygons[face_index].vertices
        })
        frame = normalization.canonical_frame(source, armature, bone, endpoint, vertex_indices)
        transform = frame["matrix_world"].inverted_safe() @ source.matrix_world
        local_points = [transform @ source.data.vertices[vertex_index].co for vertex_index in vertex_indices]
        hull = normalization.expanded_hull(
            [(float(point.x), float(point.y)) for point in local_points],
            margin_ratio,
        )
        minimum_x = min(point[0] for point in hull)
        maximum_x = max(point[0] for point in hull)
        minimum_y = min(point[1] for point in hull)
        maximum_y = max(point[1] for point in hull)
        region_min = (index - 1) / 3.0 - (0.001 if index == 1 else 0.0)
        region_max = index / 3.0 + (0.001 if index == 3 else 0.0)
        uvs = [
            [
                region_min + ((point[0] - minimum_x) / (maximum_x - minimum_x)) * (region_max - region_min),
                (point[1] - minimum_y) / (maximum_y - minimum_y),
            ]
            for point in hull
        ]
        faces = [(0, offset, offset + 1) for offset in range(1, len(hull) - 1)]
        name = f"branch_test_{index:02d}"
        mesh = bpy.data.meshes.new(name + "_ReferenceMesh")
        mesh.from_pydata([(x, y, 0.0) for x, y in hull], [], faces)
        mesh.update()
        layer = mesh.uv_layers.new(name="UVMap")
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                vertex_index = mesh.loops[loop_index].vertex_index
                layer.data[loop_index].uv = uvs[vertex_index]
        mesh.materials.append(material)
        obj = bpy.data.objects.new(name, mesh)
        objects.append(obj)
        meshes.append(mesh)
        planes.append({
            "source_mesh_id": index,
            "source_mesh_name": name + " Cutout",
            "name": name,
            "vertices": [[x, y, 0.0] for x, y in hull],
            "faces": [list(face) for face in faces],
            "uvs": uvs,
            "normals": [[0.0, 0.0, 1.0] for _ in hull],
            "attachment": {
                "source_plane_xy": [0.0, 0.0],
                "normalized_local": [0.0, 0.0, 0.0],
                "pivot_uv": [0.5, 0.0],
            },
            "topology_sha256": hashlib.sha256((name + "-topology").encode()).hexdigest(),
            "uv_sha256": hashlib.sha256((name + "-uv").encode()).hexdigest(),
        })
    reference_path = Path(output_path).resolve().with_suffix(".reference.blend")
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(reference_path), set(objects), fake_user=True)
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    if material.users == 0:
        bpy.data.materials.remove(material)
    contract = {
        "kind": "synthetic_uv_template_contract",
        "version": 1,
        "camera_spm": {"path": "synthetic_camera.spm", "sha256": "synthetic"},
        "tree_spm": {"path": "synthetic_tree.spm", "sha256": "synthetic"},
        "camera": {
            "name": "Synthetic Camera",
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "plane_normal": [0.0, 0.0, 1.0],
        },
        "material": {
            "id": 8,
            "name": "M_branch_test",
            "ordered_cutout_mesh_ids": [1, 2, 3],
            "maps": {},
        },
        "planes": planes,
        "validation": {"status": "ready", "strict_vertex_uv_topology": True},
    }
    return {
        "contract": contract,
        "contract_sha256": normalization._canonical_sha256(contract),
        "reference_blend": str(reference_path),
        "reference_blend_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "manifest_path": "synthetic_manifest.json",
        "manifest_sha256": "synthetic",
        "validation_path": "synthetic_validation.json",
        "validation_sha256": "synthetic",
        "reference_collection": "Atlas_Camera_Reference",
        "tree_file_changed_since_reference_build": False,
    }


def assert_attachment_phase_regression(normalization):
    plan_boundary = [
        (-2.2668259, 4.3656354),
        (-1.6007810, 2.7280698),
        (-0.0613291, -0.0231512),
        (-0.0562545, -0.0231954),
        (-0.0333654, -0.0225130),
        (0.0000712, -0.0213208),
        (0.0334393, -0.0199971),
        (0.0561449, -0.0189623),
        (0.0609790, -0.0185449),
        (1.5584426, 2.5376627),
        (1.6870189, 4.8453741),
        (0.5802212, 6.5503840),
        (0.2299248, 6.2995839),
    ]
    reference_boundary = [
        (-0.2896995, 0.3853699),
        (-0.2307093, 0.6979535),
        (-0.0896045, 0.9484427),
        (0.0028265, 0.9999959),
        (0.3008068, 0.6799390),
        (0.4346280, 0.4451649),
        (0.3274856, 0.2479859),
        (0.0105079, -0.0032768),
        (-0.0090492, 0.0001810),
        (-0.2964779, 0.1637978),
    ]
    minimum_x = min(point[0] for point in reference_boundary)
    maximum_x = max(point[0] for point in reference_boundary)
    minimum_y = min(point[1] for point in reference_boundary)
    maximum_y = max(point[1] for point in reference_boundary)
    reference_uvs = [
        [
            (point[0] - minimum_x) / (maximum_x - minimum_x),
            (point[1] - minimum_y) / (maximum_y - minimum_y),
        ]
        for point in reference_boundary
    ]
    reference_plane = {
        "vertices": [[x, y, 0.0] for x, y in reference_boundary],
        "faces": [
            [0, index, index + 1]
            for index in range(1, len(reference_boundary) - 1)
        ],
        "uvs": reference_uvs,
        "attachment": {"source_plane_xy": [0.0, 0.0]},
    }
    transferred, details = normalization._transfer_camera_boundary_uvs(
        plan_boundary,
        reference_plane,
        {
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        },
        plan_attachment_xy=(0.0, 0.0),
    )
    if details["candidate_selection_policy"] != (
        "attachment_origin_gate_then_outline_normalized_rms"
    ):
        raise RuntimeError("Attachment-first UV candidate policy is missing")
    if details["attachment_origin_error_normalized"] > 0.1:
        raise RuntimeError(f"Attachment phase regression escaped its origin: {details}")
    if details["determinant"] <= 0.0 or details["rotation"][0][0] < 0.9:
        raise RuntimeError(f"Attachment phase regression selected a flipped end: {details}")
    if not 0.18 < details["normalized_rms"] < details["max_normalized_rms"]:
        raise RuntimeError(
            "Attachment regression no longer proves that semantic origin wins over outline RMS"
        )
    if len(transferred) != len(plan_boundary):
        raise RuntimeError("Attachment phase regression lost exact camera UV interpolation")


def main():
    args = parse_args()
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    import speedtree_cluster_normalizer.normalization as normalization
    build_normalized_cluster_assets = normalization.build_normalized_cluster_assets
    point_in_convex_polygon = normalization.point_in_convex_polygon
    assert_attachment_phase_regression(normalization)

    source = create_scene()
    camera_uv_bundle = synthetic_camera_bundle(
        source,
        normalization,
        args.output,
        0.015,
    )
    before = snapshot_source(source)
    report = build_normalized_cluster_assets(
        bpy.context,
        source,
        "branch_test",
        "SK_branch_test",
        "Atlas_Branch_Test_Plans",
        "M_branch_test",
        plan_margin_ratio=0.015,
        replace_generated=True,
        configure_send2ue=False,
        camera_uv_bundle=camera_uv_bundle,
    )
    if report["variant_count"] != 3:
        raise RuntimeError(f"Expected three variants: {report}")
    if report["size_policy"] != "source_relative_only_no_absolute_dimensions":
        raise RuntimeError("Size-independent contract is missing")
    if snapshot_source(source) != before:
        raise RuntimeError("Source mesh or hierarchy changed")

    expected_face_counts = [12, 12, 12]
    for index, row in enumerate(report["variants"], 1):
        expected_plan = f"branch_test_{index:02d}"
        expected_asset = f"SK_branch_test_{index:02d}"
        if row["plan"] != expected_plan or row["skeletal_asset"] != expected_asset:
            raise RuntimeError(f"Naming contract mismatch: {row}")
        if row["face_count"] != expected_face_counts[index - 1]:
            raise RuntimeError(f"Face split mismatch: {row}")
        pivot = bpy.data.objects[row["pivot"]]
        armature = bpy.data.objects[row["armature"]]
        part = bpy.data.objects[row["mesh"]]
        plan = bpy.data.objects[row["plan"]]
        if row["plan_refinement_levels"] != 1 or row["plan_interior_vertices"] <= 0:
            raise RuntimeError(f"Plan internal refinement is missing: {row}")
        if row["plan_faces"] <= row["plan_boundary_vertices"] - 2:
            raise RuntimeError(f"Plan was not refined beyond boundary ear clipping: {row}")
        if pivot.type != "EMPTY" or armature.parent != pivot or part.parent != armature:
            raise RuntimeError(f"Send to Unreal hierarchy mismatch: {row}")
        if any(obj.matrix_world != Matrix.Identity(4) for obj in (pivot, armature, part, plan)):
            raise RuntimeError(f"Non-identity output transform: {row}")
        group_names = [group.name for group in part.vertex_groups]
        if group_names != ["part_root"]:
            raise RuntimeError(f"Unexpected part weights {group_names}: {row}")
        if [layer.name for layer in part.data.uv_layers] != ["uv0"]:
            raise RuntimeError(f"UV layer was not preserved: {row}")
        if [attribute.name for attribute in part.data.color_attributes] != ["color"]:
            raise RuntimeError(f"Color attribute was not preserved: {row}")
        basis = row["projection_basis"]
        right = Vector(basis["right"])
        up = Vector(basis["up"])
        normal = Vector(basis["normal"])
        if any(abs(float(vertex.co.dot(normal))) > 1.0e-6 for vertex in plan.data.vertices):
            raise RuntimeError(f"Plan is not flat in its camera basis: {row}")
        boundary_indices = normalization._ordered_boundary_indices(
            [tuple(int(value) for value in face.vertices) for face in plan.data.polygons]
        )
        polygon = [
            (
                float(plan.data.vertices[index].co.dot(right)),
                float(plan.data.vertices[index].co.dot(up)),
            )
            for index in boundary_indices
        ]
        outside = [
            vertex.index
            for vertex in part.data.vertices
            if not point_in_convex_polygon(
                (
                    float(vertex.co.dot(right)),
                    float(vertex.co.dot(up)),
                ),
                polygon,
                tolerance=1.0e-5,
            )
        ]
        if outside:
            raise RuntimeError(f"Plan coverage failed: {row['plan']} {outside[:10]}")
        if row["plan_covers_projection"] is not True:
            raise RuntimeError(f"Coverage contract missing: {row}")

    second = build_normalized_cluster_assets(
        bpy.context,
        source,
        "branch_test",
        "SK_branch_test",
        "Atlas_Branch_Test_Plans",
        "M_branch_test",
        plan_margin_ratio=0.015,
        replace_generated=True,
        configure_send2ue=False,
        camera_uv_bundle=camera_uv_bundle,
    )
    if second["variant_count"] != 3 or snapshot_source(source) != before:
        raise RuntimeError("Tagged rebuild failed or changed the source")

    protected = {
        name: bpy.data.objects[name].as_pointer()
        for name in (
            "branch_test_01",
            "branch_test_02",
            "branch_test_03",
            "SK_branch_test_01",
            "SK_branch_test_02",
            "SK_branch_test_03",
        )
    }
    original_build_plan = normalization._build_plan
    calls = {"count": 0}

    def fail_second_plan(*call_args, **call_kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("intentional transactional smoke failure")
        return original_build_plan(*call_args, **call_kwargs)

    normalization._build_plan = fail_second_plan
    try:
        try:
            build_normalized_cluster_assets(
                bpy.context,
                source,
                "branch_test",
                "SK_branch_test",
                "Atlas_Branch_Test_Plans",
                "M_branch_test",
                plan_margin_ratio=0.015,
                replace_generated=True,
                configure_send2ue=False,
                camera_uv_bundle=camera_uv_bundle,
            )
        except RuntimeError as exc:
            if "intentional transactional smoke failure" not in str(exc):
                raise
        else:
            raise RuntimeError("Intentional transactional failure did not occur")
    finally:
        normalization._build_plan = original_build_plan
    for name, pointer in protected.items():
        if bpy.data.objects.get(name) is None or bpy.data.objects[name].as_pointer() != pointer:
            raise RuntimeError(f"Transactional rollback did not preserve {name}")
    if any(obj.name.startswith("__STCLUSTER_BACKUP_") for obj in bpy.data.objects):
        raise RuntimeError("Transactional rollback left staged backup objects")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(second, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ATLAS_CLUSTER_SMOKE=" + str(output))


if __name__ == "__main__":
    main()
