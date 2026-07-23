from __future__ import annotations

import argparse
import hashlib
import json
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


def side_camera_bundle(source, pivot, normalization, output_path, margin_ratio):
    camera = {
        "name": "Synthetic YZ Side Camera",
        "right": [0.0, 1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "view_direction": [-1.0, 0.0, 0.0],
        "plane_normal": [1.0, 0.0, 0.0],
    }
    source_frame = normalization.whole_mesh_frame(source, pivot)
    frame = normalization.camera_aligned_frame(
        source,
        source_frame,
        range(len(source.data.vertices)),
        camera,
    )
    source_to_part = frame["matrix_world"].inverted_safe() @ source.matrix_world
    local_points = [source_to_part @ vertex.co for vertex in source.data.vertices]
    hull = normalization.expanded_hull(
        [(float(point.x), float(point.y)) for point in local_points]
        + [(0.0, 0.0)],
        margin_ratio,
    )
    minimum_x = min(point[0] for point in hull)
    maximum_x = max(point[0] for point in hull)
    minimum_y = min(point[1] for point in hull)
    maximum_y = max(point[1] for point in hull)
    camera_right = Vector(camera["right"])
    camera_up = Vector(camera["up"])
    reference_vertices = [
        list(camera_right * point[0] + camera_up * point[1]) for point in hull
    ]
    faces = [(0, offset, offset + 1) for offset in range(1, len(hull) - 1)]

    planes = []
    objects = []
    meshes = []
    material = bpy.data.materials.new("M_leaf_elm_side_01")
    for index in range(1, 4):
        region_min = (index - 1) / 3.0 - (0.001 if index == 1 else 0.0)
        region_max = index / 3.0 + (0.001 if index == 3 else 0.0)
        uvs = [
            [
                region_min
                + ((point[0] - minimum_x) / (maximum_x - minimum_x))
                * (region_max - region_min),
                (point[1] - minimum_y) / (maximum_y - minimum_y),
            ]
            for point in hull
        ]
        name = f"leaf_elm_side_01_{index:02d}"
        mesh = bpy.data.meshes.new(name + "_ReferenceMesh")
        mesh.from_pydata(reference_vertices, [], faces)
        mesh.update()
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                vertex_index = mesh.loops[loop_index].vertex_index
                uv_layer.data[loop_index].uv = uvs[vertex_index]
        mesh.materials.append(material)
        obj = bpy.data.objects.new(name, mesh)
        objects.append(obj)
        meshes.append(mesh)
        planes.append(
            {
                "source_mesh_id": index,
                "source_mesh_name": name + " Cutout",
                "name": name,
                "vertices": reference_vertices,
                "faces": [list(face) for face in faces],
                "uvs": uvs,
                "normals": [camera["plane_normal"] for _ in hull],
                "attachment": {
                    "source_plane_xy": [0.0, 0.0],
                    "normalized_local": [0.0, 0.0, 0.0],
                    "pivot_uv": [0.5, 0.0],
                },
                "topology_sha256": hashlib.sha256(
                    (name + "-topology").encode()
                ).hexdigest(),
                "uv_sha256": hashlib.sha256((name + "-uv").encode()).hexdigest(),
            }
        )
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
        "kind": "synthetic_side_uv_template_contract",
        "version": 1,
        "camera_spm": {"path": "synthetic_side_camera.spm", "sha256": "synthetic"},
        "tree_spm": {"path": "synthetic_tree.spm", "sha256": "synthetic"},
        "camera": camera,
        "material": {
            "id": 18,
            "name": "M_leaf_elm_side_01",
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
    from blender_cluster_normalization_smoke import create_scene

    source = create_scene()
    armature = normalization.find_source_armature(source)
    pivot = bpy.data.objects.new("SK_leaf_elm_side_01_SourcePivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.matrix_world = (
        Matrix.Translation(armature.matrix_world.translation)
        @ armature.matrix_world.to_quaternion().to_matrix().to_4x4()
    )
    armature_world = armature.matrix_world.copy()
    armature.parent = pivot
    armature.matrix_world = armature_world
    bundle = side_camera_bundle(source, pivot, normalization, args.output, 0.015)

    report = normalization.build_normalized_cluster_assets(
        bpy.context,
        source,
        "leaf_elm_side_01",
        "SK_leaf_elm_side_01",
        "Atlas_Leaf_Side_Plans",
        "M_leaf_elm_side_01",
        plan_margin_ratio=0.015,
        replace_generated=True,
        configure_send2ue=False,
        camera_uv_bundle=bundle,
        source_partition_mode="WHOLE_MESH",
        whole_mesh_pivot_object=pivot,
    )
    if report["card_count"] != 3 or report["prototype_count"] != 1:
        raise RuntimeError(f"Side card/prototype counts are wrong: {report}")
    if report["source_partition_mode"] != "WHOLE_MESH":
        raise RuntimeError(f"Whole-mesh partition was not retained: {report}")
    if {row["skeletal_asset"] for row in report["variants"]} != {
        "SK_leaf_elm_side_01_01"
    }:
        raise RuntimeError("Side cards do not share one SK prototype")
    expected_export = {
        "SK_leaf_elm_side_01_01",
        "SK_leaf_elm_side_01_01_Armature",
        "SK_leaf_elm_side_01_01_Mesh",
    }
    actual_export = {obj.name for obj in bpy.data.collections["Export"].objects}
    if actual_export != expected_export:
        raise RuntimeError(f"Side Export is not one prototype: {actual_export}")
    for row_index, row in enumerate(report["variants"]):
        plan = bpy.data.objects[row["plan"]]
        part = bpy.data.objects[row["mesh"]]
        frame_world = Matrix(row["frame_world"])
        world_geometry_error = max(
            (
                frame_world @ part.data.vertices[vertex_index].co
                - source.matrix_world @ source.data.vertices[vertex_index].co
            ).length
            for vertex_index in range(len(source.data.vertices))
        )
        if world_geometry_error > 1.0e-5:
            raise RuntimeError(f"Side camera frame changed 3D world geometry: {row}")
        basis = row["projection_basis"]
        if (
            basis.get("policy") != "camera_aligned_canonical_local_xy"
            or Vector(basis["right"]) != Vector((1.0, 0.0, 0.0))
            or Vector(basis["up"]) != Vector((0.0, 1.0, 0.0))
            or Vector(basis["normal"]) != Vector((0.0, 0.0, 1.0))
            or any(abs(float(vertex.co.z)) > 1.0e-6 for vertex in plan.data.vertices)
        ):
            raise RuntimeError(f"Side plan is not exact canonical local XY: {plan.name}")
        if row.get("frame_policy") != normalization.CAMERA_ALIGNED_FRAME_POLICY:
            raise RuntimeError(f"Side prototype did not share the camera-aligned frame: {row}")
        transfer = row["plan_uv_transfer"]
        actual_plan_uvs = _vertex_uvs(
            plan.data,
            expected_uvs=transfer["result_uvs"],
            label=plan.name,
        )
        _validate_external_camera_uv(
            plan,
            bundle["contract"]["planes"][row_index],
            bundle["contract"]["camera"],
            actual_plan_uvs,
            transfer,
        )
        attachment_index = int(transfer["attachment_vertex_index"])
        if plan.data.vertices[attachment_index].co.length > 1.0e-8:
            raise RuntimeError(f"Side plan attachment is not local origin: {plan.name}")
        if max(
            abs(
                transfer["result_uvs"][attachment_index][axis]
                - transfer["reference_pivot_uv"][axis]
            )
            for axis in range(2)
        ) > 1.0e-7:
            raise RuntimeError(f"Side plan attachment UV is not pinned: {plan.name}")
        coverage = row.get("plan_projection_coverage") or {}
        if coverage.get("covers_projection") is not True or coverage.get("outside_point_count") != 0:
            raise RuntimeError(f"Side plan measured coverage failed: {row}")
        if plan.get(normalization.COUNTERPART_KEY) != "SK_leaf_elm_side_01_01":
            raise RuntimeError(f"Side plan counterpart mismatch: {plan.name}")
    for index in range(1, 4):
        reference = bpy.data.objects[f"AtlasCameraRef_leaf_elm_side_01_{index:02d}"]
        if any(abs(float(vertex.co.x)) > 1.0e-6 for vertex in reference.data.vertices):
            raise RuntimeError(f"Synthetic side reference is not on YZ: {reference.name}")
    mapping = json.loads(bpy.context.scene[normalization.CARD_PROTOTYPE_MAP_KEY])
    if len({row["prototype_asset"] for row in mapping["cards"]}) != 1:
        raise RuntimeError("Persisted card/prototype mapping lost shared prototype lineage")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STCLUSTER_WHOLE_SIDE_SMOKE=" + str(output))


if __name__ == "__main__":
    main()
