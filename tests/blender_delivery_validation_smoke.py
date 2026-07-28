from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args(values)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_file_image(path, color):
    image = bpy.data.images.new(path.stem + "_Generated", width=4, height=4, alpha=True)
    image.pixels[:] = list(color) * 16
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return bpy.data.images.load(str(path), check_existing=False)


def make_mesh(name, vertices, faces, uvs, material):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            layer.data[loop_index].uv = uvs[vertex_index]
    mesh.materials.append(material)
    return mesh


def actual_vertex_uvs(mesh):
    layer = mesh.uv_layers["UVMap"]
    values = [None] * len(mesh.vertices)
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            values[vertex_index] = [
                float(value) for value in layer.data[loop_index].uv
            ]
    return values


def expect_failure(label, callback, phrase=None):
    try:
        callback()
    except ValueError as exc:
        if phrase is not None and phrase not in str(exc):
            raise RuntimeError(f"{label} failed for the wrong reason: {exc}") from exc
        return str(exc)
    raise RuntimeError(f"Fail-closed delivery guard accepted {label}")


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    asset_dir = output.parent / (output.stem + "_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)

    from speedtree_cluster_normalizer.atlas_handoff import (
        CAMERA_BUNDLE_KEY,
        CAMERA_CONTRACT_HASH_KEY,
        CAMERA_CONTRACT_KEY,
        CAMERA_REFERENCE_COLLECTION,
    )
    from speedtree_cluster_normalizer.delivery_validation import (
        EXPECTED_TRANSFER_POLICY,
        validate_camera_uv_delivery,
    )
    from speedtree_cluster_normalizer.attachment_contract import (
        SPM_STRUCTURAL_SEMANTIC_PROJECTION_VERSION,
        spm_structural_semantic_fingerprint,
    )
    from speedtree_cluster_normalizer.normalization import (
        ASSET_ROLE_KEY,
        CARD_PROTOTYPE_MAP_HASH_KEY,
        CARD_PROTOTYPE_MAP_KEY,
        CAMERA_REFERENCE_KEY,
        COMPOSITE_PARTS_KEY,
        COUNTERPART_KEY,
        PROJECTION_BASIS_KEY,
        PROJECTION_COVERAGE_KEY,
        PROTOTYPE_ASSET_KEY,
        PROTOTYPE_INDEX_KEY,
        PLAN_ROOT_LOCK_KEY,
        SOURCE_3D_CONTRACT_HASH_KEY,
        SOURCE_3D_CONTRACT_KEY,
        SOURCE_PARTITION_MODE_KEY,
        UV_TRANSFER_ATTACHMENT_POLICY,
        UV_TRANSFER_CANDIDATE_SELECTION_POLICY,
        UV_TRANSFER_KEY,
        UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR,
        UV_TRANSFER_MAX_NORMALIZED_RMS,
        XML_ATTACHMENT_KEY,
        _canonical_sha256,
        projection_coverage_2d,
    )

    camera_spm = asset_dir / "explicit_camera.spm"
    tree_spm = asset_dir / "explicit_tree.spm"
    source_spm = asset_dir / "SK_branch_test.spm"
    source_fbx = asset_dir / "SK_branch_test.fbx"
    source_xml = asset_dir / "SK_branch_test.xml"
    camera_spm.write_text("<SpeedTree camera='synthetic'/>", encoding="utf-8")
    tree_spm.write_text("<SpeedTree tree='synthetic'/>", encoding="utf-8")
    source_spm_text = """\
<SpeedTree>
  <Assets>
    <Material_v8 ID="1" Name="Raw Material">
      <CutoutMeshID>4</CutoutMeshID>
      <Width>16</Width><Height>16</Height>
      <Map Name="Color"><TexFilename>raw_source.tga</TexFilename></Map>
    </Material_v8>
  </Assets>
  <Preview>old-preview</Preview>
  <Generator Type="Branch">
    <Name>Synthetic Generator</Name><GUID>branch-guid</GUID>
    <Properties>
      <Property><Name>Physics:Bones</Name><Value>1</Value></Property>
      <Property><Name>Vertex Color:Red:Value</Name><Value>0</Value></Property>
      <Property><Name>Materials:Branch:0:Material</Name><Value>12</Value></Property>
    </Properties>
  </Generator>
  <Node Type="Branch">
    <GeneratorGUID>branch-guid</GeneratorGUID>
    <ParentGUID>tree-node</ParentGUID><GUID>branch-node</GUID>
    <Properties><Seed>10</Seed></Properties>
  </Node>
</SpeedTree>
"""
    source_spm.write_bytes(gzip.compress(source_spm_text.encode("utf-8")))
    source_fbx.write_bytes(b"synthetic-source-fbx")
    source_xml.write_text(
        '<SpeedTreeRaw Source="{}"><Bones>{}</Bones></SpeedTreeRaw>'.format(
            source_spm,
            "".join(
                '<Bone ID="{id}" ParentID="-1" StartX="{start}" StartY="0" '
                'StartZ="0" EndX="{start}" EndY="100" EndZ="0" '
                'Radius="10" Generator="Synthetic"/>'.format(
                    id=index - 1,
                    start=(index - 1) * 1000,
                )
                for index in range(1, 4)
            ),
        ),
        encoding="utf-8",
    )
    source_3d_contract = {
        "xml_path": str(source_xml),
        "xml_sha256": sha256(source_xml),
        "xml_mtime_ns": int(source_xml.stat().st_mtime_ns),
        "source_spm": str(source_spm),
        "source_spm_sha256": sha256(source_spm),
        "source_spm_semantic_projection_version":
            SPM_STRUCTURAL_SEMANTIC_PROJECTION_VERSION,
        "source_spm_semantic_fingerprint":
            spm_structural_semantic_fingerprint(source_spm),
        "source_fbx": str(source_fbx),
        "source_fbx_sha256": sha256(source_fbx),
        "scale": 100.0,
        "scale_scores": [{"scale": 100.0, "median_nearest_error": 0.0}],
        "root_ids": [0, 1, 2],
    }
    color_path = asset_dir / "M_branch_test_Color.png"
    opacity_path = asset_dir / "M_branch_test_Opacity.png"
    color_image = make_file_image(color_path, (0.2, 0.6, 0.1, 1.0))
    opacity_image = make_file_image(opacity_path, (1.0, 1.0, 1.0, 1.0))

    material_name = "M_branch_test"
    material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    color_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    color_node.name = "SpeedTree Color"
    color_node.image = color_image
    color_node.extension = "CLIP"
    opacity_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    opacity_node.name = "SpeedTree Opacity"
    opacity_node.image = opacity_image
    opacity_node.extension = "CLIP"

    plan_collection = bpy.data.collections.new("Atlas_Branch_Test_Plans")
    reference_collection = bpy.data.collections.new(CAMERA_REFERENCE_COLLECTION)
    export_collection = bpy.data.collections.new("Export")
    for collection in (plan_collection, reference_collection, export_collection):
        bpy.context.scene.collection.children.link(collection)

    planes = []
    plans = []
    references = []
    parts = []
    attachments = []
    root_locks = []
    plan_boundary_vertices = [
        (-1.0, -0.3, 0.0),
        (1.0, -0.3, 0.0),
        (1.0, 0.5, 0.0),
        (-1.0, 0.5, 0.0),
    ]
    reference_attachment = (0.25, -0.125)
    pivot_uv = (0.42, 0.18)
    reference_vertices = [
        (
            vertex[0] + reference_attachment[0],
            vertex[1] + reference_attachment[1],
            vertex[2],
        )
        for vertex in plan_boundary_vertices
    ]
    reference_faces = [(0, 1, 2), (0, 2, 3)]
    plan_vertices = plan_boundary_vertices + [(0.0, 0.0, 0.0)]
    plan_faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    for index in range(1, 4):
        plan_name = f"branch_test_{index:02d}"
        skeletal_name = f"SK_branch_test_{index:02d}"
        minimum_u = (index - 1) / 3.0
        maximum_u = index / 3.0
        if index == 1:
            minimum_u -= 0.01
        if index == 3:
            maximum_u += 0.01
        uvs = [
            (minimum_u, 0.05),
            (maximum_u, 0.05),
            (maximum_u, 0.95),
            (minimum_u, 0.95),
        ]
        plan_uvs = uvs + [pivot_uv]
        plane = {
            "source_mesh_id": index,
            "source_mesh_name": plan_name + " Cutout",
            "name": plan_name,
            "vertices": [list(value) for value in reference_vertices],
            "faces": [list(value) for value in reference_faces],
            "uvs": [list(value) for value in uvs],
            "normals": [[0.0, 0.0, 1.0]] * len(reference_vertices),
            "attachment": {
                "source_plane_xy": list(reference_attachment),
                "normalized_local": [0.0, 0.0, 0.0],
                "pivot_uv": list(pivot_uv),
            },
            "topology_sha256": hashlib.sha256((plan_name + " topology").encode()).hexdigest(),
            "uv_sha256": hashlib.sha256((plan_name + " uv").encode()).hexdigest(),
        }
        planes.append(plane)

        reference_mesh = make_mesh(
            plan_name + "_Reference",
            reference_vertices,
            reference_faces,
            uvs,
            material,
        )
        reference = bpy.data.objects.new("AtlasCameraRef_" + plan_name, reference_mesh)
        reference_collection.objects.link(reference)
        reference.matrix_world = Matrix.Identity(4)
        reference[CAMERA_REFERENCE_KEY] = plan_name
        references.append(reference)

        plan_mesh = make_mesh(
            plan_name,
            plan_vertices,
            plan_faces,
            plan_uvs,
            material,
        )
        plan = bpy.data.objects.new(plan_name, plan_mesh)
        plan_collection.objects.link(plan)
        plan.matrix_world = Matrix.Identity(4)
        plan[ASSET_ROLE_KEY] = "speedtree_plan"
        plan[COUNTERPART_KEY] = skeletal_name
        plan[CAMERA_REFERENCE_KEY] = reference.name
        plans.append(plan)

        pivot = bpy.data.objects.new(skeletal_name, None)
        armature_data = bpy.data.armatures.new(skeletal_name + "_ArmatureData")
        armature = bpy.data.objects.new(skeletal_name + "_Armature", armature_data)
        part_mesh = bpy.data.meshes.new(skeletal_name + "_MeshData")
        part_mesh.from_pydata(
            [(-0.8, -0.3, 0.0), (0.8, -0.3, 0.0), (0.0, 0.3, 0.0)],
            [],
            [(0, 1, 2)],
        )
        part_mesh.update()
        part = bpy.data.objects.new(skeletal_name + "_Mesh", part_mesh)
        for obj in (pivot, armature, part):
            export_collection.objects.link(obj)
            obj.matrix_world = Matrix.Identity(4)
        armature.parent = pivot
        armature.matrix_parent_inverse = Matrix.Identity(4)
        armature.matrix_basis = Matrix.Identity(4)
        part.parent = armature
        part.matrix_parent_inverse = Matrix.Identity(4)
        part.matrix_basis = Matrix.Identity(4)
        pivot[ASSET_ROLE_KEY] = "send2ue_pivot"
        armature[ASSET_ROLE_KEY] = "skeletal_armature"
        part[ASSET_ROLE_KEY] = "skeletal_mesh"
        xml_start_world = [(index - 1) * 10.0, 0.0, 0.0]
        attachment = {
            "xml_bone_id": index - 1,
            "xml_parent_id": -1,
            "xml_start_world": xml_start_world,
            "xml_end_world": [xml_start_world[0], 1.0, 0.0],
            "xml_radius_world": 0.1,
            "xml_generator": "Synthetic",
            "match_policy": "start_end_or_orphan_end_to_xml_root",
            "matched_start_bone": f"Bone_{index}_Start",
            "matched_endpoint_bone": f"Bone_{index}_End",
            "start_match_error": 0.0,
            "end_match_error": 0.0,
            "match_tolerance": 1.0e-6,
        }
        root_lock = {
            "policy": "xml_root_tangent_preserve_unexpanded_projection_support",
            "root_axis_xy": [0.0, 1.0],
            "unexpanded_root_support": -0.3,
            "unexpanded_distal_support": 0.3,
            "locked_root_support": -0.3,
            "maximum_root_margin_trim": 0.2,
            "attachment_xy": [0.0, 0.0],
            "attachment_inside_unexpanded_projection": True,
            "tolerance": math.hypot(1.6, 0.6) * 1.0e-7,
        }
        part[SOURCE_3D_CONTRACT_KEY] = json.dumps(source_3d_contract, sort_keys=True)
        part[XML_ATTACHMENT_KEY] = json.dumps(attachment, sort_keys=True)
        part["speedtree_cluster_frame_world"] = json.dumps(
            [
                [1.0, 0.0, 0.0, xml_start_world[0]],
                [0.0, 1.0, 0.0, xml_start_world[1]],
                [0.0, 0.0, 1.0, xml_start_world[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        parts.append(part)
        attachments.append(attachment)
        root_locks.append(root_lock)

    reference_blend = asset_dir / "exact_camera_reference.blend"
    bpy.data.libraries.write(str(reference_blend), set(references), fake_user=True)
    contract = {
        "kind": "synthetic_uv_template_contract",
        "version": 1,
        "camera_spm": {"path": str(camera_spm), "sha256": sha256(camera_spm)},
        "tree_spm": {"path": str(tree_spm), "sha256": sha256(tree_spm)},
        "camera": {
            "name": "Dropped XY plane camera 2",
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "plane_normal": [0.0, 0.0, 1.0],
        },
        "material": {
            "id": 8,
            "name": material_name,
            "ordered_cutout_mesh_ids": [1, 2, 3],
            "maps": {
                "Color": {"path": str(color_path), "stored": color_path.name, "size": [4, 4]},
                "Opacity": {"path": str(opacity_path), "stored": opacity_path.name, "size": [4, 4]},
            },
        },
        "planes": planes,
        "validation": {"status": "ready", "strict_vertex_uv_topology": True},
    }
    contract_hash = _canonical_sha256(contract)
    reference_hash = sha256(reference_blend)
    for plane, plan, reference, attachment, root_lock in zip(
        planes, plans, references, attachments, root_locks
    ):
        reference[CAMERA_CONTRACT_HASH_KEY] = contract_hash
        plan[CAMERA_CONTRACT_HASH_KEY] = contract_hash
        plan.data[CAMERA_CONTRACT_HASH_KEY] = contract_hash
        stored_uvs = actual_vertex_uvs(plan.data)
        projection_basis = {
            "policy": "camera_aligned_canonical_local_xy",
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "source_camera_right": [1.0, 0.0, 0.0],
            "source_camera_up": [0.0, 1.0, 0.0],
            "source_camera_normal": [0.0, 0.0, 1.0],
        }
        prototype_index = int(plane["source_mesh_id"])
        prototype_asset = f"SK_branch_test_{prototype_index:02d}"
        plan[PROTOTYPE_INDEX_KEY] = prototype_index
        plan[PROTOTYPE_ASSET_KEY] = prototype_asset
        plan[SOURCE_PARTITION_MODE_KEY] = "PER_DEFORM_ROOT"
        plan[SOURCE_3D_CONTRACT_KEY] = json.dumps(source_3d_contract, sort_keys=True)
        plan[XML_ATTACHMENT_KEY] = json.dumps(attachment, sort_keys=True)
        plan[PLAN_ROOT_LOCK_KEY] = json.dumps(root_lock, sort_keys=True)
        plan[PROJECTION_BASIS_KEY] = json.dumps(projection_basis, sort_keys=True)
        projection_coverage = projection_coverage_2d(
            [(-0.8, -0.3), (0.8, -0.3), (0.0, 0.3)],
            [(value[0], value[1]) for value in plan_boundary_vertices],
        )
        plan[PROJECTION_COVERAGE_KEY] = json.dumps(
            projection_coverage,
            sort_keys=True,
        )
        plan["speedtree_cluster_attachment_vertex_index"] = 4
        transfer = {
            "policy": EXPECTED_TRANSFER_POLICY,
            "normalized_rms": 0.01,
            "max_normalized_rms": UV_TRANSFER_MAX_NORMALIZED_RMS,
            "determinant": 1.0,
            "orientation_preserving": True,
            "scale": 1.0,
            "rotation": [[1.0, 0.0], [0.0, 1.0]],
            "translation": list(reference_attachment),
            "candidate_selection_policy": UV_TRANSFER_CANDIDATE_SELECTION_POLICY,
            "attachment_policy": UV_TRANSFER_ATTACHMENT_POLICY,
            "plan_attachment_xy": [0.0, 0.0],
            "reference_attachment_xy": list(reference_attachment),
            "reference_pivot_uv": list(pivot_uv),
            "mapped_attachment_xy": list(reference_attachment),
            "attachment_origin_error": 0.0,
            "attachment_origin_error_normalized": 0.0,
            "max_attachment_origin_error_normalized": (
                UV_TRANSFER_MAX_NORMALIZED_ATTACHMENT_ERROR
            ),
            "reference_extent_diagonal": math.hypot(2.0, 0.8),
            "contract_sha256": contract_hash,
            "reference_plane": plane["name"],
            "reference_object": reference.name,
            "source_mesh_id": plane["source_mesh_id"],
            "reference_topology_sha256": plane["topology_sha256"],
            "reference_uv_sha256": plane["uv_sha256"],
            "reference_blend": str(reference_blend),
            "reference_blend_sha256": reference_hash,
            "result_uvs": stored_uvs,
            "result_uv_sha256": _canonical_sha256(stored_uvs),
            "projection_basis": projection_basis,
            "prototype_index": prototype_index,
            "prototype_asset": prototype_asset,
            "source_partition_mode": "PER_DEFORM_ROOT",
            "plan_refinement_levels": 0,
            "attachment_vertex_index": 4,
            "attachment_vertex_uv": list(pivot_uv),
            "source_3d_contract": source_3d_contract,
            "xml_attachment": attachment,
            "plan_root_lock": root_lock,
        }
        plan[UV_TRANSFER_KEY] = json.dumps(transfer, sort_keys=True)

    manifest_path = asset_dir / "exact_camera_normalization_manifest.json"
    validation_path = asset_dir / "exact_camera_blender_validation.json"
    manifest = {
        "kind": "speedtree_cluster_card_camera_projection",
        "camera_spm": contract["camera_spm"],
        "tree_spm": contract["tree_spm"],
        "camera": contract["camera"],
        "material": contract["material"],
        "planes": contract["planes"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = {
        "status": "ready",
        "blend": {
            "path": str(reference_blend),
            "sha256": reference_hash,
            "size": reference_blend.stat().st_size,
        },
        "all_transform_identity": True,
        "all_planar": True,
        "all_uv_preserved": True,
        "all_fbx_roundtrip_preserved": True,
        "planes": [
            {
                "name": plane["name"],
                "source_mesh_id": plane["source_mesh_id"],
                "vertex_count": len(plane["vertices"]),
                "triangle_count": len(plane["faces"]),
                "uv_corner_preserved": True,
            }
            for plane in planes
        ],
    }
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    bundle = {
        "contract_sha256": contract_hash,
        "camera_spm": str(camera_spm),
        "tree_spm": str(tree_spm),
        "albedo_path": str(color_path),
        "opacity_path": str(opacity_path),
        "albedo_sha256": sha256(color_path),
        "opacity_sha256": sha256(opacity_path),
        "reference_blend": str(reference_blend),
        "reference_blend_sha256": reference_hash,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "validation_path": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "reference_collection": CAMERA_REFERENCE_COLLECTION,
    }
    scene = bpy.context.scene
    scene[CAMERA_CONTRACT_KEY] = json.dumps(contract, sort_keys=True)
    scene[CAMERA_CONTRACT_HASH_KEY] = contract_hash
    scene[CAMERA_BUNDLE_KEY] = json.dumps(bundle, sort_keys=True)
    scene[SOURCE_3D_CONTRACT_KEY] = json.dumps(source_3d_contract, sort_keys=True)
    scene[SOURCE_3D_CONTRACT_HASH_KEY] = _canonical_sha256(source_3d_contract)
    card_prototype_map = {
        "version": 1,
        "source_object": "Synthetic",
        "source_partition_mode": "PER_DEFORM_ROOT",
        "card_count": 3,
        "prototype_count": 3,
        "source_3d_contract_sha256": _canonical_sha256(source_3d_contract),
        "cards": [
            {
                "card_index": index,
                "plan": f"branch_test_{index:02d}",
                "source_mesh_id": index,
                "prototype_index": index,
                "prototype_asset": f"SK_branch_test_{index:02d}",
                "xml_bone_id": index - 1,
            }
            for index in range(1, 4)
        ],
    }
    scene[CARD_PROTOTYPE_MAP_KEY] = json.dumps(card_prototype_map, sort_keys=True)
    scene[CARD_PROTOTYPE_MAP_HASH_KEY] = _canonical_sha256(card_prototype_map)

    expected_export = {
        f"SK_branch_test_{index:02d}{suffix}"
        for index in range(1, 4)
        for suffix in ("", "_Armature", "_Mesh")
    }

    def validate(**overrides):
        values = {
            "plan_base": "branch_test",
            "expected_export_names": expected_export,
            "expected_camera_spm": camera_spm,
            "expected_camera_name": "Dropped XY plane camera 2",
            "expected_tree_spm": tree_spm,
            "expected_albedo_path": color_path,
            "expected_material_id": 8,
            "expected_card_count": 3,
            "expected_prototype_count": 3,
        }
        values.update(overrides)
        return validate_camera_uv_delivery(
            scene,
            plan_collection.name,
            material_name,
            **values,
        )

    passed = validate()
    if (
        planes[0]["attachment"]["source_plane_xy"] != list(reference_attachment)
        or planes[0]["attachment"]["pivot_uv"] != list(pivot_uv)
        or passed["planes"][0]["external_uv_validation"]["attachment_vertex_index"]
        != 4
    ):
        raise RuntimeError("Non-zero camera attachment contract was not independently validated")
    failures = {}

    original_source_spm_sha256 = source_3d_contract["source_spm_sha256"]
    cosmetic_source_spm_text = (
        source_spm_text.replace("Raw Material", "Canonical Material")
        .replace("raw_source.tga", r"D:\canonical\T_branch_color.tga")
        .replace("old-preview", "new-preview")
        .replace(
            "<Name>Vertex Color:Red:Value</Name><Value>0</Value>",
            "<Name>Vertex Color:Red:Value</Name><Value>1</Value>",
        )
        .replace(
            "<Name>Materials:Branch:0:Material</Name><Value>12</Value>",
            "<Name>Materials:Branch:0:Material</Name><Value>47</Value>",
        )
    )
    source_spm.write_bytes(gzip.compress(cosmetic_source_spm_text.encode("utf-8")))
    if (
        sha256(source_spm) == original_source_spm_sha256
        or spm_structural_semantic_fingerprint(source_spm)
        != source_3d_contract["source_spm_semantic_fingerprint"]
        or source_3d_contract["source_spm_sha256"] != original_source_spm_sha256
    ):
        raise RuntimeError(
            "Synthetic texture-only SPM rewrite did not isolate raw and semantic hashes"
        )
    validate()

    source_spm.write_bytes(
        gzip.compress(
            cosmetic_source_spm_text.replace(
                "<Name>Physics:Bones</Name><Value>1</Value>",
                "<Name>Physics:Bones</Name><Value>2</Value>",
            ).encode("utf-8")
        )
    )
    failures["source_spm_bone_change_rejected"] = expect_failure(
        "source SPM bone semantics changed",
        validate,
        "structural semantic fingerprint",
    )
    source_spm.write_bytes(gzip.compress(cosmetic_source_spm_text.encode("utf-8")))

    source_spm.write_bytes(
        gzip.compress(
            cosmetic_source_spm_text.replace(
                "<CutoutMeshID>4</CutoutMeshID>",
                "<CutoutMeshID>5</CutoutMeshID>",
            ).encode("utf-8")
        )
    )
    failures["source_spm_cutout_change_rejected"] = expect_failure(
        "source SPM cutout geometry changed",
        validate,
        "structural semantic fingerprint",
    )
    source_spm.write_bytes(gzip.compress(cosmetic_source_spm_text.encode("utf-8")))

    original_source_fbx_bytes = source_fbx.read_bytes()
    original_source_fbx_stat = source_fbx.stat()
    source_fbx.write_bytes(original_source_fbx_bytes + b"-changed")
    failures["source_fbx_change_rejected"] = expect_failure(
        "source FBX changed",
        validate,
        "Source 3D FBX hash",
    )
    source_fbx.write_bytes(original_source_fbx_bytes)
    os.utime(
        source_fbx,
        ns=(
            original_source_fbx_stat.st_atime_ns,
            original_source_fbx_stat.st_mtime_ns,
        ),
    )

    original_source_xml_text = source_xml.read_text(encoding="utf-8")
    original_source_xml_stat = source_xml.stat()
    source_xml.write_text(
        original_source_xml_text.replace(
            'Generator="Synthetic"',
            'Generator="Changed"',
        ),
        encoding="utf-8",
    )
    failures["source_xml_change_rejected"] = expect_failure(
        "source XML changed",
        validate,
        "Source 3D XML hash",
    )
    source_xml.write_text(original_source_xml_text, encoding="utf-8")
    os.utime(
        source_xml,
        ns=(
            original_source_xml_stat.st_atime_ns,
            original_source_xml_stat.st_mtime_ns,
        ),
    )

    plan = plans[0]
    part = bpy.data.objects["SK_branch_test_01_Mesh"]
    original_plan_attachment = json.loads(plan[XML_ATTACHMENT_KEY])
    original_part_attachment = json.loads(part[XML_ATTACHMENT_KEY])
    original_lineage_transfer = json.loads(plan[UV_TRANSFER_KEY])
    original_part_frame = json.loads(part["speedtree_cluster_frame_world"])
    shifted_attachment = json.loads(json.dumps(original_plan_attachment))
    delta = (1.0, 2.0, 3.0)
    for key in ("xml_start_world", "xml_end_world"):
        shifted_attachment[key] = [
            float(value) + delta[axis]
            for axis, value in enumerate(shifted_attachment[key])
        ]
    shifted_transfer = json.loads(json.dumps(original_lineage_transfer))
    shifted_transfer["xml_attachment"] = shifted_attachment
    shifted_frame = json.loads(json.dumps(original_part_frame))
    for axis in range(3):
        shifted_frame[axis][3] += delta[axis]
    plan[XML_ATTACHMENT_KEY] = json.dumps(shifted_attachment, sort_keys=True)
    part[XML_ATTACHMENT_KEY] = json.dumps(shifted_attachment, sort_keys=True)
    plan[UV_TRANSFER_KEY] = json.dumps(shifted_transfer, sort_keys=True)
    part["speedtree_cluster_frame_world"] = json.dumps(shifted_frame)
    failures["authoritative_xml_attachment_rejected"] = expect_failure(
        "coordinated attachment/frame offset with unchanged XML",
        validate,
        "XML attachment lineage mismatch",
    )
    plan[XML_ATTACHMENT_KEY] = json.dumps(original_plan_attachment, sort_keys=True)
    part[XML_ATTACHMENT_KEY] = json.dumps(original_part_attachment, sort_keys=True)
    plan[UV_TRANSFER_KEY] = json.dumps(original_lineage_transfer, sort_keys=True)
    part["speedtree_cluster_frame_world"] = json.dumps(original_part_frame)

    original_root_lock = json.loads(plan[PLAN_ROOT_LOCK_KEY])
    permissive_root_lock = dict(original_root_lock)
    permissive_root_lock["tolerance"] = 1.0e9
    permissive_transfer = json.loads(plan[UV_TRANSFER_KEY])
    permissive_transfer["plan_root_lock"] = permissive_root_lock
    plan[PLAN_ROOT_LOCK_KEY] = json.dumps(permissive_root_lock, sort_keys=True)
    plan[UV_TRANSFER_KEY] = json.dumps(permissive_transfer, sort_keys=True)
    failures["self_authored_root_tolerance_rejected"] = expect_failure(
        "self-authored permissive root-lock tolerance",
        validate,
        "root support lock",
    )
    plan[PLAN_ROOT_LOCK_KEY] = json.dumps(original_root_lock, sort_keys=True)
    plan[UV_TRANSFER_KEY] = json.dumps(original_lineage_transfer, sort_keys=True)

    original_uvs = actual_vertex_uvs(plan.data)
    layer = plan.data.uv_layers["UVMap"]
    legacy_uvs = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.5),
    )
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = plan.data.loops[loop_index].vertex_index
            layer.data[loop_index].uv = legacy_uvs[vertex_index]
    failures["legacy_bbox_uv_rejected"] = expect_failure(
        "legacy bbox UV plan",
        validate,
        "loop UV payload",
    )
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = plan.data.loops[loop_index].vertex_index
            layer.data[loop_index].uv = original_uvs[vertex_index]

    original_z = float(plan.data.vertices[0].co.z)
    plan.data.vertices[0].co.z = 0.1
    failures["tilted_plan_rejected"] = expect_failure(
        "plan outside canonical local XY",
        validate,
        "canonical local XY",
    )
    plan.data.vertices[0].co.z = original_z

    original_transfer = json.loads(plan[UV_TRANSFER_KEY])
    wrong_pivot_uv = [0.55, 0.05]
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            if plan.data.loops[loop_index].vertex_index == 4:
                layer.data[loop_index].uv = wrong_pivot_uv
    mutated_pivot_transfer = dict(original_transfer)
    mutated_result_uvs = actual_vertex_uvs(plan.data)
    mutated_pivot_transfer["result_uvs"] = mutated_result_uvs
    mutated_pivot_transfer["result_uv_sha256"] = _canonical_sha256(mutated_result_uvs)
    mutated_pivot_transfer["attachment_vertex_uv"] = mutated_result_uvs[4]
    plan[UV_TRANSFER_KEY] = json.dumps(mutated_pivot_transfer, sort_keys=True)
    failures["unpinned_pivot_uv_rejected"] = expect_failure(
        "plan attachment UV not pinned to camera pivot",
        validate,
        "not pinned",
    )
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            if plan.data.loops[loop_index].vertex_index == 4:
                layer.data[loop_index].uv = original_uvs[4]
    plan[UV_TRANSFER_KEY] = json.dumps(original_transfer, sort_keys=True)

    boundary_vertex_index = 1
    tampered_boundary_uv = [
        original_uvs[boundary_vertex_index][0] + 0.05,
        original_uvs[boundary_vertex_index][1] - 0.025,
    ]
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            if plan.data.loops[loop_index].vertex_index == boundary_vertex_index:
                layer.data[loop_index].uv = tampered_boundary_uv
    contract_tampered_transfer = dict(original_transfer)
    contract_tampered_uvs = actual_vertex_uvs(plan.data)
    contract_tampered_transfer["result_uvs"] = contract_tampered_uvs
    contract_tampered_transfer["result_uv_sha256"] = _canonical_sha256(
        contract_tampered_uvs
    )
    plan[UV_TRANSFER_KEY] = json.dumps(contract_tampered_transfer, sort_keys=True)
    failures["camera_contract_uv_tamper_rejected"] = expect_failure(
        "result UV and hash tampered together",
        validate,
        "external camera contract",
    )
    for polygon in plan.data.polygons:
        for loop_index in polygon.loop_indices:
            if plan.data.loops[loop_index].vertex_index == boundary_vertex_index:
                layer.data[loop_index].uv = original_uvs[boundary_vertex_index]
    plan[UV_TRANSFER_KEY] = json.dumps(original_transfer, sort_keys=True)

    stale_attachment_uv = dict(original_transfer)
    stale_attachment_uv["attachment_vertex_uv"] = [
        pivot_uv[0] + 0.1,
        pivot_uv[1],
    ]
    plan[UV_TRANSFER_KEY] = json.dumps(stale_attachment_uv, sort_keys=True)
    failures["stored_attachment_uv_rejected"] = expect_failure(
        "stored attachment vertex UV differs from actual pivot",
        validate,
        "not pinned",
    )
    plan[UV_TRANSFER_KEY] = json.dumps(original_transfer, sort_keys=True)

    valid_plan_mesh = plan.data
    non_triangle_mesh = make_mesh(
        "NonTrianglePlan",
        plan_vertices,
        [(0, 1, 2, 4), (0, 4, 2, 3)],
        [tuple(value) for value in original_uvs],
        material,
    )
    non_triangle_mesh[CAMERA_CONTRACT_HASH_KEY] = contract_hash
    plan.data = non_triangle_mesh
    failures["non_triangle_cdt_rejected"] = expect_failure(
        "non-triangle plan topology",
        validate,
        "not all triangles",
    )
    plan.data = valid_plan_mesh

    non_manifold_mesh = make_mesh(
        "NonManifoldPlan",
        plan_vertices,
        [(0, 1, 4), (1, 2, 4), (2, 0, 4), (2, 3, 4), (3, 0, 4)],
        [tuple(value) for value in original_uvs],
        material,
    )
    non_manifold_mesh[CAMERA_CONTRACT_HASH_KEY] = contract_hash
    plan.data = non_manifold_mesh
    failures["non_manifold_cdt_rejected"] = expect_failure(
        "non-manifold plan topology",
        validate,
        "non-manifold",
    )
    plan.data = valid_plan_mesh

    original_coverage = json.loads(plan[PROJECTION_COVERAGE_KEY])
    stale_coverage = dict(original_coverage)
    stale_coverage["outside_point_count"] = 1
    plan[PROJECTION_COVERAGE_KEY] = json.dumps(stale_coverage, sort_keys=True)
    failures["stale_projection_coverage_rejected"] = expect_failure(
        "stale stored projection coverage",
        validate,
        "coverage is stale",
    )
    plan[PROJECTION_COVERAGE_KEY] = json.dumps(original_coverage, sort_keys=True)

    mirrored_transfer = dict(original_transfer)
    mirrored_transfer["determinant"] = -1.0
    mirrored_transfer["orientation_preserving"] = False
    plan[UV_TRANSFER_KEY] = json.dumps(mirrored_transfer, sort_keys=True)
    failures["mirrored_uv_transfer_rejected"] = expect_failure(
        "mirrored UV transfer",
        validate,
        "mirrored",
    )
    plan[UV_TRANSFER_KEY] = json.dumps(original_transfer, sort_keys=True)

    opposite_end_transfer = dict(original_transfer)
    opposite_end_transfer["rotation"] = [[-1.0, 0.0], [0.0, -1.0]]
    opposite_end_transfer["translation"] = [0.0, 1.0]
    opposite_end_transfer["mapped_attachment_xy"] = [0.0, 1.0]
    opposite_end_transfer["attachment_origin_error"] = 1.0
    opposite_end_transfer["attachment_origin_error_normalized"] = (
        1.0 / math.hypot(2.0, 0.8)
    )
    plan[UV_TRANSFER_KEY] = json.dumps(opposite_end_transfer, sort_keys=True)
    failures["opposite_end_attachment_rejected"] = expect_failure(
        "orientation-preserving 180-degree attachment phase",
        validate,
        "attachment origin",
    )
    plan[UV_TRANSFER_KEY] = json.dumps(original_transfer, sort_keys=True)

    color_node.extension = "REPEAT"
    failures["texture_wrap_rejected"] = expect_failure(
        "wrapped preview texture",
        validate,
        "must use CLIP",
    )
    color_node.extension = "CLIP"

    export_collection.objects.link(plan)
    failures["export_leak_rejected"] = expect_failure(
        "plan leaked into Export",
        validate,
        "Export is not isolated",
    )
    export_collection.objects.unlink(plan)

    failures["implicit_camera_switch_rejected"] = expect_failure(
        "different explicit camera name",
        lambda: validate(expected_camera_name="Another Camera"),
        "Explicit camera name",
    )

    if validate()["contract_sha256"] != passed["contract_sha256"]:
        raise RuntimeError("Delivery validator did not recover after mutation smokes")

    shared_asset = "SK_branch_test_01"
    for index, plan in enumerate(plans, 1):
        plan[COUNTERPART_KEY] = shared_asset
        plan[PROTOTYPE_INDEX_KEY] = 1
        plan[PROTOTYPE_ASSET_KEY] = shared_asset
        plan[SOURCE_PARTITION_MODE_KEY] = "WHOLE_MESH"
        transfer = json.loads(plan[UV_TRANSFER_KEY])
        transfer["prototype_index"] = 1
        transfer["prototype_asset"] = shared_asset
        transfer["source_partition_mode"] = "WHOLE_MESH"
        plan[UV_TRANSFER_KEY] = json.dumps(transfer, sort_keys=True)
        card_prototype_map["cards"][index - 1]["prototype_index"] = 1
        card_prototype_map["cards"][index - 1]["prototype_asset"] = shared_asset
    card_prototype_map["source_partition_mode"] = "WHOLE_MESH"
    card_prototype_map["prototype_count"] = 1
    scene[CARD_PROTOTYPE_MAP_KEY] = json.dumps(card_prototype_map, sort_keys=True)
    scene[CARD_PROTOTYPE_MAP_HASH_KEY] = _canonical_sha256(card_prototype_map)
    for name in (
        "SK_branch_test_02",
        "SK_branch_test_02_Armature",
        "SK_branch_test_02_Mesh",
        "SK_branch_test_03",
        "SK_branch_test_03_Armature",
        "SK_branch_test_03_Mesh",
    ):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.name in export_collection.objects:
            export_collection.objects.unlink(obj)
    shared_export = {
        shared_asset,
        shared_asset + "_Armature",
        shared_asset + "_Mesh",
    }
    failures["whole_mesh_root_contract_rejected"] = expect_failure(
        "WHOLE_MESH card sharing under the one-XML-root-per-prototype contract",
        lambda: validate(
            expected_export_names=shared_export,
            expected_prototype_count=1,
        ),
        "Card/prototype XML root contract",
    )

    composite_parts = [
        {
            "subpart_index": index,
            "skeletal_asset_name": f"SK_branch_test_{index:02d}",
            "source_bone": f"Bone_{index}_Start",
            "endpoint_bone": f"Bone_{index}_End",
            "subpart_to_card_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "pivot_contract": "normalized_attachment_origin_0_0_0",
        }
        for index in range(1, 4)
    ]
    composite_set_id = _canonical_sha256(composite_parts)
    for index, plan in enumerate(plans, 1):
        plan[COUNTERPART_KEY] = "SK_branch_test_01"
        plan[PROTOTYPE_INDEX_KEY] = 1
        plan[PROTOTYPE_ASSET_KEY] = "SK_branch_test_01"
        plan[SOURCE_PARTITION_MODE_KEY] = "COMPOSITE_PER_DEFORM_ROOT"
        plan[COMPOSITE_PARTS_KEY] = json.dumps(composite_parts, sort_keys=True)
        plan["speedtree_cluster_composite_set_id"] = composite_set_id
        transfer = json.loads(plan[UV_TRANSFER_KEY])
        transfer["prototype_index"] = 1
        transfer["prototype_asset"] = "SK_branch_test_01"
        transfer["source_partition_mode"] = "COMPOSITE_PER_DEFORM_ROOT"
        plan[UV_TRANSFER_KEY] = json.dumps(transfer, sort_keys=True)
        composite_coverage = projection_coverage_2d(
            [(-0.8, -0.3), (0.8, -0.3), (0.0, 0.3)] * 3,
            [(value[0], value[1]) for value in plan_boundary_vertices],
        )
        plan[PROJECTION_COVERAGE_KEY] = json.dumps(
            composite_coverage,
            sort_keys=True,
        )
        card_prototype_map["cards"][index - 1].update({
            "prototype_index": 1,
            "prototype_asset": "SK_branch_test_01",
            "composite_set_id": composite_set_id,
            "composite_parts": composite_parts,
        })
    card_prototype_map.update({
        "version": 2,
        "source_partition_mode": "COMPOSITE_PER_DEFORM_ROOT",
        "prototype_count": 3,
        "composite_set_id": composite_set_id,
        "composite_parts": composite_parts,
    })
    scene[CARD_PROTOTYPE_MAP_KEY] = json.dumps(card_prototype_map, sort_keys=True)
    scene[CARD_PROTOTYPE_MAP_HASH_KEY] = _canonical_sha256(card_prototype_map)
    for name in (
        "SK_branch_test_02",
        "SK_branch_test_02_Armature",
        "SK_branch_test_02_Mesh",
        "SK_branch_test_03",
        "SK_branch_test_03_Armature",
        "SK_branch_test_03_Mesh",
    ):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.name not in export_collection.objects:
            export_collection.objects.link(obj)
    composite_export = {
        f"SK_branch_test_{index:02d}{suffix}"
        for index in range(1, 4)
        for suffix in ("", "_Armature", "_Mesh")
    }
    failures["composite_physical_root_rejected"] = expect_failure(
        "composite card frame under the physical XML root contract",
        lambda: validate(
            expected_export_names=composite_export,
            expected_prototype_count=3,
        ),
        "does not accept a composite card frame",
    )
    payload = {
        "status": "passed",
        "baseline": passed,
        "fail_closed": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STCLUSTER_DELIVERY_VALIDATION_SMOKE=" + str(output))


if __name__ == "__main__":
    main()
