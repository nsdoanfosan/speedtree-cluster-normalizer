from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atlas-albedo", required=True)
    parser.add_argument("--target-spm", required=True)
    parser.add_argument(
        "--source-object",
        default="SK_branch_elm_01_Codex_MergedSkinned_WeightsFixed",
    )
    return parser.parse_args(values)


def mesh_digest(obj):
    digest = hashlib.sha256()
    for vertex in obj.data.vertices:
        point = obj.matrix_world @ vertex.co
        digest.update(
            (",".join(f"{float(value):.9g}" for value in point) + "\n").encode("ascii")
        )
    return digest.hexdigest()


def source_snapshot(source):
    return {
        "parent": source.parent.name if source.parent else None,
        "matrix_world": [[float(value) for value in row] for row in source.matrix_world],
        "vertices": len(source.data.vertices),
        "polygons": len(source.data.polygons),
        "world_vertex_sha256": mesh_digest(source),
        "materials": [material.name if material else None for material in source.data.materials],
        "uv_layers": [layer.name for layer in source.data.uv_layers],
        "color_attributes": [attribute.name for attribute in source.data.color_attributes],
        "vertex_groups": [group.name for group in source.vertex_groups],
        "modifiers": [
            (modifier.name, modifier.type, modifier.object.name if getattr(modifier, "object", None) else None)
            for modifier in source.modifiers
        ],
    }


def main():
    args = parse_args()
    source_path = Path(args.source).resolve()
    save_to = Path(args.save_to).resolve()
    report_path = Path(args.report).resolve()
    if Path(bpy.data.filepath).resolve() != source_path:
        raise RuntimeError("Blender did not open the requested real source")
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    source = bpy.data.objects.get(args.source_object)
    if source is None or source.type != "MESH":
        raise RuntimeError(f"Real source mesh not found: {args.source_object}")
    bpy.context.view_layer.update()
    before = source_snapshot(source)
    source_names_before = sorted(obj.name for obj in bpy.data.objects)

    props = bpy.context.scene.speedtree_cluster_normalizer
    props.source_object = source
    props.plan_base_name = "branch_elm_01"
    props.skeletal_base_name = "SK_branch_elm_01"
    props.plan_collection = "Atlas_Branch_Plans"
    props.plan_material_name = "M_branch_elm_01"
    props.source_material_name = "M_branch_elm_01"
    props.source_material_id = 8
    props.plan_margin_ratio = 0.01
    props.replace_generated = True
    props.configure_send2ue = True
    props.isolate_send2ue_export = True
    props.source_reference_collection = "Cluster_Source_Reference"
    props.prepare_atlas_handoff = True
    props.atlas_albedo_path = str(Path(args.atlas_albedo).resolve())
    props.atlas_camera_spm = str(Path(args.atlas_albedo).resolve().with_suffix(".spm"))
    props.atlas_camera_name = "Dropped XY plane camera 2"
    props.atlas_target_spm = str(Path(args.target_spm).resolve())
    props.atlas_only_target = True
    props.atlas_mesh_scale = 1.0
    bpy.context.view_layer.objects.active = source
    source.select_set(True)
    result = bpy.ops.speedtree_cluster.build_normalized_assets()
    if result != {"FINISHED"}:
        raise RuntimeError(f"Cluster operator did not finish: {result}")
    report = json.loads(bpy.context.scene["speedtree_cluster_normalizer_last_report"])

    if source_snapshot(source) != before:
        raise RuntimeError("Real source object changed")
    if report["variant_count"] != 3:
        raise RuntimeError(f"Expected three real variants: {report}")
    if not report["send2ue"].get("available"):
        raise RuntimeError(f"Send to Unreal handoff was not configured: {report['send2ue']}")
    if not report.get("send2ue_export_isolated"):
        raise RuntimeError(f"Send to Unreal Export isolation was not enabled: {report}")
    if not report["atlas_handoff"].get("prepared"):
        raise RuntimeError(f"Atlas handoff was not prepared: {report['atlas_handoff']}")
    if report["atlas_handoff"].get("generated_material_name") != "M_branch_elm_01":
        raise RuntimeError(f"Atlas generated material contract mismatch: {report['atlas_handoff']}")
    if report["atlas_handoff"].get("adopt_source_material") is not True:
        raise RuntimeError(f"Atlas source material was not configured for adoption: {report['atlas_handoff']}")
    if report["atlas_handoff"].get("mesh_geometry_scale") != 1.0:
        raise RuntimeError(f"Atlas plan scale contract mismatch: {report['atlas_handoff']}")
    if sum(item["face_count"] for item in report["variants"]) != before["polygons"]:
        raise RuntimeError("Real split did not preserve the source polygon total")
    if report["mixed_face_count"] or report["unweighted_vertex_count"]:
        raise RuntimeError(f"Real source has ambiguous assignments: {report}")
    for index, item in enumerate(report["variants"], 1):
        expected_plan = f"branch_elm_01_{index:02d}"
        expected_asset = f"SK_branch_elm_01_{index:02d}"
        if item["plan"] != expected_plan or item["skeletal_asset"] != expected_asset:
            raise RuntimeError(f"Real naming mismatch: {item}")
        pivot = bpy.data.objects[item["pivot"]]
        armature = bpy.data.objects[item["armature"]]
        mesh = bpy.data.objects[item["mesh"]]
        plan = bpy.data.objects[item["plan"]]
        if pivot.type != "EMPTY" or armature.parent != pivot or mesh.parent != armature:
            raise RuntimeError(f"Real Send to Unreal hierarchy mismatch: {item}")
        if any(obj.matrix_world != Matrix.Identity(4) for obj in (pivot, armature, mesh, plan)):
            raise RuntimeError(f"Real output transform is not identity: {item}")
        if not item["plan_covers_projection"]:
            raise RuntimeError(f"Real plan coverage failed: {item}")
        if item["endpoint_policy"] != "matching_end_child":
            raise RuntimeError(f"Real Start/End relationship was not used: {item}")
        used_slots = {polygon.material_index for polygon in mesh.data.polygons}
        if used_slots != set(range(len(mesh.data.materials))):
            raise RuntimeError(f"Real part has unused material slots: {item}")
        transfer = item.get("plan_uv_transfer") or {}
        if transfer.get("policy") != "closed_loop_similarity_to_exact_camera_reference_boundary":
            raise RuntimeError(f"Real plan did not use the camera UV transfer: {item}")
        if float(transfer.get("normalized_rms", 1.0)) > float(transfer.get("max_normalized_rms", 0.0)):
            raise RuntimeError(f"Real camera boundary alignment exceeded its threshold: {item}")
        if transfer.get("orientation_preserving") is not True or float(
            transfer.get("determinant", -1.0)
        ) <= 0.0:
            raise RuntimeError(f"Real camera boundary alignment mirrored its UVs: {item}")
        if float(transfer.get("attachment_origin_error_normalized", 1.0)) > float(
            transfer.get("max_attachment_origin_error_normalized", 0.0)
        ):
            raise RuntimeError(f"Real camera boundary alignment lost its attachment: {item}")
        plan_uv = plan.data.uv_layers.get("UVMap")
        if plan_uv is None:
            raise RuntimeError(f"Real plan UVMap is missing: {item}")
        minimum_x = min(float(vertex.co.x) for vertex in plan.data.vertices)
        maximum_x = max(float(vertex.co.x) for vertex in plan.data.vertices)
        minimum_y = min(float(vertex.co.y) for vertex in plan.data.vertices)
        maximum_y = max(float(vertex.co.y) for vertex in plan.data.vertices)
        bbox_matches = []
        for polygon in plan.data.polygons:
            for loop_index in polygon.loop_indices:
                vertex = plan.data.vertices[plan.data.loops[loop_index].vertex_index]
                expected = (
                    (float(vertex.co.x) - minimum_x) / (maximum_x - minimum_x),
                    (float(vertex.co.y) - minimum_y) / (maximum_y - minimum_y),
                )
                actual = plan_uv.data[loop_index].uv
                bbox_matches.append(max(abs(float(actual[axis]) - expected[axis]) for axis in range(2)) < 1.0e-5)
        if all(bbox_matches):
            raise RuntimeError(f"Real plan fell back to legacy 0..1 bounding-box UVs: {item}")

    export_collection = bpy.data.collections.get("Export")
    expected_export_names = {
        item[key]
        for item in report["variants"]
        for key in ("pivot", "armature", "mesh")
    }
    actual_export_names = {obj.name for obj in export_collection.objects}
    if actual_export_names != expected_export_names:
        raise RuntimeError(
            f"Send to Unreal Export contains non-generated objects: {sorted(actual_export_names)}"
        )
    reference = bpy.data.collections.get("Cluster_Source_Reference")
    if reference is None or source.name not in reference.objects:
        raise RuntimeError("Real source was not preserved outside the Export collection")
    camera_reference = bpy.data.collections.get("Atlas_Camera_Reference")
    if camera_reference is None:
        raise RuntimeError("Exact camera reference collection is missing")
    contract = json.loads(bpy.context.scene["speedtree_cluster_camera_uv_contract"])
    if len(contract.get("planes") or []) != 3:
        raise RuntimeError("Persisted exact camera contract does not contain three planes")
    saw_unclamped_uv = False
    for plane in contract["planes"]:
        ref = bpy.data.objects.get("AtlasCameraRef_" + plane["name"])
        if ref is None or ref.name not in camera_reference.objects:
            raise RuntimeError(f"Exact camera reference object is missing: {plane['name']}")
        if ref.name in export_collection.objects or ref.name in bpy.data.collections[props.plan_collection].objects:
            raise RuntimeError(f"Camera reference leaked into an export/plan collection: {ref.name}")
        if len(ref.data.vertices) != len(plane["vertices"]) or len(ref.data.polygons) != len(plane["faces"]):
            raise RuntimeError(f"Camera reference topology count changed: {ref.name}")
        actual_uvs = [None] * len(ref.data.vertices)
        layer = ref.data.uv_layers.get("UVMap")
        for polygon in ref.data.polygons:
            for loop_index in polygon.loop_indices:
                vertex_index = ref.data.loops[loop_index].vertex_index
                actual_uvs[vertex_index] = tuple(float(value) for value in layer.data[loop_index].uv)
        for actual, expected in zip(actual_uvs, plane["uvs"]):
            if max(abs(actual[axis] - float(expected[axis])) for axis in range(2)) > 1.0e-6:
                raise RuntimeError(f"Exact camera UV payload was changed or clamped: {ref.name}")
            saw_unclamped_uv = saw_unclamped_uv or any(value < 0.0 or value > 1.0 for value in actual)
    if not saw_unclamped_uv:
        raise RuntimeError("Expected camera UV overscan was not preserved")
    material = bpy.data.materials.get("M_branch_elm_01")
    if material is None or not material.use_nodes:
        raise RuntimeError("Real Atlas preview material is missing")
    color_node = material.node_tree.nodes.get("SpeedTree Color")
    opacity_node = material.node_tree.nodes.get("SpeedTree Opacity")
    if color_node is None or opacity_node is None:
        raise RuntimeError("Real Atlas preview material does not use Color + Opacity")
    if color_node.extension != "CLIP" or opacity_node.extension != "CLIP":
        raise RuntimeError("Real Atlas preview textures can wrap outside the exact UV contract")

    save_to.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_to), check_existing=False)
    report["real_source_snapshot"] = before
    report["source_object_names_before"] = source_names_before
    report["saved_blend"] = str(save_to)
    report["speedtree_preconfiguration"] = report["atlas_handoff"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ATLAS_CLUSTER_REAL_SMOKE=" + str(report_path))


if __name__ == "__main__":
    main()
