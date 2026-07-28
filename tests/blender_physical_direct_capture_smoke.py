from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args(values)


REPO_ROOT = Path(__file__).resolve().parents[1]
ADDON_ROOT = REPO_ROOT / "addons"
TEST_ROOT = REPO_ROOT / "tests"
for path in (ADDON_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blender_cluster_normalization_smoke import (  # noqa: E402
    create_scene,
    snapshot_source,
    synthetic_source_xml,
)
from speedtree_cluster_normalizer import normalization  # noqa: E402
from speedtree_cluster_normalizer.atlas_handoff import (  # noqa: E402
    load_verified_unit_probe_contract,
)
from speedtree_cluster_normalizer.capture_bake import (  # noqa: E402
    DIRECT_CAPTURE_UV_SOURCE,
    MAP_SPECS,
    WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
    finalize_physical_capture_manifest,
    load_physical_capture_manifest,
    physical_capture_contract,
)
from speedtree_cluster_normalizer.delivery_validation import (  # noqa: E402
    _vertex_uvs,
    validate_cluster_delivery,
)


TOLERANCE = 2.0e-6


def assert_close(left, right, tolerance=TOLERANCE, label="value"):
    if abs(float(left) - float(right)) > tolerance:
        raise RuntimeError(f"{label} mismatch: {left} != {right}")


def projected_bounds(points, right, up):
    x = [point.dot(right) for point in points]
    y = [point.dot(up) for point in points]
    return min(x), max(x), min(y), max(y)


def make_capture_normal_attachments(source):
    armature = source.find_armature()
    if armature is None:
        raise RuntimeError("Synthetic source has no armature")
    world_offset = Vector((0.0, 0.0, 2.0))
    local_offset = (
        armature.matrix_world.to_3x3().inverted_safe() @ world_offset
    )
    local_tail = local_offset.normalized() * 0.1
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for ordinal in range(1, 4):
            start = armature.data.edit_bones[f"Bone_{ordinal}_Start"]
            end = armature.data.edit_bones[f"Bone_{ordinal}_End"]
            end.head = start.head + local_offset
            end.tail = end.head + local_tail
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        armature.select_set(False)


def make_long_dummy_root_attachments(source):
    """Reproduce a structural trunk root that begins well before visible geometry."""
    armature = source.find_armature()
    if armature is None:
        raise RuntimeError("Synthetic source has no armature")
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for ordinal in range(1, 4):
            points = [
                source.data.vertices[(ordinal - 1) * 8 + offset].co
                for offset in range(8)
            ]
            minimum = Vector(
                tuple(min(point[axis] for point in points) for axis in range(3))
            )
            maximum = Vector(
                tuple(max(point[axis] for point in points) for axis in range(3))
            )
            center = (minimum + maximum) * 0.5
            start = armature.data.edit_bones[f"Bone_{ordinal}_Start"]
            end = armature.data.edit_bones[f"Bone_{ordinal}_End"]
            start.head = (center.x, center.y, minimum.z - 10.0)
            start.tail = start.head + Vector((0.0, 0.0, 1.0))
            end.head = (center.x, center.y, maximum.z + 20.0)
            end.tail = end.head + Vector((0.0, 0.0, 1.0))
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        armature.select_set(False)


def plan_uvs(plan, row, frame):
    stored = _vertex_uvs(
        plan.data,
        expected_uvs=row["plan_uv_transfer"]["result_uvs"],
        label=plan.name,
    )
    frame_world = Matrix(row["frame_world"])
    center = Vector(frame["center"])
    right = Vector(frame["right"])
    up = Vector(frame["up"])
    width = float(frame["width"])
    height = float(frame["height"])
    expected = []
    for vertex in plan.data.vertices:
        world = frame_world @ vertex.co
        relative = world - center
        expected.append(
            (
                0.5 + relative.dot(right) / width,
                0.5 + relative.dot(up) / height,
            )
        )
    for index, (actual, target) in enumerate(zip(stored, expected)):
        for axis in range(2):
            assert_close(
                actual[axis],
                target[axis],
                tolerance=5.0e-6,
                label=f"{plan.name} UV {index}:{axis}",
            )
        if min(actual) < -TOLERANCE or max(actual) > 1.0 + TOLERANCE:
            raise RuntimeError(f"{plan.name} UV escaped direct capture: {actual}")


def assert_direct_build(report, source, before, plane):
    if snapshot_source(source) != before:
        raise RuntimeError("Physical direct capture changed source data or hierarchy")
    if (
        report["workflow_mode"] != WORKFLOW_PHYSICAL_DIRECT_CAPTURE
        or report["camera_dependency"] != "none"
        or report["direct_uv_source"] != DIRECT_CAPTURE_UV_SOURCE
        or report["variant_count"] != 3
        or report["prototype_count"] != 3
        or report["generator_size_policy"]
        != "preserve_user_authored_leaf_and_frond_dimensions"
        or not report["capture_manifest"]
        or len(report["capture_maps"]) != 8
    ):
        raise RuntimeError(f"Physical production report is incomplete: {report}")
    if (
        report["camera_reference_collection"] is not None
        or report["camera_uv_contract_sha256"] is not None
    ):
        raise RuntimeError("Physical direct capture retained a legacy camera dependency")

    contract = report["physical_capture_contract"]
    frame = contract["frame"]
    target = float(frame["target_blender_units"][0])
    assert_close(target, 0.1, tolerance=1.0e-9, label="physical target BU")
    assert_close(frame["target_meters"][0], 0.1, tolerance=1.0e-9)
    if frame["plane"] != plane:
        raise RuntimeError(f"Unexpected physical capture plane: {frame['plane']}")
    expected_rotation = 90.0 if plane == "YZ" else 0.0
    assert_close(frame["rotation_degrees"], expected_rotation, tolerance=1.0e-9)
    expected_fit = target / (
        max(frame["raw_content_width"], frame["raw_content_height"])
        * (1.0 + 2.0 * frame["padding_ratio"])
    )
    assert_close(frame["fit_scale"], expected_fit, tolerance=1.0e-9)
    assert_close(
        frame["content_width"] / frame["content_height"],
        frame["raw_content_width"] / frame["raw_content_height"],
        tolerance=1.0e-9,
        label="projected aspect ratio",
    )
    if max(frame["content_width"], frame["content_height"]) > target + TOLERANCE:
        raise RuntimeError("Physical fit exceeded the 10 cm frame")

    fitted_points = []
    for row in report["variants"]:
        pivot = bpy.data.objects[row["pivot"]]
        armature = bpy.data.objects[row["armature"]]
        part = bpy.data.objects[row["mesh"]]
        plan = bpy.data.objects[row["plan"]]
        if not all(
            obj.matrix_world == Matrix.Identity(4)
            for obj in (pivot, armature, part, plan)
        ):
            raise RuntimeError(f"Output transform is not identity: {row}")
        if row["camera_reference"] is not None:
            raise RuntimeError(f"Direct plan has a camera reference: {row}")
        if row["frame_policy"] != normalization.PHYSICAL_CAPTURE_ALIGNED_FRAME_POLICY:
            raise RuntimeError(f"Direct frame policy drifted: {row}")
        frame_world = Matrix(row["frame_world"])
        capture_normal_world = Vector(frame["normal"]).normalized()
        tangent = row["attachment_tangent_projection"]
        aligned_direction_world = Vector(
            tangent["aligned_capture_plane_world"]
        ).normalized()
        aligned_direction_local = (
            frame_world.to_3x3().inverted_safe()
            @ aligned_direction_world
        ).normalized()
        capture_normal_local = (
            frame_world.to_3x3().inverted_safe() @ capture_normal_world
        ).normalized()
        if (
            (aligned_direction_local - Vector((0.0, 1.0, 0.0))).length
            > TOLERANCE
            or (capture_normal_local - Vector((0.0, 0.0, 1.0))).length
            > TOLERANCE
        ):
            raise RuntimeError(
                f"Captured attachment axis is not local +Y/+Z after bake: {row}"
            )
        root_bone = armature.data.bones.get("part_root")
        if (
            root_bone is None
            or len(armature.data.bones) != 1
            or root_bone.head_local.length > TOLERANCE
            or (
                (root_bone.tail_local - root_bone.head_local).normalized()
                - Vector((0.0, 1.0, 0.0))
            ).length
            > TOLERANCE
        ):
            raise RuntimeError(f"Export bone disagrees with the authored axis: {row}")
        if row["plan_uv_transfer"]["policy"] != "direct_physical_capture_projection":
            raise RuntimeError(f"Direct UV policy drifted: {row}")
        assert_close(row["physical_fit_scale"], frame["fit_scale"], tolerance=1.0e-9)
        attachment_index = int(
            row["plan_uv_transfer"]["attachment_vertex_index"]
        )
        if plan.data.vertices[attachment_index].co.length > 1.0e-8:
            raise RuntimeError(f"Plan attachment is not local origin: {row['plan']}")
        if Vector(row["capture_attachment"]["normalized_local"]).length > 1.0e-9:
            raise RuntimeError(f"Pair attachment is not local origin: {row['plan']}")
        projection_basis = row["projection_basis"]
        projection_normal = Vector(projection_basis["normal"]).normalized()
        if (projection_normal - Vector((0.0, 0.0, 1.0))).length > TOLERANCE:
            raise RuntimeError(
                f"Plan normal is not canonical local +Z: {row}"
            )
        if max(
            abs(float(vertex.co.dot(projection_normal)))
            for vertex in plan.data.vertices
        ) > TOLERANCE:
            raise RuntimeError(
                f"Plan does not remain on the Blender capture plane: {row}"
            )
        plan_uvs(plan, row, frame)
        fitted_points.extend(frame_world @ vertex.co for vertex in part.data.vertices)

    right = Vector(frame["right"])
    up = Vector(frame["up"])
    x0, x1, y0, y1 = projected_bounds(fitted_points, right, up)
    if x1 - x0 > target + TOLERANCE or y1 - y0 > target + TOLERANCE:
        raise RuntimeError("Combined normalized prototypes exceed the 10 cm frame")

    attachments = contract["attachment_pivots"]
    if len(attachments) != 3:
        raise RuntimeError("Physical contract did not record all attachment pivots")
    fit_scale = float(frame["fit_scale"])
    for left in range(len(attachments)):
        for right_index in range(left + 1, len(attachments)):
            source_distance = (
                Vector(attachments[left]["source_world"])
                - Vector(attachments[right_index]["source_world"])
            ).length
            fitted_distance = (
                Vector(attachments[left]["fitted_capture_world"])
                - Vector(attachments[right_index]["fitted_capture_world"])
            ).length
            assert_close(
                fitted_distance,
                source_distance * fit_scale,
                tolerance=5.0e-6,
                label="multipart attachment spacing",
            )


def build(source, xml_path, plane, output_path):
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    contract = physical_capture_contract(
        source_collection="Source",
        scene=bpy.context.scene,
        plane=plane,
        padding_ratio=0.04,
        target_meters=0.1,
    )
    manifest_path = Path(output_path).with_name(
        f"synthetic_{plane.casefold()}_capture_manifest.json"
    )
    map_rows = []
    for role, suffix, _mode, _constant in MAP_SPECS:
        map_path = manifest_path.with_name(
            f"synthetic_{plane.casefold()}{suffix}.tga"
        )
        map_path.write_bytes(f"{plane}:{role}:synthetic-map".encode("ascii"))
        map_rows.append(
            {
                "role": role,
                "path": str(map_path),
                "size": map_path.stat().st_size,
                "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "speedtree_cluster_blender_auto_capture",
                "version": 2,
                "workflow_mode": WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
                "blend": bpy.data.filepath,
                "source_collection": "Source",
                "source_objects": contract["source_objects"],
                "excluded_exact_duplicates": contract[
                    "excluded_exact_duplicates"
                ],
                "frame": contract["frame"],
                "physical_capture_contract": contract,
                "direct_uv_source": DIRECT_CAPTURE_UV_SOURCE,
                "resolution": [512, 512],
                "prefix": f"synthetic_{plane.casefold()}",
                "maps": map_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    capture = load_physical_capture_manifest(manifest_path, contract)
    contract = capture["contract"]
    report = normalization.build_normalized_cluster_assets(
        bpy.context,
        source,
        f"direct_{plane.casefold()}",
        f"SK_direct_{plane.casefold()}",
        f"Atlas_Direct_{plane}_Plans",
        f"M_direct_{plane.casefold()}",
        plan_margin_ratio=0.01,
        replace_generated=True,
        configure_send2ue=False,
        isolate_send2ue_export=False,
        camera_uv_bundle=None,
        source_partition_mode="PER_CONNECTED_DEFORM_CLUSTER",
        source_xml_path=xml_path,
        workflow_mode=WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
        physical_capture_contract=contract,
    )
    finalized = finalize_physical_capture_manifest(
        manifest_path,
        report["physical_capture_contract"],
    )
    if (
        finalized["physical_capture_contract_sha256"]
        != report["physical_capture_contract_sha256"]
        or any(
            row.get("physical_capture_contract_sha256")
            != report["physical_capture_contract_sha256"]
            for row in finalized["maps"]
        )
    ):
        raise RuntimeError("Maps and normalized plans do not share one final contract")
    report["capture_manifest_sha256"] = finalized["manifest_sha256"]
    report["capture_maps"] = finalized["maps"]
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        persisted_manifest.get("physical_capture_contract_sha256")
        != report["physical_capture_contract_sha256"]
        or persisted_manifest["physical_capture_contract"].get("contract_sha256")
        != report["physical_capture_contract_sha256"]
    ):
        raise RuntimeError("Final capture manifest did not embed the plan contract")
    delivery = validate_cluster_delivery(
        bpy.context.scene,
        f"Atlas_Direct_{plane}_Plans",
        f"M_direct_{plane.casefold()}",
        plan_base=f"direct_{plane.casefold()}",
        expected_card_count=3,
        expected_prototype_count=3,
    )
    if (
        delivery["delivery_mode"] != "physical_direct_capture"
        or delivery["physical_capture_contract_sha256"]
        != report["physical_capture_contract_sha256"]
    ):
        raise RuntimeError("Physical delivery validator did not accept final contract")
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return contract, report


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    unit_receipt_path = output.with_name("synthetic_verified_unit_probe.json")
    unit_receipt_path.write_text(
        json.dumps(
            {
                "kind": "speedtree_fbx_spm_unit_probe",
                "version": 1,
                "status": "verified",
                "physical_target_meters": 0.1,
                "blender_units": {
                    "system": "METRIC",
                    "scale_length": 1.0,
                    "target_blender_units": 0.1,
                },
                "selected": {
                    "mesh_geometry_scale": 0.01,
                    "mesh_asset_scale": 1.0,
                    "generator_scale": 1.0,
                    "scale_location": "FBX_GEOMETRY",
                    "effective_scale": 0.01,
                },
                "generator_results": [
                    {
                        "generator_type": "Frond",
                        "status": "verified",
                        "same_unit_contract": True,
                    },
                    {
                        "generator_type": "Leaf Mesh",
                        "status": "verified",
                        "same_unit_contract": True,
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    verified_unit = load_verified_unit_probe_contract(
        str(unit_receipt_path),
        0.1,
    )
    if (
        verified_unit["scale_location"] != "FBX_GEOMETRY"
        or verified_unit["generator_scale"] != 1.0
        or not verified_unit["contract_sha256"]
    ):
        raise RuntimeError("Verified unit-probe receipt was not propagated")
    try:
        load_verified_unit_probe_contract(str(unit_receipt_path), 0.2)
    except ValueError as exc:
        if "does not match" not in str(exc):
            raise
    else:
        raise RuntimeError("Mismatched unit-probe target was accepted")

    source = create_scene()
    xml_path = synthetic_source_xml(source, output, "SK_physical_direct_test")
    before = snapshot_source(source)
    input_contract, first = build(source, xml_path, "XY", output)
    assert_direct_build(first, source, before, "XY")
    first_names = sorted(
        obj.name
        for obj in bpy.data.objects
        if obj.get("speedtree_cluster_generated")
    )
    _, second = build(source, xml_path, "XY", output)
    assert_direct_build(second, source, before, "XY")
    second_names = sorted(
        obj.name
        for obj in bpy.data.objects
        if obj.get("speedtree_cluster_generated")
    )
    if (
        first_names != second_names
        or first["physical_capture_contract_sha256"]
        != second["physical_capture_contract_sha256"]
        or any(name.endswith(".001") for name in second_names)
    ):
        raise RuntimeError("Physical direct capture rebuild is not idempotent")

    try:
        normalization.build_normalized_cluster_assets(
            bpy.context,
            source,
            "direct_reject_camera",
            "SK_direct_reject_camera",
            "Atlas_Direct_Reject_Camera",
            "M_direct_reject_camera",
            configure_send2ue=False,
            isolate_send2ue_export=False,
            camera_uv_bundle={"forbidden": True},
            source_partition_mode="PER_CONNECTED_DEFORM_CLUSTER",
            source_xml_path=xml_path,
            workflow_mode=WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
            physical_capture_contract=input_contract,
        )
    except ValueError as exc:
        if (
            "must not read" not in str(exc).lower()
            and "must not receive" not in str(exc).lower()
        ):
            raise
    else:
        raise RuntimeError("Physical direct capture accepted a legacy camera bundle")

    source = create_scene()
    make_capture_normal_attachments(source)
    xml_path = synthetic_source_xml(
        source,
        output,
        "SK_physical_capture_normal_test",
    )
    before = snapshot_source(source)
    _, capture_normal = build(source, xml_path, "XY", output)
    assert_direct_build(capture_normal, source, before, "XY")
    direction_policies = {
        row["attachment_tangent_projection"]["direction_policy"]
        for row in capture_normal["variants"]
    }
    if direction_policies != {
        normalization.PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK
    }:
        raise RuntimeError(
            "Capture-normal attachments did not use the explicit fallback: "
            + repr(sorted(direction_policies))
        )

    source = create_scene()
    make_long_dummy_root_attachments(source)
    xml_path = synthetic_source_xml(
        source,
        output,
        "SK_physical_dummy_root_test",
    )
    before = snapshot_source(source)
    _, dummy_root = build(source, xml_path, "XY", output)
    assert_direct_build(dummy_root, source, before, "XY")
    dummy_fit_scale = float(dummy_root["physical_fit_scale"])
    for row in dummy_root["variants"]:
        attachment = row["xml_attachment"]
        if (
            attachment.get("effective_attachment_policy")
            != "geometry_supported_xml_root_segment"
            or float(attachment["effective_support_distance_world"])
            <= float(attachment["effective_geometry_scale_world"])
            or float(attachment["effective_direction_length_world"])
            > float(attachment["effective_geometry_scale_world"]) + TOLERANCE
        ):
            raise RuntimeError(
                "Long dummy XML root was not resolved from visible geometry: "
                + repr(attachment)
            )
        armature = bpy.data.objects[row["armature"]]
        root_bone = armature.data.bones["part_root"]
        assert_close(
            root_bone.length,
            float(attachment["effective_direction_length_world"])
            * dummy_fit_scale,
            tolerance=5.0e-6,
            label=f"{row['armature']} geometry-limited part_root",
        )

    source = create_scene()
    xml_path = synthetic_source_xml(source, output, "SK_physical_side_test")
    before = snapshot_source(source)
    _, side = build(source, xml_path, "YZ", output)
    assert_direct_build(side, source, before, "YZ")

    output.write_text(
        json.dumps(
            {
                "status": "PASS",
                "xy_contract_sha256": second[
                    "physical_capture_contract_sha256"
                ],
                "yz_contract_sha256": side[
                    "physical_capture_contract_sha256"
                ],
                "xy_fit_scale": second["physical_fit_scale"],
                "yz_fit_scale": side["physical_fit_scale"],
                "target_meters": 0.1,
                "unit_probe_contract_sha256": verified_unit[
                    "contract_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("STCLUSTER_PHYSICAL_DIRECT_CAPTURE_SMOKE=" + str(output))


if __name__ == "__main__":
    main()
