import json
import os
import tempfile
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix, Vector


addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
from speedtree_cluster_normalizer.attachment_contract import (
    attachment_endpoint_world,
    attachment_origin_world,
    fit_attachment_to_geometry,
    load_attachment_contract,
    match_root_attachment,
    serialized_contract_source,
    spm_structural_semantic_fingerprint,
)
from speedtree_cluster_normalizer.normalization import (
    _preferred_endpoint_bone,
    _uniform_plan_triangulation,
    convex_hull_2d,
    point_in_convex_polygon,
    root_locked_expanded_hull,
)


def build_armature(name, orphan_end=False, start_only=False):
    data = bpy.data.armatures.new(name + "Data")
    armature = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    if orphan_end:
        end = data.edit_bones.new("Bone_1_End")
        end.head = (1.0, 4.0, 3.0)
        end.tail = (1.0, 4.1, 3.0)
    else:
        start = data.edit_bones.new("Bone_1_Start")
        start.head = (1.0, 2.0, 3.0)
        start.tail = (1.0, 4.0, 3.0)
        if not start_only:
            end = data.edit_bones.new("Bone_1_End")
            end.head = (1.0, 4.0, 3.0)
            end.tail = (1.0, 4.1, 3.0)
            end.parent = start
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.select_set(False)
    return armature


def root_lock_case(points, axis):
    base = convex_hull_2d(points)
    direction = Vector(axis).normalized()
    expected_support = min(Vector(point).dot(direction) for point in base)
    hull, report = root_locked_expanded_hull(points, 0.2, axis)
    actual_support = min(Vector(point).dot(direction) for point in hull)
    if abs(expected_support - actual_support) > 1.0e-8:
        raise RuntimeError((expected_support, actual_support, report))
    if not all(point_in_convex_polygon(point, hull, tolerance=1.0e-8) for point in points):
        raise RuntimeError("Root-locked margin lost a source projection point")
    if report["maximum_root_margin_trim"] <= 0.0:
        raise RuntimeError("Synthetic root lock did not trim root-side margin")
    return report


def bridged_root_lock_case(points, axis):
    base = convex_hull_2d(points)
    direction = Vector(axis).normalized()
    coverage_support = min(
        Vector(point).dot(direction)
        for point in base
    )
    hull, report = root_locked_expanded_hull(points, 0.2, axis)
    plan_support = min(
        Vector(point).dot(direction)
        for point in hull
    )
    if report["policy"] != "xml_root_forward_ray_bridge_to_projection_support":
        raise RuntimeError("Forward root gap did not use the bridge contract")
    if report["attachment_inside_unexpanded_projection"] is not False:
        raise RuntimeError("Bridged attachment was reported inside source geometry")
    if abs(report["unexpanded_root_support"] - coverage_support) > 1.0e-8:
        raise RuntimeError("Bridged source support was not preserved")
    if abs(plan_support) > 1.0e-8:
        raise RuntimeError("Bridged plan does not start at the XML attachment")
    if not point_in_convex_polygon((0.0, 0.0), hull, tolerance=1.0e-8):
        raise RuntimeError("Bridged plan does not contain the XML attachment")
    if not all(
        point_in_convex_polygon(point, hull, tolerance=1.0e-8)
        for point in points
    ):
        raise RuntimeError("Bridged plan lost a source projection point")
    return report


def shared_containment_tolerance_case():
    boundary = [
        (5.0e-9, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (5.0e-9, 1.0),
    ]
    boundary_uvs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    diagonal = 5.0 ** 0.5
    tolerance = max(diagonal * 1.0e-7, 1.0e-9)
    if not point_in_convex_polygon((0.0, 0.0), boundary, tolerance=tolerance):
        raise RuntimeError("Synthetic root-lock tolerance did not accept the origin")
    vertices, _uvs, _faces, attachment_index = _uniform_plan_triangulation(
        boundary,
        boundary_uvs,
        0,
        attachment_point=(0.0, 0.0),
        attachment_uv=(0.0, 0.5),
        containment_tolerance=tolerance,
    )
    if vertices[attachment_index] != (0.0, 0.0):
        raise RuntimeError("Canonical triangulation moved the attachment origin")
    return {
        "containment_tolerance": tolerance,
        "attachment_vertex_index": attachment_index,
    }


with tempfile.TemporaryDirectory(prefix="stcluster_xml_") as directory:
    root = Path(directory)
    spm = root / "SK_synthetic.spm"
    fbx = root / "SK_synthetic.fbx"
    xml = root / "SK_synthetic.xml"
    spm.write_text(
        "<SpeedTree><Generator Type=\"Branch\"><Properties>"
        "<Property><Name>Physics:Bones</Name><Value>1</Value></Property>"
        "</Properties></Generator></SpeedTree>",
        encoding="utf-8",
    )
    fbx.write_bytes(b"synthetic-fbx")
    xml.write_text(
        f'<SpeedTreeRaw Source="{spm}"><Bones>'
        '<Bone ID="0" ParentID="-1" StartX="100" StartY="200" StartZ="300" '
        'EndX="100" EndY="400" EndZ="300" Radius="25" Generator="Synthetic"/>'
        '</Bones></SpeedTreeRaw>',
        encoding="utf-8",
    )
    source_mesh = bpy.data.meshes.new("SyntheticSourceMesh")
    source = bpy.data.objects.new("SyntheticSource", source_mesh)
    bpy.context.scene.collection.objects.link(source)
    source["codex_source_fbx"] = str(fbx)

    start_armature = build_armature("SyntheticStartArmature")
    contract = load_attachment_contract(
        bpy.context.scene,
        source,
        start_armature,
        explicit_xml_path=str(xml),
    )
    if contract["scale"] != 100.0 or [row["id"] for row in contract["roots"]] != [0]:
        raise RuntimeError("Synthetic XML scale/root contract failed")
    serialized_contract = serialized_contract_source(contract)
    if (
        serialized_contract.get("source_spm_semantic_fingerprint")
        != spm_structural_semantic_fingerprint(spm)
        or serialized_contract.get("source_spm_semantic_projection_version") != 1
        or serialized_contract.get("source_spm_sha256")
        != contract["source_spm_sha256"]
    ):
        raise RuntimeError(
            "Source 3D contract did not preserve raw and semantic SPM fingerprints"
        )
    start_bone = start_armature.data.bones["Bone_1_Start"]
    endpoint_bone = start_armature.data.bones["Bone_1_End"]
    start_match = match_root_attachment(
        contract,
        start_armature,
        start_bone,
        endpoint_bone,
        10.0,
        set(),
    )
    if Vector(start_match["xml_start_world"]) != Vector((1.0, 2.0, 3.0)):
        raise RuntimeError("Start joint did not resolve the XML physical root")
    start_only_armature = build_armature(
        "SyntheticStartOnlyArmature",
        start_only=True,
    )
    start_only_contract = load_attachment_contract(
        bpy.context.scene,
        source,
        start_only_armature,
        explicit_xml_path=str(xml),
    )
    start_only_bone = start_only_armature.data.bones["Bone_1_Start"]
    start_only_endpoint, start_only_policy = _preferred_endpoint_bone(
        start_only_bone,
        {"Bone_1_Start"},
    )
    if (
        start_only_endpoint is not None
        or start_only_policy != "start_bone_tail_axis_endpoint"
    ):
        raise RuntimeError("Canonical Start-only axis did not use its own tail")
    start_only_match = match_root_attachment(
        start_only_contract,
        start_only_armature,
        start_only_bone,
        start_only_endpoint,
        10.0,
        set(),
    )
    if Vector(start_only_match["xml_start_world"]) != Vector((1.0, 2.0, 3.0)):
        raise RuntimeError("Start-only axis did not resolve the XML physical root")
    supported_match = fit_attachment_to_geometry(
        start_match,
        [
            Vector((0.9, 3.25, 2.9)),
            Vector((1.1, 3.25, 3.1)),
            Vector((1.0, 3.75, 3.0)),
        ],
        0.5,
    )
    if (
        (
            attachment_origin_world(supported_match)
            - Vector((1.0, 3.25, 3.0))
        ).length
        > 1.0e-9
        or (
            attachment_endpoint_world(supported_match)
            - Vector((1.0, 3.75, 3.0))
        ).length
        > 1.0e-9
        or abs(
            float(supported_match["effective_support_distance_world"]) - 1.25
        )
        > 1.0e-9
        or abs(
            float(supported_match["effective_direction_length_world"]) - 0.5
        )
        > 1.0e-9
    ):
        raise RuntimeError(
            "Geometry-supported attachment did not trim the dummy XML root span"
        )
    start_supported_match = fit_attachment_to_geometry(
        start_match,
        [
            Vector((0.9, 1.99, 2.9)),
            Vector((1.1, 2.01, 3.1)),
            Vector((1.0, 4.0, 3.0)),
        ],
        2.1,
    )
    if (
        attachment_origin_world(start_supported_match)
        - Vector((1.0, 2.0, 3.0))
    ).length > 1.0e-9:
        raise RuntimeError(
            "Geometry-supported attachment moved a valid XML root start"
        )

    comma_dir = root / "comma_decimal"
    comma_dir.mkdir()
    comma_xml = comma_dir / "SK_synthetic.xml"
    comma_xml.write_text(
        f'<SpeedTreeRaw Source="{spm}"><Bones>'
        '<Bone ID="0" ParentID="-1" StartX="100,0" StartY="200,0" '
        'StartZ="300,0" EndX="100,0" EndY="400,0" EndZ="300,0" '
        'Radius="25,0" Generator="Synthetic"/>'
        '</Bones></SpeedTreeRaw>',
        encoding="utf-8",
    )
    comma_contract = load_attachment_contract(
        bpy.context.scene,
        source,
        start_armature,
        explicit_xml_path=str(comma_xml),
    )
    comma_root = comma_contract["roots"][0]
    if (
        Vector(comma_root["start_world"]) != Vector((1.0, 2.0, 3.0))
        or Vector(comma_root["end_world"]) != Vector((1.0, 4.0, 3.0))
        or abs(float(comma_root["radius_world"]) - 0.25) > 1.0e-9
    ):
        raise RuntimeError("Comma-decimal SpeedTree XML was parsed incorrectly")

    orphan_armature = build_armature("SyntheticOrphanArmature", orphan_end=True)
    orphan_match = match_root_attachment(
        contract,
        orphan_armature,
        orphan_armature.data.bones["Bone_1_End"],
        None,
        10.0,
        set(),
    )
    if (
        orphan_match["match_policy"]
        != "xml_root_end_identifies_missing_start_joint"
        or Vector(orphan_match["xml_start_world"]) != Vector((1.0, 2.0, 3.0))
    ):
        raise RuntimeError("Orphan End did not recover the XML segment Start")

    malformed_xml = root / "SK_synthetic_bad.xml"
    malformed_xml.write_text(
        f'<SpeedTreeRaw Source="{spm}"><Bones>'
        '<Bone ID="0" ParentID="-1" StartX="100" StartY="200" StartZ="300" '
        'EndX="100" EndY="400" Radius="25" Generator="Synthetic"/>'
        '</Bones></SpeedTreeRaw>',
        encoding="utf-8",
    )
    try:
        load_attachment_contract(
            bpy.context.scene,
            source,
            start_armature,
            explicit_xml_path=str(malformed_xml),
        )
    except ValueError as exc:
        if "invalid Bone entry" not in str(exc):
            raise
    else:
        raise RuntimeError("Malformed XML Bone was silently skipped")

    negative_parent_xml = root / "SK_synthetic_negative_parent.xml"
    negative_parent_xml.write_text(
        f'<SpeedTreeRaw Source="{spm}"><Bones>'
        '<Bone ID="0" ParentID="-2" StartX="100" StartY="200" StartZ="300" '
        'EndX="100" EndY="400" EndZ="300" Radius="25" Generator="Synthetic"/>'
        '</Bones></SpeedTreeRaw>',
        encoding="utf-8",
    )
    try:
        load_attachment_contract(
            bpy.context.scene,
            source,
            start_armature,
            explicit_xml_path=str(negative_parent_xml),
        )
    except ValueError as exc:
        if "invalid negative ParentID" not in str(exc):
            raise
    else:
        raise RuntimeError("ParentID below -1 was accepted as a structural root")

    stale_time = xml.stat().st_mtime_ns + 10_000_000
    os.utime(spm, ns=(stale_time, stale_time))
    try:
        load_attachment_contract(
            bpy.context.scene,
            source,
            start_armature,
            explicit_xml_path=str(xml),
        )
    except ValueError as exc:
        if "older" not in str(exc):
            raise
    else:
        raise RuntimeError("Stale XML contract was not rejected")

reports = {
    "positive_y": root_lock_case(
        [(-1.0, 0.0), (1.0, 0.0), (1.5, 4.0), (-1.2, 4.0)],
        (0.0, 1.0),
    ),
    "negative_y": root_lock_case(
        [(-1.0, 0.0), (1.0, 0.0), (1.5, -4.0), (-1.2, -4.0)],
        (0.0, -1.0),
    ),
    "positive_x": root_lock_case(
        [(0.0, -1.0), (0.0, 1.0), (4.0, 1.5), (4.0, -1.2)],
        (1.0, 0.0),
    ),
    "forward_gap": bridged_root_lock_case(
        [(-0.4, 0.1), (0.4, 0.1), (0.6, 1.0), (-0.5, 1.0)],
        (0.0, 1.0),
    ),
    "shared_containment_tolerance": shared_containment_tolerance_case(),
}
try:
    root_locked_expanded_hull(
        [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
        0.1,
        (0.0, 1.0),
    )
except ValueError as exc:
    if "outside" not in str(exc):
        raise
else:
    raise RuntimeError("An unrelated XML attachment was silently accepted")

print(
    "__XML_ATTACHMENT_ROOT_LOCK_SMOKE__"
    + json.dumps({"status": "passed", "root_lock_cases": reports}, sort_keys=True)
)
