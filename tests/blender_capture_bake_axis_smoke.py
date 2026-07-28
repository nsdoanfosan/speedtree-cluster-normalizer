from __future__ import annotations

import sys
from pathlib import Path

from mathutils import Matrix, Vector


ADDON_ROOT = Path(__file__).resolve().parents[1] / "addons"
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

import bpy
import speedtree_cluster_normalizer as addon
from speedtree_cluster_normalizer.capture_bake import _capture_basis
from speedtree_cluster_normalizer.delivery_validation import (
    _validate_auto_capture_frame,
)
from speedtree_cluster_normalizer import operators
from speedtree_cluster_normalizer.normalization import (
    PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK,
    physical_capture_aligned_frame,
)


def assert_basis(plane, right, up, normal):
    assert right.dot(up) == 0.0
    assert right.dot(normal) == 0.0
    assert up.dot(normal) == 0.0
    assert right.length == 1.0
    assert up.length == 1.0
    assert normal.length == 1.0
    assert right.cross(up).dot(normal) == 1.0
    return plane


def main():
    xy_points = [
        Vector((-2.0, -1.0, -0.1)),
        Vector((2.0, 3.0, 0.1)),
    ]
    xz_points = [
        Vector((-2.0, -0.1, -1.0)),
        Vector((2.0, 0.1, 3.0)),
    ]
    yz_points = [
        Vector((-0.1, -2.0, -1.0)),
        Vector((0.1, 2.0, 3.0)),
    ]
    try:
        _capture_basis(xy_points, "AUTO")
    except ValueError as exc:
        assert "ambiguous" in str(exc).lower()
    else:
        raise AssertionError("AUTO capture plane must be rejected")
    for plane in ("XY", "XZ", "YZ"):
        assert assert_basis(*_capture_basis(xy_points, plane)[:4]) == plane
    for plane, right, up, normal, rotation in (
        ("XY", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.0),
        ("YZ", [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 90.0),
    ):
        frame = {
            "policy": "world_axis_locked_auto_bounds",
            "plane": plane,
            "right": right,
            "up": up,
            "normal": normal,
            "view_direction": [-value for value in normal],
            "center": [0.0, 0.0, 0.0],
            "camera_location": [normal[index] * 4.0 for index in range(3)],
            "width": 2.0,
            "height": 2.0,
            "content_width": 1.5,
            "content_height": 1.0,
            "orthogonality_error": 0.0,
            "handedness": 1.0,
            "rotation_degrees": rotation,
        }
        validated = _validate_auto_capture_frame(
            {"frame": frame, "resolution": [512, 512]},
            {"frame": frame, "resolution": [512, 512]},
        )
        assert validated["plane"] == plane
        assert validated["rotation_degrees"] == rotation
        tilted = dict(frame)
        tilted["right"] = [right[0], right[1], right[2] + 0.01]
        try:
            _validate_auto_capture_frame(
                {"frame": tilted, "resolution": [512, 512]},
                {"frame": tilted, "resolution": [512, 512]},
            )
        except ValueError as exc:
            assert "tilted" in str(exc).lower()
        else:
            raise AssertionError("Tilted auto capture basis must be rejected")

    normal_mesh = bpy.data.meshes.new("CaptureNormalAttachmentMesh")
    normal_mesh.from_pydata(
        [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    normal_source = bpy.data.objects.new(
        "CaptureNormalAttachment",
        normal_mesh,
    )
    bpy.context.scene.collection.objects.link(normal_source)
    source_frame = {
        "matrix_world": Matrix(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        "endpoint_world": [0.0, 0.0, 2.0],
        "endpoint_length": 2.0,
    }
    aligned = physical_capture_aligned_frame(
        normal_source,
        source_frame,
        range(4),
        {
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "view_direction": [0.0, 0.0, -1.0],
            "center": [0.0, 0.0, 0.0],
            "fit_scale": 0.05,
        },
    )
    tangent = aligned["attachment_tangent_projection"]
    if (
        tangent["direction_policy"]
        != PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK
        or Vector(tangent["capture_plane_world"]).length > 1.0e-9
        or (
            Vector(tangent["aligned_capture_plane_world"]).normalized()
            - Vector((0.0, 1.0, 0.0))
        ).length
        > 1.0e-9
        or (
            aligned["matrix_world"].to_3x3().col[1].normalized()
            - Vector((0.0, 1.0, 0.0))
        ).length
        > 1.0e-9
    ):
        raise AssertionError("Capture-normal attachment fallback is invalid")

    near_normal_frame = dict(source_frame)
    near_normal_frame.update(
        {
            "endpoint_world": [2.4e-6, 0.0, 3.35],
            "endpoint_length": 3.35,
            "source_world_bounds": {"size": [2.0, 2.0, 0.0]},
            "xml_attachment": {
                "match_tolerance": 2.0e-4,
                "effective_support_tolerance_world": 2.0e-4,
                "effective_geometry_scale_world": 2.0,
            },
        }
    )
    near_normal = physical_capture_aligned_frame(
        normal_source,
        near_normal_frame,
        range(4),
        {
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "view_direction": [0.0, 0.0, -1.0],
            "center": [0.0, 0.0, 0.0],
            "fit_scale": 0.05,
        },
    )
    near_tangent = near_normal["attachment_tangent_projection"]
    if (
        near_tangent["direction_policy"]
        != PHYSICAL_CAPTURE_DIRECTION_CAPTURE_UP_FALLBACK
        or near_tangent["projection_tolerance"] < 2.0e-4
        or (
            Vector(near_tangent["aligned_capture_plane_world"]).normalized()
            - Vector((0.0, 1.0, 0.0))
        ).length
        > 1.0e-9
    ):
        raise AssertionError(
            "Numerically degenerate capture-plane projection was not normalized"
        )

    addon.register()
    try:
        props = bpy.context.scene.speedtree_cluster_normalizer
        enum_items = props.bl_rna.properties["capture_plane"].enum_items
        assert {item.identifier for item in enum_items} == {"XY", "XZ", "YZ"}
        workflow_items = props.bl_rna.properties["workflow_mode"].enum_items
        assert {item.identifier for item in workflow_items} == {
            "LEGACY_CAMERA_UV",
            "PHYSICAL_DIRECT_CAPTURE",
        }
        assert props.workflow_mode == "PHYSICAL_DIRECT_CAPTURE"
        assert "unit_probe_contract_path" in props.bl_rna.properties
        props.capture_source_collection = "SpeedTree_Source"
        props.capture_output_dir = "C:/temp/stcluster_capture_smoke"
        props.capture_prefix = "leaf_elm_side_01"
        props.capture_resolution = 512
        props.capture_padding_ratio = 0.06
        props.capture_plane = "YZ"

        received = {}
        original_bake = operators.bake_capture_maps

        def fake_bake_capture_maps(**kwargs):
            received.update(kwargs)
            return {
                "frame": {"plane": "YZ"},
                "resolution": [512, 512],
                "maps": [{} for _ in range(8)],
                "manifest_path": "C:/temp/stcluster_capture_smoke/leaf_elm_side_01_auto_capture_manifest.json",
            }

        operators.bake_capture_maps = fake_bake_capture_maps
        try:
            assert bpy.ops.speedtree_cluster.bake_capture_maps() == {"FINISHED"}
        finally:
            operators.bake_capture_maps = original_bake
        assert received["output_dir"] == "C:/temp/stcluster_capture_smoke"
        assert received["prefix"] == "leaf_elm_side_01"
        assert received["source_collection"] == "SpeedTree_Source"
        assert received["resolution"] == 512
        assert abs(received["padding_ratio"] - 0.06) < 1.0e-6
        assert received["plane"] == "YZ"
        assert received["workflow_mode"] == "PHYSICAL_DIRECT_CAPTURE"
        assert abs(received["target_meters"] - 0.1) < 1.0e-6
        assert props.capture_manifest_path.endswith(
            "leaf_elm_side_01_auto_capture_manifest.json"
        )
        assert '"map_count": 8' in props.capture_last_report
    finally:
        addon.unregister()
    print("STCLUSTER_CAPTURE_AXIS_SMOKE=PASS")


if __name__ == "__main__":
    main()
