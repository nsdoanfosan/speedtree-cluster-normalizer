"""Transient meshes only: no production loads, exports, or preference saves."""

import json
from pathlib import Path
import sys

import bpy


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addons"))
from speedtree_cluster_normalizer.part_bend_payload import (  # noqa: E402
    PART_BEND_METADATA_KEY,
    build_part_bend_vertex_payload,
    resolve_part_bend_role,
    write_normalized_part_bend_payload,
)


def snapshot(mesh):
    return {
        "positions": [tuple(vertex.co) for vertex in mesh.vertices],
        "faces": [tuple(face.vertices) for face in mesh.polygons],
        "normals": [tuple(vertex.normal) for vertex in mesh.vertices],
        "colors": [tuple(item.color_srgb) for item in mesh.color_attributes[0].data],
        "uvs": [[tuple(item.uv) for item in layer.data] for layer in list(mesh.uv_layers)[:3]],
        "uv_active_index": mesh.uv_layers.active_index,
        "uv_render": next(layer.name for layer in mesh.uv_layers if layer.active_render),
        "uv_clone": next(layer.name for layer in mesh.uv_layers if layer.active_clone),
        "weights": [tuple((group.group, group.weight) for group in vertex.groups) for vertex in mesh.vertices],
    }


def make_part(name):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata([(-1, -0.25, 0), (1, 0, 0), (1, 2, 0), (-1, 4, 0)], [], [(0, 1, 2), (0, 2, 3)])
    mesh.update()
    part = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(part)
    part["speedtree_cluster_generated"] = True
    part["speedtree_cluster_asset_role"] = "skeletal_mesh"
    mesh["speedtree_cluster_generated"] = True
    for index, name in enumerate(("uv0", "blend_ao", "vertex_color_ga")):
        layer = mesh.uv_layers.new(name=name)
        for loop_index, item in enumerate(layer.data):
            item.uv = (index + loop_index * 0.1, loop_index * 0.2)
    mesh.uv_layers.active_index = 1
    mesh.uv_layers[0].active_render = True
    mesh.uv_layers[2].active_clone = True
    colors = mesh.color_attributes.new(name="color", type="BYTE_COLOR", domain="CORNER")
    for index, item in enumerate(colors.data):
        item.color_srgb = (0.2, 0.3, index / len(colors.data), 0.6)
    group = part.vertex_groups.new(name="part_root")
    group.add(list(range(len(mesh.vertices))), 1.0, "REPLACE")
    return part


def expect_failure(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Unsafe payload target was accepted")


assert resolve_part_bend_role("SK_leaf_elm_01") == "leaf"
assert resolve_part_bend_role("SK_leaf_elm_side_01") == "leaf_side"
assert resolve_part_bend_role("SK_leaf_side_elm_01") == "leaf_side"
assert resolve_part_bend_role("SK_branch_elm_01") is None
assert resolve_part_bend_role("SK_leaf_oak_01") is None
assert resolve_part_bend_role("SK_leaf_elmwood_01") is None
assert resolve_part_bend_role("SK_leaf_side_oak_01") is None
assert resolve_part_bend_role("custom", "leaf_side") == "leaf_side"
values, _report = build_part_bend_vertex_payload([(0, -2, 0), (1, 0, 1), (0, 5, 0), (0, 10, 0)], "leaf")
assert values == [(8, 0), (8, 0), (8.5, 0), (9, 0)]
expect_failure(lambda: build_part_bend_vertex_payload([(0, 0, 0), (0, -1, 0)], "leaf"))
expect_failure(lambda: build_part_bend_vertex_payload([(0, float("nan"), 0)], "leaf"))

receipts = []
for role in ("leaf", "leaf_side"):
    part = make_part(role)
    before = snapshot(part.data)
    report = write_normalized_part_bend_payload(part, role)
    assert snapshot(part.data) == before, "Protected geometry, weights, colors, UVs, or active UV changed"
    assert len(part.data.uv_layers) == 4
    expected, _ = build_part_bend_vertex_payload(before["positions"], role)
    observed = [tuple(item.uv) for item in part.data.uv_layers[3].data]
    assert observed == [expected[loop.vertex_index] for loop in part.data.loops]
    assert [1 - uv[1] for uv in observed] == [report["role_code"]] * len(observed)
    assert json.loads(part[PART_BEND_METADATA_KEY]) == report
    assert json.loads(part.data[PART_BEND_METADATA_KEY]) == report
    assert write_normalized_part_bend_payload(part, role) == report
    assert snapshot(part.data) == before
    receipts.append({"role": role, "mask_range": [report["mask_minimum"], report["mask_maximum"]], "protected_streams_equal": True})

part = make_part("unsafe")
before = snapshot(part.data)
part["speedtree_cluster_asset_role"] = "plan"
expect_failure(lambda: write_normalized_part_bend_payload(part, "leaf"))
part["speedtree_cluster_asset_role"] = "skeletal_mesh"
part.location.x = 1
bpy.context.view_layer.update()
expect_failure(lambda: write_normalized_part_bend_payload(part, "leaf"))
part.location.x = 0
bpy.context.view_layer.update()
part.data.uv_layers.new(name="unrelated_uv3")
expect_failure(lambda: write_normalized_part_bend_payload(part, "leaf"))
assert snapshot(part.data) == before
assert PART_BEND_METADATA_KEY not in part
print("PART_BEND_PAYLOAD_SMOKE " + json.dumps({"ok": True, "receipts": receipts, "rejected_unsafe_targets": True}))
