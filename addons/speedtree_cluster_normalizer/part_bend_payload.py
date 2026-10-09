"""Durable, origin-anchored bend weights for normalized 3D leaf parts."""

from array import array
import hashlib
import json
import math
import re


PART_BEND_UV_NAME = "nanite_part_bend"
PART_BEND_UV_INDEX = 3
PART_BEND_U_TAG = 8.0
PART_BEND_METADATA_KEY = "speedtree_cluster_part_bend_payload"
PART_BEND_ROLE_CODES = {"leaf": 1, "leaf_side": 2}


def resolve_part_bend_role(asset_name, requested_role="AUTO"):
    """AUTO enables Elm leaf parts; other species retain their existing UVs."""
    requested_role = str(requested_role or "AUTO").strip().lower()
    if requested_role == "auto":
        tokens = re.sub(r"^sk_", "", str(asset_name).lower()).split("_")
        if not tokens or tokens[0] != "leaf":
            return None
        family = [token for token in tokens[1:] if token != "side"]
        if not family or family[0] != "elm":
            return None
        return "leaf_side" if "side" in tokens[1:] else "leaf"
    if requested_role in {"none", "branch"}:
        return None
    if requested_role not in PART_BEND_ROLE_CODES:
        raise ValueError(f"Unsupported normalized part bend role: {requested_role}")
    return requested_role


def build_part_bend_vertex_payload(positions, role):
    """Use +Y/maxY, not (Y-minY)/size: the attachment at Y=0 stays fixed."""
    if role not in PART_BEND_ROLE_CODES:
        raise ValueError(f"Unsupported normalized part bend role: {role}")
    positions = [tuple(float(value) for value in position) for position in positions]
    if not positions or any(
        len(position) != 3 or not all(math.isfinite(value) for value in position)
        for position in positions
    ):
        raise ValueError("Part bend payload requires nonempty finite 3D positions.")
    maximum_y = max(position[1] for position in positions)
    if maximum_y <= 0.0:
        raise ValueError("Normalized leaf part must extend from its origin toward +Y.")
    weights = [min(1.0, max(0.0, position[1] / maximum_y)) for position in positions]
    role_code = PART_BEND_ROLE_CODES[role]
    # UE's skeletal FBX importer flips V. The imported channel is (8+weight, role).
    values = [(PART_BEND_U_TAG + weight, 1.0 - role_code) for weight in weights]
    return values, {
        "schema_version": 1,
        "role": role,
        "role_code": role_code,
        "uv_name": PART_BEND_UV_NAME,
        "uv_index": PART_BEND_UV_INDEX,
        "blender_encoding": "U=8+clamp(Y/maxY,0,1);V=1-role_code",
        "unreal_encoding": "U=8+mask;V=role_code",
        "growth_axis_blender": "+Y",
        "growth_axis_unreal": "-Y",
        "anchor_local": [0.0, 0.0, 0.0],
        "maximum_y": maximum_y,
        "minimum_y": min(position[1] for position in positions),
        "mask_minimum": min(weights),
        "mask_maximum": max(weights),
        "vertex_count": len(positions),
    }


def _uv_hash(layer):
    values = array("f", [0.0]) * (len(layer.data) * 2)
    layer.data.foreach_get("uv", values)
    return hashlib.sha256(values.tobytes()).hexdigest()


def write_normalized_part_bend_payload(part, role):
    """Append UV3 only on an owned, generated, identity-space skeletal part.

    This also supports saved normalized Export meshes; the source mesh and
    generated 2D plans are rejected. It never exports or saves a Blender file.
    """
    if (
        getattr(part, "type", None) != "MESH"
        or not part.get("speedtree_cluster_generated")
        or part.get("speedtree_cluster_asset_role") != "skeletal_mesh"
        or not part.data.get("speedtree_cluster_generated")
    ):
        raise ValueError("Bend payload target must be a generated normalized 3D part.")
    if part.data.users != 1:
        raise ValueError("Bend payload cannot mutate a shared mesh datablock.")
    if any(
        abs(float(part.matrix_world[row][column]) - float(row == column)) > 1.0e-6
        for row in range(4) for column in range(4)
    ):
        raise ValueError("Bend payload target must use its normalized identity frame.")
    mesh = part.data
    values, report = build_part_bend_vertex_payload(
        [vertex.co for vertex in mesh.vertices], role
    )
    layers = mesh.uv_layers
    target = layers.get(PART_BEND_UV_NAME)
    if target is None and len(layers) != PART_BEND_UV_INDEX:
        raise ValueError("Part bend UV3 requires exactly three existing UV channels.")
    if target is not None and list(layers).index(target) != PART_BEND_UV_INDEX:
        raise ValueError("Existing part bend payload must occupy UV3.")
    protected = [
        (index, layer, _uv_hash(layer))
        for index, layer in enumerate(layers) if layer != target
    ]
    active_index = layers.active_index
    render_layer = next((layer for layer in layers if layer.active_render), None)
    clone_layer = next((layer for layer in layers if layer.active_clone), None)
    if target is None:
        target = layers.new(name=PART_BEND_UV_NAME, do_init=False)
    try:
        loop_uvs = array("f", (
            component for loop in mesh.loops for component in values[loop.vertex_index]
        ))
        target.data.foreach_set("uv", loop_uvs)
        mesh.update()
    finally:
        layers.active_index = active_index
        if render_layer is not None:
            render_layer.active_render = True
        if clone_layer is not None:
            clone_layer.active_clone = True
    if _uv_hash(target) != hashlib.sha256(loop_uvs.tobytes()).hexdigest():
        raise RuntimeError("Part bend UV3 write did not preserve its expected payload.")
    if any(_uv_hash(layer) != digest for _index, layer, digest in protected):
        raise RuntimeError("Part bend payload changed a protected UV channel.")
    report["loop_count"] = len(mesh.loops)
    report["protected_uv_sha256"] = {
        str(index): digest for index, _layer, digest in protected
    }
    serialized = json.dumps(report, sort_keys=True)
    part[PART_BEND_METADATA_KEY] = serialized
    mesh[PART_BEND_METADATA_KEY] = serialized
    return report
