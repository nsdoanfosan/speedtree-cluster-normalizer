from __future__ import annotations

import addon_utils
import bpy
from mathutils import Vector


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("speedtree_cluster_normalizer", default_set=False)
    from speedtree_cluster_normalizer import normalization

    armature_data = bpy.data.armatures.new("ConnectedClusterArmatureData")
    armature = bpy.data.objects.new("ConnectedClusterArmature", armature_data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root = armature_data.edit_bones.new("Root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, 0.5, 0.0)
    for index in range(1, 13):
        start = armature_data.edit_bones.new(f"Bone_{index}_Start")
        start.head = (0.0, float(index), 0.0)
        start.tail = (0.0, float(index) + 0.25, 0.0)
        start.parent = root
        end = armature_data.edit_bones.new(f"Bone_{index}_End")
        end.head = (0.0, float(index) + 1.0, 0.0)
        end.tail = (0.0, float(index) + 1.25, 0.0)
        end.parent = start
    bpy.ops.object.mode_set(mode="OBJECT")

    # Faces 1..10 share vertex 0, so they form one topology component while
    # retaining distinct dominant deform assignments.  Faces 11 and 12 are
    # isolated and therefore form two more complete clusters.
    vertices = [(0.0, 0.0, 0.0)]
    faces = []
    face_vertices = []
    for index in range(1, 11):
        first = len(vertices)
        vertices.extend(
            [
                (float(index), 1.0, 0.0),
                (float(index), 1.0, 1.0),
            ]
        )
        faces.append((0, first, first + 1))
        face_vertices.append((index, [first, first + 1]))
    for index, x_value in ((11, 20.0), (12, 24.0)):
        first = len(vertices)
        vertices.extend(
            [
                (x_value, 0.0, 0.0),
                (x_value + 1.0, 0.0, 0.0),
                (x_value, 1.0, 1.0),
            ]
        )
        faces.append((first, first + 1, first + 2))
        face_vertices.append((index, [first, first + 1, first + 2]))
    mesh = bpy.data.meshes.new("ConnectedClusterMeshData")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    source = bpy.data.objects.new("ConnectedClusterMesh", mesh)
    bpy.context.scene.collection.objects.link(source)
    modifier = source.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature

    native_root = source.vertex_groups.new(name="Root")
    native_root.add(range(len(vertices)), 0.75, "REPLACE")
    shared = source.vertex_groups.new(name="Bone_1_Start")
    shared.add([0], 0.25, "REPLACE")
    for index, indices in face_vertices:
        group = source.vertex_groups.get(f"Bone_{index}_Start")
        if group is None:
            group = source.vertex_groups.new(name=f"Bone_{index}_Start")
        group.add(indices, 0.25, "REPLACE")

    weights = normalization._vertex_bone_weights(source, armature)
    native_assignments = normalization._face_group_assignments(source, weights)
    if set(native_assignments["faces"]) != {"Root"}:
        raise RuntimeError("Fixture did not reproduce the native shared Root dominance")
    rows = normalization._speedtree_prototype_bone_rows(armature.data.bones)
    prototype_names = {bone.name for _ordinal, bone in rows}
    prototype_weights = {
        vertex_index: {
            name: weight for name, weight in row.items() if name in prototype_names
        }
        for vertex_index, row in weights.items()
    }
    assignments = normalization._face_group_assignments(source, prototype_weights)
    groups = normalization._connected_deform_clusters(source, assignments, rows)
    actual = [group["bone_names"] for group in groups]
    expected = [
        [f"Bone_{index}_Start" for index in range(1, 11)],
        ["Bone_11_Start"],
        ["Bone_12_Start"],
    ]
    if actual != expected:
        raise RuntimeError(f"Connected deform grouping mismatch: {actual}")
    if [len(group["face_indices"]) for group in groups] != [10, 1, 1]:
        raise RuntimeError("Connected deform grouping lost or duplicated faces")
    print("STCLUSTER_CONNECTED_DEFORM_CLUSTER_SMOKE=ok")


if __name__ == "__main__":
    main()
