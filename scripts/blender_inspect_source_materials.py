import json

import bpy


def main():
    collection = bpy.data.collections.get("SpeedTree_Source")
    if collection is None:
        raise RuntimeError("SpeedTree_Source collection is missing")
    objects = [
        obj
        for obj in collection.all_objects
        if obj.type == "MESH" and len(obj.data.vertices) > 0
    ]
    payload = []
    for obj in objects:
        materials = []
        for material in obj.data.materials:
            if material is None:
                continue
            nodes = []
            if material.use_nodes and material.node_tree is not None:
                for node in material.node_tree.nodes:
                    image = getattr(node, "image", None)
                    nodes.append(
                        {
                            "name": node.name,
                            "label": node.label,
                            "type": node.bl_idname,
                            "image": (
                                bpy.path.abspath(image.filepath) if image is not None else None
                            ),
                        }
                    )
            links = []
            if material.use_nodes and material.node_tree is not None:
                for link in material.node_tree.links:
                    links.append(
                        {
                            "from_node": link.from_node.name,
                            "from_socket": link.from_socket.name,
                            "to_node": link.to_node.name,
                            "to_socket": link.to_socket.name,
                        }
                    )
            materials.append({"name": material.name, "nodes": nodes, "links": links})
        payload.append(
            {
                "object": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "materials": materials,
            }
        )
    print("SOURCE_MATERIALS=" + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
