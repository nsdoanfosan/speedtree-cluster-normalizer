import json

import bpy


def id_properties(data_block):
    return {
        key: data_block[key]
        for key in data_block.keys()
        if isinstance(data_block[key], (str, int, float, bool, type(None)))
    }


def main():
    collection_names = {
        "Atlas_Cluster_Cards",
        "Atlas_Auto_Capture",
        "Atlas_Camera_Reference",
    }
    payload = {
        "blend": bpy.data.filepath,
        "collections": {
            collection.name: id_properties(collection)
            for collection in bpy.data.collections
            if collection.name in collection_names
        },
        "scene": id_properties(bpy.context.scene),
        "plans": {
            obj.name: id_properties(obj)
            for obj in bpy.data.objects
            if obj.name.startswith("leaf_elm_01")
        },
    }
    print("ATLAS_SCOPE_STATE=" + json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
