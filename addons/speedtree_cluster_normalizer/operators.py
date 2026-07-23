import json

import addon_utils
import bpy
from bpy.types import Operator, Panel

from .atlas_handoff import prepare_atlas_handoff, resolve_camera_uv_contract
from .normalization import build_normalized_cluster_assets


def _enable_send2ue_if_requested(props):
    if not props.configure_send2ue or hasattr(bpy.types.Scene, "send2ue"):
        return
    try:
        addon_utils.enable("send2ue", default_set=False)
    except Exception:
        pass


class STCLUSTER_OT_build(Operator):
    bl_idname = "speedtree_cluster.build_normalized_assets"
    bl_label = "Build Normalized 3D Parts + Plans"
    bl_description = (
        "Split the skinned source by dominant deform group, normalize each Start/End "
        "parent-child frame to the origin, and build a covering SpeedTree plan"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "speedtree_cluster_normalizer", None)
        active = getattr(context.view_layer.objects, "active", None)
        source = props.source_object if props else None
        return bool(
            props
            and (
                (source is not None and source.type == "MESH")
                or (active is not None and active.type == "MESH")
            )
        )

    def execute(self, context):
        props = context.scene.speedtree_cluster_normalizer
        source = props.source_object or context.view_layer.objects.active
        if source is None or source.type != "MESH":
            self.report({"ERROR"}, "Set a skinned 3D Cluster Source mesh.")
            return {"CANCELLED"}
        if not props.prepare_atlas_handoff:
            self.report(
                {"ERROR"},
                "Prepare Atlas SPM Handoff must be enabled for the exact camera UV contract.",
            )
            return {"CANCELLED"}
        try:
            camera_uv_bundle = resolve_camera_uv_contract(
                props,
                props.plan_base_name.strip().rstrip("_"),
            )
            _enable_send2ue_if_requested(props)
            report = build_normalized_cluster_assets(
                context,
                source,
                props.plan_base_name,
                props.skeletal_base_name,
                props.plan_collection,
                props.plan_material_name,
                plan_margin_ratio=props.plan_margin_ratio,
                replace_generated=props.replace_generated,
                configure_send2ue=props.configure_send2ue,
                isolate_send2ue_export=props.isolate_send2ue_export,
                source_reference_collection_name=props.source_reference_collection,
                camera_uv_bundle=camera_uv_bundle,
                source_partition_mode=props.source_partition_mode,
                whole_mesh_pivot_object=props.whole_mesh_pivot_object,
                plan_refinement_levels=props.plan_refinement_levels,
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        atlas = {"available": False, "prepared": False, "reason": "disabled"}
        if props.prepare_atlas_handoff:
            try:
                atlas = prepare_atlas_handoff(
                    context,
                    props,
                    report["plan_collection"],
                    camera_uv_bundle,
                )
            except Exception as exc:
                atlas = {
                    "available": hasattr(bpy.types.Scene, "atlas_leaf_builder"),
                    "prepared": False,
                    "reason": str(exc),
                }
                self.report(
                    {"WARNING"},
                    f"Normalized assets were built, but Atlas handoff failed: {exc}",
                )

        report["geometry_committed"] = True
        report["atlas_handoff"] = atlas
        context.scene["speedtree_cluster_normalizer_last_report"] = json.dumps(
            report,
            ensure_ascii=False,
        )
        props.source_object = source
        props.last_report = json.dumps(
            {
                "source": source.name,
                "variants": report["variant_count"],
                "plans": [item["plan"] for item in report["variants"]],
                "skeletal_assets": [item["skeletal_asset"] for item in report["variants"]],
                "send2ue": report["send2ue"],
                "atlas": atlas,
            },
            ensure_ascii=False,
        )
        bpy.ops.object.select_all(action="DESELECT")
        for item in report["variants"]:
            pivot = bpy.data.objects.get(item["pivot"])
            if pivot is not None:
                pivot.select_set(True)
        if report["variants"]:
            context.view_layer.objects.active = bpy.data.objects.get(
                report["variants"][0]["pivot"]
            )
        self.report(
            {"INFO"},
            f"Built {report['prototype_count']} normalized 3D prototype(s) and "
            f"{report['card_count']} covering card(s).",
        )
        return {"FINISHED"}


class STCLUSTER_PT_panel(Panel):
    bl_label = "SpeedTree Cluster Normalizer"
    bl_idname = "STCLUSTER_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Cluster Normalize"

    def draw(self, context):
        layout = self.layout
        props = context.scene.speedtree_cluster_normalizer
        layout.prop(props, "source_object")
        layout.prop(props, "source_partition_mode")
        if props.source_partition_mode in {
            "AUTO",
            "WHOLE_MESH",
            "COMPOSITE_PER_DEFORM_ROOT",
        }:
            layout.prop(props, "whole_mesh_pivot_object")
        names = layout.column(align=True)
        names.prop(props, "plan_base_name")
        names.prop(props, "skeletal_base_name")
        names.prop(props, "plan_collection")
        names.prop(props, "plan_material_name")
        names.prop(props, "source_material_name")
        names.prop(props, "source_material_id")
        layout.prop(props, "plan_margin_ratio")
        layout.prop(props, "plan_refinement_levels")
        options = layout.row(align=True)
        options.prop(props, "replace_generated")
        options.prop(props, "configure_send2ue")
        layout.prop(props, "isolate_send2ue_export")
        if props.isolate_send2ue_export:
            layout.prop(props, "source_reference_collection")
        atlas = layout.box()
        atlas.prop(props, "prepare_atlas_handoff")
        atlas.prop(props, "atlas_albedo_path")
        atlas.prop(props, "atlas_camera_spm")
        atlas.prop(props, "atlas_camera_name")
        atlas.prop(props, "atlas_target_spm")
        atlas.prop(props, "atlas_only_target")
        atlas.prop(props, "atlas_mesh_scale")
        layout.label(text="Start to End becomes +Y; every output origin is (0,0,0).")
        layout.operator("speedtree_cluster.build_normalized_assets", icon="MOD_ARMATURE")
        if props.last_report:
            layout.label(text="Last report stored on Scene.")
