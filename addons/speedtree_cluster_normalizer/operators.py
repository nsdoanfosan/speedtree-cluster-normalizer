import json

import addon_utils
import bpy
from bpy.types import Operator, Panel

from .atlas_handoff import prepare_atlas_handoff, resolve_camera_uv_contract
from .cluster_handoff import (
    load_verified_unit_probe_contract,
    prepare_cluster_handoff,
)
from .capture_bake import (
    WORKFLOW_LEGACY_CAMERA_UV,
    WORKFLOW_PHYSICAL_DIRECT_CAPTURE,
    bake_capture_maps,
    finalize_physical_capture_manifest,
    load_physical_capture_manifest,
    physical_capture_contract,
)
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
        if (
            props.workflow_mode == WORKFLOW_LEGACY_CAMERA_UV
            and not props.prepare_atlas_handoff
        ):
            self.report(
                {"ERROR"},
                "Prepare Atlas SPM Handoff must be enabled for the exact camera UV contract.",
            )
            return {"CANCELLED"}
        try:
            camera_uv_bundle = None
            capture_contract = None
            capture_evidence = None
            verified_unit_probe = None
            if props.workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
                camera_uv_bundle = resolve_camera_uv_contract(
                    props,
                    props.plan_base_name.strip().rstrip("_"),
                )
            elif props.workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
                capture_contract = physical_capture_contract(
                    source_collection=props.capture_source_collection.strip(),
                    scene=context.scene,
                    plane=props.capture_plane,
                    padding_ratio=props.capture_padding_ratio,
                    target_meters=props.capture_target_meters,
                )
                if props.prepare_atlas_handoff:
                    capture_evidence = load_physical_capture_manifest(
                        props.capture_manifest_path,
                        capture_contract,
                    )
                    capture_contract = capture_evidence["contract"]
                    props.atlas_albedo_path = capture_evidence["maps"]["Color"][
                        "path"
                    ]
                    verified_unit_probe = load_verified_unit_probe_contract(
                        props.unit_probe_contract_path,
                        props.capture_target_meters,
                    )
            else:
                raise ValueError(
                    f"Unsupported production workflow: {props.workflow_mode}"
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
                source_xml_path=props.source_xml_path,
                workflow_mode=props.workflow_mode,
                physical_capture_contract=capture_contract,
            )
            if (
                props.workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                and props.prepare_atlas_handoff
            ):
                finalized_capture = finalize_physical_capture_manifest(
                    props.capture_manifest_path,
                    report["physical_capture_contract"],
                )
                if (
                    finalized_capture["physical_capture_contract_sha256"]
                    != report["physical_capture_contract_sha256"]
                ):
                    raise ValueError(
                        "Final capture manifest hash differs from normalized plans."
                    )
                report["capture_manifest_sha256"] = finalized_capture[
                    "manifest_sha256"
                ]
                report["capture_maps"] = finalized_capture["maps"]
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        handoff = {"available": False, "prepared": False, "reason": "disabled"}
        if props.prepare_atlas_handoff:
            try:
                if props.workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
                    handoff = prepare_cluster_handoff(
                        context,
                        props,
                        report["plan_collection"],
                        verified_unit_probe["contract"],
                        finalized_capture["maps"],
                    )
                else:
                    handoff = prepare_atlas_handoff(
                        context,
                        props,
                        report["plan_collection"],
                        camera_uv_bundle,
                        None,
                    )
            except Exception as exc:
                handoff = {
                    "available": False,
                    "prepared": False,
                    "reason": str(exc),
                }
                self.report(
                    (
                        {"ERROR"}
                        if props.workflow_mode
                        == WORKFLOW_PHYSICAL_DIRECT_CAPTURE
                        else {"WARNING"}
                    ),
                    f"Normalized assets were built, but SPM handoff failed: {exc}",
                )
                if props.workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
                    return {"CANCELLED"}

        report["geometry_committed"] = True
        if verified_unit_probe is not None:
            report["unit_probe_receipt"] = {
                "path": verified_unit_probe["receipt_path"],
                "file_sha256": verified_unit_probe["receipt_file_sha256"],
                "contract_sha256": verified_unit_probe["contract_sha256"],
                "scale_location": verified_unit_probe["scale_location"],
                "mesh_geometry_scale": verified_unit_probe[
                    "mesh_geometry_scale"
                ],
                "mesh_asset_scale": verified_unit_probe["mesh_asset_scale"],
                "generator_scale": verified_unit_probe["generator_scale"],
            }
        report["cluster_handoff"] = handoff
        context.scene["speedtree_cluster_normalizer_last_report"] = json.dumps(
            report,
            ensure_ascii=False,
        )
        props.source_object = source
        props.last_report = json.dumps(
            {
                "workflow_mode": props.workflow_mode,
                "source": source.name,
                "variants": report["variant_count"],
                "plans": [item["plan"] for item in report["variants"]],
                "skeletal_assets": [item["skeletal_asset"] for item in report["variants"]],
                "send2ue": report["send2ue"],
                "cluster_handoff": handoff,
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


class STCLUSTER_OT_bake_capture_maps(Operator):
    bl_idname = "speedtree_cluster.bake_capture_maps"
    bl_label = "Bake 8 World-Axis Capture Maps"
    bl_description = (
        "Bake eight Color/Opacity/Normal/Gloss/Subsurface/AO/Height maps from the "
        "selected source collection using an explicit world-axis capture plane"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.speedtree_cluster_normalizer
        source_collection = props.capture_source_collection.strip()
        output_dir = props.capture_output_dir.strip()
        prefix = props.capture_prefix.strip()
        if not source_collection:
            self.report({"ERROR"}, "Set Capture Source Collection.")
            return {"CANCELLED"}
        if not output_dir:
            self.report({"ERROR"}, "Set Capture Output Folder.")
            return {"CANCELLED"}
        if not prefix:
            self.report({"ERROR"}, "Set Capture Map Prefix.")
            return {"CANCELLED"}
        try:
            report = bake_capture_maps(
                output_dir=output_dir,
                prefix=prefix,
                source_collection=source_collection,
                resolution=props.capture_resolution,
                padding_ratio=props.capture_padding_ratio,
                plane=props.capture_plane,
                workflow_mode=props.workflow_mode,
                target_meters=props.capture_target_meters,
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        props.capture_last_report = json.dumps(
            {
                "source_collection": source_collection,
                "plane": report["frame"]["plane"],
                "workflow_mode": report.get("workflow_mode"),
                "physical_target_meters": (
                    report["frame"].get("target_meters")
                ),
                "fit_scale": report["frame"].get("fit_scale"),
                "resolution": report["resolution"],
                "map_count": len(report["maps"]),
                "manifest": report["manifest_path"],
            },
            ensure_ascii=False,
        )
        props.capture_manifest_path = report["manifest_path"]
        self.report(
            {"INFO"},
            f"Baked {len(report['maps'])} maps with {report['frame']['plane']} world-axis capture.",
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
        layout.prop(props, "workflow_mode")
        layout.prop(props, "source_object")
        layout.prop(props, "source_xml_path")
        layout.prop(props, "source_partition_mode")
        if props.source_partition_mode in {
            "AUTO",
            "WHOLE_MESH",
            "COMPOSITE_PER_DEFORM_ROOT",
            "PER_CONNECTED_DEFORM_CLUSTER",
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
        if props.workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
            atlas.prop(props, "atlas_camera_spm")
            atlas.prop(props, "atlas_camera_name")
        atlas.prop(props, "atlas_target_spm")
        if props.workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
            atlas.prop(props, "unit_probe_contract_path")
        atlas.prop(props, "atlas_only_target")
        if props.workflow_mode == WORKFLOW_LEGACY_CAMERA_UV:
            atlas.prop(props, "atlas_mesh_scale")
            atlas.prop(props, "atlas_mesh_asset_scale")
        else:
            atlas.label(text="FBX/SPM scales come only from the verified unit probe.")
        capture = layout.box()
        capture.label(text="Blender Auto Capture (8 Maps)")
        capture.prop(props, "capture_source_collection")
        capture.prop(props, "capture_output_dir")
        capture.prop(props, "capture_prefix")
        capture.prop(props, "capture_resolution")
        if props.workflow_mode == WORKFLOW_PHYSICAL_DIRECT_CAPTURE:
            capture.prop(props, "capture_target_meters")
            capture.prop(props, "capture_manifest_path")
        capture.prop(props, "capture_padding_ratio")
        capture.prop(props, "capture_plane")
        capture.label(text="Side clusters: choose YZ (explicit 90° basis).")
        capture.operator("speedtree_cluster.bake_capture_maps", icon="RENDER_STILL")
        if props.capture_last_report:
            capture.label(text="Last capture report stored on Scene.")
        layout.label(text="XML root is shared by 3D + plan; every output origin is (0,0,0).")
        layout.operator("speedtree_cluster.build_normalized_assets", icon="MOD_ARMATURE")
        if props.last_report:
            layout.label(text="Last report stored on Scene.")
