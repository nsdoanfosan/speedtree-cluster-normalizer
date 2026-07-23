import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


def mesh_source_poll(_self, obj):
    return (
        obj is not None
        and obj.type == "MESH"
        and not obj.get("speedtree_cluster_generated")
        and not obj.get("atlas_leaf_cluster_generated")
    )


class STCLUSTER_Properties(PropertyGroup):
    source_object: PointerProperty(
        name="3D Cluster Source",
        type=bpy.types.Object,
        poll=mesh_source_poll,
        description=(
            "Skinned mesh whose populated deform groups and matching Start/End bones "
            "define normalized 3D parts and covering plans"
        ),
    )
    source_xml_path: StringProperty(
        name="Source 3D XML",
        subtype="FILE_PATH",
        default="",
        description=(
            "SpeedTree Raw XML exported with the 3D cluster SPM. When empty, the "
            "explicit XML loaded by SpeedTree Bone Weight Repair is used"
        ),
    )
    source_partition_mode: EnumProperty(
        name="Source Partition",
        items=(
            (
                "AUTO",
                "Auto",
                "Use a one-root-per-prototype layout only when every populated deform root has an explicit consecutive *_N_Start contract",
            ),
            (
                "PER_DEFORM_ROOT",
                "Per Deform Root",
                "Build one normalized SK prototype per populated *_N_Start/*_N_End pair",
            ),
            (
                "PER_CONNECTED_DEFORM_CLUSTER",
                "Per Connected 3D Cluster",
                "Merge deform roots that share source topology, then build one normalized SK prototype per complete cluster and require a 1:1 camera-card match",
            ),
        ),
        default="PER_CONNECTED_DEFORM_CLUSTER",
    )
    whole_mesh_pivot_object: PointerProperty(
        name="Whole Mesh Pivot",
        type=bpy.types.Object,
        description=(
            "Optional explicit ancestor used only to validate WHOLE_MESH hierarchy; "
            "the physical attachment still comes from Source 3D XML"
        ),
    )
    plan_base_name: StringProperty(
        name="Plan Base Name",
        default="branch_elm_01",
        description="Base name for SpeedTree plan variants such as branch_elm_01_01",
    )
    skeletal_base_name: StringProperty(
        name="Skeletal Base Name",
        default="SK_branch_elm_01",
        description="Base name for Send to Unreal pivot assets such as SK_branch_elm_01_01",
    )
    plan_collection: StringProperty(
        name="Plan Collection",
        default="Atlas_Branch_Plans",
        description="Collection handed to the existing Atlas Target SPM workflow",
    )
    plan_material_name: StringProperty(
        name="Plan Material",
        default="M_branch_elm_01",
        description="Existing SpeedTree Material_v8 adopted in place to own the generated plans",
    )
    source_material_name: StringProperty(
        name="Source Material",
        default="M_branch_elm_01",
        description="Existing SpeedTree Material_v8 whose Generator binding will be replaced",
    )
    source_material_id: IntProperty(
        name="Source Material ID",
        default=0,
        min=0,
        description="Optional verified Material_v8 ID; zero disables the ID check",
    )
    plan_margin_ratio: FloatProperty(
        name="Plan Margin Ratio",
        default=0.01,
        min=0.0,
        soft_max=0.1,
        precision=4,
        subtype="FACTOR",
        description=(
            "Relative expansion of the projected convex outline; never an absolute scene-unit distance"
        ),
    )
    plan_refinement_levels: IntProperty(
        name="Plan Internal Subdivision",
        default=1,
        min=0,
        max=2,
        description=(
            "Constrained near-uniform interior triangulation for SpeedTree fold/curl; "
            "keeps the camera-authored boundary UV without fan-like edge concentration"
        ),
    )
    replace_generated: BoolProperty(
        name="Replace Generated",
        default=True,
        description="Replace only outputs tagged by this add-on; unmanaged name conflicts stop the build",
    )
    configure_send2ue: BoolProperty(
        name="Configure Send to Unreal",
        default=True,
        description="Enable Use Object Origin and Use Immediate Parent Name when Send to Unreal is installed",
    )
    isolate_send2ue_export: BoolProperty(
        name="Isolate Generated Export",
        default=True,
        description=(
            "Move pre-existing direct Export objects to a preserved reference collection so Send to Unreal collects only the normalized outputs"
        ),
    )
    source_reference_collection: StringProperty(
        name="Source Reference Collection",
        default="Cluster_Source_Reference",
        description="Collection that preserves objects moved out of Send to Unreal's Export collection",
    )
    prepare_atlas_handoff: BoolProperty(
        name="Prepare Atlas SPM Handoff",
        default=True,
        description="Fill the existing Atlas add-on collection, texture, material, target, and source mapping settings",
    )
    atlas_albedo_path: StringProperty(
        name="Atlas Albedo",
        subtype="FILE_PATH",
        default="",
        description="Existing atlas texture used by the Atlas add-on; matching opacity and maps are resolved there",
    )
    atlas_camera_spm: StringProperty(
        name="Camera SPM",
        subtype="FILE_PATH",
        default="",
        description=(
            "Explicit non-SK SpeedTree camera SPM that authored the cutout UV templates"
        ),
    )
    atlas_camera_name: StringProperty(
        name="Camera Name",
        default="Dropped XY plane camera 2",
        description="Exact orthographic camera name read from the explicit Camera SPM",
    )
    atlas_target_spm: StringProperty(
        name="Target SPM",
        subtype="FILE_PATH",
        default="",
        description="Existing SpeedTree SPM to add to the Atlas Target SPM list",
    )
    atlas_only_target: BoolProperty(
        name="Use Only This Target",
        default=False,
        description="Replace the Atlas target list with this SPM; use only in a dedicated cluster blend",
    )
    atlas_mesh_scale: FloatProperty(
        name="Atlas Mesh Scale",
        default=1.0,
        min=0.000001,
        soft_max=10.0,
        precision=6,
        description=(
            "Geometry scale passed to Atlas; 1.0 preserves the normalized 3D/plan size relationship"
        ),
    )
    last_report: StringProperty(name="Last Report", default="")
