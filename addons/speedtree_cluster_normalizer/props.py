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
    workflow_mode: EnumProperty(
        name="Production Workflow",
        items=(
            (
                "LEGACY_CAMERA_UV",
                "Legacy Camera UV",
                "Preserve the existing SpeedTree camera-template UV transfer workflow",
            ),
            (
                "PHYSICAL_DIRECT_CAPTURE",
                "Physical Direct Capture",
                "Uniformly fit the complete source layout into a physical target and derive plan UVs from the same Blender capture",
            ),
        ),
        default="PHYSICAL_DIRECT_CAPTURE",
        description=(
            "Explicitly isolates legacy SpeedTree camera UV transfer from the "
            "physical Blender direct-capture production contract"
        ),
    )
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
    unit_probe_contract_path: StringProperty(
        name="Verified Unit Probe",
        subtype="FILE_PATH",
        default="",
        description=(
            "Verified role-independent Blender-to-SpeedTree FBX/SPM unit-probe "
            "receipt required by Physical Direct Capture production"
        ),
    )
    atlas_only_target: BoolProperty(
        name="Use Only This Target",
        default=False,
        description="Replace the Atlas target list with this SPM; use only in a dedicated cluster blend",
    )
    atlas_mesh_scale: FloatProperty(
        name="Atlas FBX Geometry Scale",
        default=0.01,
        min=0.000001,
        soft_max=10.0,
        precision=6,
        description=(
            "Bake the Blender-to-SpeedTree unit conversion into the generated "
            "FBX geometry so every SpeedTree generator type exports consistently"
        ),
    )
    atlas_mesh_asset_scale: FloatProperty(
        name="SpeedTree Mesh Asset Scale",
        default=1.0,
        min=0.000001,
        soft_max=1.0,
        precision=6,
        description=(
            "Scale written to every generated SpeedTree Mesh asset; keep at 1.0 "
            "when the unit conversion is baked into the generated FBX geometry"
        ),
    )
    capture_source_collection: StringProperty(
        name="Capture Source Collection",
        default="SpeedTree_Source",
        description=(
            "Collection containing the original 3D source meshes rendered into the "
            "eight SpeedTree map files"
        ),
    )
    capture_output_dir: StringProperty(
        name="Capture Output Folder",
        subtype="DIR_PATH",
        default="",
        description="Folder that receives the eight Color, Opacity, Normal, Gloss, Subsurface Color/Amount, AO, and Height TGA maps",
    )
    capture_manifest_path: StringProperty(
        name="Capture Manifest",
        subtype="FILE_PATH",
        default="",
        description=(
            "Manifest from the exact Blender map bake used by Physical Direct "
            "Capture plan UV generation and SpeedTree handoff"
        ),
    )
    capture_prefix: StringProperty(
        name="Capture Map Prefix",
        default="",
        description="Filename stem shared by the eight generated maps, for example leaf_elm_01",
    )
    capture_resolution: IntProperty(
        name="Capture Resolution",
        default=2048,
        min=1,
        soft_min=256,
        soft_max=8192,
        description="Square resolution used for every generated TGA map",
    )
    capture_target_meters: FloatProperty(
        name="Physical Capture Target",
        default=0.1,
        min=0.000001,
        soft_min=0.01,
        soft_max=1.0,
        precision=4,
        subtype="DISTANCE",
        unit="LENGTH",
        description=(
            "Physical square capture side in meters; with METRIC scale_length=1.0, "
            "0.1 meters equals 0.1 Blender Unit"
        ),
    )
    capture_padding_ratio: FloatProperty(
        name="Capture Padding",
        default=0.04,
        min=0.0,
        soft_max=0.25,
        precision=4,
        subtype="FACTOR",
        description="Relative world-axis capture-frame padding around the source bounds",
    )
    capture_plane: EnumProperty(
        name="Capture Plane",
        items=(
            (
                "XY",
                "XY (Top)",
                "World-axis XY capture with no roll; use for top-facing clusters.",
            ),
            (
                "XZ",
                "XZ (Front)",
                "World-axis XZ capture with no roll; use when the cluster faces the XZ plane.",
            ),
            (
                "YZ",
                "YZ (Side, 90°)",
                "World-axis YZ side capture with no roll; its capture basis is the explicit 90° side orientation.",
            ),
        ),
        default="XY",
        description="Explicit world-axis projection. Automatic plane selection is intentionally rejected as ambiguous.",
    )
    capture_last_report: StringProperty(
        name="Last Capture Report",
        default="",
        description="Summary of the most recent eight-map Blender capture",
    )
    last_report: StringProperty(name="Last Report", default="")
