# SpeedTree Cluster Normalizer

Dedicated Blender 5.x add-on for turning differently placed skinned SpeedTree cluster geometry into normalized 3D skeletal parts plus covering plan meshes.

## Responsibility boundary

This add-on owns only:

- deform-group part detection,
- validated attachment-origin normalization with preserved `Start`/`End` lineage,
- origin-safe 3D part and plan construction,
- exact camera-SPM cutout contract validation and boundary UV transfer,
- Send to Unreal hierarchy preparation,
- configuration handoff to the existing Atlas add-on.

It does not export directly to Unreal and does not duplicate the Atlas add-on's SPM transaction code. Unreal delivery must use Send to Unreal. SPM modification must use Atlas Leaf Mesh Builder's existing `Target SPMs` and `Build/Update Target SPMs` operation.

It can additionally bake a self-contained eight-map capture from the source 3D meshes. This is a Blender-side texture capture only: it does not overwrite an SPM, alter Atlas cutout geometry, or bypass Send to Unreal.

## Blender UI

```text
3D Viewport > Sidebar > Cluster Normalize > SpeedTree Cluster Normalizer
```

### Blender Auto Capture (8 Maps)

`Blender Auto Capture (8 Maps)` renders the selected original source collection to the same output prefix with these files:

```text
<prefix>.tga
<prefix>_Opacity.tga
<prefix>_Normal.tga
<prefix>_Gloss.tga
<prefix>_Subsurface.tga
<prefix>_SubsurfaceAmount.tga
<prefix>_AO.tga
<prefix>_Height.tga
```

Choose the source collection, output folder, prefix, square resolution, padding, and a capture plane in the panel. Only explicit world-axis `XY`, `XZ`, and `YZ` planes are available. `AUTO` is deliberately rejected because a bound-based choice can be ambiguous. Use `YZ (Side, 90°)` for a side cluster; it is an explicit 90° world-axis basis with no roll, not a best-fit camera rotation. The emitted manifest records that frame and the exact map fingerprints.

The saved auto-capture delivery is a first-class validation mode, not a legacy-camera bypass. Before Atlas or Send to Unreal can run, the validator rechecks the manifest and all eight map hashes, the visible orthographic Camera/Area rig, the exact world-axis basis, every plan UV against its normalized frame, the XML-root pivot at local `0,0,0`, the plan-to-3D counterpart lineage, and the isolated `Export` hierarchy. `XY` must remain `+X/+Y`, view `-Z`, rotation `0°`; `YZ` side must remain `+Y/+Z`, view `-X`, rotation exactly `90°`. Any tilt, roll, stale UV, missing Color/Opacity node, or mixed legacy camera contract fails closed before delivery.

For each populated deform group, the operator prefers a same-prefix `*_End` child of `*_Start`. The validated Start/pivot attachment remains the origin, while every 3D prototype and its plan share one rigid camera-aligned canonical frame (`camera right`, `camera up`, `camera normal`). The original bone frame is retained as metadata. No rule depends on one example's dimensions.

`Source 3D XML` is the authoritative attachment source. The add-on verifies that the XML, source FBX, and source SPM belong to the same asset, records their SHA-256 hashes and freshness, then reads every structural root whose `ParentID == -1`. XML coordinates are matched against armature bone heads in world space to select the source scale instead of assuming one fixed unit or axis conversion. Each generated part is translated by the exact XML root `Start`, so the physical stem attachment becomes local `(0, 0, 0)`. If an FBX omits a root `*_Start` bone, the corresponding `*_End` head is matched to the XML root `End` only to recover segment identity; the pivot still uses the XML `Start`. Root count and ordinals come from the hierarchy, so the same contract supports branch, leaf, side, and future tree-cluster layouts without elm-specific constants.

Generated Unreal staging hierarchy:

```text
Export
  SK_branch_elm_01_01
    SK_branch_elm_01_01_Armature
      SK_branch_elm_01_01_Mesh
```

The top Empty supplies the asset name through Send to Unreal's `Use Immediate Parent Name`. The mesh geometry, part armature, and root bone are already normalized; the Empty is not treated as a substitute for transform normalization.

With `Isolate Generated Export` enabled, objects that were already linked directly to `Export` are preserved in `Cluster_Source_Reference`. This prevents a source rig or an older export from being collected together with the three normalized outputs. The collection move participates in the same rollback transaction as asset generation.

Generated plans such as `branch_elm_01_01` are identity-transform convex projection hulls in `Atlas_Branch_Plans`. Their mesh vertices are always on exact local XY (`Z=0`) in the same canonical space as the normalized 3D counterpart. The margin is a dimensionless ratio. Coverage is measured from every projected counterpart vertex, persisted, and independently recomputed during delivery validation; the plan does not copy an old cutout's point count, area, or absolute dimensions. Margin expansion is locked on the physical root support line: it may expand the outer silhouette, but it cannot grow behind the XML attachment point and recreate the old stem-to-plan offset.

UVs do not use the plan bounding box and are never fitted to opacity pixels. `Camera SPM` and `Camera Name` are explicit inputs; the Color texture stem is not used to guess either one. Before geometry is changed, the operator reads Atlas' public `cluster_card_pipeline.read_uv_template_contract` wrapper from that camera SPM and the explicit tree SPM, then proves the selected Color map is the material contract's Color path. It verifies the camera/material/Cutout payload against the normalization manifest and the validated reference `.blend` hash. Exact reference objects are appended into `Atlas_Camera_Reference`, outside both `Export` and the plan collection. A NumPy closed-loop similarity fit maps each new normalized outline to its corresponding exact camera boundary with the normalized attachment origin constrained exactly to the camera contract's `source_plane_xy`. The origin is forced as a CDT vertex and its UV is pinned to the contract's `pivot_uv`; all other boundary/interior UVs are transferred without clamping the source overscan. Delivery validation independently repeats the camera-boundary transfer and deterministic CDT rebuild from the external contract, rejects non-triangle/non-manifold cards, and does not accept a modified `result_uvs` payload merely because its stored hash was updated too.

Variant pairing is also explicit. Every populated deform bone must be a complete `*_N_Start` axis bone (its own tail is the endpoint); legacy inputs may instead provide a direct matching `*_N_End` child or a validated orphan `*_N_End` marker. Ordinals must be unique and consecutive `1..N`, and must match the camera plane suffixes (`_01`, `_02`, ...); missing or ambiguous numbers stop the build instead of guessing a zip order.

Production delivery uses one physical XML root per normalized prototype (`PER_CONNECTED_DEFORM_CLUSTER`, or the equivalent proven per-root layout). Legacy whole-mesh and composite shared-frame requests fail closed because they cannot preserve independent attachment roots without introducing a second pivot convention.

The full contract and all relevant camera/reference/texture hashes are persisted on the Scene and generated plans. A SpeedTree Camera Export is accepted only through Batch Tools' request/finalize receipt: the request freezes the camera SPM, camera transform/GUID, material, resolution, external mesh dependencies, and the Color/Opacity fingerprints used by the plan. Finalization requires Color and Opacity to have been rewritten after the request while the SPM and dependencies remain unchanged; Normal/Gloss/Subsurface/AO/Height outputs remain optional evidence. The actual image export remains SpeedTree's Camera Export action because Modeler 10.1 has no camera-image command-line export. On first normalization without a valid receipt, the add-on writes `*_capture_request.json` beside the normalization manifest and stops with the exact path. After Camera Export, running normalization again finalizes the receipt automatically. This is also the supported rebuild path after Atlas adopts the source material and deletes the old embedded cutout IDs.

After the camera contract preflight and geometry transaction, the add-on calls Atlas Leaf Mesh Builder's public integration API to fill the existing collection, texture, target SPM, Generator Source Mapping, and two separate scale fields. The canonical adapter bakes the effective `0.01` unit conversion into the generated SpeedTree-only FBX geometry and writes `SpeedTree Mesh Asset Scale=1.0`. This preserves the same effective size while avoiding generator-dependent FBX export behavior for Mesh Asset Scale. The normalized 3D source, its plan, pivots, UVs, and Send to Unreal assets remain in their shared canonical cluster space. The handoff explicitly adopts the existing source material (for example `M_branch_elm_01`) in place instead of creating a separate `_plan` material; Atlas snapshots the original cutouts for reversible removal. It also builds the real Color + Opacity preview material and reloads same-path texture updates. The actual SPM update remains Atlas' `Build/Update Target SPMs` operation.

The handoff also requests Atlas' explicit `ensure_all_material_cutouts` generator policy. Every normalized `01/02/03...` cutout must be referenced by a real Frond/Leaf generator Material+Mesh child slot; merely listing a mesh under `SupplementalCutoutMeshIDs` is rejected. This keeps branch, leaf, and side on the same scalable rule while preserving authored duplicate slots that intentionally weight a variation.

## Package

```text
addons/speedtree_cluster_normalizer/
  __init__.py
  attachment_contract.py
  atlas_handoff.py
  delivery_validation.py
  normalization.py
  operators.py
  props.py
tests/
  blender_atlas_spm_handoff_smoke.py
  blender_composite_side_smoke.py
  blender_connected_deform_cluster_smoke.py
  blender_cluster_normalization_smoke.py
  blender_xml_attachment_root_lock_smoke.py
  blender_cluster_normalization_real_smoke.py
  blender_delivery_validation_smoke.py
  blender_delivery_validation_real_smoke.py
  blender_whole_mesh_side_smoke.py
  render_camera_uv_material_qa.py
  render_cluster_preview.py
scripts/
  blender_build_cluster_delivery.py
  blender_finalize_cluster_delivery.py
  blender_update_atlas_target_spm.py
```

## Validation

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --factory-startup --background --python 'tests\blender_cluster_normalization_smoke.py' -- --output 'cluster_smoke.json'
```

The real Atlas handoff smoke test mutates the SPM passed with `--target-spm`, so it must receive an isolated copy. To add an explicit production-file guard without embedding a workstation-specific path in the repository, set `SPEEDTREE_CLUSTER_PRODUCTION_SPM` or pass `--production-spm`. Existing calls that already supply the required smoke-test arguments continue to work unchanged.

```powershell
$env:SPEEDTREE_CLUSTER_PRODUCTION_SPM = 'D:\path\to\production_tree.spm'
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --factory-startup --background --python 'tests\blender_atlas_spm_handoff_smoke.py' -- --normalized-blend 'normalized.blend' --target-spm 'test_outputs\isolated_tree.spm' --albedo 'atlas_color.png' --report 'test_outputs\atlas_handoff.json'
```
