# SpeedTree Cluster Normalizer

Dedicated Blender 5.x add-on for turning differently placed skinned SpeedTree cluster geometry into normalized 3D skeletal parts plus covering plan meshes.

## Responsibility boundary

This add-on owns only:

- deform-group part detection,
- `Start`/`End` parent-child frame normalization,
- origin-safe 3D part and plan construction,
- exact camera-SPM cutout contract validation and boundary UV transfer,
- Send to Unreal hierarchy preparation,
- configuration handoff to the existing Atlas add-on.

It does not export directly to Unreal and does not duplicate the Atlas add-on's SPM transaction code. Unreal delivery must use Send to Unreal. SPM modification must use Atlas Leaf Mesh Builder's existing `Target SPMs` and `Build/Update Target SPMs` operation.

## Blender UI

```text
3D Viewport > Sidebar > Cluster Normalize > SpeedTree Cluster Normalizer
```

For each populated deform group, the operator prefers a same-prefix `*_End` child of `*_Start`. The Start head becomes the origin and the Start-to-End vector becomes local `+Y`. Width and projection normal come from that part's covariance, so the workflow never depends on one example's dimensions.

Generated Unreal staging hierarchy:

```text
Export
  SK_branch_elm_01_01
    SK_branch_elm_01_01_Armature
      SK_branch_elm_01_01_Mesh
```

The top Empty supplies the asset name through Send to Unreal's `Use Immediate Parent Name`. The mesh geometry, part armature, and root bone are already normalized; the Empty is not treated as a substitute for transform normalization.

With `Isolate Generated Export` enabled, objects that were already linked directly to `Export` are preserved in `Cluster_Source_Reference`. This prevents a source rig or an older export from being collected together with the three normalized outputs. The collection move participates in the same rollback transaction as asset generation.

Generated plans such as `branch_elm_01_01` are identity-transform convex projection hulls in `Atlas_Branch_Plans`. The margin is a dimensionless ratio. A plan is validated by covering its own normalized 3D part projection; it does not copy an old cutout's point count, area, or absolute dimensions.

UVs do not use the plan bounding box and are never fitted to opacity pixels. `Camera SPM` and `Camera Name` are explicit inputs; the Color texture stem is not used to guess either one. Before geometry is changed, the operator reads Atlas' public `cluster_card_pipeline.read_uv_template_contract` wrapper from that camera SPM and the explicit tree SPM, then proves the selected Color map is the material contract's Color path. It verifies the camera/material/Cutout payload against the normalization manifest and the validated reference `.blend` hash. Exact reference objects are appended into `Atlas_Camera_Reference`, outside both `Export` and the plan collection. A NumPy closed-loop similarity fit (cyclic shift, rotation, and reflection) maps each new normalized outline to its corresponding exact camera boundary, then transfers the reference boundary UVs without clamping the source overscan.

Variant pairing is also explicit. Every populated deform bone must end in `*_N_Start`, have its direct matching `*_N_End` child, and use unique consecutive ordinals `1..N`. Those ordinals must match the camera plane suffixes (`_01`, `_02`, ...); missing or ambiguous numbers stop the build instead of guessing a zip order.

The full contract and all relevant camera/reference/Color/Opacity hashes are persisted on the Scene and generated plans. This is also the supported rebuild path after Atlas adopts the source material and deletes the old embedded cutout IDs: the saved contract is reused only when every independent camera/reference/texture hash still validates. Missing or stale camera evidence cancels the operator before normalized geometry is built.

After the camera contract preflight and geometry transaction, the add-on calls Atlas Leaf Mesh Builder's public integration API to fill the existing collection, texture, target SPM, Generator Source Mapping, and mesh-scale fields. `Atlas Mesh Scale` defaults to `1.0`, preserving the normalized 3D/plan size relationship. The handoff explicitly adopts the existing source material (for example `M_branch_elm_01`) in place instead of creating a separate `_plan` material; Atlas snapshots the original cutouts for reversible removal. It also builds the real Color + Opacity preview material. Texture nodes use `CLIP`, while the exact UV payload remains unclamped. A post-build configuration failure is reported as a warning without discarding committed geometry. The actual SPM update remains Atlas' `Build/Update Target SPMs` operation.

## Package

```text
addons/speedtree_cluster_normalizer/
  __init__.py
  atlas_handoff.py
  delivery_validation.py
  normalization.py
  operators.py
  props.py
tests/
  blender_atlas_spm_handoff_smoke.py
  blender_composite_side_smoke.py
  blender_cluster_normalization_smoke.py
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
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --factory-startup --background --python 'tests\blender_cluster_normalization_smoke.py' -- --output 'cluster_smoke.json'
```

The real Atlas handoff smoke test mutates the SPM passed with `--target-spm`, so it must receive an isolated copy. To add an explicit production-file guard without embedding a workstation-specific path in the repository, set `SPEEDTREE_CLUSTER_PRODUCTION_SPM` or pass `--production-spm`. Existing calls that already supply the required smoke-test arguments continue to work unchanged.

```powershell
$env:SPEEDTREE_CLUSTER_PRODUCTION_SPM = 'D:\path\to\production_tree.spm'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --factory-startup --background --python 'tests\blender_atlas_spm_handoff_smoke.py' -- --normalized-blend 'normalized.blend' --target-spm 'test_outputs\isolated_tree.spm' --albedo 'atlas_color.png' --report 'test_outputs\atlas_handoff.json'
```
