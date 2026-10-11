import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("payload", Path(__file__).resolve().parents[1] /
    "addons/speedtree_cluster_normalizer/part_bend_payload.py")
payload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payload)


class AspectPayloadTests(unittest.TestCase):
    def test_square_keeps_existing_curve_and_root(self):
        values, report = payload.build_part_bend_vertex_payload(
            [(-1, 0, 0), (1, 1, 0), (0, 2, 0)], "leaf")
        self.assertEqual(values, [(8, 0), (8.5, 0), (9, 0)])
        self.assertEqual(report["aspect_gain"], 1)

    def test_tall_gain_is_linear_after_reference_exponent(self):
        values, report = payload.build_part_bend_vertex_payload(
            [(-1, 0, 0), (1, 4, 0), (0, 8, 0)], "leaf")
        self.assertEqual(values, [(8, 0), (9, 0), (10, 0)])
        self.assertEqual(report["aspect_ratio"], 4)
        self.assertEqual((values[-1][0] - 8) ** 2, 4)

    def test_side_and_units_preserve_gain_and_role(self):
        points = [(-1, 0, 0), (1, 4, 0), (0, 8, 0)]
        a, ar = payload.build_part_bend_vertex_payload(points, "leaf")
        b, br = payload.build_part_bend_vertex_payload(
            [(z * 100, y * 100, x * 100) for x, y, z in points], "leaf_side")
        self.assertEqual([x for x, y in a], [x for x, y in b])
        self.assertEqual([y for x, y in b], [-1] * 3)
        self.assertEqual(ar["aspect_correction"], br["aspect_correction"])

    def test_negative_y_remains_pinned_and_degenerate_width_fails(self):
        values, report = payload.build_part_bend_vertex_payload(
            [(-1, -1, 0), (1, 0, 0), (0, 4, 0)], "leaf")
        self.assertEqual(values[:2], [(8, 0), (8, 0)])
        for points in ([(0, 0, 0), (0, 1, 0)], [(0, math.nan, 0)]):
            with self.assertRaises(ValueError):
                payload.build_part_bend_vertex_payload(points, "leaf")

    def test_species_independent_and_branch_excluded(self):
        for species in ("elm", "oak", "elmwood", "reed", "Weeping_Willow"):
            self.assertEqual(payload.resolve_part_bend_role(f"SK_leaf_{species}_01"), "leaf")
            self.assertEqual(payload.resolve_part_bend_role(f"SK_leaf_{species}_side_01"), "leaf_side")
            self.assertIsNone(payload.resolve_part_bend_role(f"SK_branch_{species}_01"))

    def test_cluster_auto_preserves_semantic_role_without_branch_capture_inference(self):
        for name in ("SK_cluster_reed_01", "cluster_grass_02", "SK_cluster_side_reed_01"):
            self.assertEqual(payload.resolve_part_bend_role(name), "cluster")
        self.assertEqual(payload.resolve_part_bend_role("custom", "cluster"), "cluster")
        for name in ("SK_branch_cluster_reed_01", "SK_branch_leaf_reed_01",
                     "SK_clustered_reed_01", "SK_cluster", "SK_cluster__01", "other_reed"):
            self.assertIsNone(payload.resolve_part_bend_role(name))
        for requested in ("NONE", "branch"):
            self.assertIsNone(payload.resolve_part_bend_role("SK_cluster_reed_01", requested))

    def test_explicit_source_role_precedes_any_asset_name(self):
        points = [(-1, 0, 0), (1, 4, 0), (0, 8, 0)]
        for role in ("leaf", "leaf_side", "cluster"):
            expected = payload.build_part_bend_vertex_payload(points, role)
            for name in ("custom_normalized_part", "ImportedPart42", "SK_leaf_side_family_01",
                         "SK_cluster_family_01", "SK_branch_named_source_01"):
                resolved = payload.resolve_part_bend_role(name, role)
                self.assertEqual(resolved, role)
                self.assertEqual(payload.build_part_bend_vertex_payload(points, resolved), expected)

    def test_auto_unknown_and_explicit_disabled_roles_stay_disabled(self):
        for name in ("custom_normalized_part", "ImportedPart42", "SK_branch_family_01"):
            self.assertIsNone(payload.resolve_part_bend_role(name))
        for name in ("custom_normalized_part", "SK_leaf_family_01", "SK_cluster_family_01"):
            for role in ("NONE", "branch"):
                self.assertIsNone(payload.resolve_part_bend_role(name, role))

    def test_reserved_tag_distinguishes_pinned_root_from_uninitialized_uv3(self):
        points = [(-1, 0, 0), (1, 4, 0), (0, 8, 0)]
        for role, code in (("leaf", 1), ("leaf_side", 2), ("cluster", 1)):
            blender_uvs, report = payload.build_part_bend_vertex_payload(points, role)
            unreal_uvs = [(u, 1 - v) for u, v in blender_uvs]
            self.assertEqual(unreal_uvs[0], (8, code))
            self.assertNotEqual(unreal_uvs[0], (0, 0))
            self.assertEqual([v for u, v in unreal_uvs], [code] * len(points))
            self.assertEqual(report["mask_minimum"], 0)
            self.assertEqual((unreal_uvs[0][0] - payload.PART_BEND_U_TAG), 0)
            self.assertEqual((unreal_uvs[-1][0] - payload.PART_BEND_U_TAG) ** 2,
                             report["aspect_gain"])

    def test_cluster_uses_ordinary_code_and_identical_leaf_math_metadata(self):
        points = [(-0.5, -0.125, 0), (0.5, 0, 0), (0, 4, 0), (0, 8, 0)]
        leaf_values, leaf_report = payload.build_part_bend_vertex_payload(points, "leaf")
        cluster_values, cluster_report = payload.build_part_bend_vertex_payload(points, "cluster")
        self.assertEqual(cluster_values, leaf_values)
        self.assertEqual(cluster_values[:2], [(8, 0), (8, 0)])
        self.assertGreater(cluster_values[-1][0], 9)
        self.assertEqual(cluster_report["role"], "cluster")
        self.assertEqual(cluster_report["role_code"], 1)
        self.assertEqual({key: value for key, value in cluster_report.items() if key != "role"},
                         {key: value for key, value in leaf_report.items() if key != "role"})

    def test_cluster_lateral_width_scale_and_invalid_geometry_match_leaf(self):
        points = [(0, -0.25, -1), (0, 0, 1), (0, 6, 0)]
        values, report = payload.build_part_bend_vertex_payload(points, "cluster")
        scaled_values, scaled_report = payload.build_part_bend_vertex_payload(
            [(x * 0.01, y * 0.01, z * 0.01) for x, y, z in points], "cluster")
        for actual, expected in zip(scaled_values, values):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
        self.assertAlmostEqual(report["aspect_correction"], scaled_report["aspect_correction"])
        for positions in ([(0, 0, 0), (0, 1, 0)], [(0, 0, 0), (1, -1, 0)],
                          [(0, math.nan, 0)], [(0, 0, 0), (1, 1e20, 0)]):
            with self.assertRaises(ValueError):
                payload.build_part_bend_vertex_payload(positions, "cluster")

    def test_large_valid_gain_has_no_artistic_cap_but_overflow_fails(self):
        values, report = payload.build_part_bend_vertex_payload(
            [(0, 0, 0), (1, 10000, 0)], "leaf")
        self.assertEqual(report["aspect_gain"], 10000)
        self.assertEqual(values[-1][0], 108)
        with self.assertRaises(ValueError):
            payload.build_part_bend_vertex_payload([(0, 0, 0), (1, 1e20, 0)], "leaf")


if __name__ == "__main__":
    unittest.main()
