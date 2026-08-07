from __future__ import annotations

import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "blender_build_physical_capture_delivery.py"
)
CLUSTER_HANDOFF = (
    Path(__file__).resolve().parents[1]
    / "addons"
    / "speedtree_cluster_normalizer"
    / "cluster_handoff.py"
)


class PhysicalDeliveryScriptContractTests(unittest.TestCase):
    def test_physical_delivery_uses_current_cluster_handoff_key(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('build.get("cluster_handoff")', source)
        self.assertNotIn('build.get("atlas_handoff")', source)

    def test_external_atlas_transaction_preserves_cluster_material_name(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "execute_external_target_transaction"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        value = keywords.get("preserve_explicit_material_name")
        self.assertIsInstance(value, ast.Constant)
        self.assertIs(value.value, True)

        operator_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build_speedtree_spm"
        ]
        self.assertEqual(operator_calls, [])

    def test_physical_handoff_rewires_only_an_explicit_different_source(self):
        source = CLUSTER_HANDOFF.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "configure_external_plan_target"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        adoption = keywords.get("adopt_source_material")
        self.assertIsInstance(adoption, ast.Constant)
        self.assertIs(adoption.value, False)
        connection = keywords.get("connect_generators")
        self.assertIsInstance(connection, ast.Compare)

    def test_physical_delivery_accepts_distinct_source_material_identity(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--source-material")', source)
        self.assertIn('parser.add_argument("--source-material-id", type=int)', source)
        self.assertIn("args.source_material or args.material", source)

    def test_transaction_report_serializes_path_values(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "json.dumps(payload, ensure_ascii=False, indent=2, default=str)",
            source,
        )

    def test_normalized_blend_is_saved_before_atlas_fingerprints_it(self):
        source = SCRIPT.read_text(encoding="utf-8")
        save_offset = source.index("bpy.ops.wm.save_as_mainfile")
        transaction_offset = source.index("execute_external_target_transaction(")
        self.assertLess(save_offset, transaction_offset)
        self.assertEqual(source.count("bpy.ops.wm.save_as_mainfile"), 1)

    def test_exact_publish_keeps_shared_provider_peer_registrations(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("props.atlas_only_target = False", source)
        self.assertNotIn("props.atlas_only_target = True", source)


if __name__ == "__main__":
    unittest.main()
