"""Exercise CLI role selection independently of Blender or production files."""

import argparse
import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/blender_build_physical_capture_delivery.py"


def parse_delivery_args(extra=()):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    parser = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    required = ("--mode", "stage", "--blend", "input.blend", "--save-to", "output.blend",
                "--report", "report.json", "--role", "branch", "--source-object", "source",
                "--source-xml", "source.xml", "--plan-base", "cluster_test",
                "--skeletal-base", "SK_cluster_test", "--plan-collection", "Atlas_Branch_Plans",
                "--material", "M_cluster_test", "--material-id", "1", "--capture-plane", "XY")
    scope = {"argparse": argparse, "sys": SimpleNamespace(argv=["blender", "--", *required, *extra])}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(SCRIPT), "exec"), scope)
    return scope["parse_args"]()


class PartBendRoleControlTests(unittest.TestCase):
    def test_cluster_bend_is_independent_of_branch_capture_role(self):
        args = parse_delivery_args(("--part-bend-role", "cluster"))
        self.assertEqual(args.part_bend_role, "cluster")
        self.assertEqual(args.role, "branch")
        self.assertEqual(args.plan_collection, "Atlas_Branch_Plans")

    def test_default_and_explicit_controls(self):
        self.assertEqual(parse_delivery_args().part_bend_role, "AUTO")
        for role in ("AUTO", "NONE", "leaf", "leaf_side", "cluster"):
            self.assertEqual(parse_delivery_args(("--part-bend-role", role)).part_bend_role, role)

    def test_cluster_is_not_a_capture_role_and_unknown_bend_fails(self):
        for extra in (("--role", "cluster"), ("--part-bend-role", "branch")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_delivery_args(extra)

    def test_selected_role_is_forwarded_through_property_and_build_operator(self):
        script = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        assignments = [node for node in ast.walk(script) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Attribute) and target.attr == "part_bend_role"
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "args.part_bend_role")
        operator = ast.parse((ROOT / "addons/speedtree_cluster_normalizer/operators.py")
                             .read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(operator) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "build_normalized_cluster_assets"]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertEqual(ast.unparse(keywords["part_bend_role"]), "props.part_bend_role")


if __name__ == "__main__":
    unittest.main()
