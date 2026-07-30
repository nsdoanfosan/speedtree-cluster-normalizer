import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "addons"
    / "speedtree_cluster_normalizer"
    / "generator_delivery_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generator_delivery_contract_test_module",
    MODULE_PATH,
)
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


def spm_root(*, generator_hidden=False, node_hidden=False, mesh_id=63):
    return ET.fromstring(
        f"""
        <SpeedTree>
          <Generators>
            <Generator Type="Frond">
              <Name>Frond 1</Name>
              <GUID>generator-guid</GUID>
              <Hidden>{str(generator_hidden).lower()}</Hidden>
              <Properties>
                <Property>
                  <Name>Material:Frond:0:Material</Name>
                  <Value>7</Value>
                </Property>
                <Property>
                  <Name>Material:Frond:0:Mesh</Name>
                  <Value>{mesh_id}</Value>
                </Property>
              </Properties>
            </Generator>
          </Generators>
          <Nodes>
            <Node>
              <GeneratorGUID>generator-guid</GeneratorGUID>
              <Hidden>{str(node_hidden).lower()}</Hidden>
              <Extra>
                <m_bDeleted>false</m_bDeleted>
                <m_bCulled>false</m_bCulled>
              </Extra>
            </Node>
          </Nodes>
        </SpeedTree>
        """
    )


class GeneratorDeliveryContractTests(unittest.TestCase):
    def test_hidden_or_non_exporting_generator_is_not_live_delivery(self):
        hidden = delivery.live_export_generator_bindings(
            spm_root(generator_hidden=True)
        )
        self.assertFalse(hidden[0]["export_participates"])

        node_hidden = delivery.live_export_generator_bindings(
            spm_root(node_hidden=True)
        )
        self.assertFalse(node_hidden[0]["export_participates"])

    def test_exact_declared_and_live_binding_is_render_connected(self):
        live = delivery.live_export_generator_bindings(spm_root())
        result = delivery.classify_generator_delivery(
            spm="target.spm",
            connection={
                "requested": True,
                "complete": True,
                "generator_variant_policy":
                    "ensure_all_material_cutouts",
                "bindings": [{
                    "generator_index": 0,
                    "generator_guid": "generator-guid",
                    "slot_prefix": "Material:Frond:0",
                    "target_material_id": 7,
                    "target_mesh_id": 63,
                }],
            },
            target_material_id=7,
            normalized_target_mesh_ids=[63],
            live_bindings=live,
        )
        self.assertEqual(
            result["delivery_mode"],
            delivery.DELIVERY_MODE_RENDER_CONNECTED,
        )
        self.assertEqual(result["errors"], [])

    def test_registration_only_is_explicit_and_mesh_drift_fails_closed(self):
        registration = delivery.classify_generator_delivery(
            spm="target.spm",
            connection={
                "requested": False,
                "complete": False,
                "bindings": [],
            },
            target_material_id=7,
            normalized_target_mesh_ids=[63],
            live_bindings=[],
        )
        self.assertEqual(
            registration["delivery_mode"],
            delivery.DELIVERY_MODE_ASSET_REGISTRATION_ONLY,
        )

        drift = delivery.classify_generator_delivery(
            spm="target.spm",
            connection={
                "requested": True,
                "complete": True,
                "generator_variant_policy":
                    "ensure_all_material_cutouts",
                "bindings": [{
                    "generator_index": 0,
                    "slot_prefix": "Material:Frond:0",
                    "target_material_id": 7,
                    "target_mesh_id": 64,
                }],
            },
            target_material_id=7,
            normalized_target_mesh_ids=[63],
            live_bindings=delivery.live_export_generator_bindings(
                spm_root()
            ),
        )
        self.assertEqual(
            drift["delivery_mode"],
            delivery.DELIVERY_MODE_CONNECTION_INCOMPLETE,
        )
        self.assertIn(
            "normalized_and_declared_target_mesh_sets_differ",
            drift["errors"],
        )


if __name__ == "__main__":
    unittest.main()
