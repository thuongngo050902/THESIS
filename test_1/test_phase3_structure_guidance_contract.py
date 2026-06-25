from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Phase3StructureGuidanceContractTest(unittest.TestCase):
    def test_structure_module_declares_adaptive_components(self):
        text = (ROOT / "networks/structure_guidance.py").read_text(encoding="utf-8")
        self.assertIn("class StructureInputBuilder", text)
        self.assertIn("class StructureEncoder", text)
        self.assertIn("class StructureAwareAttentionBias", text)
        self.assertIn("class StructureResidualAdapter", text)
        self.assertIn("class MaskSeverityGate", text)


if __name__ == "__main__":
    unittest.main()