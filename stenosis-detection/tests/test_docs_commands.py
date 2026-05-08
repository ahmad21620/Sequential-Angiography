from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "PIPELINE.md",
    PROJECT_ROOT / "docs" / "CADICA_BENCHMARKING.md",
    PROJECT_ROOT / "docs" / "BENCHMARKING.md",
]
YOLO_SNIPPETS = [
    "keyframes -> YOLO stenosis detection -> temporal fusion -> multiview fusion",
    "python scripts/run_yolo_train.py",
    "python scripts/run_stenosis_detection.py",
    "--detector yolo",
    "python scripts/run_temporal_fusion.py",
    "python scripts/run_multiview_fusion.py",
    "python scripts/run_stenosis_temporal_sweep.py",
    "--frame-detector yolo",
    "python scripts/run_cadica_benchmark.py",
]
SCRIPT_REFERENCES = [
    "scripts/run_yolo_train.py",
    "scripts/run_stenosis_detection.py",
    "scripts/run_temporal_fusion.py",
    "scripts/run_multiview_fusion.py",
    "scripts/run_stenosis_temporal_sweep.py",
    "scripts/run_cadica_benchmark.py",
    "scripts/run_sweep_benchmark.py",
]


class DocsCommandTests(unittest.TestCase):
    def test_yolo_route_commands_are_documented(self) -> None:
        combined_docs = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)

        for snippet in YOLO_SNIPPETS:
            self.assertIn(snippet, combined_docs)

    def test_documented_yolo_scripts_exist(self) -> None:
        for script in SCRIPT_REFERENCES:
            self.assertTrue((PROJECT_ROOT / script).is_file(), script)


if __name__ == "__main__":
    unittest.main()
