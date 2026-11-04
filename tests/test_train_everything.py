"""Tests for the resumable training runner."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.train_everything import CLASSIFIERS, DATASET_ORDER, build_runs, load_status, runs_to_do, save_status


class RunListTests(unittest.TestCase):
    def test_every_algorithm_runs_on_every_dataset_type(self):
        names = {run.name for run in build_runs()}
        for dataset in DATASET_ORDER:
            for task in ("plant_gate", "detection", "segmentation", "cyclegan"):
                self.assertIn(f"{dataset}/{task}", names)
            for model in CLASSIFIERS:
                self.assertIn(f"{dataset}/classification/{model}", names)
        self.assertEqual(len(names), 4 * (8 + 4))
        self.assertEqual(len(CLASSIFIERS), 8)

    def test_outputs_never_collide_between_dataset_types(self):
        outputs = []
        for run in build_runs():
            flag = "--output" if "--output" in run.args else "--output-root"
            output = run.args[run.args.index(flag) + 1]
            if "--model" in run.args:
                output += "/" + run.args[run.args.index("--model") + 1]
            outputs.append(output)
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_cyclegan_comes_last(self):
        names = [run.name for run in build_runs()]
        first = min(i for i, n in enumerate(names) if n.endswith("cyclegan"))
        self.assertTrue(all(n.endswith("cyclegan") for n in names[first:]))

    def test_yolo_uses_the_lower_learning_rate(self):
        for run in build_runs():
            if run.name.endswith(("detection", "segmentation")):
                self.assertEqual(run.args[run.args.index("--lr0") + 1], "0.001")


class ResumeTests(unittest.TestCase):
    def test_only_finished_runs_are_skipped(self):
        runs = build_runs()
        status = {runs[0].name: {"state": "done"}, runs[1].name: {"state": "failed"}, runs[2].name: {"state": "running"}}
        todo = [run.name for run in runs_to_do(runs, status)]
        self.assertNotIn(runs[0].name, todo)
        self.assertEqual(todo[:2], [runs[1].name, runs[2].name])
        self.assertEqual(len(todo), len(runs) - 1)

    def test_status_survives_a_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "status.json"
            save_status({"a": {"state": "done"}}, path)
            self.assertEqual(load_status(path), {"a": {"state": "done"}})
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_missing_status_means_nothing_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_status(Path(tmp) / "none.json"), {})


if __name__ == "__main__":
    unittest.main()
