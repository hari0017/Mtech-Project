import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from validation_pipeline import GeneFeatures, run_experiment


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.genes = [gene for gene in ("APP", "PSEN1", "PSEN2")
                      for label in (0, 1) for index in range(10)]
        self.labels = [label for gene in ("APP", "PSEN1", "PSEN2")
                       for label in (0, 1) for index in range(10)]
        self.sequences = ["ACGT" * 20 + ("A" if label == 0 else "T") + "C" * index
                          for index, label in enumerate(self.labels)]

    def run_quietly(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_experiment(self.sequences, self.labels, self.genes, **kwargs)

    def test_fold_coverage_and_fitted_preprocessing(self):
        fitted_records = []
        original_fit = GeneFeatures.fit

        def observed_fit(transformer, records, labels):
            fitted_records.append(set(records[:, 0]))
            return original_fit(transformer, records, labels)

        with patch.object(GeneFeatures, "fit", observed_fit):
            result = self.run_quietly(model_name="lr", feature_select=12)
        self.assertEqual((result["split"]["train"], result["split"]["test"]), (42, 18))
        test_indices = set(result["test_indices"])
        validated = []
        for fold_number, (train, valid) in enumerate(result["fold_indices"]):
            self.assertFalse(set(train) & set(valid))
            self.assertFalse(test_indices & (set(train) | set(valid)))
            self.assertEqual(fitted_records[fold_number], {self.sequences[index] for index in train})
            validated.extend(valid)
        self.assertEqual(sorted(validated), sorted(result["train_indices"]))
        self.assertEqual(len(fitted_records), 6)
        pipeline = result["pipeline"]
        self.assertEqual(pipeline.named_steps["scale"].n_samples_seen_, 42)
        self.assertEqual(pipeline.named_steps["select"].get_support().sum(), 12)
        train_set = set(result["train_indices"])
        for gene, reference in pipeline.named_steps["features"].references_.items():
            self.assertTrue(any(self.sequences[index] == reference and self.labels[index] == 0
                                and self.genes[index] == gene for index in train_set))
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 18)

    def test_test_sequences_cannot_change_cv_or_selection(self):
        with patch("importlib.util.find_spec", return_value=None):
            first = self.run_quietly(model_name="all")
            for index in first["test_indices"]:
                self.sequences[index] = "T" * 90
            second = self.run_quietly(model_name="all")
        self.assertEqual(first["comparison"], second["comparison"])
        self.assertEqual(first["best_model"], second["best_model"])
        expected = max(first["comparison"], key=lambda row: row["f1_macro"])["name"]
        self.assertEqual(first["best_model"], expected)

    def test_references_are_not_learned_during_transform(self):
        records = np.array([["ACGTACGT", "APP"], ["ACGTTCGT", "APP"]], dtype=object)
        transformer = GeneFeatures().fit(records[:1], [0])
        with contextlib.redirect_stdout(io.StringIO()):
            transformed = transformer.transform(records[1:])
        self.assertEqual(transformed.shape, (1, 532))
        self.assertEqual(transformed[0, 262], 1)
        self.assertEqual(transformer.references_, {"APP": "ACGTACGT"})

    def test_smote_runs_only_inside_training_folds(self):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            self.skipTest("Optional imbalanced-learn is not installed")
        calls = []
        original = SMOTE.fit_resample

        def observed(sampler, features, labels):
            calls.append(len(labels))
            return original(sampler, features, labels)

        with patch.object(SMOTE, "fit_resample", observed):
            result = self.run_quietly(model_name="lr", smote=True)
        self.assertEqual(calls, [fold["train"] for fold in result["cv"]["folds"]] + [42])

    def test_invalid_sizes_and_missing_class(self):
        with self.assertRaises(ValueError):
            self.run_quietly(test_size=0.9)
        self.labels = [0] * len(self.labels)
        with self.assertRaisesRegex(ValueError, "Both healthy"):
            self.run_quietly()

    def test_tuning_and_ensemble_use_validation_selection(self):
        with patch("validation_pipeline.get_hyperparameter_space", side_effect=lambda name:
                   {"C": [0.1, 1.0]} if name == "lr" else {}):
            result = self.run_quietly(model_name="lr", tune=True, ensemble=True)
        self.assertEqual({row["name"] for row in result["comparison"]}, {"LR", "ENSEMBLE"})
        self.assertIn("lr:all", result["tuning"])
        self.assertTrue(all(len(row["folds"]) == 5 for row in result["comparison"]))
        self.assertEqual(result["best_model"], max(result["comparison"], key=lambda row: row["f1_macro"])["name"])

    def test_flask_and_cli_defaults(self):
        from app import app
        from gene_classifier import parse_args
        with patch("sys.argv", ["gene_classifier.py"]):
            self.assertEqual(parse_args().test_size, 0.3)
        client = app.test_client()
        self.assertEqual(client.post("/train", json=[]).status_code, 400)
        self.assertEqual(client.post("/train", json={"kmer": "bad"}).status_code, 400)
        with patch("app.load_data", return_value=(self.sequences, self.labels, [], self.genes)):
            with contextlib.redirect_stdout(io.StringIO()):
                response = client.post("/train", json={"model": "lr"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["split"]["test"], 18)
        self.assertEqual(len(data["cv"]["folds"]), 5)
        self.assertIn(b'value="0.3"', client.get("/").data)
        with patch("app.load_data", return_value=(self.sequences, self.labels, [], self.genes)):
            audit = client.post("/audit", json={})
            self.assertEqual(audit.status_code, 200)
            self.assertEqual(audit.get_json()["total"], len(self.sequences))
        self.assertEqual(client.post("/audit", json=[]).status_code, 400)
        with patch("app.load_data", return_value=([], [], [], [])):
            self.assertEqual(client.post("/audit", json={}).status_code, 400)

    def test_upload_reuses_pipeline_and_rejects_invalid_records(self):
        from app import app, _state
        result = self.run_quietly(model_name="lr", feature_select=12)
        pipeline = result["pipeline"]
        sequence = pipeline.named_steps["features"].references_["APP"]
        client = app.test_client()
        def upload(text, gene="APP"):
            return client.post("/predict", data={"gene": gene, "file": (io.BytesIO(text.encode()), "sample.fasta")})
        with patch.dict(_state, {"pipeline": pipeline}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            response = upload(f">first\n{sequence}\n>second\n{sequence}")
            self.assertEqual(response.status_code, 200, response.get_json())
            expected = pipeline.predict(np.array([[sequence, "APP"]], dtype=object))[0]
            self.assertEqual(response.get_json()["predictions"][0]["class"], int(expected))
            self.assertEqual(response.get_json()["total"], 2)
            for text, gene in ((">short\nACGT", "APP"), (">invalid\nACNT", "APP"),
                               (sequence, "APP"), (">empty", "APP"), (f">seq\n{sequence}", "BRCA1")):
                self.assertEqual(upload(text, gene).status_code, 400)
        with patch.dict(_state, {}, clear=True):
            self.assertEqual(upload(f">seq\n{sequence}").status_code, 400)
        with patch.dict(_state, {"pipeline": pipeline}, clear=True):
            response = client.post("/predict", data={"gene": "APP", "file": (io.BytesIO(b'>seq\n\xff'), "bad.fasta")})
            self.assertEqual(response.status_code, 400)
            with patch.dict(app.config, {"MAX_CONTENT_LENGTH": 64}):
                response = upload(f">seq\n{sequence}")
                self.assertEqual(response.status_code, 413)
                self.assertIn("error", response.get_json())

    def test_audit_exact_duplicates_conflicts_and_overlap(self):
        from validation_pipeline import audit_dataset
        result = audit_dataset(["ACGT", "acgt", "AAAA", "AAAA"], [0, 1, 0, 0],
                               ["APP"] * 4, ["one", "two", "three", "four"], [0, 2], [1, 3])
        self.assertEqual(result["unique_sequences"], 2)
        self.assertEqual(result["duplicate_extra_copies"], 2)
        self.assertEqual(result["conflicting_groups"], 1)
        self.assertEqual(result["holdout_overlap_groups"], 2)

    def test_feature_comparison_uses_identical_training_folds(self):
        first = self.run_quietly(model_name="lr", compare_features=True)
        self.assertEqual([row["features"] for row in first["comparison"]], [256, 262, 529, 532])
        self.assertEqual({row["feature_set"] for row in first["comparison"]},
                         {"kmer", "composition", "mutation", "all"})
        for index in first["test_indices"]:
            self.sequences[index] = "G" * 90
        second = self.run_quietly(model_name="lr", compare_features=True)
        self.assertEqual(first["comparison"], second["comparison"])
        self.assertEqual(first["best_features"], second["best_features"])
        winner = max(first["comparison"], key=lambda row: row["f1_macro"])
        self.assertEqual(first["best_features"], winner["feature_set"])

    def test_audit_only_cli_never_trains(self):
        from gene_classifier import main
        filenames = [str(index) for index in range(len(self.sequences))]
        with patch("sys.argv", ["gene_classifier.py", "--audit-only"]), \
             patch("gene_classifier.load_data", return_value=(self.sequences, self.labels, filenames, self.genes)), \
             patch("validation_pipeline.run_experiment") as train, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        train.assert_not_called()
        self.assertIn('"unique_sequences": 60', output.getvalue())


if __name__ == "__main__":
    unittest.main()