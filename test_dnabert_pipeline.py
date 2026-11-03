import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dnabert_model import PredictionResult, DNABertClassifier
from main import stage_evaluation, split_labelled_inputs, run_training_experiment, stage_prediction
from tokenizer_module import KmerTokenizer
from utils import PipelineConfig


class DNABertEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.encoded = [SimpleNamespace(token_ids=[index], attention_mask=[1])
                        for index in range(40)]
        self.labels = [index % 2 for index in range(40)]
        self.indices = list(range(40))

    def test_without_labels_has_no_metrics(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            metrics = stage_evaluation([])
        self.assertEqual(metrics, {})
        self.assertIn("unavailable", output.getvalue().lower())

    def test_evaluation_does_not_tune_on_test_labels(self):
        results = [
            PredictionResult(index, [0.0, 0.0], [0.6, 0.4], 0, "BENIGN")
            for index in range(4)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = stage_evaluation(results, {0: 0, 1: 1, 2: 1, 3: 1}, True)
        self.assertEqual(metrics["accuracy"], 0.25)
        self.assertEqual(metrics["f1"], 0.0)

    def test_single_class_evaluation_has_no_undefined_auc(self):
        result = PredictionResult(0, [0.0, 0.0], [0.6, 0.4], 0, "BENIGN")
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = stage_evaluation([result], {0: 0}, True)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])

    def test_inference_uses_saved_threshold_for_displayed_labels(self):
        result = PredictionResult(0, [0.0, 0.0], [0.6, 0.4], 0, "BENIGN")
        with patch("main.DNABertClassifier") as classifier_type:
            classifier_type.return_value.predict.return_value = [result]
            with contextlib.redirect_stdout(io.StringIO()):
                _, results = stage_prediction(PipelineConfig(), KmerTokenizer(k=3),
                                              self.encoded[:1], threshold=0.35)
        self.assertEqual(results[0].predicted_class, 1)
        self.assertEqual(results[0].predicted_label, "PATHOGENIC")

    def test_split_is_reproducible_disjoint_and_stratified(self):
        cfg = PipelineConfig()
        first = split_labelled_inputs(cfg, self.encoded, self.indices, self.labels)
        self.assertEqual(first, split_labelled_inputs(cfg, self.encoded, self.indices, self.labels))
        self.assertEqual(sorted(sum(first.values(), [])), self.indices)
        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [22, 6, 12])
        for partition in first.values():
            self.assertEqual({self.labels[index] for index in partition}, {0, 1})

    def test_duplicate_inputs_stay_together_and_conflicts_fail(self):
        self.encoded.append(self.encoded[0])
        self.indices.append(40)
        self.labels.append(0)
        partitions = split_labelled_inputs(PipelineConfig(), self.encoded, self.indices, self.labels)
        self.assertTrue(any(0 in partition and 40 in partition for partition in partitions.values()))
        self.labels[-1] = 1
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            split_labelled_inputs(PipelineConfig(), self.encoded, self.indices, self.labels)

    def test_tiny_datasets_fail_instead_of_reusing_training_data(self):
        with self.assertRaisesRegex(ValueError, "Not enough distinct"):
            split_labelled_inputs(PipelineConfig(), self.encoded[:4], self.indices[:4], self.labels[:4])

    def test_checkpoint_vocabulary_is_used_and_kmer_mismatch_rejected(self):
        vocab = KmerTokenizer(k=3).get_vocab()
        checkpoint_vocab = {token: len(vocab) - 1 - token_id for token, token_id in vocab.items()}
        checkpoint = Mock()
        checkpoint.get_vocab.return_value = checkpoint_vocab
        with patch("transformers.BertTokenizer.from_pretrained", return_value=checkpoint):
            tokenizer = KmerTokenizer.from_pretrained("local-test", k=3)
            self.assertEqual(tokenizer.get_vocab(), checkpoint_vocab)
            with self.assertRaisesRegex(ValueError, "Match --kmer"):
                KmerTokenizer.from_pretrained("local-test", k=6)

    def test_checkpoint_failure_does_not_create_random_model(self):
        with patch("dnabert_model.BertForSequenceClassification.from_pretrained", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "No random-model fallback"):
                DNABertClassifier("missing-model")

    def test_new_classification_head_cannot_predict(self):
        model = Mock()
        model.config.max_position_embeddings = 128
        model.config.vocab_size = 69
        model.parameters.return_value = []
        with patch("dnabert_model.BertForSequenceClassification.from_pretrained",
                   return_value=(model, {"missing_keys": ["classifier.weight", "classifier.bias"]})):
            classifier = DNABertClassifier("test", vocab_size=69)
        with self.assertRaisesRegex(ValueError, "untrained"):
            classifier.predict([[1]], [[1]])

    def test_local_tiny_model_can_train_save_reload_and_predict(self):
        from transformers import BertConfig, BertForSequenceClassification

        with tempfile.TemporaryDirectory() as directory:
            config = BertConfig(vocab_size=69, hidden_size=8, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=16,
                                max_position_embeddings=8, num_labels=2)
            BertForSequenceClassification(config).save_pretrained(directory)
            classifier = DNABertClassifier(directory, vocab_size=69, max_seq_length=4, device="cpu")
            history = classifier.train_model([[2, 5, 3, 0], [2, 6, 3, 0]],
                                             [[1, 1, 1, 0]] * 2, [0, 1], epochs=1,
                                             batch_size=2, val_split=0)
            self.assertEqual(len(history["train_loss"]), 1)
            self.assertEqual(history["val_acc"], [])
            classifier.save_model(directory)
            reloaded = DNABertClassifier(directory, vocab_size=69, max_seq_length=4, device="cpu")
            result = reloaded.predict([[2, 5, 3, 0]], [[1, 1, 1, 0]])[0]
            self.assertAlmostEqual(sum(result.probabilities), 1.0, places=5)

    def test_training_threshold_and_test_calls_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = PipelineConfig(output_dir=directory)
            cfg.gfa_path = str(Path(__file__))
            cfg.labels_file = str(Path(__file__))
            partitions = split_labelled_inputs(cfg, self.encoded, self.indices, self.labels)
            classifier = Mock()
            classifier.train_model.return_value = {"train_loss": [0.5]}

            def predict(token_ids, masks):
                return [PredictionResult(position, [0.0, 0.0], [0.6, 0.4], 0, "BENIGN")
                        for position in range(len(token_ids))]

            classifier.predict.side_effect = predict
            with contextlib.redirect_stdout(io.StringIO()):
                record = run_training_experiment(cfg, classifier, self.encoded,
                                                 dict(enumerate(self.labels)), partitions)
            train_call = classifier.train_model.call_args.kwargs
            self.assertEqual(train_call["token_ids_batch"], [[index] for index in partitions["train"]])
            self.assertEqual(train_call["val_split"], 0.0)
            self.assertEqual([call.args[0] for call in classifier.predict.call_args_list],
                             [[[index] for index in partitions[name]] for name in ("validation", "test")])
            self.assertEqual(record["metrics"]["samples"], len(partitions["test"]))
            self.assertEqual(json.loads((Path(directory) / "experiment.json").read_text()), record)
            changed_labels = dict(enumerate(self.labels))
            for index in partitions["test"]:
                changed_labels[index] = 1 - changed_labels[index]
            with contextlib.redirect_stdout(io.StringIO()):
                second = run_training_experiment(cfg, classifier, self.encoded, changed_labels, partitions)
            self.assertEqual(record["threshold"], second["threshold"])


if __name__ == "__main__":
    unittest.main()