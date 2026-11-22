import unittest

import numpy as np

from src.gnn_evaluation import evaluate_predictions, wilson_interval


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        # Class 2 is unseen in training. Its prediction as 0 is a false positive
        # in full evaluation, but must disappear in seen-only evaluation.
        self.y = np.array([0, 1, 0, 1, 0, 1, 2])
        self.pred = np.array([0, 1, 0, 1, 0, 1, 0])
        self.masks = {key: np.isin(np.arange(7), indices) for key, indices in
                      [('train', [0, 1]), ('valid', [2, 3]), ('test', [4, 5, 6])]}

    def bundle(self, pred=None):
        return evaluate_predictions(self.y, self.pred if pred is None else pred,
                                    self.masks, ['A', 'B', 'C'], 'GCN', '1_ogb', 1, 2, 100)

    def test_filtered_f1_recomputes_false_positives(self):
        bundle = self.bundle()
        test = bundle['groups'].query("split == 'test'").set_index('group')
        self.assertAlmostEqual(test.loc['full', 'accuracy'], 2 / 3)
        self.assertEqual(test.loc['seen', 'accuracy'], 1)
        self.assertEqual(test.loc['seen', 'macro_f1_present'], 1)
        # The old, invalid shortcut would retain precision 1/2 for class A.
        self.assertAlmostEqual(bundle['details']['test'].loc[0, 'precision'], 0.5)
        self.assertAlmostEqual(bundle['details']['test'].loc[0, 'recall'], 1)

    def test_filter_does_not_restrict_predicted_labels(self):
        pred = self.pred.copy()
        pred[4] = 2  # A seen-category product wrongly predicted as unseen.
        group = self.bundle(pred)['groups'].query("split == 'test' and group == 'seen'").iloc[0]
        self.assertEqual(group.accuracy, 0.5)
        self.assertEqual(group.products, 2)

    def test_missing_support_has_undefined_interval_and_delta(self):
        row = self.bundle()['comparison'].query('category_id == 2').iloc[0]
        self.assertTrue(np.isnan(row.recall_ci95_lower_valid))
        self.assertTrue(np.isnan(row.recall_test_minus_valid))

    def test_common_mixture_excludes_unsupported_true_classes(self):
        standard = self.bundle()['standardised']
        self.assertEqual(set(standard.common_categories), {2})
        self.assertTrue(np.allclose(standard.accuracy, 1))

    def test_prediction_exports_exclude_training_labels(self):
        frame = self.bundle()['predictions']
        self.assertEqual(set(frame.node_id), {2, 3, 4, 5, 6})
        self.assertEqual(set(frame.split), {'valid', 'test'})

    def test_small_sample_interval_is_wide(self):
        lo, hi = wilson_interval([1, 100, 0], [1, 100, 0])
        self.assertGreater(hi[0] - lo[0], hi[1] - lo[1])
        self.assertTrue(np.isnan(lo[2]))
        self.assertTrue(np.isnan(hi[2]))


if __name__ == '__main__':
    unittest.main()
