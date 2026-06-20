import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "implementacja"))

from data import CSVDataError, inspect_csv, load_csv


class CSVLoaderTests(unittest.TestCase):
    def _csv(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "dataset.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_numeric_regression_with_named_target(self):
        path = self._csv("age,income,target\n20,2000,4\n30,3000,6\n40,4000,8\n")

        inspection = inspect_csv(path)
        X, y, info = load_csv(path, "target", "regression", return_info=True)

        self.assertEqual(inspection.suggested_target, "target")
        self.assertEqual(X.shape, (3, 2))
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_array_equal(y, [4.0, 6.0, 8.0])
        self.assertEqual(info.target_name, "target")

    def test_semicolon_decimal_comma_and_missing_value_imputation(self):
        path = self._csv(
            "height;weight;y\n1,70;60;1\n1,80;;2\n1,90;80;3\n2,00;90;\n"
        )

        X, y, info = load_csv(path, "y", "regression", return_info=True)

        self.assertEqual(X.shape, (3, 2))
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(info.imputed_values, 1)
        self.assertEqual(info.rows_dropped, 1)
        np.testing.assert_array_equal(y, [1.0, 2.0, 3.0])

    def test_categorical_features_and_binary_target_are_encoded(self):
        path = self._csv(
            "size,color,label\n1,red,no\n2,blue,yes\n3,red,no\n4,,yes\n"
        )

        X, y, info = load_csv(path, "label", "classification", return_info=True)

        self.assertEqual(X.shape, (4, 4))
        self.assertEqual(info.categorical_columns, ["color"])
        self.assertEqual(info.class_mapping, {"no": 0.0, "yes": 1.0})
        np.testing.assert_array_equal(y, [0.0, 1.0, 0.0, 1.0])

    def test_headerless_csv_uses_last_column_by_default(self):
        path = self._csv("1,2,3\n2,3,4\n3,4,5\n")

        X, y = load_csv(path)

        self.assertEqual(X.shape, (3, 2))
        np.testing.assert_array_equal(y, [3.0, 4.0, 5.0])

    def test_rejects_inconsistent_rows(self):
        path = self._csv("x,y,target\n1,2,0\n3,1\n")

        with self.assertRaisesRegex(CSVDataError, "different column counts"):
            load_csv(path)

    def test_rejects_multiclass_target_for_logistic_regression(self):
        path = self._csv("x,label\n1,a\n2,b\n3,c\n")

        with self.assertRaisesRegex(CSVDataError, "exactly 2 target classes"):
            load_csv(path, "label", "classification")


if __name__ == "__main__":
    unittest.main()
