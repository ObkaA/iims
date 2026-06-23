"""
Dataset generators and CSV loader.
All implementations from scratch — pure NumPy, no sklearn, no scipy.
"""
import numpy as np
import csv
from dataclasses import dataclass, field
from pathlib import Path


def generate_linear_regression(n_samples: int = 200, noise: float = 0.3, seed: int = 42):
    """y = 2.5·x + 1 + ε,   ε ~ N(0, (3·noise)²)"""
    rng = np.random.default_rng(seed)
    X   = rng.uniform(-3, 3, (n_samples, 1))
    y   = 2.5 * X[:, 0] + 1.0 + rng.normal(0, noise * 3, n_samples)
    return X, y


def generate_ill_conditioned_regression(
    n_samples: int = 2000,
    feature_scale: float = 10.0,
    noise: float = 0.3,
    seed: int = 42,
):
    """Regression with an elongated loss surface for demonstrating Adam."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, feature_scale, size=(n_samples, 1))
    y = (
        1.0
        + (3.0 / feature_scale) * X[:, 0]
        + rng.normal(0.0, noise, size=n_samples)
    )
    return X, y


def generate_nonlinear_regression(n_samples: int = 200, noise: float = 0.3, seed: int = 42):
    """y = 2·sin(x) + ε"""
    rng = np.random.default_rng(seed)
    X   = rng.uniform(-3, 3, (n_samples, 1))
    y   = np.sin(X[:, 0]) * 2 + rng.normal(0, noise, n_samples)
    return X, y


def generate_logistic_regression(n_samples: int = 300, noise: float = 0.15, seed: int = 42):
    """
    Two 2-D Gaussian blobs: class 0 centred at (+1,+1), class 1 at (−1,−1).

    Sampling without numpy.random.multivariate_normal:
        x = L·z + μ,   z ~ N(0,I),   L = Cholesky(Σ)
    """
    rng    = np.random.default_rng(seed)
    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    def sample_mvn(mean, cov, n):
        L = np.linalg.cholesky(np.array(cov, dtype=np.float64))
        return rng.standard_normal((n, 2)) @ L.T + np.array(mean)

    X0 = sample_mvn([ 1,  1], [[1, 0.3], [0.3, 1]],  n_class_0)
    X1 = sample_mvn([-1, -1], [[1,-0.3], [-0.3, 1]], n_class_1)
    X  = np.vstack([X0, X1])
    y  = np.hstack([np.zeros(n_class_0), np.ones(n_class_1)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_circles(n_samples: int = 300, noise: float = 0.05, seed: int = 42):
    """
    Two concentric circles — implemented from scratch (no sklearn).

    Inner ring: radius 0.5,  label 0
    Outer ring: radius 1.0,  label 1

    Points sampled uniformly by angle θ ~ U[0, 2π]:
        x = r·cos(θ) + ε,   y = r·sin(θ) + ε
    """
    rng    = np.random.default_rng(seed)
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    θ0 = rng.uniform(0, 2 * np.pi, n_inner)
    θ1 = rng.uniform(0, 2 * np.pi, n_outer)

    X0 = np.column_stack([
        0.5 * np.cos(θ0) + rng.normal(0, noise, n_inner),
        0.5 * np.sin(θ0) + rng.normal(0, noise, n_inner),
    ])
    X1 = np.column_stack([
        1.0 * np.cos(θ1) + rng.normal(0, noise, n_outer),
        1.0 * np.sin(θ1) + rng.normal(0, noise, n_outer),
    ])

    X   = np.vstack([X0, X1])
    y   = np.hstack([np.zeros(n_inner), np.ones(n_outer)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


DATASETS = {
    "Linear Data":       generate_linear_regression,
    "Noisy Linear Data": lambda n_samples=200: generate_linear_regression(
        n_samples=n_samples, noise=1.2
    ),
    "Ill-conditioned Linear (Adam demo)": generate_ill_conditioned_regression,
    "Logistic 2D":       generate_logistic_regression,
    "Circles":           generate_circles,
}

DATASET_TASKS = {
    "Linear Data": "regression",
    "Noisy Linear Data": "regression",
    "Ill-conditioned Linear (Adam demo)": "regression",
    "Logistic 2D": "classification",
    "Circles": "classification",
}


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    task: str = "regression",
    seed: int = 42,
):
    """Deterministically split data; preserve class proportions for classification."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and contain the same number of rows as y.")
    if len(y) < 4:
        raise ValueError("At least 4 samples are required for a train/test split.")
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("Test fraction must be between 5% and 50%.")

    rng = np.random.default_rng(seed)
    if task == "classification":
        test_parts = []
        train_parts = []
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            if len(indices) < 2:
                raise ValueError(f"Class {label!r} needs at least 2 samples for splitting.")
            indices = rng.permutation(indices)
            n_test = min(len(indices) - 1, max(1, round(len(indices) * test_fraction)))
            test_parts.append(indices[:n_test])
            train_parts.append(indices[n_test:])
        test_indices = rng.permutation(np.concatenate(test_parts))
        train_indices = rng.permutation(np.concatenate(train_parts))
    elif task == "regression":
        indices = rng.permutation(len(y))
        n_test = min(len(y) - 1, max(1, round(len(y) * test_fraction)))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        raise ValueError("Task must be 'regression' or 'classification'.")

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_from_training(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_full: np.ndarray,
):
    """Standardize all feature sets using statistics calculated only on training data."""
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (
        (X_train - mean) / scale,
        (X_test - mean) / scale,
        (X_full - mean) / scale,
    )


MISSING_VALUES = {"", "?", "na", "n/a", "nan", "null", "none", "missing"}
TARGET_NAMES = {"target", "y", "label", "class", "output", "result", "wartosc_docelowa"}
MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_CATEGORIES = 50
MAX_OUTPUT_FEATURES = 500


class CSVDataError(ValueError):
    """Raised when a CSV cannot be safely converted to a model dataset."""


@dataclass(frozen=True)
class CSVInspection:
    columns: list[str]
    suggested_target: str
    delimiter: str
    has_header: bool
    row_count: int


@dataclass
class CSVLoadInfo:
    feature_names: list[str]
    target_name: str
    delimiter: str
    rows_loaded: int
    rows_dropped: int = 0
    imputed_values: int = 0
    categorical_columns: list[str] = field(default_factory=list)
    class_mapping: dict[str, float] = field(default_factory=dict)


def _normalise_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_VALUES


def _parse_number(value: str, delimiter: str) -> float:
    cleaned = value.strip().replace("\u00a0", "").replace(" ", "")
    if delimiter != "," and cleaned.count(",") == 1 and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    result = float(cleaned)
    if not np.isfinite(result):
        raise ValueError("non-finite number")
    return result


def _read_csv(path: str | Path):
    csv_path = Path(path)
    if not csv_path.is_file():
        raise CSVDataError("The selected CSV file does not exist.")
    if csv_path.stat().st_size == 0:
        raise CSVDataError("The selected CSV file is empty.")
    if csv_path.stat().st_size > MAX_CSV_BYTES:
        raise CSVDataError("CSV is larger than 50 MB. Please reduce it before loading.")

    text = None
    for encoding in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            text = csv_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None or "\x00" in text:
        raise CSVDataError("CSV encoding is unsupported or the file is not plain text.")

    sample = text[:8192]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","

    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(text.splitlines(), delimiter=delimiter)
        if any(cell.strip() for cell in row)
    ]
    if len(rows) < 2:
        raise CSVDataError("CSV must contain at least two non-empty rows.")

    width = len(rows[0])
    if width < 2:
        raise CSVDataError("CSV must contain at least one feature and one target column.")
    invalid_rows = [index + 1 for index, row in enumerate(rows) if len(row) != width]
    if invalid_rows:
        shown = ", ".join(map(str, invalid_rows[:5]))
        raise CSVDataError(f"Rows have different column counts (for example: {shown}).")

    normalised_first = {_normalise_name(value) for value in rows[0]}
    try:
        sniffed_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        sniffed_header = False
    has_header = bool(normalised_first & TARGET_NAMES) or sniffed_header

    if has_header:
        raw_columns = rows.pop(0)
        if any(not value for value in raw_columns):
            raise CSVDataError("Every CSV column must have a non-empty header.")
    else:
        raw_columns = [f"column_{index + 1}" for index in range(width)]

    columns = []
    used = set()
    for index, value in enumerate(raw_columns):
        name = value.strip() or f"column_{index + 1}"
        base = name
        suffix = 2
        while name.casefold() in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name.casefold())
        columns.append(name)

    if not rows:
        raise CSVDataError("CSV contains headers but no data rows.")
    return columns, rows, delimiter, has_header


def inspect_csv(path: str | Path) -> CSVInspection:
    """Inspect columns before loading so the UI can ask for a target column."""
    columns, rows, delimiter, has_header = _read_csv(path)
    suggested = next(
        (name for name in columns if _normalise_name(name) in TARGET_NAMES),
        columns[-1],
    )
    return CSVInspection(columns, suggested, delimiter, has_header, len(rows))


def _target_index(columns: list[str], target_column: str | int | None) -> int:
    if target_column is None:
        for index, name in enumerate(columns):
            if _normalise_name(name) in TARGET_NAMES:
                return index
        return len(columns) - 1
    if isinstance(target_column, int):
        if not 0 <= target_column < len(columns):
            raise CSVDataError("Selected target column is outside the CSV column range.")
        return target_column
    for index, name in enumerate(columns):
        if name == target_column:
            return index
    raise CSVDataError(f"Target column '{target_column}' was not found in the CSV.")


def load_csv(
    path: str | Path,
    target_column: str | int | None = None,
    task: str = "regression",
    *,
    standardize: bool = True,
    return_info: bool = False,
):
    """Convert a CSV into the same ``(X, y)`` format as built-in generators."""
    if task not in {"regression", "classification"}:
        raise CSVDataError("Task must be 'regression' or 'classification'.")

    columns, rows, delimiter, _ = _read_csv(path)
    target_index = _target_index(columns, target_column)
    target_name = columns[target_index]
    usable_rows = [row for row in rows if not _is_missing(row[target_index])]
    rows_dropped = len(rows) - len(usable_rows)
    if len(usable_rows) < 2:
        raise CSVDataError("At least two rows with a non-empty target are required.")

    raw_target = [row[target_index].strip() for row in usable_rows]
    class_mapping = {}
    if task == "regression":
        try:
            y = np.array([_parse_number(value, delimiter) for value in raw_target])
        except ValueError as exc:
            raise CSVDataError("Regression target must contain only numeric values.") from exc
    else:
        labels = sorted(set(raw_target), key=str.casefold)
        if len(labels) != 2:
            raise CSVDataError(
                f"Logistic Regression requires exactly 2 target classes; found {len(labels)}."
            )
        class_mapping = {label: float(index) for index, label in enumerate(labels)}
        y = np.array([class_mapping[value] for value in raw_target], dtype=np.float64)

    feature_arrays = []
    feature_names = []
    categorical_columns = []
    imputed_values = 0

    for column_index, column_name in enumerate(columns):
        if column_index == target_index:
            continue
        values = [row[column_index].strip() for row in usable_rows]
        non_missing = [value for value in values if not _is_missing(value)]
        if not non_missing:
            continue

        parsed = []
        numeric_count = 0
        for value in values:
            if _is_missing(value):
                parsed.append(None)
                continue
            try:
                parsed.append(_parse_number(value, delimiter))
                numeric_count += 1
            except ValueError:
                parsed.append(None)

        if numeric_count / len(non_missing) >= 0.8:
            valid = np.array([value for value in parsed if value is not None])
            median = float(np.median(valid))
            imputed_values += sum(value is None for value in parsed)
            array = np.array([median if value is None else value for value in parsed])
            spread = float(array.std())
            if spread < 1e-12:
                continue
            if standardize:
                array = (array - array.mean()) / spread
            feature_arrays.append(array[:, None])
            feature_names.append(column_name)
            continue

        categories = sorted(
            {"(missing)" if _is_missing(value) else value for value in values},
            key=str.casefold,
        )
        if len(categories) > MAX_CATEGORIES:
            raise CSVDataError(
                f"Column '{column_name}' has {len(categories)} categories. "
                f"The safe limit is {MAX_CATEGORIES}; remove identifier/text columns first."
            )
        categorical_columns.append(column_name)
        clean_values = ["(missing)" if _is_missing(value) else value for value in values]
        encoded = np.column_stack([
            np.array([1.0 if value == category else 0.0 for value in clean_values])
            for category in categories
        ])
        feature_arrays.append(encoded)
        feature_names.extend(f"{column_name}={category}" for category in categories)

    if not feature_arrays:
        raise CSVDataError("No usable feature columns remain after CSV preparation.")
    X = np.hstack(feature_arrays).astype(np.float64, copy=False)
    if X.shape[1] > MAX_OUTPUT_FEATURES:
        raise CSVDataError(
            f"CSV expands to {X.shape[1]} features; the safe limit is {MAX_OUTPUT_FEATURES}."
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise CSVDataError("Prepared data contains NaN or infinite values.")

    info = CSVLoadInfo(
        feature_names=feature_names,
        target_name=target_name,
        delimiter=delimiter,
        rows_loaded=len(y),
        rows_dropped=rows_dropped,
        imputed_values=imputed_values,
        categorical_columns=categorical_columns,
        class_mapping=class_mapping,
    )
    if return_info:
        return X, y, info
    return X, y
