import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DE Drive end accelarometer
# FE Fan End accelerometer
# BA base accelerometer
# RPM toerental
#
# Comparison script: train on 0.007" / 0.014" / 0.028" fault sizes, test on
# held-out 0.021" (completely unknown defect severity to the model).
# See BearingFault.py for the load-based split (unknown 2 HP load).

# Resolve paths relative to this file so the script works from any cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Portfolio export folder (README / LaTeX / PNGs live here).
PORTFOLIO_DIR = os.path.join(
    os.path.dirname(BASE_DIR), "portfolio", "BearingFaultPlotsandLatex"
)
FIGURES_DIR = os.path.join(PORTFOLIO_DIR, "figures")
# Set SAVE_FIGURES=0 to skip writing PNGs (interactive exploration only).
SAVE_FIGURES = os.environ.get("SAVE_FIGURES", "1") != "0"


def save_figure(filename):
    """Save the current matplotlib figure into the portfolio figures folder."""
    if not SAVE_FIGURES:
        return
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def maybe_show():
    """Show the current figure unless running headless (MPLBACKEND=Agg)."""
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    else:
        plt.close()


TRAIN_DEFECT_SIZES = {"007", "014", "028"}
TEST_DEFECT_SIZES = {"021"}

# (mat file, label, DE_time key)
# Missing CWRU files (not downloaded): OR014 @ 3:00 & 12:00, all OR028
DATASET_SPECS = [
    ("Normal0.mat", "Normal0", "X097_DE_time"),

    ("B0070.mat", "B0070", "X118_DE_time"),
    ("B0071.mat", "B0071", "X119_DE_time"),
    ("B0072.mat", "B0072", "X120_DE_time"),
    ("B0073.mat", "B0073", "X121_DE_time"),
    ("IR0070.mat", "IR0070", "X105_DE_time"),
    ("IR0071.mat", "IR0071", "X106_DE_time"),
    ("IR0072.mat", "IR0072", "X107_DE_time"),
    ("IR0073.mat", "IR0073", "X108_DE_time"),
    ("OR00760.mat", "OR00760", "X130_DE_time"),
    ("OR00761.mat", "OR00761", "X131_DE_time"),
    ("OR00762.mat", "OR00762", "X132_DE_time"),
    ("OR00763.mat", "OR00763", "X133_DE_time"),
    ("OR00730.mat", "OR00730", "X144_DE_time"),
    ("OR00731.mat", "OR00731", "X145_DE_time"),
    ("OR00732.mat", "OR00732", "X146_DE_time"),
    ("OR00733.mat", "OR00733", "X147_DE_time"),
    ("OR007120.mat", "OR007120", "X156_DE_time"),
    ("OR007121.mat", "OR007121", "X158_DE_time"),
    ("OR007122.mat", "OR007122", "X159_DE_time"),
    ("OR007123.mat", "OR007123", "X160_DE_time"),

    ("B0140.mat", "B0140", "X185_DE_time"),
    ("B0141.mat", "B0141", "X186_DE_time"),
    ("B0142.mat", "B0142", "X187_DE_time"),
    ("B0143.mat", "B0143", "X188_DE_time"),
    ("IR0140.mat", "IR0140", "X169_DE_time"),
    ("IR0141.mat", "IR0141", "X170_DE_time"),
    ("IR0142.mat", "IR0142", "X171_DE_time"),
    ("IR0143.mat", "IR0143", "X172_DE_time"),
    ("OR01460.mat", "OR01460", "X197_DE_time"),
    ("OR01461.mat", "OR01461", "X198_DE_time"),
    ("OR01462.mat", "OR01462", "X199_DE_time"),
    ("OR01463.mat", "OR01463", "X200_DE_time"),

    ("B0210.mat", "B0210", "X222_DE_time"),
    ("B0211.mat", "B0211", "X223_DE_time"),
    ("B0212.mat", "B0212", "X224_DE_time"),
    ("B0213.mat", "B0213", "X225_DE_time"),
    ("IR0210.mat", "IR0210", "X209_DE_time"),
    ("IR0211.mat", "IR0211", "X210_DE_time"),
    ("IR0212.mat", "IR0212", "X211_DE_time"),
    ("IR0213.mat", "IR0213", "X212_DE_time"),
    ("OR02160.mat", "OR02160", "X234_DE_time"),
    ("OR02161.mat", "OR02161", "X235_DE_time"),
    ("OR02162.mat", "OR02162", "X236_DE_time"),
    ("OR02163.mat", "OR02163", "X237_DE_time"),
    ("OR02130.mat", "OR02130", "X246_DE_time"),
    ("OR02131.mat", "OR02131", "X247_DE_time"),
    ("OR02132.mat", "OR02132", "X248_DE_time"),
    ("OR02133.mat", "OR02133", "X249_DE_time"),
    ("OR021120.mat", "OR021120", "X258_DE_time"),
    ("OR021121.mat", "OR021121", "X259_DE_time"),
    ("OR021122.mat", "OR021122", "X260_DE_time"),
    ("OR021123.mat", "OR021123", "X261_DE_time"),

    ("B0280.mat", "B0280", "X048_DE_time"),
    ("B0281.mat", "B0281", "X049_DE_time"),
    ("B0282.mat", "B0282", "X050_DE_time"),
    ("B0283.mat", "B0283", "X051_DE_time"),
    ("IR0280.mat", "IR0280", "X056_DE_time"),
    ("IR0281.mat", "IR0281", "X057_DE_time"),
    ("IR0282.mat", "IR0282", "X058_DE_time"),
    ("IR0283.mat", "IR0283", "X059_DE_time"),
]


def defect_size(label: str) -> str | None:
    """Return the 3-digit defect size from a dataset label, or None for Normal."""
    if label.startswith("Normal"):
        return None
    if label.startswith("B"):
        return label[1:4]
    if label.startswith("IR") or label.startswith("OR"):
        return label[2:5]
    raise ValueError(f"Unknown label: {label}")


def dataset_split(label: str) -> str:
    """Assign train or test split from defect size."""
    size = defect_size(label)
    if size is None:
        return "train"
    if size in TRAIN_DEFECT_SIZES:
        return "train"
    if size in TEST_DEFECT_SIZES:
        return "test"
    raise ValueError(f"Unexpected defect size {size} for label {label}")


def load_de_signal(mat_file: str, de_key: str | None = None) -> np.ndarray:
    """Load the Drive End time signal from a CWRU .mat file."""
    path = os.path.join(BASE_DIR, mat_file)
    data = loadmat(path)
    if de_key is None:
        de_key = next(k for k in data if not k.startswith("__") and "DE_time" in k)
    return data[de_key]


def dataset_title(label: str) -> str:
    """Human-readable title for plots and logs."""
    if label.startswith("Normal"):
        return f"{label} — Drive End (0 HP)"
    if label.startswith("B"):
        size = label[1:4]
        load_hp = label[4]
        return f"{label} — Drive End (ball defect {size}\", {load_hp} HP)"
    if label.startswith("IR"):
        size = label[2:5]
        load_hp = label[5]
        return f"{label} — Drive End (inner race defect {size}\", {load_hp} HP)"
    if label.startswith("OR"):
        size = label[2:5]
        rest = label[5:]
        load_hp = rest[-1]
        clock = rest[:-1]
        return (
            f"{label} — Drive End (outer race defect {size}\", "
            f"{clock}:00, {load_hp} HP)"
        )
    return f"{label} — Drive End"


loaded_datasets: dict[str, np.ndarray] = {}
train_sources: list[tuple[np.ndarray, str, str]] = []
test_sources: list[tuple[np.ndarray, str, str]] = []

for mat_file, label, de_key in DATASET_SPECS:
    path = os.path.join(BASE_DIR, mat_file)
    if not os.path.isfile(path):
        print(f"Skipping missing file: {mat_file}")
        continue
    signal = load_de_signal(mat_file, de_key)
    loaded_datasets[label] = signal
    entry = (signal, label, dataset_title(label))
    split = dataset_split(label)
    if split == "train":
        train_sources.append(entry)
    else:
        test_sources.append(entry)

print(
    f"Loaded {len(loaded_datasets)} datasets "
    f"({len(train_sources)} train @ {sorted(TRAIN_DEFECT_SIZES)}\", "
    f"{len(test_sources)} test @ {sorted(TEST_DEFECT_SIZES)}\")"
)
print(
    "Note: OR014 @ 3:00/12:00 and all OR028 files are unavailable in CWRU — "
    "only OR014 @ 6:00 is used for training. Normal baseline is 0 HP only (train)."
)

N = 2000
FS = 12000  # CWRU Drive End sampling rate (Hz)
WINDOW_SIZE = 500  # samples per segment (~24 Hz FFT resolution at 12 kHz)

# Baseline 0.007 inch / 0 HP datasets for plots and statistics (OR at 6:00 only).
plot_labels = ["Normal0", "B0070", "IR0070", "OR00760"]
plot_sources = [
    (loaded_datasets[label], label, dataset_title(label))
    for label in plot_labels
    if label in loaded_datasets
]

FAULT_CLASSES = ["Normal", "Ball", "IR", "OR"]


def fault_type(file_label: str) -> str:
    """Map a dataset label to one of the four bearing fault types."""
    if file_label.startswith("Normal"):
        return "Normal"
    if file_label.startswith("B"):
        return "Ball"
    if file_label.startswith("IR"):
        return "IR"
    if file_label.startswith("OR"):
        return "OR"
    raise ValueError(f"Unknown label: {file_label}")


signals = [
    (src.flatten()[:N], name, title) for src, name, title in plot_sources
]

header = (
    f"{'Signal':<10} {'Mean':>10} {'Std':>10} {'Variance':>12} {'Max':>10} {'Min':>10} "
    f"{'RMS':>10} {'Kurtosis':>10} {'Crest':>10} {'P2P':>10}"
)
print("\nDrive End statistics (first 2000 samples, 0.007\" baseline @ 0 HP)")
print(header)
print("-" * len(header))
for signal, name, _ in signals:
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    crest = peak / rms if rms > 0 else np.nan
    kurt = kurtosis(signal)
    p2p = np.max(signal) - np.min(signal)
    print(
        f"{name:<10} {mean:>10.4f} {std:>10.4f} {var:>12.4f} "
        f"{np.max(signal):>10.4f} {np.min(signal):>10.4f} {rms:>10.4f} "
        f"{kurt:>10.4f} {crest:>10.4f} {p2p:>10.4f}"
    )
print()

plot_ranges = [
    (0, 2000),
    (10000, 10500),
    (50000, 50500),
]

for start, end in plot_ranges:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    x = np.arange(start, end)
    for ax, (src, _, title) in zip(axes.flat, plot_sources):
        ax.plot(x, src.flatten()[start:end])
        ax.set_title(title)
        ax.set_ylabel("Acceleration (g)")
        ax.grid(alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Sample index")

    fig.suptitle(
        f"Drive End time signals — Samples {start} to {end}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    # EDA time/FFT plots are saved once from BearingFault.py; skip duplicates here.
    maybe_show()


def fft_features(segment: np.ndarray) -> np.ndarray:
    """Magnitude spectrum (single-sided FFT), normalized by window length."""
    return np.abs(rfft(segment)) / len(segment)


def time_domain_features(segment: np.ndarray) -> np.ndarray:
    """Per-window time-domain statistics: RMS, kurtosis, crest factor, peak-to-peak."""
    rms = np.sqrt(np.mean(segment ** 2))
    peak = np.max(np.abs(segment))
    crest = peak / rms if rms > 0 else 0.0
    return np.array([rms, kurtosis(segment), crest, np.max(segment) - np.min(segment)])


def window_features(segment: np.ndarray) -> np.ndarray:
    """Concatenate FFT magnitude bins with time-domain statistics."""
    return np.concatenate([fft_features(segment), time_domain_features(segment)])


def build_feature_dataset(
    signal: np.ndarray,
    label: str,
    region_start: int,
    region_end: int,
    window_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Slice a time region into non-overlapping windows and compute features."""
    features = []
    labels = []
    for start in range(region_start, region_end - window_size + 1, window_size):
        segment = signal[start : start + window_size]
        features.append(window_features(segment))
        labels.append(label)
    return np.array(features), labels


for start, end in plot_ranges:
    n_samples = end - start
    freqs = rfftfreq(n_samples, 1 / FS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (src, _, title) in zip(axes.flat, plot_sources):
        segment = src.flatten()[start:end]
        spectrum = np.abs(rfft(segment)) / n_samples
        ax.plot(freqs, spectrum)
        ax.set_title(title)
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Frequency (Hz)")

    fig.suptitle(
        f"Drive End FFT magnitude — Samples {start} to {end}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    maybe_show()


# --- FFT + time-domain Random Forest: 4-class bearing fault classification ---
N_FFT_BINS = WINDOW_SIZE // 2 + 1
N_TIME_FEATURES = 4  # RMS, kurtosis, crest factor, peak-to-peak
print(
    f"\nFFT + time-domain Random Forest — fault types: {', '.join(FAULT_CLASSES)} "
    f"(window={WINDOW_SIZE} samples, train={sorted(TRAIN_DEFECT_SIZES)}\", "
    f"test={sorted(TEST_DEFECT_SIZES)}\")"
)

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for src, label, _ in train_sources:
    signal = src.flatten()
    class_label = fault_type(label)
    if len(signal) < WINDOW_SIZE:
        raise ValueError(f"{label}: not enough samples for training windows.")
    X_tr, y_tr = build_feature_dataset(signal, class_label, 0, len(signal), WINDOW_SIZE)
    X_train_list.append(X_tr)
    y_train_list.extend(y_tr)

for src, label, _ in test_sources:
    signal = src.flatten()
    class_label = fault_type(label)
    if len(signal) < WINDOW_SIZE:
        print(f"Warning: {label} has too few samples for test windows — skipped.")
        continue
    X_te, y_te = build_feature_dataset(signal, class_label, 0, len(signal), WINDOW_SIZE)
    X_test_list.append(X_te)
    y_test_list.extend(y_te)

X_train = np.vstack(X_train_list)
y_train = np.array(y_train_list)
n_features = N_FFT_BINS + N_TIME_FEATURES
X_test = np.vstack(X_test_list) if X_test_list else np.empty((0, n_features))
y_test = np.array(y_test_list)

print(f"Training windows: {len(y_train)}  ({len(train_sources)} recordings)")
print(f"Test windows:     {len(y_test)}  ({len(test_sources)} recordings)")
for cls in FAULT_CLASSES:
    print(f"  {cls:<8} train={np.sum(y_train == cls):>4}  test={np.sum(y_test == cls):>4}")
print(
    f"Features per window: {X_train.shape[1]} "
    f"({N_FFT_BINS} FFT bins + {N_TIME_FEATURES} time-domain stats)"
)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=FAULT_CLASSES))

cm = confusion_matrix(y_test, y_pred, labels=FAULT_CLASSES)
print("Confusion matrix (rows=true, cols=predicted):")
print(f"{'':>10}", "  ".join(f"{lbl:>8}" for lbl in FAULT_CLASSES))
for i, true_label in enumerate(FAULT_CLASSES):
    row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(FAULT_CLASSES)))
    print(f"{true_label:>10}  {row}")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="YlOrBr", origin="lower")
ax.set_xticks(range(len(FAULT_CLASSES)))
ax.set_yticks(range(len(FAULT_CLASSES)))
ax.set_xticklabels(FAULT_CLASSES)
ax.set_yticklabels(FAULT_CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(
    "FFT + time-domain RF confusion matrix (4 fault types)\n"
    f"train: {sorted(TRAIN_DEFECT_SIZES)}\", test: {sorted(TEST_DEFECT_SIZES)}\" "
    f"(unknown severity), window={WINDOW_SIZE}",
)
text_threshold = cm.max() / 2
for i in range(len(FAULT_CLASSES)):
    for j in range(len(FAULT_CLASSES)):
        text_color = "white" if cm[i, j] > text_threshold else "black"
        ax.text(j, i, cm[i, j], ha="center", va="center", color=text_color)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
save_figure("cm_unknown_severity_fft.png")
maybe_show()
