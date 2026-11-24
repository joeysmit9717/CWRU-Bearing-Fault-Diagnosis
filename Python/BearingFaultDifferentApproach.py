import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
from scipy.signal import butter, hilbert, sosfiltfilt
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


# CWRU nominal shaft RPM by motor load (HP).
NOMINAL_RPM_BY_LOAD = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
N_HARMONICS = 3  # BPFI / BSF harmonics
N_BPFO_HARMONICS = 5  # extra BPFO harmonics for OR discrimination
N_RATIO_FEATURES = 2  # BPFO_sum/total, BPFO_1/BPFO_sum
N_ENVELOPE_STATS = 2  # envelope kurtosis, crest factor
BAND_HALF_WIDTH_HZ = 10.0  # ±10 Hz around each harmonic line


def load_hp_from_label(label: str) -> int:
    """Motor load (HP) is the trailing digit in every dataset label."""
    return int(label[-1])


def nominal_rpm(load_hp: int) -> float:
    """CWRU nominal shaft speed for a given load: 0/1/2/3 HP → 1797/1772/1750/1730."""
    return float(NOMINAL_RPM_BY_LOAD[load_hp])


def load_de_recording(
    mat_file: str, label: str, de_key: str | None = None
) -> tuple[np.ndarray, float]:
    """Load Drive End time signal; RPM from nominal speed for the file's HP load."""
    path = os.path.join(BASE_DIR, mat_file)
    data = loadmat(path)
    if de_key is None:
        de_key = next(k for k in data if not k.startswith("__") and "DE_time" in k)
    signal = data[de_key]
    rpm = nominal_rpm(load_hp_from_label(label))
    return signal, rpm


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


# CWRU Drive End bearing (SKF 6205-2RS JEM) geometry → fault-frequency multiples of fr.
# BPFO ≈ 3.5848·fr, BPFI ≈ 5.4152·fr, BSF ≈ 2.357·fr
N_BALLS = 9
BALL_DIAMETER_IN = 0.3126
PITCH_DIAMETER_IN = 1.537
CONTACT_ANGLE_RAD = 0.0
_BD_OVER_PD = BALL_DIAMETER_IN / PITCH_DIAMETER_IN
BPFO_ORDER = (N_BALLS / 2.0) * (1.0 - _BD_OVER_PD * np.cos(CONTACT_ANGLE_RAD))
BPFI_ORDER = (N_BALLS / 2.0) * (1.0 + _BD_OVER_PD * np.cos(CONTACT_ANGLE_RAD))
BSF_ORDER = (PITCH_DIAMETER_IN / (2.0 * BALL_DIAMETER_IN)) * (
    1.0 - (_BD_OVER_PD * np.cos(CONTACT_ANGLE_RAD)) ** 2
)

loaded_datasets: dict[str, np.ndarray] = {}
loaded_rpms: dict[str, float] = {}
train_sources: list[tuple[np.ndarray, str, str, float]] = []
test_sources: list[tuple[np.ndarray, str, str, float]] = []

for mat_file, label, de_key in DATASET_SPECS:
    path = os.path.join(BASE_DIR, mat_file)
    if not os.path.isfile(path):
        print(f"Skipping missing file: {mat_file}")
        continue
    signal, rpm = load_de_recording(mat_file, label, de_key)
    loaded_datasets[label] = signal
    loaded_rpms[label] = rpm
    entry = (signal, label, dataset_title(label), rpm)
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
    "only OR014 @ 6:00 is used for training. Normal baseline is 0 HP only (train). "
    "RPM from nominal map by load: 0/1/2/3 HP → 1797/1772/1750/1730."
)

N = 2000
FS = 12000  # CWRU Drive End sampling rate (Hz)
# Longer windows → finer envelope bins for ±10 Hz bands around BPFO/BPFI/BSF.
WINDOW_SIZE = 2048  # ~5.86 Hz resolution at 12 kHz
# Resonance bandpass before Hilbert envelope (Hz). Nyquist at 12 kHz is 6 kHz.
BANDPASS_LOW_HZ = 2000.0
BANDPASS_HIGH_HZ = 5000.0
BANDPASS_ORDER = 4
_BANDPASS_SOS = butter(
    BANDPASS_ORDER,
    [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
    btype="bandpass",
    fs=FS,
    output="sos",
)

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


def fault_orders_hz(rpm: float) -> dict[str, np.ndarray]:
    """BPFO / BPFI / BSF harmonic lines (Hz) for the given shaft RPM."""
    fr = rpm / 60.0
    return {
        "BPFO": BPFO_ORDER * fr * np.arange(1, N_BPFO_HARMONICS + 1, dtype=float),
        "BPFI": BPFI_ORDER * fr * np.arange(1, N_HARMONICS + 1, dtype=float),
        "BSF": BSF_ORDER * fr * np.arange(1, N_HARMONICS + 1, dtype=float),
    }


def band_energy(spectrum: np.ndarray, freqs: np.ndarray, center_hz: float) -> float:
    """Sum envelope-spectrum magnitude in ±BAND_HALF_WIDTH_HZ around center_hz."""
    mask = (freqs >= center_hz - BAND_HALF_WIDTH_HZ) & (
        freqs <= center_hz + BAND_HALF_WIDTH_HZ
    )
    if not np.any(mask):
        return 0.0
    return float(np.sum(spectrum[mask]))


def characteristic_frequency_features(segment: np.ndarray, rpm: float) -> np.ndarray:
    """Bandpass → Hilbert envelope FFT → absolute + relative BPFO/BPFI/BSF features.

    Feature order:
      BPFO_1..5, BPFI_1..3, BSF_1..3 (absolute band energies),
      BPFO_sum / (BPFO+BPFI+BSF), BPFO_1 / BPFO_sum (ratios),
      envelope kurtosis, envelope crest factor.
    """
    filtered = sosfiltfilt(_BANDPASS_SOS, segment)
    envelope = np.abs(hilbert(filtered))
    spectrum = np.abs(rfft(envelope)) / len(envelope)
    freqs = rfftfreq(len(envelope), 1.0 / FS)

    orders = fault_orders_hz(rpm)
    bpfo = np.array(
        [band_energy(spectrum, freqs, hz) for hz in orders["BPFO"]], dtype=float
    )
    bpfi = np.array(
        [band_energy(spectrum, freqs, hz) for hz in orders["BPFI"]], dtype=float
    )
    bsf = np.array(
        [band_energy(spectrum, freqs, hz) for hz in orders["BSF"]], dtype=float
    )

    bpfo_sum = float(np.sum(bpfo))
    bpfi_sum = float(np.sum(bpfi))
    bsf_sum = float(np.sum(bsf))
    total = bpfo_sum + bpfi_sum + bsf_sum
    bpfo_frac = bpfo_sum / total if total > 0 else 0.0
    bpfo_fund_frac = bpfo[0] / bpfo_sum if bpfo_sum > 0 else 0.0

    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    env_peak = float(np.max(envelope))
    env_kurtosis = float(kurtosis(envelope))
    env_crest = env_peak / env_rms if env_rms > 0 else 0.0

    return np.concatenate(
        [bpfo, bpfi, bsf, [bpfo_frac, bpfo_fund_frac, env_kurtosis, env_crest]]
    )


def window_features(segment: np.ndarray, rpm: float) -> np.ndarray:
    """Bandpass-envelope features at characteristic fault frequencies only."""
    return characteristic_frequency_features(segment, rpm)


def build_feature_dataset(
    signal: np.ndarray,
    label: str,
    rpm: float,
    region_start: int,
    region_end: int,
    window_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Slice a time region into non-overlapping windows and compute features."""
    features = []
    labels = []
    for start in range(region_start, region_end - window_size + 1, window_size):
        segment = signal[start : start + window_size]
        features.append(window_features(segment, rpm))
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


# --- Characteristic-frequency envelope RF: 4-class bearing fault classification ---
N_FEATURES = (
    N_BPFO_HARMONICS + 2 * N_HARMONICS + N_RATIO_FEATURES + N_ENVELOPE_STATS
)
print(
    f"\nBPFO/BPFI/BSF narrowband envelope RF — fault types: {', '.join(FAULT_CLASSES)} "
    f"(window={WINDOW_SIZE}, bandpass={BANDPASS_LOW_HZ:.0f}–{BANDPASS_HIGH_HZ:.0f} Hz, "
    f"±{BAND_HALF_WIDTH_HZ:.0f} Hz, BPFO=1..{N_BPFO_HARMONICS}, "
    f"BPFI/BSF=1..{N_HARMONICS}, +ratios +env kurt/crest, "
    f"train={sorted(TRAIN_DEFECT_SIZES)}\", test={sorted(TEST_DEFECT_SIZES)}\")"
)
print(
    f"  Fault orders (× shaft freq): BPFO={BPFO_ORDER:.4f}, "
    f"BPFI={BPFI_ORDER:.4f}, BSF={BSF_ORDER:.4f}"
)

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for src, label, _, rpm in train_sources:
    signal = src.flatten()
    class_label = fault_type(label)
    if len(signal) < WINDOW_SIZE:
        raise ValueError(f"{label}: not enough samples for training windows.")
    X_tr, y_tr = build_feature_dataset(
        signal, class_label, rpm, 0, len(signal), WINDOW_SIZE
    )
    X_train_list.append(X_tr)
    y_train_list.extend(y_tr)

for src, label, _, rpm in test_sources:
    signal = src.flatten()
    class_label = fault_type(label)
    if len(signal) < WINDOW_SIZE:
        print(f"Warning: {label} has too few samples for test windows — skipped.")
        continue
    X_te, y_te = build_feature_dataset(
        signal, class_label, rpm, 0, len(signal), WINDOW_SIZE
    )
    X_test_list.append(X_te)
    y_test_list.extend(y_te)

X_train = np.vstack(X_train_list)
y_train = np.array(y_train_list)
X_test = np.vstack(X_test_list) if X_test_list else np.empty((0, N_FEATURES))
y_test = np.array(y_test_list)

print(f"Training windows: {len(y_train)}  ({len(train_sources)} recordings)")
print(f"Test windows:     {len(y_test)}  ({len(test_sources)} recordings)")
for cls in FAULT_CLASSES:
    print(f"  {cls:<8} train={np.sum(y_train == cls):>4}  test={np.sum(y_test == cls):>4}")
print(
    f"Features per window: {X_train.shape[1]} "
    f"(BPFO_1..{N_BPFO_HARMONICS} + BPFI/BSF_1..{N_HARMONICS} absolute, "
    f"+ BPFO ratios + envelope kurtosis/crest)"
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
    "Bandpass→envelope BPFO/BPFI/BSF RF (+ ratios, env kurt/crest)\n"
    f"train: {sorted(TRAIN_DEFECT_SIZES)}\", test: {sorted(TEST_DEFECT_SIZES)}\" "
    f"(unknown severity), BPFO=1..{N_BPFO_HARMONICS}, "
    f"{BANDPASS_LOW_HZ:.0f}–{BANDPASS_HIGH_HZ:.0f} Hz",
)
text_threshold = cm.max() / 2
for i in range(len(FAULT_CLASSES)):
    for j in range(len(FAULT_CLASSES)):
        text_color = "white" if cm[i, j] > text_threshold else "black"
        ax.text(j, i, cm[i, j], ha="center", va="center", color=text_color)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
save_figure("cm_unknown_severity_envelope.png")
maybe_show()

