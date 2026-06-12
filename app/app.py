import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from obspy import read
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import tempfile
from pathlib import Path

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(r"D:\Seismic_Monitoring_AI")
MODEL_PATH = PROJECT_DIR / "models" / "best_seismic_model_v3.pth"
EXAMPLES_DIR = PROJECT_DIR / "examples"

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Seismic Monitoring AI",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI-Powered Seismic Monitoring System")

st.markdown(
    """
    Upload a MiniSEED file and classify the event as:

    - Noise
    - Avalanche
    - Earthquake
    """
)

# ==================================================
# LABEL MAP
# ==================================================

LABEL_MAP = {
    0: "Noise",
    1: "Avalanche",
    2: "Earthquake"
}

# ==================================================
# MODEL ARCHITECTURE
# ==================================================

class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================================================
# DEVICE
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    model = SeismicCNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()
st.success("CNN Model Loaded Successfully")

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Home",
        "Classify Files",
        "Model Performance",
        "Example Events",
        "About Project"
    ]
)

# ==================================================
# HELPER FUNCTIONS
# ==================================================

WINDOW_SIZE = 2000


def preprocess_window(window):
    window = window.astype(np.float32)
    window = window - window.mean()
    window = window / (window.std() + 1e-8)

    if len(window) > WINDOW_SIZE:
        window = window[:WINDOW_SIZE]
    elif len(window) < WINDOW_SIZE:
        window = np.pad(window, (0, WINDOW_SIZE - len(window)))

    return window


def predict_window(window):
    window = preprocess_window(window)
    window = torch.tensor(window, dtype=torch.float32)

    # (2000,) -> (1, 1, 2000)
    window = window.unsqueeze(0)
    window = window.unsqueeze(0)

    window = window.to(device)

    with torch.no_grad():
        outputs = model(window)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)

    return LABEL_MAP[pred.item()], probs.cpu().numpy()[0]


def extract_high_energy_window(signal, window_size=WINDOW_SIZE):
    signal = signal.astype(np.float32)

    energy = signal ** 2
    kernel = np.ones(window_size) / window_size
    rolling_energy = np.convolve(energy, kernel, mode="same")

    center = np.argmax(rolling_energy)

    left = center - window_size // 2
    right = center + window_size // 2

    if left < 0:
        left = 0
        right = window_size

    if right > len(signal):
        right = len(signal)
        left = right - window_size

    window = signal[left:right]

    if len(window) < window_size:
        window = np.pad(window, (0, window_size - len(window)))

    return window, center


def plot_spectrogram(signal, fs=200):
    signal = signal - np.mean(signal)
    signal = signal / (np.std(signal) + 1e-8)

    f, t, Sxx = spectrogram(
        signal,
        fs=fs,
        nperseg=256,
        noverlap=128
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(
        t, f, Sxx_db,
        shading="gouraud",
        cmap="viridis"
    )

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Improved Spectrogram")
    ax.set_ylim(0, 80)

    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    st.pyplot(fig)

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":
    st.header("AI-Powered Seismic Monitoring System")

    st.markdown(
        """
        This project uses deep learning to classify seismic signals into:

        - Noise
        - Avalanche
        - Earthquake

        Features included in this version:

        ✅ MiniSEED Support  
        ✅ CNN Classification  
        ✅ Spectrogram Analysis  
        ✅ Batch Processing  
        ✅ CSV Export  
        ✅ Professional Dashboard Layout  
        """
    )

    st.info("Go to **Classify Files** from the sidebar to upload and analyze MiniSEED files.")

# ==================================================
# CLASSIFY FILES PAGE
# ==================================================

elif page == "Classify Files":

    uploaded_files = st.file_uploader(
        "Upload MiniSEED Files",
        type=["mseed"],
        accept_multiple_files=True
    )

    all_results = []

    if uploaded_files:

        for uploaded_file in uploaded_files:

            st.markdown("---")
            st.header(f"📂 {uploaded_file.name}")

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mseed"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            tr = read(temp_path)[0]
            signal = tr.data.astype(np.float32)

            st.success("MiniSEED Loaded Successfully")

            # ----------------------------------------------
            # Metadata
            # ----------------------------------------------

            st.subheader("Waveform Metadata")

            st.write(
                {
                    "Station": tr.stats.station,
                    "Channel": tr.stats.channel,
                    "Sampling Rate": tr.stats.sampling_rate,
                    "Samples": len(signal),
                    "Start Time": str(tr.stats.starttime),
                }
            )

            # ----------------------------------------------
            # Extract Event Window
            # ----------------------------------------------

            window, picked_center = extract_high_energy_window(signal)

            # ----------------------------------------------
            # Waveform Plot
            # ----------------------------------------------

            st.subheader("Waveform")

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(signal, linewidth=1)
            ax.axvline(
                picked_center,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Detected Event"
            )
            ax.set_title("Seismic Waveform")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.legend()
            st.pyplot(fig)

            # ----------------------------------------------
            # Prediction
            # ----------------------------------------------

            prediction, probs = predict_window(window)
            confidence = float(probs.max())

            all_results.append({
                "File": uploaded_file.name,
                "Prediction": prediction,
                "Confidence": round(confidence, 4)
            })

            st.subheader("AI Classification")
            st.success(f"Predicted Event: {prediction}")

            # ----------------------------------------------
            # Confidence Chart
            # ----------------------------------------------

            st.subheader("Confidence Scores")

            confidence_df = pd.DataFrame(
                {
                    "Class": ["Noise", "Avalanche", "Earthquake"],
                    "Probability": probs
                }
            )

            st.bar_chart(confidence_df.set_index("Class"))

            # ----------------------------------------------
            # Spectrogram
            # ----------------------------------------------

            st.subheader("Spectrogram")
            plot_spectrogram(window)

        # ==================================================
        # BATCH RESULTS
        # ==================================================

        if len(all_results) > 0:

            st.markdown("---")
            st.header("📊 Batch Classification Results")

            results_df = pd.DataFrame(all_results)

            st.dataframe(
                results_df,
                use_container_width=True
            )

            csv = results_df.to_csv(index=False)

            st.download_button(
                label="⬇ Download Results CSV",
                data=csv,
                file_name="seismic_results.csv",
                mime="text/csv"
            )

    else:
        st.info("Please upload one or more MiniSEED files to begin classification.")

# ==================================================
# MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.header("📊 Model Performance")

    st.markdown(
        """
        Performance of the final CNN model on the test dataset.
        """
    )

    # =====================================
    # Top Metrics
    # =====================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "80.02%")

    with col2:
        st.metric("Macro Precision", "80%")

    with col3:
        st.metric("Macro Recall", "61%")

    with col4:
        st.metric("Macro F1", "67%")

    st.markdown("---")

    # =====================================
    # Class-wise Metrics
    # =====================================

    st.subheader("Class-wise Performance")

    performance_df = pd.DataFrame({
        "Class": [
            "Noise",
            "Avalanche",
            "Earthquake"
        ],
        "Precision": [
            0.81,
            0.88,
            0.72
        ],
        "Recall": [
            0.95,
            0.44,
            0.45
        ],
        "F1 Score": [
            0.87,
            0.59,
            0.55
        ],
        "Support": [
            652,
            84,
            185
        ]
    })

    st.dataframe(performance_df, use_container_width=True)

    st.markdown("---")

    # =====================================
    # Confusion Matrix
    # =====================================

    st.subheader("Confusion Matrix")

    cm = np.array([
        [617, 3, 32],
        [47, 37, 0],
        [100, 2, 83]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    classes = [
        "Noise",
        "Avalanche",
        "Earthquake"
    ]

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # =====================================
    # Training Summary
    # =====================================

    st.subheader("Training Summary")

    training_df = pd.DataFrame({
        "Metric": [
            "Final Training Accuracy",
            "Final Training Loss",
            "Test Accuracy",
            "Training Samples",
            "Test Samples",
            "Number of Classes",
            "Stations Used"
        ],
        "Value": [
            "80.56%",
            "0.4840",
            "80.02%",
            "3684",
            "921",
            "3",
            "5"
        ]
    })

    st.table(training_df)

# ==================================================
# EXAMPLE EVENTS PAGE
# ==================================================

elif page == "Example Events":

    st.header("📚 Example Events")

    example = st.selectbox(
        "Choose Event",
        [
            "Earthquake",
            "Avalanche",
            "Noise"
        ]
    )

    example_files = {
        "Earthquake": EXAMPLES_DIR / "earthquake.mseed",
        "Avalanche": EXAMPLES_DIR / "avalanche.mseed",
        "Noise": EXAMPLES_DIR / "noise.mseed"
    }

    file_path = example_files[example]

    if not file_path.exists():
        st.error(f"Example file not found: {file_path}")
    else:
        tr = read(str(file_path))[0]
        signal = tr.data.astype(np.float32)

        window, picked_center = extract_high_energy_window(signal)
        prediction, probs = predict_window(window)

        st.success(f"Prediction: {prediction}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(signal)
        ax.axvline(
            picked_center,
            color="red",
            linestyle="--"
        )
        ax.set_title(f"{example} Example Waveform")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        confidence_df = pd.DataFrame({
            "Class": [
                "Noise",
                "Avalanche",
                "Earthquake"
            ],
            "Probability": probs
        })

        st.bar_chart(
            confidence_df.set_index("Class")
        )

        plot_spectrogram(window)

# ==================================================
# ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.header("About This Project")

    st.markdown(
        """
        Developed using:

        - Python
        - PyTorch
        - Streamlit
        - ObsPy
        - NumPy
        - Pandas

        Dataset:
        Multi-station seismic recordings containing:

        - Noise
        - Avalanche
        - Earthquake
        """
    )