import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import gdown

# ══════════════════════════════════════════════════════════════════════════════
# ✅ STEP 1: PASTE YOUR GOOGLE DRIVE FILE IDs HERE
# For each .pth file: right-click → Share → Copy link
# Extract only the ID part: drive.google.com/file/d/THIS_PART/view
# ══════════════════════════════════════════════════════════════════════════════
RESNET_GDRIVE_ID  = "1LgHU0-FxIs4UGykU6WsH6QsW-p7QwgiV"
ALEXNET_GDRIVE_ID = "17vYkPxlvebAdCa9QIX5dg454ep1rI_MM"
CUSTOMCNN_GDRIVE_ID  = "157T3ZhZYGGOprNrhuzWPnp_QkiF1uVla"
EFFICIENT_GDRIVE_ID  = "1EckVZ1T8Dm-E-ec_82-9xQySrs8fWBS1"

RESNET_PATH    = "best_model.pth"
ALEXNET_PATH   = "best_alexnet.pth"
CUSTOMCNN_PATH = "custom_best_model.pth"           # we convert zip → single .pth
EFFICIENT_PATH = "best_efficientnet.pth"           # we convert zip → single .pth

# ══════════════════════════════════════════════════════════════════════════════
# ✅ STEP 2: CONFIRM CLASS ORDER
# Run in your notebook: print(full_train_dataset.classes)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ['FAKE', 'REAL']

# ══════════════════════════════════════════════════════════════════════════════
# ✅ STEP 3: PASTE YOUR REAL METRICS FROM NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
PRECOMPUTED_METRICS = {
    "ResNet18": {
        "accuracy": 0.9250, "precision": 0.9310, "recall": 0.9180,
        "f1": 0.9244, "specificity": 0.9320, "roc_auc": 0.9780,
        "conf_matrix": np.array([[4650, 350], [410, 4590]]),
        "fpr": None, "tpr": None,
    },
    "AlexNet": {
        "accuracy": 0.8950, "precision": 0.8900, "recall": 0.9010,
        "f1": 0.8954, "specificity": 0.8890, "roc_auc": 0.9520,
        "conf_matrix": np.array([[4445, 555], [495, 4505]]),
        "fpr": None, "tpr": None,
    },
    "CustomCNN": {
        "accuracy": 0.8700, "precision": 0.8650, "recall": 0.8750,
        "f1": 0.8700, "specificity": 0.8650, "roc_auc": 0.9300,
        "conf_matrix": np.array([[4325, 675], [625, 4375]]),
        "fpr": None, "tpr": None,
    },
    "EfficientNet": {
        "accuracy": 0.9400, "precision": 0.9450, "recall": 0.9350,
        "f1": 0.9400, "specificity": 0.9450, "roc_auc": 0.9850,
        "conf_matrix": np.array([[4725, 275], [325, 4675]]),
        "fpr": None, "tpr": None,
    },
}

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu")  # Streamlit Cloud has no GPU


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CNN ARCHITECTURE
# Reverse-engineered from your saved weights:
#   features.0  → Conv2d(3,  32, 3, padding=1)
#   features.1  → BatchNorm2d(32) + ReLU
#   features.2  → MaxPool2d(2,2)
#   features.4  → Conv2d(32, 64, 3, padding=1)
#   features.5  → BatchNorm2d(64) + ReLU
#   features.6  → MaxPool2d(2,2)
#   features.8  → Conv2d(64,128, 3, padding=1)
#   features.9  → BatchNorm2d(128) + ReLU
#   features.10 → MaxPool2d(2,2)
#   classifier.1 → Linear(8192, 256)
#   classifier.4 → Linear(256, 2)
# ══════════════════════════════════════════════════════════════════════════════
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 0
            nn.BatchNorm2d(32),                # 1
            nn.ReLU(inplace=True),             # 2
            nn.MaxPool2d(2, 2),                # 3
            nn.Conv2d(32, 64, 3, padding=1),   # 4
            nn.BatchNorm2d(64),                # 5
            nn.ReLU(inplace=True),             # 6
            nn.MaxPool2d(2, 2),                # 7
            nn.Conv2d(64, 128, 3, padding=1),  # 8
            nn.BatchNorm2d(128),               # 9
            nn.ReLU(inplace=True),             # 10
            nn.MaxPool2d(2, 2),                # 11
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 0
            nn.Linear(8192, 256),              # 1  (128 * 8 * 8 after 3x pool on 64px input)
            nn.ReLU(inplace=True),             # 2
            nn.Dropout(0.5),                   # 3
            nn.Linear(256, 2),                 # 4
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPER — handles large files & virus-scan warning
# ══════════════════════════════════════════════════════════════════════════════
def download_if_missing(path, gdrive_id):
    if os.path.exists(path):
        return

    st.info(f"⬇️ Downloading {path} from Google Drive (first run only)...")

    for url in [
        f"https://drive.google.com/uc?export=download&confirm=t&id={gdrive_id}",
        f"https://drive.google.com/uc?id={gdrive_id}",
        f"https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
    ]:
        try:
            gdown.download(url, path, quiet=False, fuzzy=True)
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                st.success(f"✅ Downloaded {path}")
                return
        except Exception:
            pass

    if os.path.exists(path):
        os.remove(path)

    st.error(
        f"❌ Could not download `{path}`.\n\n"
        f"Check: File ID is `{gdrive_id}` — make sure sharing is "
        f"**'Anyone on the internet with this link can view'**.\n\n"
        f"Test URL: `https://drive.google.com/uc?id={gdrive_id}`"
    )
    st.stop()


def download_zip_and_extract_pth(zip_path, pth_path, gdrive_id):
    """Download a zipped PyTorch model folder and save as a flat .pth file."""
    if os.path.exists(pth_path):
        return

    import zipfile, shutil, pickle, sys

    zip_file    = zip_path + ".zip"
    extract_dir = f"/tmp/extract_{zip_path}/"

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    download_if_missing(zip_file, gdrive_id)

    st.info(f"📦 Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(extract_dir)

    # Find the subfolder that contains data.pkl
    model_folder = None
    for root, dirs, files in os.walk(extract_dir):
        if "data.pkl" in files:
            model_folder = root
            break

    if model_folder is None:
        st.error(f"❌ No data.pkl found inside {zip_file}.")
        st.stop()

    st.info(f"🔄 Loading weights from extracted folder...")

    # PyTorch saves models as a folder (zip internally) when using torch.save on newer versions.
    # We load data.pkl manually then resolve the tensor storage files.
    data_dir = os.path.join(model_folder, "data")
    pkl_path = os.path.join(model_folder, "data.pkl")

    class StorageLoader(pickle.Unpickler):
        """Custom unpickler that loads tensor data from the data/ subfolder."""
        def persistent_load(self, pid):
            # pid format: ('storage', storage_type, key, location, size)
            storage_type, key, location, size = pid[1], pid[2], pid[3], pid[4]
            filename = os.path.join(data_dir, str(key))
            nbytes = size * torch.finfo(torch.float32).bits // 8
            with open(filename, "rb") as f:
                data = f.read()
            storage = torch.ByteStorage.from_buffer(data, byte_order="little")
            return torch.tensor([], dtype=torch.float32).set_(storage).view(-1)[:size]

    # Repack the extracted folder as a proper torch-compatible zip file
    # This is the most reliable approach across all PyTorch versions
    repacked = pth_path + ".tmp.pt"
    folder_name = os.path.basename(model_folder)
    with zipfile.ZipFile(repacked, "w", compression=zipfile.ZIP_STORED) as zout:
        for root, dirs, files in os.walk(model_folder):
            for file in sorted(files):
                filepath = os.path.join(root, file)
                # arcname must be: foldername/filename or foldername/data/key
                arcname = folder_name + filepath[len(model_folder):]
                zout.write(filepath, arcname)
    try:
        state_dict = torch.load(repacked, map_location="cpu", weights_only=False)
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        st.stop()
    finally:
        if os.path.exists(repacked):
            os.remove(repacked)

    torch.save(state_dict, pth_path)
    shutil.rmtree(extract_dir)
    st.success(f"✅ Model ready: {pth_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resnet():
    download_if_missing(RESNET_PATH, RESNET_GDRIVE_ID)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(RESNET_PATH, map_location="cpu"))
    return m.eval()


@st.cache_resource(show_spinner=False)
def load_alexnet():
    download_if_missing(ALEXNET_PATH, ALEXNET_GDRIVE_ID)
    m = models.alexnet(weights=None)
    m.classifier[6] = nn.Linear(4096, 2)
    m.load_state_dict(torch.load(ALEXNET_PATH, map_location="cpu"))
    return m.eval()


@st.cache_resource(show_spinner=False)
def load_customcnn():
    download_zip_and_extract_pth("custom_best_model", CUSTOMCNN_PATH, CUSTOMCNN_GDRIVE_ID)
    m = CustomCNN()
    m.load_state_dict(torch.load(CUSTOMCNN_PATH, map_location="cpu"))
    return m.eval()


@st.cache_resource(show_spinner=False)
def load_efficientnet():
    download_zip_and_extract_pth("best_efficientnet_cifake", EFFICIENT_PATH, EFFICIENT_GDRIVE_ID)
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(EFFICIENT_PATH, map_location="cpu"))
    return m.eval()


MODEL_LOADERS = {
    "ResNet18":    load_resnet,
    "AlexNet":     load_alexnet,
    "CustomCNN":   load_customcnn,
    "EfficientNet": load_efficientnet,
}


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.clone()       # clone prevents inplace view error

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].clone() # clone prevents inplace view error

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()

    def generate(self, img_tensor, class_idx=None):
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()
        img_tensor = img_tensor.clone()
        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire.")
        grads = self.gradients[0].detach()
        acts  = self.activations[0].detach()
        w   = grads.mean(dim=(1, 2), keepdim=True)
        cam = (w * acts).sum(dim=0).relu().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_gradcam_layer(model, model_name):
    """Return the best target layer for each architecture."""
    if model_name == "ResNet18":
        return model.layer4[-1]
    elif model_name == "AlexNet":
        return model.features[10]
    elif model_name == "CustomCNN":
        return model.features[8]   # last Conv2d
    elif model_name == "EfficientNet":
        return model.features[7]   # last conv block
    return None


def gradcam_figure(cam_np, img_np_224):
    cam_r    = cv2.resize(cam_np, (224, 224))
    heatmap  = plt.get_cmap('jet')(cam_r)[:, :, :3]
    overlay  = (0.45 * heatmap + 0.55 * img_np_224).clip(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [img_np_224, heatmap, overlay],
                                ["Original", "Grad-CAM", "Overlay"]):
        ax.imshow(data); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CIFAKE Detector", page_icon="🔍", layout="wide")
st.title("🔍 CIFAKE: Real vs AI-Generated Image Detector")
st.markdown("Detect whether an image is **REAL** or **AI-Generated** using 4 models.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Select Mode", ["🖼️ Single Model Prediction", "📊 Model Comparison"], index=0)
    st.markdown("---")
    if "Single" in mode:
        selected_model = st.selectbox("Choose Model",
                                      ["ResNet18", "AlexNet", "CustomCNN", "EfficientNet"])
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    st.markdown("---")
    st.markdown("**Dataset:** [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE MODEL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "Single" in mode:
    st.subheader(f"🤖 Predict with {selected_model}")
    uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img    = Image.open(uploaded).convert("RGB")
        img_np = np.array(img.resize((224, 224))) / 255.0

        col1, col2 = st.columns([1, 2])
        col1.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner(f"Loading {selected_model} and predicting..."):
            net   = MODEL_LOADERS[selected_model]()
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(net(img_t), dim=1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        with col2:
            badge = "🟢" if pred_label == "REAL" else "🔴"
            st.markdown(f"### {badge} Prediction: **{pred_label}**")
            st.progress(confidence / 100)
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
            st.markdown("**Class Probabilities:**")
            for i, name in enumerate(CLASS_NAMES):
                st.write(f"- {name}: `{probs[i]*100:.2f}%`")

        if show_gradcam:
            st.markdown("---")
            st.subheader("🌡️ Grad-CAM Explanation")
            with st.spinner("Generating Grad-CAM..."):
                try:
                    target = get_gradcam_layer(net, selected_model)
                    if target is None:
                        st.warning("Grad-CAM not supported for this model.")
                    else:
                        gc = GradCAM(net, target)
                        cam_np, _ = gc.generate(transform(img).unsqueeze(0).to(device), pred_idx)
                        gc.remove_hooks()
                        fig = gradcam_figure(cam_np, img_np)
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.warning(f"⚠️ Grad-CAM failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("📊 All 4 Models — Comparison Dashboard")
    st.info("Uses pre-computed training metrics. Scroll down for live side-by-side prediction.")

    keys   = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    model_names = list(PRECOMPUTED_METRICS.keys())
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorchid']

    # ── Metric table ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Metrics Summary")
    hdr = st.columns([2] + [1]*len(keys))
    hdr[0].markdown("**Model**")
    for c, lbl in zip(hdr[1:], labels):
        c.markdown(f"**{lbl}**")

    for name in model_names:
        m = PRECOMPUTED_METRICS[name]
        row = st.columns([2] + [1]*len(keys))
        row[0].markdown(f"**{name}**")
        for i, k in enumerate(keys):
            best_val = max(PRECOMPUTED_METRICS[n][k] for n in model_names)
            trophy = " 🏆" if m[k] == best_val else ""
            row[i+1].markdown(f"`{m[k]:.4f}`{trophy}")

    # Winner by F1
    winner = max(model_names, key=lambda n: PRECOMPUTED_METRICS[n]['f1'])
    st.success(f"🏆 **Overall Winner (F1-Score): {winner}**")
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    has_roc = any(PRECOMPUTED_METRICS[n]['fpr'] is not None for n in model_names)
    tab_names = ["📊 Bar Chart"] + (["📈 ROC Curves"] if has_roc else []) + ["🔢 Confusion Matrices"]
    tabs = st.tabs(tab_names)

    # Bar chart
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(13, 5))
        n_models = len(model_names)
        x  = np.arange(len(labels))
        bw = 0.8 / n_models
        for i, (name, color) in enumerate(zip(model_names, colors)):
            vals   = [PRECOMPUTED_METRICS[name][k] for k in keys]
            offset = (i - n_models/2 + 0.5) * bw
            bars   = ax.bar(x + offset, vals, bw, label=name, color=color, alpha=0.85)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                        f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.18)
        ax.set_title('All Models — Performance Metrics', fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # ROC curves (if available)
    if has_roc:
        with tabs[1]:
            fig, ax = plt.subplots(figsize=(7, 6))
            for name, color in zip(model_names, colors):
                m = PRECOMPUTED_METRICS[name]
                if m['fpr'] is not None:
                    ax.plot(m['fpr'], m['tpr'], lw=2, color=color,
                            label=f"{name} (AUC={m['roc_auc']:.4f})")
            ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('ROC Curves'); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    # Confusion matrices
    with tabs[-1]:
        cols = st.columns(4)
        cmaps = ['Blues', 'Reds', 'Greens', 'Purples']
        for col, name, cmap_n in zip(cols, model_names, cmaps):
            cm_mat = PRECOMPUTED_METRICS[name]['conf_matrix']
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(cm_mat, annot=True, fmt='d', cmap=cmap_n,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        ax=ax, cbar=False)
            ax.set_title(name, fontweight='bold', fontsize=10)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            col.pyplot(fig); plt.close(fig)

    # ── Live side-by-side ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🖼️ Live 4-Model Comparison")
    uploaded2 = st.file_uploader("Upload an image to run all 4 models",
                                  type=["jpg","jpeg","png"], key="compare_uploader")

    if uploaded2 is not None:
        img2  = Image.open(uploaded2).convert("RGB")
        img2_t = transform(img2).unsqueeze(0).to(device)

        with st.spinner("Loading all 4 models and predicting..."):
            results = {}
            for name, loader in MODEL_LOADERS.items():
                net = loader()
                with torch.no_grad():
                    probs = F.softmax(net(img2_t), dim=1)[0].cpu().numpy()
                results[name] = probs

        st.image(img2, caption="Uploaded Image", width=200)
        cols = st.columns(4)
        for col, (name, probs) in zip(cols, results.items()):
            pred   = CLASS_NAMES[int(np.argmax(probs))]
            conf   = float(max(probs)) * 100
            badge  = "🟢" if pred == "REAL" else "🔴"
            col.markdown(f"### {name}")
            col.markdown(f"**{badge} {pred}**")
            col.progress(conf / 100)
            col.markdown(f"`{conf:.1f}%` confident")
            for i, n in enumerate(CLASS_NAMES):
                col.write(f"- {n}: `{probs[i]*100:.2f}%`")

        # Agreement check
        preds = [CLASS_NAMES[int(np.argmax(p))] for p in results.values()]
        if len(set(preds)) == 1:
            st.success(f"✅ All 4 models agree: **{preds[0]}**")
        else:
            votes = {c: preds.count(c) for c in set(preds)}
            majority = max(votes, key=votes.get)
            st.warning(f"⚠️ Models disagree — majority vote: **{majority}** ({votes.get(majority,0)}/4 models)")
