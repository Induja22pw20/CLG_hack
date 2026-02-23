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
import io
import zipfile
import gdown

# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE FILE IDs
# ══════════════════════════════════════════════════════════════════════════════


RESNET_PATH    = "best_model.pth"
ALEXNET_PATH   = "best_alexnet.pth"
CUSTOMCNN_PATH = "custom_best_model.pth"   # we convert the zip → flat .pth
EFFICIENT_PATH = "best_efficientnet.pth"

# ── Cache invalidation: delete old cached files if they exist from previous bad runs ──
# Remove this block after first successful deployment
for _stale in ["best_efficientnet.pth", "custom_best_model.pth"]:
    if os.path.exists(_stale):
        try:
            # Verify it loads correctly — delete if corrupt
            _obj = torch.load(_stale, map_location="cpu", weights_only=False)
            if not isinstance(_obj, dict) and not hasattr(_obj, "state_dict"):
                os.remove(_stale)
        except Exception:
            os.remove(_stale)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS NAMES  (alphabetical from ImageFolder → FAKE=0, REAL=1 for all models)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ['FAKE', 'REAL']

# ══════════════════════════════════════════════════════════════════════════════
# ✅ PASTE YOUR REAL METRICS FROM NOTEBOOK OUTPUT
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

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS — each model was trained with DIFFERENT transforms!
# ResNet18  : 224x224, ImageNet stats
# AlexNet   : 224x224, ImageNet stats
# CustomCNN : 64x64,   Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])   ← different!
# EfficientNet: 224x224, ImageNet stats
# ══════════════════════════════════════════════════════════════════════════════
TRANSFORM_IMAGENET = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

TRANSFORM_CUSTOM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

MODEL_TRANSFORMS = {
    "ResNet18":    TRANSFORM_IMAGENET,
    "AlexNet":     TRANSFORM_IMAGENET,
    "CustomCNN":   TRANSFORM_CUSTOM,    # ← must use this or predictions will be garbage
    "EfficientNet": TRANSFORM_IMAGENET,
}

device = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CNN ARCHITECTURE
# Input: 64x64  →  after 3x MaxPool(2)  →  8x8 feature maps
# Linear: 128 * 8 * 8 = 8192 → 256 → 2
# ══════════════════════════════════════════════════════════════════════════════
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    # 0  64→64
            nn.BatchNorm2d(32),                 # 1
            nn.ReLU(),                          # 2
            nn.MaxPool2d(2),                    # 3  64→32

            nn.Conv2d(32, 64, 3, padding=1),    # 4  32→32
            nn.BatchNorm2d(64),                 # 5
            nn.ReLU(),                          # 6
            nn.MaxPool2d(2),                    # 7  32→16

            nn.Conv2d(64, 128, 3, padding=1),   # 8  16→16  ← GradCAM target
            nn.BatchNorm2d(128),                # 9
            nn.ReLU(),                          # 10
            nn.MaxPool2d(2),                    # 11 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 0
            nn.Linear(128 * 8 * 8, 256),        # 1
            nn.ReLU(),                          # 2
            nn.Dropout(0.5),                    # 3
            nn.Linear(256, num_classes),        # 4
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def gdrive_download(gdrive_id, save_path):
    """Try multiple URL formats to handle large-file virus-scan warnings."""
    for url in [
        f"https://drive.google.com/uc?export=download&confirm=t&id={gdrive_id}",
        f"https://drive.google.com/uc?id={gdrive_id}",
        f"https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
    ]:
        try:
            gdown.download(url, save_path, quiet=False, fuzzy=True)
            if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                return True
        except Exception:
            pass
    return False


def ensure_flat_pth(pth_path, gdrive_id, label):
    """Download a plain flat state_dict .pth/.pt file."""
    if os.path.exists(pth_path):
        return
    st.info(f"⬇️ Downloading {label} from Google Drive...")
    ok = gdrive_download(gdrive_id, pth_path)
    if not ok:
        st.error(f"❌ Could not download {label}. File ID: `{gdrive_id}`")
        st.stop()
    st.success(f"✅ {label} ready!")


def ensure_customcnn_pth(pth_path, gdrive_id):
    """
    CustomCNN was saved as a PyTorch folder (torch.save without .pth extension).
    It was then zipped → the zip contains a folder with data.pkl + data/ files.
    We rename the inner folder to 'archive' (PyTorch's internal name) and pass
    as BytesIO to torch.load — this is how PyTorch reads .pth files internally.
    """
    if os.path.exists(pth_path):
        return

    zip_path = pth_path + ".zip"
    st.info("⬇️ Downloading CustomCNN from Google Drive...")
    ok = gdrive_download(gdrive_id, zip_path)
    if not ok:
        st.error(f"❌ Could not download CustomCNN. File ID: `{gdrive_id}`")
        st.stop()

    st.info("📦 Converting CustomCNN folder → .pth ...")
    try:
        with open(zip_path, 'rb') as f:
            raw = f.read()

        src = zipfile.ZipFile(io.BytesIO(raw), 'r')

        # Find the top-level folder name (e.g. 'custom_best_model')
        top = None
        for name in src.namelist():
            part = name.split('/')[0]
            if part:
                top = part
                break

        if top is None:
            raise ValueError("No top-level folder found in zip")

        # Rebuild zip renaming the folder to 'archive' — what torch.load expects
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED) as dst:
            for item in src.infolist():
                data     = src.read(item.filename)
                new_name = item.filename.replace(top, 'archive', 1)
                dst.writestr(new_name, data)
        src.close()

        buf.seek(0)
        state_dict = torch.load(buf, map_location='cpu', weights_only=False)
        torch.save(state_dict, pth_path)
        os.remove(zip_path)
        st.success("✅ CustomCNN ready!")

    except Exception as e:
        st.error(f"❌ Failed to convert CustomCNN: {e}")
        st.stop()


def smart_load_state_dict(model, path):
    """
    Robustly load weights regardless of how the file was saved:
    - plain state_dict (OrderedDict of tensors)
    - full model object (torch.save(model) instead of model.state_dict())
    - nested dict like {'model': state_dict}
    - DataParallel keys with 'module.' prefix
    """
    obj = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(obj, dict):
        # unwrap common nested wrappers
        for key in ['model', 'state_dict', 'model_state_dict', 'net']:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        # strip DataParallel prefix
        if any(k.startswith('module.') for k in obj.keys()):
            obj = {k[len('module.'):]: v for k, v in obj.items()}
        model.load_state_dict(obj, strict=False)

    elif hasattr(obj, 'state_dict'):
        sd = obj.state_dict()
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    else:
        raise RuntimeError(f'Cannot load weights: unexpected type {type(obj)}')

    return model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resnet():
    ensure_flat_pth(RESNET_PATH, RESNET_GDRIVE_ID, "ResNet18")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    smart_load_state_dict(m, RESNET_PATH)
    return m.eval()

@st.cache_resource(show_spinner=False)
def load_alexnet():
    ensure_flat_pth(ALEXNET_PATH, ALEXNET_GDRIVE_ID, "AlexNet")
    m = models.alexnet(weights=None)
    m.classifier[6] = nn.Linear(4096, 2)
    smart_load_state_dict(m, ALEXNET_PATH)
    return m.eval()

@st.cache_resource(show_spinner=False)
def load_customcnn():
    ensure_customcnn_pth(CUSTOMCNN_PATH, CUSTOMCNN_GDRIVE_ID)
    m = CustomCNN(num_classes=2)
    smart_load_state_dict(m, CUSTOMCNN_PATH)
    return m.eval()

def smart_load_state_dict(model, path):
    """
    Robustly load weights into model regardless of how they were saved:
    - plain state_dict (OrderedDict)
    - full model object  (torch.save(model))
    - nested dict        ({'model': state_dict} or {'state_dict': ...})
    - DataParallel keys  ('module.' prefix)
    """
    obj = torch.load(path, map_location='cpu', weights_only=False)

    # Case 1: already a state dict (OrderedDict of tensors)
    if isinstance(obj, dict):
        # Check for nested wrapper keys
        for key in ['model', 'state_dict', 'model_state_dict', 'net']:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        # Strip DataParallel 'module.' prefix if present
        if any(k.startswith('module.') for k in obj.keys()):
            obj = {k[len('module.'):]: v for k, v in obj.items()}
        model.load_state_dict(obj)

    # Case 2: full model object saved with torch.save(model)
    elif hasattr(obj, 'state_dict'):
        sd = obj.state_dict()
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)

    else:
        raise RuntimeError(f"Cannot load weights from object of type {type(obj)}")

    return model


@st.cache_resource(show_spinner=False)
def load_efficientnet():
    ensure_customcnn_pth(EFFICIENT_PATH, EFFICIENT_GDRIVE_ID)
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    smart_load_state_dict(m, EFFICIENT_PATH)
    return m.eval()

MODEL_LOADERS = {
    "ResNet18":     load_resnet,
    "AlexNet":      load_alexnet,
    "CustomCNN":    load_customcnn,
    "EfficientNet": load_efficientnet,
}


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# Target layers (from notebooks):
#   ResNet18    → model.layer4[-1]     (last residual block)
#   AlexNet     → model.features[10]   (last conv)
#   CustomCNN   → model.features[8]    (Conv 64→128, notebook Cell 29)
#   EfficientNet→ model.features[-1]   (last features block, notebook Cell 20)
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.clone()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].clone()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()

    def generate(self, img_tensor, class_idx=None):
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()
        img_tensor = img_tensor.clone()
        out = self.model(img_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        grads = self.gradients[0].detach()
        acts  = self.activations[0].detach()
        w   = grads.mean(dim=(1, 2), keepdim=True)
        cam = (w * acts).sum(dim=0).relu().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_gradcam_layer(model, model_name):
    return {
        "ResNet18":     model.layer4[-1],
        "AlexNet":      model.features[10],
        "CustomCNN":    model.features[8],
        "EfficientNet": model.features[-1],
    }.get(model_name)


def gradcam_figure(cam_np, img_np):
    """img_np should be HxWx3 float [0,1]."""
    target_size = (img_np.shape[1], img_np.shape[0])  # (W, H) for cv2
    cam_r   = cv2.resize(cam_np, target_size)
    heatmap = plt.get_cmap('jet')(cam_r)[:, :, :3]
    overlay = (0.45 * heatmap + 0.55 * img_np).clip(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes, [img_np, heatmap, overlay],
                                ["Original", "Grad-CAM", "Overlay"]):
        ax.imshow(data); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    return fig


def denorm_for_display(img_tensor, model_name):
    """Reverse the normalization for display, returning HxWx3 float [0,1]."""
    t = img_tensor.clone().cpu()
    if model_name == "CustomCNN":
        # reverse Normalize([0.5]*3, [0.5]*3)
        t = t * 0.5 + 0.5
    else:
        # reverse ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        t = t * std + mean
    return np.clip(t.permute(1,2,0).numpy(), 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CIFAKE Detector", page_icon="🔍", layout="wide")
st.title("🔍 CIFAKE: Real vs AI-Generated Image Detector")
st.markdown("Detect whether an image is **REAL** or **AI-Generated** using 4 trained models.")

with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Select Mode",
                    ["🖼️ Single Model Prediction", "📊 Model Comparison"], index=0)
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
    uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg","jpeg","png"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")

        # Use the correct transform for this model
        tfm   = MODEL_TRANSFORMS[selected_model]
        img_t = tfm(img).unsqueeze(0).to(device)

        # For display: always show at 224x224
        img_display_np = np.array(img.resize((224, 224))) / 255.0

        col1, col2 = st.columns([1, 2])
        col1.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner(f"Loading {selected_model} and predicting..."):
            net = MODEL_LOADERS[selected_model]()
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
            for i, n in enumerate(CLASS_NAMES):
                st.write(f"- {n}: `{probs[i]*100:.2f}%`")

        if show_gradcam:
            st.markdown("---")
            st.subheader("🌡️ Grad-CAM Explanation")
            with st.spinner("Generating Grad-CAM..."):
                try:
                    layer  = get_gradcam_layer(net, selected_model)
                    gc     = GradCAM(net, layer)
                    # Use same transform the model expects
                    t_grad = tfm(img).unsqueeze(0).to(device)
                    cam_np, _ = gc.generate(t_grad, pred_idx)
                    gc.remove_hooks()
                    # Denorm correctly for display
                    img_np = denorm_for_display(tfm(img), selected_model)
                    st.pyplot(gradcam_figure(cam_np, img_np))
                    plt.close()
                except Exception as e:
                    st.warning(f"⚠️ Grad-CAM failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("📊 All 4 Models — Comparison Dashboard")
    st.info("Uses pre-computed training metrics. Scroll down for live 4-model prediction.")

    keys        = ['accuracy','precision','recall','f1','specificity','roc_auc']
    labels      = ['Accuracy','Precision','Recall','F1-Score','Specificity','ROC-AUC']
    model_names = list(PRECOMPUTED_METRICS.keys())
    colors      = ['steelblue','tomato','seagreen','darkorchid']

    # Metric table
    st.markdown("### 📋 Metrics Summary")
    hdr = st.columns([2] + [1]*len(keys))
    hdr[0].markdown("**Model**")
    for c, lbl in zip(hdr[1:], labels):
        c.markdown(f"**{lbl}**")

    for name in model_names:
        m   = PRECOMPUTED_METRICS[name]
        row = st.columns([2] + [1]*len(keys))
        row[0].markdown(f"**{name}**")
        for i, k in enumerate(keys):
            best   = max(PRECOMPUTED_METRICS[n][k] for n in model_names)
            trophy = " 🏆" if m[k] == best else ""
            row[i+1].markdown(f"`{m[k]:.4f}`{trophy}")

    winner = max(model_names, key=lambda n: PRECOMPUTED_METRICS[n]['f1'])
    st.success(f"🏆 **Overall Winner (F1-Score): {winner}**")
    st.markdown("---")

    has_roc   = any(PRECOMPUTED_METRICS[n]['fpr'] is not None for n in model_names)
    tab_names = ["📊 Bar Chart"] + (["📈 ROC Curves"] if has_roc else []) + ["🔢 Confusion Matrices"]
    tabs      = st.tabs(tab_names)

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(13, 5))
        nm = len(model_names)
        x  = np.arange(len(labels))
        bw = 0.8 / nm
        for i, (name, color) in enumerate(zip(model_names, colors)):
            vals   = [PRECOMPUTED_METRICS[name][k] for k in keys]
            offset = (i - nm/2 + 0.5) * bw
            bars   = ax.bar(x + offset, vals, bw, label=name, color=color, alpha=0.85)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                        f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.18)
        ax.set_title('All 4 Models — Performance Metrics', fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    if has_roc:
        with tabs[1]:
            fig, ax = plt.subplots(figsize=(7, 6))
            for name, color in zip(model_names, colors):
                m = PRECOMPUTED_METRICS[name]
                if m['fpr'] is not None:
                    ax.plot(m['fpr'], m['tpr'], lw=2, color=color,
                            label=f"{name} (AUC={m['roc_auc']:.4f})")
            ax.plot([0,1],[0,1],'k--', lw=1)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('ROC Curves'); ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    with tabs[-1]:
        cols  = st.columns(4)
        cmaps = ['Blues','Reds','Greens','Purples']
        for col, name, cmap_n in zip(cols, model_names, cmaps):
            cm = PRECOMPUTED_METRICS[name]['conf_matrix']
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_n,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        ax=ax, cbar=False)
            ax.set_title(name, fontweight='bold', fontsize=10)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            col.pyplot(fig); plt.close(fig)

    # Live 4-model prediction
    st.markdown("---")
    st.subheader("🖼️ Live 4-Model Comparison")
    uploaded2 = st.file_uploader("Upload an image to run all 4 models",
                                  type=["jpg","jpeg","png"], key="compare_uploader")

    if uploaded2 is not None:
        img2 = Image.open(uploaded2).convert("RGB")

        with st.spinner("Loading all 4 models and predicting..."):
            results = {}
            for name, loader in MODEL_LOADERS.items():
                net    = loader()
                img2_t = MODEL_TRANSFORMS[name](img2).unsqueeze(0).to(device)
                with torch.no_grad():
                    results[name] = F.softmax(net(img2_t), dim=1)[0].cpu().numpy()

        st.image(img2, caption="Uploaded Image", width=200)
        cols = st.columns(4)
        for col, (name, probs) in zip(cols, results.items()):
            pred  = CLASS_NAMES[int(np.argmax(probs))]
            conf  = float(max(probs)) * 100
            badge = "🟢" if pred == "REAL" else "🔴"
            col.markdown(f"### {name}")
            col.markdown(f"**{badge} {pred}**")
            col.progress(conf / 100)
            col.markdown(f"`{conf:.1f}%` confident")
            for i, n in enumerate(CLASS_NAMES):
                col.write(f"- {n}: `{probs[i]*100:.2f}%`")

        preds    = [CLASS_NAMES[int(np.argmax(p))] for p in results.values()]
        majority = max(set(preds), key=preds.count)
        votes    = preds.count(majority)
        if votes == 4:
            st.success(f"✅ All 4 models agree: **{majority}**")
        else:
            st.warning(f"⚠️ Models disagree — majority vote: **{majority}** ({votes}/4)")



