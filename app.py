import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
import os
import gdown

# ──────────────────────────────────────────────
# CONFIG — PASTE YOUR GOOGLE DRIVE FILE IDs HERE
# ──────────────────────────────────────────────
RESNET_GDRIVE_ID  = "https://drive.google.com/file/d/1LgHU0-FxIs4UGykU6WsH6QsW-p7QwgiV/view?usp=sharing"
ALEXNET_GDRIVE_ID = "https://drive.google.com/file/d/17vYkPxlvebAdCa9QIX5dg454ep1rI_MM/view?usp=sharing"

RESNET_PATH  = "resnet_model.pth"
ALEXNET_PATH = "alexnet_model.pth"

CLASS_NAMES = ['FAKE', 'REAL']   # adjust if your training used different order

# ── Pre-computed metrics from your training (paste your actual values) ──────
# These are shown in the Comparison tab — no retraining needed!
PRECOMPUTED_METRICS = {
    "ResNet18": {
        "accuracy":    0.9250,   # <-- replace with your actual values
        "precision":   0.9310,
        "recall":      0.9180,
        "f1":          0.9244,
        "specificity": 0.9320,
        "roc_auc":     0.9780,
        "conf_matrix": np.array([[4650, 350], [410, 4590]]),  # example
        "fpr": np.linspace(0, 1, 100),
        "tpr": np.linspace(0, 1, 100) ** 0.5,
    },
    "AlexNet": {
        "accuracy":    0.8950,
        "precision":   0.8900,
        "recall":      0.9010,
        "f1":          0.8954,
        "specificity": 0.8890,
        "roc_auc":     0.9520,
        "conf_matrix": np.array([[4445, 555], [495, 4505]]),
        "fpr": np.linspace(0, 1, 100),
        "tpr": np.linspace(0, 1, 100) ** 0.65,
    },
}

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loaders ─────────────────────────────────────────────────────────────
def download_if_missing(path, gdrive_id):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {path} from Google Drive..."):
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, path, quiet=False)

@st.cache_resource
def load_resnet():
    download_if_missing(RESNET_PATH, RESNET_GDRIVE_ID)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    m.to(device).eval()
    return m

@st.cache_resource
def load_alexnet():
    download_if_missing(ALEXNET_PATH, ALEXNET_GDRIVE_ID)
    m = models.alexnet(weights=None)
    m.classifier[6] = nn.Linear(4096, 2)
    m.load_state_dict(torch.load(ALEXNET_PATH, map_location=device))
    m.to(device).eval()
    return m


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def generate(self, img_tensor, class_idx=None):
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.eval()
        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        grads = self.gradients[0]
        acts  = self.activations[0]
        w = grads.mean(dim=(1, 2), keepdim=True)
        cam = (w * acts).sum(dim=0).relu()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx


def apply_gradcam_overlay(cam_np, orig_img_np):
    cam_resized = cv2.resize(cam_np, (224, 224))
    cmap = plt.get_cmap('jet')
    heatmap = cmap(cam_resized)[:, :, :3]
    overlay = (0.45 * heatmap + 0.55 * orig_img_np).clip(0, 1)
    return heatmap, overlay


def denorm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = tensor.cpu().clone() * std + mean
    return np.clip(x.permute(1, 2, 0).numpy(), 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="CIFAKE Detector", page_icon="🔍", layout="wide")

st.title("🔍 CIFAKE: Real vs AI-Generated Image Detector")
st.markdown("Detect whether an image is **REAL** or **AI-Generated** using ResNet18 or AlexNet.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio(
        "Select Mode",
        ["🖼️ Single Model Prediction", "📊 Model Comparison"],
        index=0
    )
    st.markdown("---")
    if "Single" in mode:
        selected_model = st.selectbox("Choose Model", ["ResNet18", "AlexNet"])
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Models trained on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1 — SINGLE MODEL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "Single" in mode:
    st.subheader(f"🤖 Predict with {selected_model}")
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_display = img.resize((224, 224))
        img_np = np.array(img_display) / 255.0

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner(f"Loading {selected_model} and predicting..."):
            model = load_resnet() if selected_model == "ResNet18" else load_alexnet()
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs  = F.softmax(output, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx]) * 100

        with col2:
            color = "🟢" if pred_label == "REAL" else "🔴"
            st.markdown(f"### {color} Prediction: **{pred_label}**")
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
                    if selected_model == "ResNet18":
                        gc = GradCAM(model, model.layer4[-1])
                    else:
                        gc = GradCAM(model, model.features[10])

                    img_tensor_grad = transform(img).unsqueeze(0).to(device)
                    cam_np, pred_cls = gc.generate(img_tensor_grad)
                    heatmap, overlay = apply_gradcam_overlay(cam_np, img_np)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
                    axes[1].imshow(heatmap); axes[1].set_title("Heatmap"); axes[1].axis("off")
                    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — MODEL COMPARISON (uses pre-computed metrics — no retraining!)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("📊 ResNet18 vs AlexNet — Comparison Dashboard")
    st.info("ℹ️ Comparison uses pre-computed metrics from training. Upload an image below to compare live predictions from both models.")

    m_r = PRECOMPUTED_METRICS["ResNet18"]
    m_a = PRECOMPUTED_METRICS["AlexNet"]
    keys   = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    winner = "ResNet18" if m_r['f1'] >= m_a['f1'] else "AlexNet"

    # ── Metric table ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Metrics Summary")
    cols = st.columns(len(keys))
    for col, k, label in zip(cols, keys, labels):
        w = "🏆" if m_r[k] >= m_a[k] else ""
        col.metric(f"ResNet18 {label} {w}", f"{m_r[k]:.4f}")
    cols2 = st.columns(len(keys))
    for col, k, label in zip(cols2, keys, labels):
        w = "🏆" if m_a[k] > m_r[k] else ""
        col.metric(f"AlexNet {label} {w}", f"{m_a[k]:.4f}")

    st.success(f"🏆 **Overall Winner (F1-Score): {winner}**")
    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 ROC Curves", "🔢 Confusion Matrices"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        vr = [m_r[k] for k in keys]
        va = [m_a[k] for k in keys]
        x  = np.arange(len(labels))
        w  = 0.35
        b1 = ax.bar(x - w/2, vr, w, label='ResNet18', color='steelblue', alpha=0.85)
        b2 = ax.bar(x + w/2, va, w, label='AlexNet',  color='tomato',    alpha=0.85)
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.set_title('Performance Metrics — ResNet18 vs AlexNet', fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(m_r['fpr'], m_r['tpr'], 'b-', lw=2, label=f"ResNet18 (AUC={m_r['roc_auc']:.4f})")
        ax.plot(m_a['fpr'], m_a['tpr'], 'r-', lw=2, label=f"AlexNet  (AUC={m_a['roc_auc']:.4f})")
        ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves'); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        for col, name, cm_mat, cmap in [(c1, "ResNet18", m_r['conf_matrix'], 'Blues'),
                                         (c2, "AlexNet",  m_a['conf_matrix'], 'Reds')]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_mat, annot=True, fmt='d', cmap=cmap,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        ax=ax, cbar=False)
            ax.set_title(f'{name} Confusion Matrix', fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            col.pyplot(fig); plt.close()

    # ── Live comparison on uploaded image ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🖼️ Live Comparison — Upload an Image")
    uploaded = st.file_uploader("Upload image for side-by-side prediction", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with st.spinner("Running both models..."):
            rn = load_resnet()
            ax = load_alexnet()
            with torch.no_grad():
                out_r = F.softmax(rn(img_tensor), dim=1)[0].cpu().numpy()
                out_a = F.softmax(ax(img_tensor), dim=1)[0].cpu().numpy()

        col1, col2, col3 = st.columns(3)
        col1.image(img, caption="Uploaded Image", use_container_width=True)
        pred_r = CLASS_NAMES[int(np.argmax(out_r))]
        pred_a = CLASS_NAMES[int(np.argmax(out_a))]
        col2.markdown(f"### ResNet18\n**{pred_r}** `{max(out_r)*100:.1f}%`")
        for i, n in enumerate(CLASS_NAMES):
            col2.write(f"- {n}: `{out_r[i]*100:.2f}%`")
        col3.markdown(f"### AlexNet\n**{pred_a}** `{max(out_a)*100:.1f}%`")
        for i, n in enumerate(CLASS_NAMES):
            col3.write(f"- {n}: `{out_a[i]*100:.2f}%`")
