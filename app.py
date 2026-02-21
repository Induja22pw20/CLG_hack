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
# ══════════════════════════════════════════════════════════════════════════════
RESNET_GDRIVE_ID  = "1LgHU0-FxIs4UGykU6WsH6QsW-p7QwgiV"
ALEXNET_GDRIVE_ID = "17vYkPxlvebAdCa9QIX5dg454ep1rI_MM"

RESNET_PATH  = "best_model.pth"
ALEXNET_PATH = "alexnet_model.pth"

# ══════════════════════════════════════════════════════════════════════════════
# ✅ STEP 2: SET YOUR CLASS ORDER
# Run this in your notebook:  print(full_train_dataset.classes)
# It prints ['FAKE', 'REAL'] or ['REAL', 'FAKE'] — match it exactly below
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ['FAKE', 'REAL']

# ══════════════════════════════════════════════════════════════════════════════
# ✅ STEP 3: PASTE YOUR ACTUAL METRICS FROM NOTEBOOK OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
PRECOMPUTED_METRICS = {
    "ResNet18": {
        "accuracy":    0.9250,   # <-- replace with your real values
        "precision":   0.9310,
        "recall":      0.9180,
        "f1":          0.9244,
        "specificity": 0.9320,
        "roc_auc":     0.9780,
        "conf_matrix": np.array([[4650, 350], [410, 4590]]),  # [[TN,FP],[FN,TP]]
        "fpr": None,   # set to your fpr array if you have it, else leave None
        "tpr": None,   # set to your tpr array if you have it, else leave None
    },
    "AlexNet": {
        "accuracy":    0.8950,
        "precision":   0.8900,
        "recall":      0.9010,
        "f1":          0.8954,
        "specificity": 0.8890,
        "roc_auc":     0.9520,
        "conf_matrix": np.array([[4445, 555], [495, 4505]]),
        "fpr": None,
        "tpr": None,
    },
}

# ── Transform (must exactly match what was used during training) ───────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu")   # Streamlit Cloud has no GPU — always use CPU


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD & LOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_if_missing(path, gdrive_id):
    """Download model from Google Drive if not already on disk.
    Tries multiple URL formats to handle large files and virus-scan warnings."""
    if os.path.exists(path):
        return  # already downloaded

    st.info(f"⬇️ Downloading {path} from Google Drive (first run only, may take a minute)...")

    # Try 1: confirm=t bypasses the "too large to virus scan" warning page
    try:
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={gdrive_id}"
        gdown.download(url, path, quiet=False, fuzzy=True)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            st.success(f"✅ Downloaded {path}")
            return
    except Exception:
        pass

    # Try 2: fuzzy=True alone
    try:
        url2 = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url2, path, quiet=False, fuzzy=True)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            st.success(f"✅ Downloaded {path}")
            return
    except Exception:
        pass

    # Try 3: full share link format with fuzzy
    try:
        url3 = f"https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing"
        gdown.download(url3, path, quiet=False, fuzzy=True)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            st.success(f"✅ Downloaded {path}")
            return
    except Exception:
        pass

    # All failed — clean up any partial file and show help
    if os.path.exists(path):
        os.remove(path)

    st.error(
        f"❌ **Could not download `{path}` from Google Drive.**\n\n"
        f"**Most likely causes:**\n"
        f"1. File sharing is not set to **'Anyone on the internet with this link can view'** — "
        f"make sure it's fully public, not just 'your organisation'\n"
        f"2. File ID `{gdrive_id}` is wrong — double-check by opening the link: "
        f"`https://drive.google.com/file/d/{gdrive_id}/view`\n"
        f"3. Google Drive download quota exceeded — try sharing via a different account\n\n"
        f"**Quick test:** Open this URL in a browser — if it asks you to log in, the sharing is wrong:\n"
        f"`https://drive.google.com/uc?id={gdrive_id}`"
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def load_resnet():
    download_if_missing(RESNET_PATH, RESNET_GDRIVE_ID)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(RESNET_PATH, map_location="cpu"))
    m.to(device).eval()
    return m


@st.cache_resource(show_spinner=False)
def load_alexnet():
    download_if_missing(ALEXNET_PATH, ALEXNET_GDRIVE_ID)
    m = models.alexnet(weights=None)
    m.classifier[6] = nn.Linear(4096, 2)
    m.load_state_dict(torch.load(ALEXNET_PATH, map_location="cpu"))
    m.to(device).eval()
    return m


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

        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire — check target layer.")

        grads = self.gradients[0].detach()   # C, H, W
        acts  = self.activations[0].detach() # C, H, W
        w   = grads.mean(dim=(1, 2), keepdim=True)
        cam = (w * acts).sum(dim=0).relu().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def gradcam_figure(cam_np, img_np_224):
    cam_r = cv2.resize(cam_np, (224, 224))
    heatmap = plt.get_cmap('jet')(cam_r)[:, :, :3]
    overlay = (0.45 * heatmap + 0.55 * img_np_224).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [img_np_224, heatmap, overlay],
                                ["Original", "Grad-CAM", "Overlay"]):
        ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CIFAKE Detector", page_icon="🔍", layout="wide")
st.title("🔍 CIFAKE: Real vs AI-Generated Image Detector")
st.markdown("Upload any image to detect whether it is **REAL** or **AI-Generated**.")

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
        show_gradcam   = st.checkbox("Show Grad-CAM Heatmap", value=True)
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
        img_np = np.array(img.resize((224, 224))) / 255.0  # for display + gradcam

        col1, col2 = st.columns([1, 2])
        col1.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner(f"Loading {selected_model} and predicting..."):
            net = load_resnet() if selected_model == "ResNet18" else load_alexnet()
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
                    target = net.layer4[-1] if selected_model == "ResNet18" else net.features[10]
                    gc     = GradCAM(net, target)
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
    st.subheader("📊 ResNet18 vs AlexNet — Comparison Dashboard")
    st.info("Uses your pre-computed training metrics. Scroll down to run a live side-by-side prediction too.")

    m_r = PRECOMPUTED_METRICS["ResNet18"]
    m_a = PRECOMPUTED_METRICS["AlexNet"]
    keys   = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    winner = "ResNet18" if m_r['f1'] >= m_a['f1'] else "AlexNet"

    # ── Metric table ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Metrics Summary")
    hdr = st.columns([2] + [1]*len(keys))
    hdr[0].markdown("**Model**")
    for c, lbl in zip(hdr[1:], labels):
        c.markdown(f"**{lbl}**")

    row_r = st.columns([2] + [1]*len(keys))
    row_r[0].markdown("🔵 **ResNet18**")
    for c, k in zip(row_r[1:], keys):
        row_r[keys.index(k)+1].markdown(f"`{m_r[k]:.4f}`" + (" 🏆" if m_r[k] >= m_a[k] else ""))

    row_a = st.columns([2] + [1]*len(keys))
    row_a[0].markdown("🔴 **AlexNet**")
    for c, k in zip(row_a[1:], keys):
        row_a[keys.index(k)+1].markdown(f"`{m_a[k]:.4f}`" + (" 🏆" if m_a[k] > m_r[k] else ""))

    st.success(f"🏆 **Overall Winner (F1-Score): {winner}**")
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    has_roc = m_r['fpr'] is not None and m_a['fpr'] is not None
    tab_names = ["📊 Bar Chart"] + (["📈 ROC Curves"] if has_roc else []) + ["🔢 Confusion Matrices"]
    tabs = st.tabs(tab_names)

    # Bar chart tab
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(11, 5))
        vr, va  = [m_r[k] for k in keys], [m_a[k] for k in keys]
        x, bw   = np.arange(len(labels)), 0.35
        b1 = ax.bar(x - bw/2, vr, bw, label='ResNet18', color='steelblue', alpha=0.85)
        b2 = ax.bar(x + bw/2, va, bw, label='AlexNet',  color='tomato',    alpha=0.85)
        for b in list(b1) + list(b2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.set_title('ResNet18 vs AlexNet — Performance Metrics', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # ROC tab (only if arrays provided)
    if has_roc:
        with tabs[1]:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(m_r['fpr'], m_r['tpr'], 'b-', lw=2, label=f"ResNet18 (AUC={m_r['roc_auc']:.4f})")
            ax.plot(m_a['fpr'], m_a['tpr'], 'r-', lw=2, label=f"AlexNet  (AUC={m_a['roc_auc']:.4f})")
            ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title('ROC Curves'); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    # Confusion matrix tab
    with tabs[-1]:
        c1, c2 = st.columns(2)
        for col, name, cm_mat, cmap_n in [
            (c1, "ResNet18", m_r['conf_matrix'], 'Blues'),
            (c2, "AlexNet",  m_a['conf_matrix'], 'Reds'),
        ]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_mat, annot=True, fmt='d', cmap=cmap_n,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                        ax=ax, cbar=False)
            ax.set_title(f'{name} Confusion Matrix', fontweight='bold')
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            col.pyplot(fig); plt.close(fig)

    # ── Live side-by-side ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🖼️ Live Side-by-Side Prediction")
    uploaded2 = st.file_uploader("Upload an image to compare both models",
                                  type=["jpg","jpeg","png"], key="compare_uploader")

    if uploaded2 is not None:
        img2  = Image.open(uploaded2).convert("RGB")
        img2_t = transform(img2).unsqueeze(0).to(device)

        with st.spinner("Running both models..."):
            rnet = load_resnet()
            anet = load_alexnet()
            with torch.no_grad():
                out_r = F.softmax(rnet(img2_t), dim=1)[0].cpu().numpy()
                out_a = F.softmax(anet(img2_t), dim=1)[0].cpu().numpy()

        pred_r = CLASS_NAMES[int(np.argmax(out_r))]
        pred_a = CLASS_NAMES[int(np.argmax(out_a))]
        col1, col2, col3 = st.columns(3)

        col1.image(img2, caption="Uploaded Image", use_container_width=True)

        col2.markdown("### 🔵 ResNet18")
        col2.markdown(f"**{'🟢' if pred_r=='REAL' else '🔴'} {pred_r}** — `{max(out_r)*100:.1f}%`")
        for i, n in enumerate(CLASS_NAMES):
            col2.write(f"- {n}: `{out_r[i]*100:.2f}%`")

        col3.markdown("### 🔴 AlexNet")
        col3.markdown(f"**{'🟢' if pred_a=='REAL' else '🔴'} {pred_a}** — `{max(out_a)*100:.1f}%`")
        for i, n in enumerate(CLASS_NAMES):
            col3.write(f"- {n}: `{out_a[i]*100:.2f}%`")

        if pred_r == pred_a:
            st.success(f"✅ Both models agree: **{pred_r}**")
        else:
            st.warning(f"⚠️ Models disagree — ResNet18: **{pred_r}** | AlexNet: **{pred_a}**")
