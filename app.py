import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import cv2

st.set_page_config(layout="wide")

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# DATA (Only Test Set Needed)
# =====================================================
test_dir = "test"   # keep test folder in repo

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
class_names  = test_dataset.classes

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():

    # ResNet18
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load("best_model.pth", map_location=device))
    resnet = resnet.to(device)
    resnet.eval()

    # AlexNet
    alexnet = models.alexnet(pretrained=False)
    alexnet.classifier[6] = nn.Linear(
        alexnet.classifier[6].in_features, 2
    )
    alexnet.load_state_dict(torch.load("best_alexnet.pth", map_location=device))
    alexnet = alexnet.to(device)
    alexnet.eval()

    return resnet, alexnet


resnet_model, alexnet_model = load_models()

# =====================================================
# METRICS FUNCTION
# =====================================================
def compute_metrics(model):

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs   = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs[:,1].cpu().numpy())

    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    y_scores = np.array(y_scores)

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return acc, cm, fpr, tpr, roc_auc


# =====================================================
# SIDEBAR MENU
# =====================================================
st.sidebar.title("Model Selection")
option = st.sidebar.radio(
    "Choose Mode",
    ["ResNet18", "AlexNet", "Model Comparison"]
)

st.title("CIFAKE Detection Dashboard")

# =====================================================
# RESNET VIEW
# =====================================================
if option == "ResNet18":

    st.header("ResNet18 Performance")

    acc, cm, fpr, tpr, roc_auc = compute_metrics(resnet_model)

    st.subheader(f"Accuracy: {acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = plt.figure()
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title("Confusion Matrix")
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1],'--')
        plt.legend()
        plt.title("ROC Curve")
        st.pyplot(fig2)


# =====================================================
# ALEXNET VIEW
# =====================================================
elif option == "AlexNet":

    st.header("AlexNet Performance")

    acc, cm, fpr, tpr, roc_auc = compute_metrics(alexnet_model)

    st.subheader(f"Accuracy: {acc:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = plt.figure()
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title("Confusion Matrix")
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1],'--')
        plt.legend()
        plt.title("ROC Curve")
        st.pyplot(fig2)


# =====================================================
# COMPARISON VIEW
# =====================================================
else:

    st.header("Model Comparison")

    acc_r, cm_r, fpr_r, tpr_r, auc_r = compute_metrics(resnet_model)
    acc_a, cm_a, fpr_a, tpr_a, auc_a = compute_metrics(alexnet_model)

    st.write("### Accuracy Comparison")
    st.write(f"ResNet18: {acc_r:.4f}")
    st.write(f"AlexNet : {acc_a:.4f}")

    fig = plt.figure()
    plt.plot(fpr_r, tpr_r, label=f"ResNet (AUC={auc_r:.3f})")
    plt.plot(fpr_a, tpr_a, label=f"AlexNet (AUC={auc_a:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Comparison")
    st.pyplot(fig)