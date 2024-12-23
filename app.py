import os
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import plotly.graph_objects as go
from io import BytesIO
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.UNet import UNet
from src.utils.utils import load_checkpoint

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
DATA_DIR = "./data/processed"

# Get tissue types
tissue_types = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# Semantic mask color mapping
SEMANTIC_CLASSES = {
    0: "Background",
    1: "Neoplastic cells",
    2: "Inflammatory cells",
    3: "Connective/Soft tissue cells",
    4: "Dead cells",
    5: "Epithelial cells",
}
COLORMAP = ListedColormap(["black", "red", "blue", "green", "yellow", "purple"])

# Initialize model
@st.cache_resource
def load_model(checkpoint_path):
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model

# Image preprocessing for model
def preprocess_image(image):
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ])
    
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is float32 and in range [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply transforms
    transformed = transform(image=image)
    image = transformed["image"]
    
    return image.unsqueeze(0)

def get_model_prediction(model, image):
    with torch.no_grad():
        image = image.to(DEVICE)
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()

        return preds.cpu().numpy().squeeze()

def overlay_prediction(image, pred_mask, alpha=0.5):
    """
    Overlay binary prediction mask on the original image, excluding background
    
    Args:
        image: Original image
        pred_mask: Binary prediction mask
        alpha: Transparency of the overlay (0-1)
    
    Returns:
        Overlay image with predictions in red, excluding background
    """
    # Convert image to float32 in range [0,1]
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    # Create output array starting with the original image
    output = image.copy()
    
    # Only modify pixels where prediction is positive (foreground)
    foreground_mask = pred_mask > 0
    
    # Create red overlay only for foreground pixels
    output[foreground_mask] = (
        image[foreground_mask] * (1 - alpha) +  # Original image contribution
        np.array([1, 0, 0]) * alpha             # Red overlay contribution
    )
    
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


# Existing functions
def get_slide_info(filename):
    match = re.match(r'img_.*?(\d+)\d+.jpg', filename)
    if match:
        return int(match.group(1))
    return None

def overlay_masks(image, mask, alpha=0.5):
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    output = image.copy()
    mask_colored = COLORMAP(mask)
    non_zero_mask = mask > 0
    
    for c in range(3):
        output[non_zero_mask, c] = (1 - alpha) * image[non_zero_mask, c] + \
                                  alpha * mask_colored[non_zero_mask, c]
    
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)

def create_interactive_image(image, mask):
    fig = go.Figure()
    
    fig.add_trace(
        go.Image(z=image)
    )
    
    hover_text = [
        [SEMANTIC_CLASSES.get(val, "Unknown") for val in row] for row in mask
    ]
    
    y_coords, x_coords = np.indices(mask.shape)
    fig.add_trace(
        go.Scatter(
            x=x_coords.flatten(),
            y=y_coords.flatten(),
            mode='markers',
            marker=dict(size=0, opacity=0),
            hoverinfo='text',
            text=np.array(hover_text).flatten(),
        )
    )
    
    fig.update_layout(
        width=800,
        height=800,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    return fig

# Streamlit app
st.title("PanNuke Dataset Explorer with UNet Segmentation")
st.sidebar.header("Navigation")

# Model checkpoint selection
checkpoint_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="checkpoints/checkpoint.pth"
)

# Load model if checkpoint exists
model = None
if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.warning("Model checkpoint not found. Please provide a valid path.")

# Select tissue type
tissue_type = st.sidebar.selectbox("Select Tissue Type", tissue_types)

# Load tissue type directory
tissue_dir = os.path.join(DATA_DIR, tissue_type)
images_dir = os.path.join(tissue_dir, "images")
sem_masks_dir = os.path.join(tissue_dir, "sem_masks")
inst_masks_dir = os.path.join(tissue_dir, "inst_masks")

# Get all image files
image_files = sorted(os.listdir(images_dir))

# Select visualization mode
visualization_mode = st.sidebar.radio(
    "Visualization Mode",
    ("Single Image", "Multiple Images")
)

if visualization_mode == "Single Image":
    idx = st.sidebar.slider("Select Image Index", 0, len(image_files) - 1, 0)
    
    # Load image and masks
    image_path = os.path.join(images_dir, image_files[idx])
    sem_mask_path = os.path.join(sem_masks_dir, image_files[idx].replace('img', 'sem'))
    
    image = Image.open(image_path)
    sem_mask = np.array(Image.open(sem_mask_path))
    
    # Model prediction
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            overlay_alpha = st.slider("Ground Truth Overlay Transparency", 0.0, 1.0, 0.5)
            overlay = overlay_masks(image, sem_mask, alpha=overlay_alpha)
            st.image(overlay, use_container_width=True)
        
        with col2:
            st.subheader("UNet Segmentation")
            pred_alpha = st.slider("Prediction Overlay Transparency", 0.0, 1.0, 0.5)
            
            # Get model prediction
            preprocessed_image = preprocess_image(image)
            prediction = get_model_prediction(model, preprocessed_image)
            
            # Display prediction overlay
            pred_overlay = overlay_prediction(image, prediction, alpha=pred_alpha)
            st.image(pred_overlay, use_container_width=True)
            
            # Display metrics
            if st.checkbox("Show Metrics"):
                intersection = np.logical_and(prediction, sem_mask > 0)
                union = np.logical_or(prediction, sem_mask > 0)
                iou = np.sum(intersection) / (np.sum(union) + 1e-8)
                st.metric("IoU Score", f"{iou:.4f}")
    else:
        st.subheader(f"Tissue Type: {tissue_type} | Crop: {idx} | File: {image_files[idx]}")
        overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
        overlay = overlay_masks(np.array(image), sem_mask, alpha=overlay_alpha)
        fig = create_interactive_image(overlay, sem_mask)
        st.plotly_chart(fig, use_container_width=True)

elif visualization_mode == "Multiple Images":
    n_cols = st.sidebar.slider("Number of columns", 1, 5, 3)
    n_rows = st.sidebar.slider("Number of rows", 1, 5, 3)
    start_idx = st.sidebar.slider("Starting Index", 0, len(image_files) - n_rows * n_cols, 0)
    
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
    
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = start_idx + i * n_cols + j
            if idx < len(image_files):
                with cols[j]:
                    image = Image.open(os.path.join(images_dir, image_files[idx]))
                    sem_mask = np.array(Image.open(os.path.join(sem_masks_dir, 
                                                              image_files[idx].replace('img', 'sem'))))
                    
                    if model is not None:
                        preprocessed_image = preprocess_image(image)
                        prediction = get_model_prediction(model, preprocessed_image)
                        pred_overlay = overlay_prediction(image, prediction, alpha=overlay_alpha)
                        st.image(pred_overlay, caption=f"File: {image_files[idx]} (UNet)", use_container_width=True)
                    else:
                        overlay = overlay_masks(np.array(image), sem_mask, alpha=overlay_alpha)
                        st.image(overlay, caption=f"File: {image_files[idx]}", use_container_width=True)

# Legend for semantic classes
st.sidebar.subheader("Semantic Mask Legend")
for class_id, class_name in SEMANTIC_CLASSES.items():
    st.sidebar.markdown(f"{class_id}: {class_name}")