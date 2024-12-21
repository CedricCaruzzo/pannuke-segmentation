import os
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import plotly.graph_objects as go
from io import BytesIO

# Set paths
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

def get_slide_info(filename):
    """Extract whole slide number from filename"""
    match = re.match(r'img_.*?_(\d+)_\d+\.jpg', filename)
    if match:
        return int(match.group(1))
    return None

def overlay_masks(image, mask, alpha=0.5):
    """
    Overlays only the non-zero parts of the semantic mask on the original image.
    """
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    # Create output image
    output = image.copy()
    
    # Create colored mask only for non-zero values
    mask_colored = COLORMAP(mask)
    non_zero_mask = mask > 0
    
    # Apply overlay only where mask is non-zero
    for c in range(3):
        output[non_zero_mask, c] = (1 - alpha) * image[non_zero_mask, c] + \
                                  alpha * mask_colored[non_zero_mask, c]
    
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)

def create_interactive_image(image, mask):
    """Create an interactive image with hover information"""
    fig = go.Figure()

    # Add the base image
    fig.add_trace(
        go.Image(z=image)
    )

    # Prepare hover annotations
    hover_text = [
        [SEMANTIC_CLASSES.get(val, "Unknown") for val in row] for row in mask
    ]

    # Add scatter layer for hover text
    y_coords, x_coords = np.indices(mask.shape)
    fig.add_trace(
        go.Scatter(
            x=x_coords.flatten(),
            y=y_coords.flatten(),
            mode='markers',
            marker=dict(size=0, opacity=0),  # Invisible markers
            hoverinfo='text',
            text=np.array(hover_text).flatten(),  # Flattened annotations
        )
    )

    # Update layout for better visualization
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
st.title("PanNuke Dataset Explorer")
st.sidebar.header("Navigation")

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
    #("Single Image", "Multiple Images", "Whole Slide")
)

if visualization_mode == "Single Image":
    idx = st.sidebar.slider("Select Image Index", 0, len(image_files) - 1, 0)
    
    # Load image and masks
    image_path = os.path.join(images_dir, image_files[idx])
    sem_mask_path = os.path.join(sem_masks_dir, image_files[idx].replace('img', 'sem'))
    
    image = np.array(Image.open(image_path))
    sem_mask = np.array(Image.open(sem_mask_path))
    
    # Interactive visualization
    st.subheader(f"Tissue Type: {tissue_type} | Crop: {idx} | File: {image_files[idx]}")
    
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
    overlay = overlay_masks(image, sem_mask, alpha=overlay_alpha)
    
    fig = create_interactive_image(overlay, sem_mask)
    st.plotly_chart(fig, use_container_width=True)

elif visualization_mode == "Multiple Images":
    # Select number of images to display
    n_cols = st.sidebar.slider("Number of columns", 1, 5, 3)
    n_rows = st.sidebar.slider("Number of rows", 1, 5, 3)
    
    start_idx = st.sidebar.slider("Starting Index", 0, len(image_files) - n_rows * n_cols, 0)
    
    overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
    
    # Create grid of images
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = start_idx + i * n_cols + j
            if idx < len(image_files):
                with cols[j]:
                    image = np.array(Image.open(os.path.join(images_dir, image_files[idx])))
                    sem_mask = np.array(Image.open(os.path.join(sem_masks_dir, 
                                                              image_files[idx].replace('img', 'sem'))))
                    overlay = overlay_masks(image, sem_mask, alpha=overlay_alpha)
                    st.image(overlay, caption=f"File: {image_files[idx]}", use_container_width=True)

# =============================================================================
# elif visualization_mode == "Whole Slide":
#     # Calculate grid dimensions based on filename patterns
#     all_positions = []
#     for filename in image_files:
#         match = re.match(r'img_.*?_\d+_(\d+)\.jpg', filename)
#         if match:
#             position = int(match.group(1))
#             all_positions.append(position)
#     
#     grid_size = int(np.ceil(np.sqrt(max(all_positions) + 1)))
#     
#     # Create stitched image
#     stitched_image = np.zeros((grid_size * 256, grid_size * 256, 3), dtype=np.uint8)
#     stitched_mask = np.zeros((grid_size * 256, grid_size * 256), dtype=np.uint8)
#     
#     progress_bar = st.progress(0)
#     for idx, img_file in enumerate(image_files):
#         match = re.match(r'img_.*?_\d+_(\d+)\.jpg', img_file)
#         if match:
#             position = int(match.group(1))
#             row = position // grid_size
#             col = position % grid_size
#             
#             image = np.array(Image.open(os.path.join(images_dir, img_file)))
#             sem_mask = np.array(Image.open(os.path.join(sem_masks_dir, 
#                                                       img_file.replace('img', 'sem'))))
#             
#             if len(image.shape) == 2:
#                 image = np.stack((image,) * 3, axis=-1)
#             
#             stitched_image[row * 256:(row + 1) * 256, col * 256:(col + 1) * 256, :] = image
#             stitched_mask[row * 256:(row + 1) * 256, col * 256:(col + 1) * 256] = sem_mask
#             
#             progress_bar.progress((idx + 1) / len(image_files))
#     
#     overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5)
#     stitched_overlay = overlay_masks(stitched_image, stitched_mask, alpha=overlay_alpha)
#     
#     # Display stitched results
#     fig = create_interactive_image(stitched_overlay, stitched_mask)
#     st.plotly_chart(fig, use_container_width=True)
# =============================================================================

# Legend for semantic classes
st.sidebar.subheader("Semantic Mask Legend")
for class_id, class_name in SEMANTIC_CLASSES.items():
    st.sidebar.markdown(f"**{class_id}:** {class_name}")