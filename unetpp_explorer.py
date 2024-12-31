import streamlit as st
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import albumentations as A
from albumentations.pytorch import ToTensorV2
import networkx as nx
import os
from src.models.UNetpp import UNetPlusPlus
from src.utils.utils import load_checkpoint

st.set_page_config(layout="wide")

# Constants and Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHECKPOINT_DIR = "checkpoints/UNet++/"

def get_available_checkpoints():
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    return [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks for encoder blocks
        for i, block in enumerate(self.model.encoder_blocks):
            handle = block.register_forward_hook(get_hook(f'encoder_{i}'))
            self.hook_handles.append(handle)
        
        # Register hooks for nested blocks
        for name, block in self.model.nested_convs.items():
            handle = block.register_forward_hook(get_hook(f'nested_{name}'))
            self.hook_handles.append(handle)
    
    def forward(self, x):
        return self.model(x)
    
    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def create_architecture_graph():
    G = nx.DiGraph()
    features = [64, 128, 256, 512]
    
    # Add nodes for each level
    for i in range(len(features)):
        for j in range(len(features) - i):
            node_id = f"x{i}_{j}"
            G.add_node(node_id, feature_size=features[j])
    
    # Add edges
    for i in range(len(features)):
        for j in range(len(features) - i):
            node = f"x{i}_{j}"
            
            # Add skip connection if not in first row
            if i > 0:
                for k in range(i):
                    G.add_edge(f"x{k}_{j}", node)
            
            # Add upsampling connection if not in first column
            if j < len(features) - i - 1:
                G.add_edge(node, f"x{i}_{j+1}")
    
    return G

def plot_architecture(G):
    pos = nx.spring_layout(G)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        feature_size = G.nodes[node]['feature_size']
        node_text.append(f"Node: {node}<br>Features: {feature_size}")
        node_size.append(feature_size / 10)
    
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color='#1f77b4',
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edges_trace, nodes_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def plot_feature_maps(features, n_features=16, figsize=(20, 12)):
    """Enhanced feature map visualization with larger figures"""
    fig = plt.figure(figsize=figsize)
    for i in range(n_features):
        plt.subplot(4, n_features//4, i + 1)
        plt.imshow(features[0, i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i}', pad=20)
    plt.tight_layout()
    return fig

def plot_deep_supervision_outputs(outputs):
    """Plot deep supervision outputs horizontally"""
    n_outputs = len(outputs)
    fig, axes = plt.subplots(1, n_outputs, figsize=(20, 4))
    
    if n_outputs == 1:
        axes = [axes]
    
    for i, (output, ax) in enumerate(zip(outputs, axes)):
        output_vis = torch.sigmoid(output).squeeze().cpu().numpy()
        ax.imshow(output_vis, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Output Level {i+1}')
    
    plt.tight_layout()
    return fig

def main():
    st.title("UNet++ Architecture Explorer")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Model checkpoint selection from available checkpoints
    available_checkpoints = get_available_checkpoints()
    if not available_checkpoints:
        st.sidebar.warning("No checkpoints found in the checkpoints/UNet++/ directory")
    else:
        selected_checkpoint = st.sidebar.selectbox(
            "Select checkpoint",
            available_checkpoints
        )
        model_path = os.path.join(CHECKPOINT_DIR, selected_checkpoint)
    
    tab1, tab2, tab3 = st.tabs(["Architecture", "Feature Maps", "Deep Supervision"])
    
    with tab1:
        st.header("UNet++ Architecture Visualization")
        
        # Create and display architecture graph
        G = create_architecture_graph()
        fig = plot_architecture(G)
        st.plotly_chart(fig, use_container_width=True)
        
        # Architecture explanation
        st.markdown("""
        ### Key Components:
        1. **Dense Skip Connections**: Multiple skip pathways between encoder and decoder
        2. **Nested Structure**: Progressive feature fusion at each level
        3. **Deep Supervision**: Multiple segmentation heads for different scales
        """)
    
    with tab2:
        st.header("Feature Maps Analysis")
        
        if available_checkpoints:
            if st.sidebar.button("Load Model"):
                try:
                    model = UNetPlusPlus(
                        in_channels=3,
                        out_channels=1,
                        features=[64, 128, 256, 512],
                        deep_supervision=True
                    ).to(DEVICE)
                    load_checkpoint(model_path, model)
                    st.session_state.model = FeatureExtractor(model)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        # Image upload and processing
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        if uploaded_file is not None and 'model' in st.session_state:
            image = Image.open(uploaded_file)
            
            # Preprocess image
            transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(),
                ToTensorV2(),
            ])
            
            img_array = np.array(image)
            transformed = transform(image=img_array)
            input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)
            
            # Get feature maps
            with torch.no_grad():
                _ = st.session_state.model(input_tensor)
            
            # Enhanced feature map visualization
            col1, col2 = st.columns([1, 3])
            
            with col1:
                feature_type = st.selectbox(
                    "Select feature type",
                    ["encoder", "nested"]
                )
                
                available_maps = [k for k in st.session_state.model.features.keys() 
                                if k.startswith(feature_type)]
                
                selected_map = st.selectbox("Select feature map", available_maps)
                
                # Visualization options
                n_features = st.slider("Number of features to display", 4, 32, 16, 4)
                fig_height = st.slider("Figure height", 8, 20, 12)
            
            with col2:
                if selected_map:
                    features = st.session_state.model.features[selected_map]
                    fig = plot_feature_maps(features, n_features=n_features, 
                                         figsize=(20, fig_height))
                    st.pyplot(fig)
    
    with tab3:
        st.header("Deep Supervision Analysis")
        if 'model' in st.session_state and uploaded_file is not None:
            with torch.no_grad():
                outputs = st.session_state.model(input_tensor)
            
            if isinstance(outputs, list):
                # Plot all outputs horizontally
                fig = plot_deep_supervision_outputs(outputs)
                st.pyplot(fig)
                
                # Display metrics in columns
                cols = st.columns(len(outputs))
                for i, (output, col) in enumerate(zip(outputs, cols)):
                    with col:
                        output_vis = torch.sigmoid(output).squeeze().cpu().numpy()
                        st.write(f"**Output {i+1}**")
                        st.write(f"Shape: {output.shape}")
                        st.write(f"Range: [{output_vis.min():.3f}, {output_vis.max():.3f}]")

if __name__ == "__main__":
    main()