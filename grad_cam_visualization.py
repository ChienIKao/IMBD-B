#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM Visualization for 1D-CNN Time Series Analysis
基於 final_full_prompt.md 的完整實現
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端，適合服務器環境

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
import seaborn as sns
import joblib
import random
import argparse
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 載入專案自定義模組
from src import data_loader, model

# === 設定可復現性 ===
def set_random_seeds():
    """設定所有隨機種子以確保復現性"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Global Parameters ===
WINDOW_SIZE = 500
STEP_SIZE = 50
NUM_FEATURES = 35  # TODO: Modify based on actual project, taking 35 features starting from column C

# === 1. Load Model ===
def load_model(model_dir: str = "./models") -> torch.nn.Module:
    """
    Load trained model
    
    Args:
        model_dir: Model directory path
        
    Returns:
        Loaded model instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入 scaler 以獲得特徵數量
    scaler_path = os.path.join(model_dir, "final_scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    num_features = scaler.n_features_in_
    
    # 初始化模型
    model_instance = model.CNC_1D_CNN(
        num_features=num_features, 
        window_size=WINDOW_SIZE
    ).to(device)
    
    # 載入模型權重
    model_path = os.path.join(model_dir, "final_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()  # Set to evaluation mode
    
    print(f"✅ Successfully loaded model: {model_path}")
    print(f"✅ Number of features: {num_features}")
    print(f"✅ Using device: {device}")
    
    return model_instance

# === 2. Load Scaler ===
def load_scaler(model_dir: str = "./models"):
    """
    Load StandardScaler saved during training
    
    Args:
        model_dir: Model directory path
        
    Returns:
        Loaded scaler instance
    """
    scaler_path = os.path.join(model_dir, "final_scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    print(f"✅ Successfully loaded Scaler: {scaler_path}")
    
    return scaler

# === 3. Read and Preprocess CSV ===
def read_and_preprocess_csv(csv_path: str, scaler) -> Tuple[np.ndarray, List[str]]:
    """
    Read single CSV and preprocess (must be completely consistent)
    
    1. Skip first two rows (column names and units)
    2. Take 35 features starting from column C  
    3. Apply StandardScaler saved during training (cannot refit)
    
    Args:
        csv_path: CSV file path
        scaler: Trained StandardScaler
        
    Returns:
        Tuple of (standardized_data (T, num_features), feature_names)
    """
    # Read CSV to get both data and column names
    df = pd.read_csv(csv_path, header=0, skiprows=[1])  # Skip units row
    
    # Get feature names (starting from column C, index 2)
    feature_names = df.columns[2:2+NUM_FEATURES].tolist()
    
    # Use existing project data loading function
    data_array = data_loader.load_single_csv(csv_path)
    
    # Apply scaler (only use transform, cannot refit)
    scaled_data = scaler.transform(data_array)
    
    print(f"✅ Read CSV: {os.path.basename(csv_path)}")
    print(f"   Original shape: {data_array.shape}")
    print(f"   Standardized shape: {scaled_data.shape}")
    print(f"   Feature names: {feature_names[:5]}..." if len(feature_names) > 5 else f"   Feature names: {feature_names}")
    
    return scaled_data, feature_names

# === 4. Extract Sliding Windows ===
def extract_windows(data_array: np.ndarray, 
                   window_length: int = WINDOW_SIZE, 
                   stride: int = STEP_SIZE) -> Tuple[np.ndarray, List[int]]:
    """
    Extract sliding windows
    
    Args:
        data_array: Standardized data (T, num_features)
        window_length: Window length, default 500
        stride: Stride, default 50
        
    Returns:
        windows: (N_windows, num_features, window_length) for PyTorch Conv1D
        start_indices: Global start indices for each window
    """
    T, num_features = data_array.shape
    windows = []
    start_indices = []
    
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        window = data_array[start:end, :]  # (window_length, num_features)
        
        # Convert to PyTorch Conv1D format (num_features, window_length)
        window_conv1d = window.transpose(1, 0)  
        
        windows.append(window_conv1d)
        start_indices.append(start)
    
    if len(windows) == 0:
        raise ValueError(f"Cannot extract windows: data length {T} < window length {window_length}")
    
    windows = np.array(windows)  # (N_windows, num_features, window_length)
    
    print(f"✅ Window extraction completed")
    print(f"   Number of windows: {len(windows)}")
    print(f"   Window shape: {windows.shape}")
    
    return windows, start_indices

# === 5. Grad-CAM Computation for Single Window ===
def compute_grad_cam_for_window(model_instance: torch.nn.Module, 
                               window: np.ndarray,
                               target_layer_name: str = "conv3") -> np.ndarray:
    """
    Compute Grad-CAM for single window (standard procedure)
    
    Args:
        model_instance: Loaded model
        window: Single window (1, num_features, window_length)
        target_layer_name: Target layer name, default "conv3"
        
    Returns:
        cam: Normalized CAM (window_length,)
    """
    device = next(model_instance.parameters()).device
    
    # Convert to tensor and move to device
    window_tensor = torch.tensor(window, dtype=torch.float32).to(device)
    window_tensor.requires_grad_(True)
    
    # Containers for storing activations and gradients
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    # Register hooks to target layer (last Conv1D)
    target_layer = getattr(model_instance, target_layer_name)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    try:
        # 1. Forward pass
        output = model_instance(window_tensor)
        
        # 2. Binary classification has only one logit → output[0,0], do not apply sigmoid first
        target_logit = output[0, 0]
        
        # 3. Backward w.r.t target layer output
        model_instance.zero_grad()
        target_logit.backward()
        
        # 4. Calculate α = GAP(gradient)
        # activations: (1, channels, length), gradients: (1, channels, length)
        alpha = torch.mean(gradients, dim=2, keepdim=True)  # (1, channels, 1)
        
        # 5. CAM calculation
        cam = torch.sum(alpha * activations, dim=1)  # (1, length)
        cam = F.relu(cam)  # Keep only positive values
        cam = cam.squeeze(0)  # (length,)
        
        # 6. Interpolate to length 500 (if needed)
        if cam.size(0) != WINDOW_SIZE:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),  # (1, 1, length)
                size=WINDOW_SIZE,
                mode='linear',
                align_corners=False
            ).squeeze()  # (500,)
        
        # 7. Per-window min-max normalize → [0,1]
        cam_np = cam.detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)
        
        return cam_normalized
        
    finally:
        # Clean up hooks
        forward_handle.remove()
        backward_handle.remove()

# === 6. Multi-window Global CAM Aggregation ===
def aggregate_global_cam(windows: np.ndarray, 
                        start_indices: List[int],
                        model_instance: torch.nn.Module,
                        total_length: int,
                        target_layer_name: str = "conv3") -> np.ndarray:
    """
    Aggregate CAMs from multiple windows (must use averaging)
    
    Args:
        windows: All windows (N_windows, num_features, window_length)
        start_indices: Start indices for each window
        model_instance: Loaded model
        total_length: Total length of original time series
        target_layer_name: Target layer name
        
    Returns:
        global_cam: Global CAM (total_length,)
    """
    global_cam = np.zeros(total_length)
    count = np.zeros(total_length)
    
    print("🔄 Starting Grad-CAM computation for each window...")
    
    for k, (window, start) in enumerate(zip(windows, start_indices)):
        # Compute CAM for each window individually (cannot batch process)
        window_batch = window[np.newaxis, :]  # (1, num_features, window_length)
        cam_k = compute_grad_cam_for_window(model_instance, window_batch, target_layer_name)
        
        # Accumulate to global CAM
        for i in range(WINDOW_SIZE):
            global_idx = start + i
            if global_idx < total_length:
                global_cam[global_idx] += cam_k[i]
                count[global_idx] += 1
        
        if (k + 1) % 10 == 0 or k == len(windows) - 1:
            print(f"   Progress: {k+1}/{len(windows)} windows completed")
    
    # Calculate average (avoid division by zero)
    mask = count > 0
    global_cam[mask] /= count[mask]
    
    # Final min-max normalization
    cam_min, cam_max = global_cam.min(), global_cam.max()
    if cam_max > cam_min:
        global_cam = (global_cam - cam_min) / (cam_max - cam_min)
    else:
        global_cam = np.zeros_like(global_cam)
    
    print(f"✅ Global CAM computation completed, shape: {global_cam.shape}")
    
    return global_cam

# === 6.5. Regional Attention Enhancement ===
def enhance_attention_regionally(attention_values: np.ndarray, 
                               window_size: int = 500) -> np.ndarray:
    """
    Enhance attention values based on regional maxima within sliding windows
    Each time point is enhanced based on its surrounding window context
    
    Args:
        attention_values: Raw attention values (T,)
        window_size: Size of the sliding window (default 500 to match training windows)
        
    Returns:
        Enhanced attention values with regional focus
    """
    T = len(attention_values)
    enhanced_attention = np.zeros_like(attention_values)
    
    print(f"   Applying sliding regional enhancement with window_size={window_size}")
    
    # Process every time point with its surrounding context
    for center_idx in range(T):
        # Define window boundaries around current time point
        half_window = window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(T, center_idx + half_window)
        
        # Extract the regional window around current time point
        region = attention_values[start_idx:end_idx]
        
        if len(region) == 0:
            enhanced_attention[center_idx] = attention_values[center_idx]
            continue
        
        # Find the maximum attention in this region
        max_attention = np.max(region)
        
        # Calculate relative position of center within this region
        center_relative = center_idx - start_idx
        
        if max_attention > 0.05:  # Lower threshold for more sensitivity
            # Find peak position within the region
            max_idx = np.argmax(region)
            
            # Calculate distance from current point to regional peak
            distance_to_peak = abs(center_relative - max_idx)
            
            # Create Gaussian enhancement based on proximity to regional peak
            enhancement_factor = np.exp(-distance_to_peak**2 / (2 * (window_size/8)**2))
            
            # Enhancement formula: original value + regional boost
            enhancement = 0.3 * enhancement_factor * max_attention
            enhanced_value = attention_values[center_idx] * (1 + enhancement)
            
            # Ensure values stay in [0, 1]
            enhanced_attention[center_idx] = np.clip(enhanced_value, 0, 1)
        else:
            # If region has low attention, keep original value
            enhanced_attention[center_idx] = attention_values[center_idx]
    
    print(f"   Sliding regional enhancement completed for all {T} time points")
    return enhanced_attention

# === 7. Channel-wise Attention Computation ===
def compute_channel_attention(data_array: np.ndarray, 
                             global_cam: np.ndarray) -> np.ndarray:
    """
    Compute channel-wise attention (must use this method)
    
    importance[t,c] = |X[t,c]| * A[t]
    channel_attention[c,:] = minmax(importance[:,c])
    
    Args:
        data_array: Standardized original data (T, num_features)
        global_cam: Global CAM (T,)
        
    Returns:
        channel_attention: (num_features, T)
    """
    T, num_features = data_array.shape
    
    # Calculate importance[t,c] = |X[t,c]| * A[t]
    importance = np.abs(data_array) * global_cam[:, np.newaxis]  # (T, num_features)
    
    # Min-max normalization for each channel
    channel_attention = np.zeros((num_features, T))
    
    for c in range(num_features):
        importance_c = importance[:, c]  # (T,)
        
        # Min-max normalization
        imp_min, imp_max = importance_c.min(), importance_c.max()
        if imp_max > imp_min:
            channel_attention[c, :] = (importance_c - imp_min) / (imp_max - imp_min)
        else:
            channel_attention[c, :] = np.zeros_like(importance_c)
    
    print(f"✅ Channel-wise attention computation completed, shape: {channel_attention.shape}")
    
    return channel_attention

# === 7.5. Downsample and Interpolate Function ===
def downsample_and_interpolate(values: np.ndarray, 
                              sample_step: int = 200,
                              local_window: int = 200) -> np.ndarray:
    """
    Downsample values by taking maximum in local windows around sampling points, 
    then interpolate back to original length
    
    Args:
        values: Original values array (T,)
        sample_step: Sampling step (default 200)
        local_window: Window size around each sampling point to find maximum (default 200)
        
    Returns:
        Interpolated values with same length as input
    """
    T = len(values)
    half_window = local_window // 2
    
    # Create sampling indices
    sample_indices = np.arange(0, T, sample_step)
    
    # Ensure we include the last point
    if sample_indices[-1] != T - 1:
        sample_indices = np.append(sample_indices, T - 1)
    
    # Extract maximum values in local windows around each sampling point
    sampled_values = []
    
    for idx in sample_indices:
        # Define window boundaries around sampling point
        start_idx = max(0, idx - half_window)
        end_idx = min(T, idx + half_window + 1)
        
        # Find maximum value in this window
        local_window_values = values[start_idx:end_idx]
        max_value = np.max(local_window_values)
        
        sampled_values.append(max_value)
    
    sampled_values = np.array(sampled_values)
    
    # Interpolate back to original length
    interpolated = np.interp(np.arange(T), sample_indices, sampled_values)
    
    print(f"   Downsampled from {T} to {len(sample_indices)} points (step={sample_step}, local_max_window={local_window}), then interpolated back")
    
    return interpolated

# === 8. Line Plot with Attention Background ===
def plot_line_with_attention(data_array: np.ndarray,
                           channel_attention: np.ndarray,
                           channel_idx: int,
                           feature_name: str,
                           save_path: str,
                           csv_name: str):
    """
    Plot upper half: line plot + red attention background
    
    Args:
        data_array: Original data (T, num_features)
        channel_attention: (num_features, T)  
        channel_idx: Channel index to plot
        feature_name: Feature name for display
        save_path: Save path
        csv_name: CSV file name (for title)
    """
    T = data_array.shape[0]
    time_axis = np.arange(T)
    
    # Create figure with single plot layout
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    
    # Get attention values for this channel
    raw_attention_values = channel_attention[channel_idx, :]
    
    # Apply regional enhancement to preserve temporal locality
    enhanced_attention_values = enhance_attention_regionally(
        raw_attention_values, 
        window_size=500  # Match the training window size
    )
    
    # Apply downsampling with local maximum and interpolation for smoother visualization
    attention_values = downsample_and_interpolate(
        enhanced_attention_values,
        sample_step=200,      # Every 200 time units
        local_window=200      # Find max in 200-unit window around each sample point
    )
    
    # Step 1: First plot signal to establish y-axis limits
    ax1.plot(time_axis, data_array[:, channel_idx], 
            color='blue', linewidth=0.8, alpha=0.9,
            label=f'Signal ({feature_name})', rasterized=True)
    
    # Get y-axis limits for heatmap extent
    y_min, y_max = ax1.get_ylim()
    
    # Step 2: Create continuous heatmap background using imshow
    # Critical: reshape to (1, T) for proper imshow display
    attention_heatmap = attention_values.reshape(1, -1)  # shape: (1, T)
    
    # Plot continuous smooth heatmap background
    im_bg = ax1.imshow(attention_heatmap, 
                      cmap='Reds', 
                      aspect='auto', 
                      interpolation='bilinear',  # Essential for smooth gradients
                      alpha=0.6,  # Increased for more visible heatmap
                      extent=[0, T, y_min, y_max],  # (left, right, bottom, top)
                      zorder=0)  # Background layer
    
    # Step 3: Re-plot signal line on top of heatmap
    ax1.plot(time_axis, data_array[:, channel_idx], 
            color='blue', linewidth=0.8, alpha=0.9,
            label=f'Signal ({feature_name})', rasterized=True, zorder=2)
    
    # Step 4: Create secondary y-axis for attention score
    ax1_twin = ax1.twinx()
    
    # Step 5: Plot red attention score line (local max sampled & interpolated)
    attention_line = ax1_twin.plot(time_axis, attention_values, 
                                  color='red', linewidth=2.0, alpha=0.5,
                                  label='Attention Score (Local Max)', zorder=3)
    
    # Configure axes styling
    ax1.set_ylabel('Signal Value', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    ax1_twin.set_ylabel('Attention Score', fontsize=12, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim(0, 1.0)
    
    # Enhanced title
    ax1.set_title(f'{feature_name} - Local Max Sampling Analysis (200-unit sampling)\nFile: {csv_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Final layout adjustments
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Saved plot: {save_path}")

# === 9. Attention Heatmap ===
def plot_attention_heatmap(channel_attention: np.ndarray, 
                          channel_idx: int,
                          feature_name: str,
                          ax):
    """
    Plot lower half: Enhanced multi-row heatmap (like the reference image)
    
    Args:
        channel_attention: (num_features, T)
        channel_idx: Channel index to plot  
        feature_name: Feature name for display
        ax: matplotlib axis
    """
    # Extract single channel attention
    attention_data = channel_attention[channel_idx, :]
    T = len(attention_data)
    
    # Create multi-row heatmap by repeating the attention pattern
    # This creates the continuous color block effect like in the reference image
    num_rows = 50  # Height of the heatmap in pixels/rows
    attention_heatmap = np.tile(attention_data, (num_rows, 1))
    
    # Enhanced heatmap with better color mapping
    im = ax.imshow(attention_heatmap, cmap='Reds', aspect='auto', 
                   vmin=0, vmax=1, interpolation='bilinear',
                   extent=[0, T, 0, 1])  # Set proper extent
    
    # Enhanced styling
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Attention', fontsize=12)
    ax.set_title(f'{feature_name} - Attention Heatmap', fontsize=12, fontweight='bold')
    
    # Remove y-axis ticks for cleaner look
    ax.set_yticks([])
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

# === 10. Feature Importance Computation ===
def compute_feature_importance(channel_attention: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute Feature Importance (must calculate this way)
    
    Args:
        channel_attention: (num_features, T)
        
    Returns:
        importance_dict: Dictionary containing three types of importance scores
    """
    num_features = channel_attention.shape[0]
    
    # 1. Maximum Single Feature Attention
    max_scores = np.max(channel_attention, axis=1)  # (num_features,)
    
    # 2. Single/Overall Ratio  
    mean_of_max = np.mean(max_scores)
    ratio_scores = max_scores / mean_of_max if mean_of_max > 0 else np.zeros_like(max_scores)
    
    # 3. Mean Single Feature Attention
    mean_scores = np.mean(channel_attention, axis=1)  # (num_features,)
    
    importance_dict = {
        'max_scores': max_scores,
        'ratio_scores': ratio_scores,  
        'mean_scores': mean_scores
    }
    
    print("✅ Feature Importance computation completed")
    
    return importance_dict

def print_top_features(importance_dict: Dict[str, np.ndarray], feature_names: List[str], top_k: int = 10):
    """
    Print top K features for each ranking
    
    Args:
        importance_dict: feature importance dictionary
        feature_names: list of feature names
        top_k: show top K features, default 10
    """
    max_scores = importance_dict['max_scores']
    ratio_scores = importance_dict['ratio_scores']
    mean_scores = importance_dict['mean_scores']
    
    print(f"\n{'='*60}")
    print("📊 FEATURE IMPORTANCE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # 1. Maximum Single Feature Attention top K
    max_ranking = np.argsort(max_scores)[::-1][:top_k]
    print(f"\n🏆 Maximum Single Feature Attention (Top {top_k}):")
    print("-" * 70)
    for i, feat_idx in enumerate(max_ranking):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
        print(f"  {i+1:2d}. {feat_name:20s}: {max_scores[feat_idx]:.6f}")
    
    # 2. Single/Overall Ratio top K
    ratio_ranking = np.argsort(ratio_scores)[::-1][:top_k]
    print(f"\n📈 Single/Overall Ratio (Top {top_k}):")
    print("-" * 70) 
    for i, feat_idx in enumerate(ratio_ranking):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
        print(f"  {i+1:2d}. {feat_name:20s}: {ratio_scores[feat_idx]:.6f}")
    
    # 3. Mean Single Feature Attention top K
    mean_ranking = np.argsort(mean_scores)[::-1][:top_k]
    print(f"\n📊 Mean Single Feature Attention (Top {top_k}):")
    print("-" * 70)
    for i, feat_idx in enumerate(mean_ranking):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
        print(f"  {i+1:2d}. {feat_name:20s}: {mean_scores[feat_idx]:.6f}")
    
    print(f"\n{'='*60}")

# === 11. Main Function ===
def main():
    """Main function: integrate all steps"""
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for 1D-CNN')
    parser.add_argument('--csv', type=str, required=True, 
                       help='Input CSV file path')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Model directory path (default: ./models)')
    parser.add_argument('--output-dir', type=str, default='./grad_cam_results',
                       help='Output directory path (default: ./grad_cam_results)')
    parser.add_argument('--target-layer', type=str, default='conv3',
                       help='Target Conv layer name (default: conv3)')
    parser.add_argument('--feature-idx', type=int, default=None,
                       help='Specify feature index to visualize (default: auto-select based on importance)')
    parser.add_argument('--feature-name', type=str, default=None,
                       help='Specify feature name to visualize (e.g., POS3DC.1)')
    
    args = parser.parse_args()
    
    # 設定可復現性
    set_random_seeds()
    
    print("🚀 Starting Grad-CAM visualization analysis")
    print(f"📁 CSV file: {args.csv}")
    print(f"📁 Model directory: {args.model_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Target layer: {args.target_layer}")
    
    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. Load model and scaler
        model_instance = load_model(args.model_dir)
        scaler = load_scaler(args.model_dir)
        
        # 2. Read and preprocess CSV
        data_array, feature_names = read_and_preprocess_csv(args.csv, scaler)
        
        # 3. Extract sliding windows
        windows, start_indices = extract_windows(data_array)
        
        # 4. Compute global CAM
        total_length = data_array.shape[0]
        global_cam = aggregate_global_cam(
            windows, start_indices, model_instance, 
            total_length, args.target_layer
        )
        
        # 5. Compute channel-wise attention
        channel_attention = compute_channel_attention(data_array, global_cam)
        
        # 6. Compute feature importance
        importance_dict = compute_feature_importance(channel_attention)
        
        # 7. Print importance rankings
        print_top_features(importance_dict, feature_names)
        
        # 8. Determine target feature to visualize
        target_feature = None
        target_feature_name = None
        
        if args.feature_name is not None:
            # Find feature by name
            try:
                target_feature = feature_names.index(args.feature_name)
                target_feature_name = args.feature_name
                print(f"\n🎯 User specified feature by name: {target_feature_name} (index: {target_feature})")
            except ValueError:
                print(f"\n⚠️ Feature name '{args.feature_name}' not found. Available features: {feature_names[:10]}...")
                target_feature = np.argmax(importance_dict['max_scores'])
                target_feature_name = feature_names[target_feature] if target_feature < len(feature_names) else f"Feature_{target_feature}"
                print(f"\n🎯 Auto-selected most important feature: {target_feature_name} (index: {target_feature})")
        elif args.feature_idx is not None:
            target_feature = args.feature_idx
            target_feature_name = feature_names[target_feature] if target_feature < len(feature_names) else f"Feature_{target_feature}"
            print(f"\n🎯 User specified feature by index: {target_feature_name} (index: {target_feature})")
        else:
            # Auto-select most important feature based on max_scores
            target_feature = np.argmax(importance_dict['max_scores'])
            target_feature_name = feature_names[target_feature] if target_feature < len(feature_names) else f"Feature_{target_feature}"
            print(f"\n🎯 Auto-selected most important feature: {target_feature_name} (index: {target_feature})")
        
        # 9. Create visualization
        csv_name = os.path.splitext(os.path.basename(args.csv))[0]
        safe_feature_name = target_feature_name.replace('.', '_').replace('/', '_')
        save_path = os.path.join(args.output_dir, f'{csv_name}_{safe_feature_name}_grad_cam.png')
        
        plot_line_with_attention(
            data_array, channel_attention, target_feature, target_feature_name,
            save_path, csv_name
        )
        
        # 10. Save numerical results
        results_path = os.path.join(args.output_dir, f'{csv_name}_results.npz')
        np.savez(results_path,
                global_cam=global_cam,
                channel_attention=channel_attention,
                feature_names=np.array(feature_names),
                max_scores=importance_dict['max_scores'],
                ratio_scores=importance_dict['ratio_scores'], 
                mean_scores=importance_dict['mean_scores'])
        
        print(f"✅ Numerical results saved: {results_path}")
        print("🎉 Grad-CAM analysis completed!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()