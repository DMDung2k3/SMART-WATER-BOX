import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Ignite imports
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
# Updated import path to avoid deprecation warning
from ignite.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from ignite.contrib.metrics import GpuInfo

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Check for GPU only once at script level
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
CONFIG_PATH = "D:/pythonlearn/A_bot/water_meter.yaml"

# Load YAML configuration
with open(CONFIG_PATH, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Paths
TRAIN_DIR = yaml_data['train']
VALID_DIR = yaml_data['val']
TEST_DIR = yaml_data['test']

# Class information
NUM_CLASSES = yaml_data['nc']
CLASS_NAMES = yaml_data['names']

# Training parameters
BATCH_SIZE = 16
IMAGE_SIZE = 416
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1000
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]  # Grid sizes: 13, 26, 52

# Learning rate scheduler parameters
SCHEDULER_TYPE = "onecycle"  # Options: "step", "cosine", "onecycle", "plateau"
WARMUP_EPOCHS = 5  # Number of epochs for learning rate warmup
WARMUP_FACTOR = 0.1  # Initial learning rate = LEARNING_RATE * WARMUP_FACTOR
STEP_SIZE = 30  # For step scheduler: epochs to decay learning rate
GAMMA = 0.1  # For step scheduler: learning rate decay factor
PATIENCE = 5  # For plateau scheduler: epochs to wait before reducing learning rate
LR_MIN_FACTOR = 0.01  # For cosine and onecycle: minimum learning rate as a fraction of LEARNING_RATE

# Extract directory names
TRAIN_LABEL_DIR = TRAIN_DIR.replace('images', 'labels')
VALID_LABEL_DIR = VALID_DIR.replace('images', 'labels')
TEST_LABEL_DIR = TEST_DIR.replace('images', 'labels')

# Anchors - You may need to adjust these for your specific dataset
# These are typically determined using k-means clustering on your training set bounding boxes
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],     # For 13×13 grid
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],    # For 26×26 grid
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],    # For 52×52 grid
]  # Anchors are normalized to [0, 1]

# Model save directory
MODEL_SAVE_DIR = "D:/pythonlearn/A_bot/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Logs directory
LOGS_DIR = os.path.join(MODEL_SAVE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Flag to determine if we're training a DPU-compatible model
DPU_COMPATIBLE = True

# Model Architecture
#############################################

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        # Use exact LeakyReLU slope as required by DPU: 0.1015625
        self.leaky = nn.LeakyReLU(0.1015625 if DPU_COMPATIBLE else 0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class DPUCompatibleScalePrediction(nn.Module):
    """DPU-compatible scale prediction that avoids reshape/view operations"""
    def __init__(self, in_channels, num_classes):
        super(DPUCompatibleScalePrediction, self).__init__()
        self.num_classes = num_classes
        
        # Common feature extractor
        self.common = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1)
        
        # Separate prediction heads for each anchor box to avoid reshape operations
        # Each will output (num_classes + 5) channels: objectness, x, y, w, h, class_scores
        self.anchor1_pred = CNNBlock(2 * in_channels, num_classes + 5, bn_act=False, kernel_size=1)
        self.anchor2_pred = CNNBlock(2 * in_channels, num_classes + 5, bn_act=False, kernel_size=1)
        self.anchor3_pred = CNNBlock(2 * in_channels, num_classes + 5, bn_act=False, kernel_size=1)

    def forward(self, x):
        # Extract features
        features = self.common(x)
        
        # Get predictions for each anchor box separately
        pred1 = self.anchor1_pred(features)
        pred2 = self.anchor2_pred(features)
        pred3 = self.anchor3_pred(features)
        
        return (pred1, pred2, pred3)


class StandardScalePrediction(nn.Module):
    """Original scale prediction with view operation (not DPU compatible)"""
    def __init__(self, in_channels, num_classes):
        super(StandardScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        batch_size, _, grid_h, grid_w = x.shape
        pred_result = self.pred(x)
        fused_result = pred_result.view(batch_size, 3, grid_h, grid_w, self.num_classes + 5)
        return fused_result


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # DarkNet-53 architecture configuration
        self.config = [
            (32, 3, 1),
            (64, 3, 2),
            ["B", 1],
            (128, 3, 2),
            ["B", 2],
            (256, 3, 2),
            ["B", 8],
            (512, 3, 2),
            ["B", 8],
            (1024, 3, 2),
            ["B", 4],  # To this point is Darknet-53

            (512, 1, 1),
            (1024, 3, 1),
            "S",
            (256, 1, 1),
            "U",
            (256, 1, 1),
            (512, 3, 1),
            "S",
            (128, 1, 1),
            "U",
            (128, 1, 1),
            (256, 3, 1),
            "S",
        ]
        
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, (StandardScalePrediction, DPUCompatibleScalePrediction)):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in self.config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0
                ))
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                    ]
                    
                    # Use appropriate prediction module based on DPU compatibility
                    if DPU_COMPATIBLE:
                        layers.append(DPUCompatibleScalePrediction(in_channels // 2, num_classes=self.num_classes))
                    else:
                        layers.append(StandardScalePrediction(in_channels // 2, num_classes=self.num_classes))
                    
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers


# Post-processing function for DPU outputs
def process_dpu_outputs(scale_outputs, anchors, S, image_size=416):
    """
    Process outputs from DPU-compatible model
    
    Args:
        scale_outputs: List of tuples, each containing 3 tensors (one per anchor) for each scale
        anchors: List of anchors for each scale
        S: List of grid sizes for each scale
        
    Returns:
        Processed outputs in same format as original model for compatibility
    """
    processed_outputs = []
    
    for scale_idx, scale_preds in enumerate(scale_outputs):
        anchor1_pred, anchor2_pred, anchor3_pred = scale_preds
        grid_size = S[scale_idx]
        
        batch_size, channels, height, width = anchor1_pred.shape
        
        # Process each anchor prediction
        anchor_outputs = []
        for anchor_idx, anchor_pred in enumerate([anchor1_pred, anchor2_pred, anchor3_pred]):
            # Extract components
            objectness = anchor_pred[:, 0:1, :, :]
            box_xy = anchor_pred[:, 1:3, :, :]
            box_wh = anchor_pred[:, 3:5, :, :]
            classes = anchor_pred[:, 5:, :, :]
            
            # Reshape to match original model output format
            # This is done only after the model has run on DPU
            anchor_output = torch.cat([objectness, box_xy, box_wh, classes], dim=1)
            anchor_outputs.append(anchor_output)
        
        # Stack anchor outputs along a new dimension to create [batch, 3, channels, height, width]
        stacked = torch.stack(anchor_outputs, dim=1)
        
        # Reshape to [batch, 3, height, width, channels] to match original format
        # This is not done in the DPU but in post-processing
        batch_size, num_anchors, channels, height, width = stacked.shape
        reshaped = stacked.permute(0, 1, 3, 4, 2)
        
        processed_outputs.append(reshaped)
    
    return processed_outputs


# Utility Functions
#############################################

def iou_width_height(boxes1, boxes2):
    """
    Calculates IoU for width and height of anchor boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union (IoU) for bounding boxes
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Non-max suppression for bounding boxes
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def process_dpu_outputs_for_cells_to_bboxes(scale_outputs, batch_size, grid_size, num_classes):
    """
    Convert DPU outputs to the format expected by cells_to_bboxes
    
    Args:
        scale_outputs: Tuple of 3 tensors (one per anchor) with shape [batch_size, num_channels, grid_h, grid_w]
        batch_size: Batch size
        grid_size: Grid size for this scale (S)
        num_classes: Number of classes
        
    Returns:
        Tensor with shape [batch_size, 3, grid_size, grid_size, num_classes+5]
    """
    anchor1_out, anchor2_out, anchor3_out = scale_outputs
    
    # Each anchor output has shape [batch_size, num_classes+5, grid_h, grid_w]
    # Need to permute to [batch_size, 1, grid_h, grid_w, num_classes+5]
    anchor1_permuted = anchor1_out.permute(0, 2, 3, 1).unsqueeze(1)
    anchor2_permuted = anchor2_out.permute(0, 2, 3, 1).unsqueeze(1)
    anchor3_permuted = anchor3_out.permute(0, 2, 3, 1).unsqueeze(1)
    
    # Concatenate along the anchor dimension
    combined = torch.cat([anchor1_permuted, anchor2_permuted, anchor3_permuted], dim=1)
    
    return combined


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Converts YOLO cells to bounding boxes
    
    Parameters:
        predictions: tensor of shape (batch_size, num_anchors, S, S, num_classes+5)
                     or tuple of (anchor1, anchor2, anchor3) tensors for DPU model
        anchors: tensor of shape (num_anchors, 2) for width and height anchors
        S: int, grid size
        is_preds: bool, whether predictions are model outputs or ground truth
        
    Returns:
        converted_bboxes: list of lists of bounding boxes [class, confidence, x, y, w, h]
    """
    # Check if predictions is a tuple (DPU model output)
    if isinstance(predictions, tuple) and DPU_COMPATIBLE:
        # Process DPU outputs to match the expected format
        batch_size = predictions[0].shape[0]
        predictions = process_dpu_outputs_for_cells_to_bboxes(predictions, batch_size, S, NUM_CLASSES)
    
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    
    # Error checking for shapes
    expected_shape = (BATCH_SIZE, num_anchors, S, S, predictions.shape[-1])
    if predictions.shape != expected_shape:
        print(f"Warning: Prediction shape mismatch. Expected {expected_shape}, got {predictions.shape}")
        # Try to reshape if needed
        try:
            predictions = predictions.reshape(expected_shape)
        except RuntimeError as e:
            print(f"Reshape failed: {e}")
            print(f"Batch size: {BATCH_SIZE}, num_anchors: {num_anchors}, S: {S}")
            print(f"Total elements: {predictions.numel()}, expected: {np.prod(expected_shape)}")
    
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], num_anchors, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    
    try:
        converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
        converted_bboxes = converted_bboxes.tolist()
    except RuntimeError as e:
        # Handle reshape error safely
        print(f"Error in reshaping: {e}")
        print(f"Shapes - best_class: {best_class.shape}, scores: {scores.shape}, x: {x.shape}, y: {y.shape}, w_h: {w_h.shape}")
        
        # Try a safer approach - process each item in the batch separately
        converted_bboxes = []
        for b in range(BATCH_SIZE):
            # Flatten and concatenate
            batch_boxes = []
            for a in range(num_anchors):
                for i in range(S):
                    for j in range(S):
                        box = [
                            best_class[b, a, i, j, 0].item(),
                            scores[b, a, i, j, 0].item(),
                            x[b, a, i, j, 0].item(),
                            y[b, a, i, j, 0].item(),
                            w_h[b, a, i, j, 0].item(),
                            w_h[b, a, i, j, 1].item()
                        ]
                        batch_boxes.append(box)
            converted_bboxes.append(batch_boxes)
    
    return converted_bboxes


def plot_image(image, boxes, class_names=CLASS_NAMES, save_path=None):
    """
    Plots predicted bounding boxes on the image
    """
    # Create color map
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]
    
    # Convert image if it's a PIL image
    if isinstance(image, Image.Image):
        im = np.array(image)
    else:
        im = image
        
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        conf = box[1]
        box = box[2:]
        
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[class_pred],
            facecolor="none",
        )
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        # Add class label and confidence
        class_name = class_names[class_pred]
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=f"{class_name} {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[class_pred], "pad": 0},
        )

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_dataset_csv(image_dir, label_dir, output_csv_path):
    """
    Creates a CSV file containing image and label file paths
    """
    # Check if CSV already exists
    if os.path.exists(output_csv_path):
        print(f"Using existing dataset CSV: {output_csv_path}")
        return output_csv_path
        
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                         glob.glob(os.path.join(image_dir, "*.png")))
    
    data = []
    for img_path in image_files:
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Construct label path with .txt extension
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        # Only include if label file exists
        if os.path.exists(label_path):
            # Get relative paths
            rel_img_path = os.path.basename(img_path)
            rel_label_path = os.path.basename(label_path)
            
            data.append([rel_img_path, rel_label_path])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["image", "label"])
    df.to_csv(output_csv_path, index=False)
    
    print(f"Created dataset CSV at {output_csv_path} with {len(df)} entries")
    return output_csv_path


# Dataset and DataLoader
#############################################

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors,
                 image_size=416, S=[13,26,52], C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # For all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        
        # Check if label file exists and is not empty
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            # Load bounding boxes - format: class_id, x, y, width, height
            bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
            # If only one box, reshape to make it 2D
            if len(bboxes.shape) == 1:
                bboxes = np.expand_dims(bboxes, 0)
        else:
            # No labels, create empty array
            bboxes = np.zeros((0, 5))

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # Apply image transformations
            image = self.transform(image)

        # Create targets for each scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]  # objectness, x, y, w, h, class

        for box in bboxes:
            # box format: [class_id, x, y, width, height]
            iou_anchors = iou_width_height(torch.tensor([box[3], box[4]]).float(), self.anchors) # IOU from height and width
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # Sorting such that the first is the best anchor

            class_label, x, y, width, height = box
            has_anchor = [False] * 3  # For each scale
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # Cell coordinates
                
                # Boundary checking
                if i >= S or j >= S:
                    continue
                
                # Check if this cell already has an anchor assigned
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] == 1
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    # Set objectness to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    
                    # Calculate cell-relative coordinates
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell, height_cell = width * S, height * S
                    
                    # Set box coordinates
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    
                    # Set class label
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    
                    has_anchor[scale_idx] = True
                
                # If IoU is high but not the best, ignore this prediction during training
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # Ignore prediction

        return image, tuple(targets)


def get_loaders(train_csv_path, valid_csv_path, test_csv_path=None):
    """
    Creates and returns data loaders for training, validation, and testing
    """
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = YOLODataset(
        train_csv_path,
        TRAIN_DIR,
        TRAIN_LABEL_DIR,
        anchors=ANCHORS,
        image_size=IMAGE_SIZE,
        S=S,
        C=NUM_CLASSES,
        transform=transform,
    )
    
    valid_dataset = YOLODataset(
        valid_csv_path,
        VALID_DIR,
        VALID_LABEL_DIR,
        anchors=ANCHORS,
        image_size=IMAGE_SIZE,
        S=S, 
        C=NUM_CLASSES,
        transform=transform,
    )
    
    # Create loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    # Create test loader if test CSV is provided
    test_loader = None
    if test_csv_path:
        test_dataset = YOLODataset(
            test_csv_path,
            TEST_DIR,
            TEST_LABEL_DIR,
            anchors=ANCHORS,
            image_size=IMAGE_SIZE,
            S=S,
            C=NUM_CLASSES,
            transform=transform,
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
    
    return train_loader, valid_loader, test_loader


# Loss Function
#############################################

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        # Constants for weighting different components of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # If DPU-compatible model, process the predictions first
        if DPU_COMPATIBLE and isinstance(predictions, tuple):
            # Convert tuple of anchor predictions to standard format for loss calculation
            anchor1, anchor2, anchor3 = predictions
            batch_size, _, grid_h, grid_w = anchor1.shape
            
            # Reshape to match standard format [B, 3, H, W, C]
            anchor1_reshaped = anchor1.permute(0, 2, 3, 1).unsqueeze(1)
            anchor2_reshaped = anchor2.permute(0, 2, 3, 1).unsqueeze(1)
            anchor3_reshaped = anchor3.permute(0, 2, 3, 1).unsqueeze(1)
            
            # Stack along anchor dimension
            predictions = torch.cat([anchor1_reshaped, anchor2_reshaped, anchor3_reshaped], dim=1)
        
        # Identify which cells have objects and which don't
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss - binary cross entropy for cells without objects
        no_object_loss = self.bce(predictions[..., 0:1][noobj], target[..., 0:1][noobj])

        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        
        # Transform predictions to match target format
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], 
            dim=-1
        )
        
        # Calculate IoU between predicted and target boxes
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        
        # Object loss - binary cross entropy weighted by IoU
        object_loss = self.bce(predictions[..., 0:1][obj], ious * target[..., 0:1][obj])

        # Box coordinate loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        
        # Mean squared error for box coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # Class loss
        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())

        # Combine all loss components with their respective weights
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

# Evaluation Metrics
#############################################

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device=DEVICE,
    max_batches=10  # Limit number of batches for faster evaluation
):
    """
    Get bounding boxes from the model for evaluation
    """
    # Set model to evaluation mode
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    
    # Process limited number of batches for speed
    num_batches = min(max_batches, len(loader))
    print(f"Processing {num_batches} batches for mAP calculation...")
    
    # Process each batch
    for batch_idx, (x, labels) in enumerate(tqdm(loader, total=num_batches)):
        if batch_idx >= max_batches:
            break
            
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        
        # For each scale
        for i in range(3):
            # Get the shape based on the type of output
            if DPU_COMPATIBLE and isinstance(predictions[i], tuple):
                # For DPU model, get grid size from the first anchor tensor's shape
                anchor1, _, _ = predictions[i]
                S = anchor1.shape[2]  # Grid height (assumed square grid)
            else:
                # For standard model
                S = predictions[i].shape[2]
                
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            
            # Get bounding boxes for this scale
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            
            # Add boxes to corresponding image
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # Get ground truth boxes - Use the same scale as the model's output
        for i in range(3):
            # Get grid size consistently with above
            if DPU_COMPATIBLE and isinstance(predictions[i], tuple):
                anchor1, _, _ = predictions[i]
                S = anchor1.shape[2]
            else:
                S = predictions[i].shape[2]
                
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            
            # Convert labels to bounding boxes
            true_boxes_scale_i = cells_to_bboxes(
                labels[i], anchor, S=S, is_preds=False
            )
            
            # Process each image in the batch
            for idx in range(batch_size):
                # Apply non-max suppression to predictions
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                    box_format=box_format,
                )

                # Add prediction boxes with batch index
                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                # Add ground truth boxes with batch index
                for box in true_boxes_scale_i[idx]:
                    if box[1] > threshold:
                        all_true_boxes.append([train_idx] + box)
                
                train_idx += 1

    # Set model back to training mode
    model.train()
    
    print(f"Collected {len(all_pred_boxes)} predictions and {len(all_true_boxes)} ground truths")
    return all_pred_boxes, all_true_boxes


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASSES, max_samples=300
):
    """
    Calculate mean average precision across all classes with timeout and optimization
    """
    # Limit the number of boxes to process to improve performance
    if len(pred_boxes) > max_samples:
        print(f"Limiting prediction boxes from {len(pred_boxes)} to {max_samples} for faster mAP calculation")
        # Sort by confidence and keep the highest confidence predictions
        pred_boxes.sort(key=lambda x: x[2], reverse=True)  
        pred_boxes = pred_boxes[:max_samples]
    
    if len(true_boxes) > max_samples:
        print(f"Limiting ground truth boxes from {len(true_boxes)} to {max_samples} for faster mAP calculation")
        # Keep a balanced distribution across images
        img_ids = set(box[0] for box in true_boxes)
        if len(img_ids) > max_samples:
            img_ids = list(img_ids)[:max_samples]
            true_boxes = [box for box in true_boxes if box[0] in img_ids]
        else:
            true_boxes = true_boxes[:max_samples]
    
    # List to store AP for each class
    average_precisions = []
    epsilon = 1e-6

    # Calculate AP for each class
    print(f"Calculating mAP across {num_classes} classes...")
    for c in tqdm(range(num_classes), desc="Processing classes"):
        detections = []
        ground_truths = []

        # Filter detections and ground truths for current class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # If no ground truths exist for this class, skip
        if len(ground_truths) == 0:
            continue

        # Count number of ground truths per image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # Convert to tensor of zeros
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x[2], reverse=True)
        
        # Initialize TP and FP counters
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # Process each detection
        for detection_idx, detection in enumerate(detections):
            # Get ground truths for current image
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = -1

            # Find ground truth with highest IoU
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # If IoU exceeds threshold, it's a potential TP
            if best_iou > iou_threshold and best_gt_idx >= 0:
                # Check if this ground truth has already been detected
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # True positive
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # False positive (duplicate detection)
                    FP[detection_idx] = 1
            else:
                # False positive (IoU too low)
                FP[detection_idx] = 1

        # Calculate cumulative sums
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        # Calculate recall and precision
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        
        # Add start points for numerical integration
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        # Calculate average precision using trapezoidal rule
        average_precisions.append(torch.trapz(precisions, recalls))

    # Return mean of average precisions across all classes
    if not average_precisions:
        print("No valid classes for mAP calculation")
        return 0.0
    
    mean_ap = sum(average_precisions) / len(average_precisions)
    print(f"Completed mAP calculation: {mean_ap:.4f}")
    return mean_ap


def quick_eval_metrics(model, val_loader, device, max_batches=5):
    """
    Calculate simple classification and confidence metrics without the expensive mAP calculation.
    This is a much faster alternative to full mAP calculation.
    """
    model.eval()
    
    # Initialize as tensors instead of integers
    class_correct = torch.tensor(0.0, device=device)
    class_total = torch.tensor(0.0, device=device)
    obj_correct = torch.tensor(0.0, device=device)
    obj_total = torch.tensor(0.0, device=device)
    noobj_correct = torch.tensor(0.0, device=device)
    noobj_total = torch.tensor(0.0, device=device)
    
    conf_scores = []
    
    print("Calculating quick evaluation metrics...")
    batch_count = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, total=min(max_batches, len(val_loader))):
            if batch_count >= max_batches:
                break
                
            x = x.to(device)
            y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
            
            out = model(x)
            
            # Check accuracy for one scale only (scale with middle resolution)
            i = 1  # Use middle scale for faster processing
            
            # Handle DPU-compatible model outputs
            if DPU_COMPATIBLE and isinstance(out[i], tuple):
                # For DPU model, out[i] is a tuple of (anchor1, anchor2, anchor3) tensors
                anchor1, anchor2, anchor3 = out[i]
                
                # Use the first anchor tensor for metrics
                # The objectness score is in the first channel
                objectness = anchor1[:, 0, :, :]  # Shape [batch_size, grid_h, grid_w]
                obj_preds = torch.sigmoid(objectness) > 0.5
                
                # Get target shape info for y1
                batch_size, num_anchors, grid_h, grid_w, _ = y1.shape
                
                # Process only the first anchor for simplicity
                y1_first_anchor = y1[:, 0, :, :, :]  # Shape [batch_size, grid_h, grid_w, 6]
                
                # Identify objects and non-objects
                obj_mask = (y1_first_anchor[..., 0] == 1)  # Shape [batch_size, grid_h, grid_w]
                noobj_mask = (y1_first_anchor[..., 0] == 0)  # Shape [batch_size, grid_h, grid_w]
                
                # Calculate objectness accuracy
                if torch.sum(obj_mask) > 0:
                    obj_correct += torch.sum((obj_preds[obj_mask] == 1).float())
                    obj_total += torch.sum(obj_mask)
                
                if torch.sum(noobj_mask) > 0:
                    noobj_correct += torch.sum((obj_preds[noobj_mask] == 0).float())
                    noobj_total += torch.sum(noobj_mask)
                
                # Calculate class accuracy if there are objects
                if torch.sum(obj_mask) > 0:
                    # Find where the objects are
                    obj_indices = torch.nonzero(obj_mask, as_tuple=True)
                    
                    # Get class predictions for those positions
                    # Class channels start at index 5
                    class_channels = anchor1[:, 5:, :, :]  # Shape [batch_size, num_classes, grid_h, grid_w]
                    
                    all_class_preds = []
                    all_class_targets = []
                    
                    for idx in range(len(obj_indices[0])):
                        b, y_pos, x_pos = obj_indices[0][idx], obj_indices[1][idx], obj_indices[2][idx]
                        
                        # Get class prediction
                        class_pred = torch.argmax(class_channels[b, :, y_pos, x_pos]).item()
                        all_class_preds.append(class_pred)
                        
                        # Get class target
                        class_target = int(y1_first_anchor[b, y_pos, x_pos, 5].item())
                        all_class_targets.append(class_target)
                        
                        # If prediction is correct, record confidence
                        if class_pred == class_target:
                            conf_score = torch.sigmoid(objectness[b, y_pos, x_pos]).item()
                            conf_scores.append(conf_score)
                    
                    # Convert to tensors for counting
                    all_class_preds = torch.tensor(all_class_preds, device=device)
                    all_class_targets = torch.tensor(all_class_targets, device=device)
                    
                    # Count correct classifications
                    class_correct += torch.sum((all_class_preds == all_class_targets).float())
                    class_total += len(all_class_preds)
            else:
                # Original model outputs handling
                obj = y1[..., 0] == 1
                noobj = y1[..., 0] == 0
                
                # Get predictions
                obj_preds = torch.sigmoid(out[i][..., 0]) > 0.5
                
                # Handle class predictions only if there are objects
                if torch.sum(obj) > 0:
                    class_preds = torch.argmax(out[i][..., 5:][obj], dim=-1)
                    class_targets = y1[..., 5][obj].long()  # Ensure it's long type for comparison
                    
                    # Calculate class accuracy
                    class_correct += torch.sum(class_preds == class_targets)
                    class_total += torch.sum(obj)
                    
                    # Record confidence scores of correct predictions
                    correct_mask = class_preds == class_targets
                    if torch.sum(correct_mask) > 0:
                        scores = torch.sigmoid(out[i][..., 0][obj][correct_mask])
                        conf_scores.extend(scores.cpu().tolist())
                
                # Calculate objectness accuracy
                obj_correct += torch.sum(obj_preds[obj] == y1[..., 0][obj])
                obj_total += torch.sum(obj)
                
                noobj_correct += torch.sum(obj_preds[noobj] == y1[..., 0][noobj])
                noobj_total += torch.sum(noobj)
            
            batch_count += 1
    
    # Calculate metrics - already using tensor operations
    class_acc = (class_correct / (class_total + 1e-16)) * 100
    obj_acc = (obj_correct / (obj_total + 1e-16)) * 100
    noobj_acc = (noobj_correct / (noobj_total + 1e-16)) * 100
    
    # Calculate average confidence score for correct predictions
    avg_confidence = sum(conf_scores) / (len(conf_scores) + 1e-16) if conf_scores else 0.0
    
    # Calculate a simplified composite score (replaces mAP)
    # Balance between classification and localization performance
    composite_score = (class_acc.item() * 0.5 + obj_acc.item() * 0.3 + noobj_acc.item() * 0.2) / 100
    
    model.train()
    
    return {
        'class_acc': class_acc.item(),
        'obj_acc': obj_acc.item(),
        'noobj_acc': noobj_acc.item(),
        'avg_confidence': avg_confidence,
        'composite_score': composite_score
    }


# Ignite Training Setup
#############################################

def prepare_batch(batch, device, non_blocking=True):
    """Prepare batch for training/evaluation."""
    x, y = batch
    # Convert tensors to device silently (no print statements)
    x = x.to(device, non_blocking=non_blocking)
    y0 = y[0].to(device, non_blocking=non_blocking)
    y1 = y[1].to(device, non_blocking=non_blocking) 
    y2 = y[2].to(device, non_blocking=non_blocking)
    return x, (y0, y1, y2)


def create_trainer(model, optimizer, loss_fn, scaled_anchors, device, scaler):
    """Create training engine with Ignite."""
    
    def _update(engine, batch):
        model.train()
        
        x, (y0, y1, y2) = prepare_batch(batch, device)
        
        # Zero gradients first
        optimizer.zero_grad()
        
        # Use mixed precision training
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
        
        # Use gradient scaling
        scaler.scale(loss).backward()
        
        # Step optimizer (before scheduler to avoid warnings)
        scaler.step(optimizer)
        scaler.update()
        
        # Update OneCycleLR scheduler if it exists
        if hasattr(engine.state, "schedulers") and engine.state.schedulers:
            for scheduler in engine.state.schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
        
        return {'loss': loss.item()}
    
    trainer = Engine(_update)
    
    # Attach metrics to trainer
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    
    return trainer


def create_evaluator(model, loss_fn, scaled_anchors, device):
    """Create evaluation engine with Ignite."""
    
    def _evaluate(engine, batch):
        model.eval()
        with torch.no_grad():
            x, (y0, y1, y2) = prepare_batch(batch, device)
            
            # Forward pass
            out = model(x)
            
            # Calculate loss
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            
            return {'loss': loss.item()}
    
    evaluator = Engine(_evaluate)
    
    # Attach metrics to evaluator
    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    
    return evaluator


def calculate_map(
    model, val_loader, anchors, device, iou_threshold=0.5, threshold=0.05, box_format="midpoint", num_classes=NUM_CLASSES
):
    """Calculate mAP for validation set with optimized performance."""
    
    try:
        print("Starting mAP calculation (optimized for speed)...")
        # Get bounding boxes for evaluation (limited number of batches for speed)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            val_loader,
            model,
            iou_threshold=NMS_IOU_THRESH,
            anchors=anchors,
            threshold=threshold,
            device=device,
            max_batches=10  # Limit to 10 batches maximum
        )
        
        # If no boxes were detected or found in ground truth
        if not pred_boxes or not true_boxes:
            print("Warning: No boxes found for mAP calculation")
            return 0.0
        
        # Calculate mAP with sample limiting for speed
        map_value = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=iou_threshold,
            box_format=box_format,
            num_classes=num_classes,
            max_samples=300  # Limit number of samples for faster calculation
        )
        
        return map_value
    
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # Return 0.0 to allow training to continue


# Main Function
#############################################

def main():
    # Suppress PyTorch warnings about scheduler order
    import warnings
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
    
    # Create CSV files for datasets
    train_csv = os.path.join(MODEL_SAVE_DIR, "train_data.csv")
    valid_csv = os.path.join(MODEL_SAVE_DIR, "valid_data.csv")
    test_csv = os.path.join(MODEL_SAVE_DIR, "test_data.csv")
    
    create_dataset_csv(TRAIN_DIR, TRAIN_LABEL_DIR, train_csv)
    create_dataset_csv(VALID_DIR, VALID_LABEL_DIR, valid_csv)
    create_dataset_csv(TEST_DIR, TEST_LABEL_DIR, test_csv)

    # Check for existing checkpoint to resume training
    resume_training = False
    checkpoint_path = None
    
    # Look for the latest checkpoint
    checkpoints = glob.glob(os.path.join(MODEL_SAVE_DIR, "yolov3_water_meter_epoch*.pth"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        response = input(f"Found checkpoint {latest_checkpoint}. Resume training? (y/n): ")
        if response.lower() == 'y':
            resume_training = True
            checkpoint_path = latest_checkpoint
    
    # Initialize model
    model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Get data loaders - create these first as they're needed for scheduler setup
    print("Creating data loaders...")
    train_loader, valid_loader, test_loader = get_loaders(train_csv, valid_csv, test_csv)
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if resume_training and checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Load optimizer state if resuming training
    if resume_training and checkpoint_path:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Initialize learning rate scheduler - after train_loader is created
    if SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=STEP_SIZE, 
            gamma=GAMMA
        )
    elif SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS - WARMUP_EPOCHS,
            eta_min=LEARNING_RATE * LR_MIN_FACTOR
        )
    elif SCHEDULER_TYPE == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE * 10,  # Peak learning rate
            total_steps=NUM_EPOCHS * len(train_loader),
            pct_start=0.3,  # Percentage of training to increase LR
            div_factor=10,  # Initial learning rate = max_lr/div_factor
            final_div_factor=1/(LR_MIN_FACTOR),  # Final learning rate = initial_lr/final_div_factor
            anneal_strategy='cos'
        )
    elif SCHEDULER_TYPE == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=GAMMA,
            patience=PATIENCE,
            verbose=True,
            min_lr=LEARNING_RATE * LR_MIN_FACTOR
        )
    else:
        scheduler = None
        print(f"Unknown scheduler type: {SCHEDULER_TYPE}, proceeding without a scheduler")
    
    # Create warmup scheduler wrapper if needed
    if SCHEDULER_TYPE != "onecycle" and WARMUP_EPOCHS > 0 and scheduler is not None:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=WARMUP_FACTOR, 
            end_factor=1.0, 
            total_iters=WARMUP_EPOCHS * len(train_loader)
        )
        
        # Chain warmup with main scheduler
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[WARMUP_EPOCHS * len(train_loader)]
        )
    
    # Load scheduler state if resuming training
    if resume_training and checkpoint_path and 'scheduler_state_dict' in checkpoint:
        if checkpoint.get('scheduler_type', '') == SCHEDULER_TYPE:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state")
        else:
            print(f"Warning: Checkpoint used {checkpoint.get('scheduler_type', 'unknown')} scheduler, "
                  f"but current configuration uses {SCHEDULER_TYPE}. Scheduler state not loaded.")
    
    # Initialize loss function
    loss_fn = YoloLoss()
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Load scaler state if resuming training
    if resume_training and checkpoint_path and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    # Scale anchors for each prediction scale
    scaled_anchors = (
        torch.tensor(ANCHORS) 
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)
    
    # Create Ignite engines
    trainer = create_trainer(model, optimizer, loss_fn, scaled_anchors, DEVICE, scaler)
    evaluator = create_evaluator(model, loss_fn, scaled_anchors, DEVICE)
    
    # Store scheduler in trainer state to access it in training loop
    trainer.state.schedulers = []
    
    # Add scheduler step handlers based on scheduler type
    if scheduler is not None:
        if SCHEDULER_TYPE == "plateau":
            # ReduceLROnPlateau needs validation loss
            @trainer.on(Events.EPOCH_COMPLETED)
            def update_plateau_scheduler(engine):
                evaluator.run(valid_loader)
                val_loss = evaluator.state.metrics['loss']
                scheduler.step(val_loss)
        elif SCHEDULER_TYPE == "onecycle":
            # Store the scheduler in engine state for access in update function
            trainer.state.schedulers.append(scheduler)
        else:
            # Other schedulers update every epoch
            @trainer.on(Events.EPOCH_COMPLETED)
            def update_scheduler(engine):
                scheduler.step()
    
    # Add progress bar for training with description
    pbar = ProgressBar(desc="Training")
    pbar.attach(trainer, metric_names=['loss'])
    
    # Add running average metrics to evaluator
    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    
    # Add progress bar for validation with description
    pbar_eval = ProgressBar(desc="Validation")
    pbar_eval.attach(evaluator, metric_names=['loss'])
    
    # Set up checkpoint handler
    checkpoint_dir = os.path.join(MODEL_SAVE_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # To store objects for checkpointing
    to_save = {
        'model': model,
        'optimizer': optimizer,
        'trainer': trainer,
        'scaler': scaler
    }
    
    # Add scheduler to checkpoint if it exists
    if scheduler is not None:
        to_save['scheduler'] = scheduler
    
    checkpoint_handler = Checkpoint(
        to_save, 
        DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        n_saved=3,
        score_function=lambda engine: -engine.state.metrics['loss'],
        score_name='val_loss',
        filename_prefix='best'
    )
    
    # Set up tensorboard logger
    tb_logger = TensorboardLogger(log_dir=LOGS_DIR)
    
    # Attach handler for model weights & optimizer parameters only when metrics exist
    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_metrics(engine):
        if 'loss' in engine.state.metrics:
            tb_logger.writer.add_scalar("training/loss", engine.state.metrics['loss'], engine.state.iteration)
            # Log learning rate
            lr = optimizer.param_groups[0]['lr']
            tb_logger.writer.add_scalar("training/learning_rate", lr, engine.state.iteration)
    
    # Track epoch metrics and timing
    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start(engine):
        engine.state.epoch_start_time = time.time()
        engine.state.epoch_metrics = {"train_loss": []}

    # Track batch losses during training
    @trainer.on(Events.ITERATION_COMPLETED)
    def track_batch_loss(engine):
        if hasattr(engine.state, "epoch_metrics"):
            if "output" in engine.state.output and "loss" in engine.state.output:
                engine.state.epoch_metrics["train_loss"].append(engine.state.output["loss"])
    
    # Epoch completed handler to print detailed epoch summary
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_epoch_summary(engine):
        epoch = engine.state.epoch
        epoch_time = time.time() - engine.state.epoch_start_time
        
        # Calculate training metrics
        if hasattr(engine.state, "epoch_metrics") and engine.state.epoch_metrics["train_loss"]:
            train_losses = engine.state.epoch_metrics["train_loss"]
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # Get current learning rate
            lr = optimizer.param_groups[0]['lr']
            
            # Print header for epoch summary
            print("\n" + "="*80)
            print(f"Epoch {epoch}/{NUM_EPOCHS} Summary - Completed in {epoch_time:.2f}s")
            print("-"*80)
            
            # Print training metrics
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Learning Rate: {lr:.8f}")
            
            # Run evaluator for validation metrics
            evaluator.run(valid_loader)
            val_loss = evaluator.state.metrics.get('loss', float('nan'))
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Estimate remaining time
            if epoch > 1 and hasattr(engine.state, 'epoch_times'):
                engine.state.epoch_times.append(epoch_time)
            else:
                engine.state.epoch_times = [epoch_time]
                
            avg_epoch_time = sum(engine.state.epoch_times) / len(engine.state.epoch_times)
            remaining_epochs = NUM_EPOCHS - epoch
            est_remaining_time = avg_epoch_time * remaining_epochs
            
            # Convert to hours, minutes, seconds
            hours, remainder = divmod(est_remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Log metrics to tensorboard
            tb_logger.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
            tb_logger.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            tb_logger.writer.add_scalar("epoch/learning_rate", lr, epoch)
            
            # Return to a clean state for the next epoch
            engine.state.epoch_metrics = {"train_loss": []}
            
            print("="*80)
    
    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED
    )
    
    # GPU information logging - make it optional
    if DEVICE == "cuda":
        try:
            # Adjuntar GpuInfo como métrica al entrenador
            gpu_metrics = GpuInfo()
            gpu_metrics.attach(trainer, name="gpu")
            
            # Luego configurar el manejador de registros para capturar las métricas
            # Usa los nombres exactos de métricas en lugar de comodines
            tb_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="training", 
                    # Cambia 'gpu:*' por los nombres exactos de las métricas
                    metric_names=["loss", "gpu:0 mem(%)", "gpu:0 util(%)"]
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            
            print("GPU monitoring attached successfully")
            
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Could not attach GPU monitoring: {e}")
            print("Training will continue without GPU monitoring")
            print("If you want GPU monitoring, try: pip install pynvml==11.5.0")
            print("You may also need to restart your Python environment after installation")
    
    # Track best model based on validation mAP
    best_map = 0.0
    best_epoch = 0
    
    # Evaluation function to run at the end of each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        nonlocal best_map, best_epoch
        
        epoch = engine.state.epoch
        
        # Run validation engine on validation dataset
        evaluator.run(valid_loader)
        val_loss = evaluator.state.metrics['loss']
        
        print(f"Validation Results - Epoch: {epoch}  Avg loss: {val_loss:.4f}")
        
        # Run every 5 epochs or on the last epoch
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print("\nEvaluating model performance...")
            
            # Use quick evaluation metrics instead of slow mAP calculation
            metrics = quick_eval_metrics(model, valid_loader, DEVICE, max_batches=5)
            
            # Print detailed classification performance
            print(f"Classification Performance:")
            print(f"  - Class accuracy: {metrics['class_acc']:.2f}%")
            print(f"  - Object detection accuracy: {metrics['obj_acc']:.2f}%")
            print(f"  - No-object accuracy: {metrics['noobj_acc']:.2f}%")
            print(f"  - Average confidence: {metrics['avg_confidence']:.4f}")
            print(f"  - Composite score: {metrics['composite_score']:.4f}")
            
            # Log to tensorboard
            tb_logger.writer.add_scalar("metrics/class_accuracy", metrics['class_acc'], epoch)
            tb_logger.writer.add_scalar("metrics/obj_accuracy", metrics['obj_acc'], epoch)
            tb_logger.writer.add_scalar("metrics/no_obj_accuracy", metrics['noobj_acc'], epoch)
            tb_logger.writer.add_scalar("metrics/avg_confidence", metrics['avg_confidence'], epoch)
            tb_logger.writer.add_scalar("metrics/composite_score", metrics['composite_score'], epoch)
            
            # Calculate full mAP very infrequently (optional)
            use_map = False  # Set to True if you want to occasionally calculate mAP
            if use_map and (epoch % 20 == 0 or epoch == NUM_EPOCHS):
                try:
                    print("\nCalculating full mAP (this may take a while)...")
                    map_value = calculate_map(model, valid_loader, ANCHORS, DEVICE)
                    print(f"  - Validation mAP: {map_value:.4f}")
                    tb_logger.writer.add_scalar("metrics/mAP", map_value, epoch)
                except Exception as e:
                    print(f"Error calculating mAP: {e}")
                    map_value = 0.0
            else:
                # Use composite score as a proxy for mAP
                map_value = metrics['composite_score']
            
            # Save best model
            if map_value > best_map:
                best_map = map_value
                best_epoch = epoch
                
                # Save best model
                best_model_path = os.path.join(MODEL_SAVE_DIR, f"yolov3_water_meter_best_map{map_value:.4f}.pth")
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': map_value,
                    'loss': val_loss,
                    'scaler': scaler.state_dict(),
                }
                
                # Add scheduler state if it exists
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                    checkpoint_data['scheduler_type'] = SCHEDULER_TYPE
                
                torch.save(checkpoint_data, best_model_path)
                
                print(f"Saved best model with mAP {best_map:.4f} at epoch {best_epoch}")
    
    # Save model at regular intervals
    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def save_model(engine):
        epoch = engine.state.epoch
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }
        
        # Add scheduler state if it exists
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_data['scheduler_type'] = SCHEDULER_TYPE
        
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"yolov3_water_meter_epoch{epoch}.pth")
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"\n💾 Checkpoint saved at epoch {epoch} to {checkpoint_path}")
    
    # Handler for NaN values in loss
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    
    # Run the training
    print("\n" + "="*80)
    print(f"🚀 Starting YOLOv3 training for {NUM_EPOCHS} epochs")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Scheduler: {SCHEDULER_TYPE}")
    print(f"   - Device: {DEVICE}")
    print(f"   - Training samples: {len(train_loader.dataset)}")
    print(f"   - Validation samples: {len(valid_loader.dataset)}")
    if test_loader:
        print(f"   - Test samples: {len(test_loader.dataset)}")
    print("="*80 + "\n")
    
    # Start timer for total training time
    total_start_time = time.time()
    
    trainer.run(train_loader, max_epochs=NUM_EPOCHS)
    
    # Calculate total training time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print(f"🏁 Training complete!")
    print(f"   - Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"   - Best model: mAP {best_map:.4f} at epoch {best_epoch}")
    print("="*80)
    
    # Close tensorboard logger
    tb_logger.close()
    
    # Final evaluation on test set
    if test_loader:
        print("\n" + "="*80)
        print("📊 Evaluating on test set...")
        # Load best model
        print(f"   Loading best model from epoch {best_epoch} with score {best_map:.4f}")
        checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, f"yolov3_water_meter_best_score{best_map:.4f}.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Use quick evaluation metrics
        test_metrics = quick_eval_metrics(model, test_loader, DEVICE, max_batches=10)
        
        print(f"\nTest Results Summary:")
        print(f"   - Test class accuracy: {test_metrics['class_acc']:.2f}%")
        print(f"   - Test object detection accuracy: {test_metrics['obj_acc']:.2f}%")
        print(f"   - Test no-object accuracy: {test_metrics['noobj_acc']:.2f}%")
        print(f"   - Test composite score: {test_metrics['composite_score']:.4f}")
        
        # Print comparison with validation performance
        score_diff = test_metrics['composite_score'] - best_map
        print(f"\nPerformance Delta (Test vs. Validation):")
        print(f"   - Score difference: {score_diff:.4f} ({'better' if score_diff > 0 else 'worse'})")
        
        # Save test results
        test_results_path = os.path.join(MODEL_SAVE_DIR, "test_results.txt")
        with open(test_results_path, "w") as f:
            f.write(f"Test Results - Best model from epoch {best_epoch}\n")
            f.write(f"Test composite score: {test_metrics['composite_score']:.4f}\n")
            f.write(f"Test class accuracy: {test_metrics['class_acc']:.2f}%\n")
            f.write(f"Test object detection accuracy: {test_metrics['obj_acc']:.2f}%\n")
            f.write(f"Test no-object accuracy: {test_metrics['noobj_acc']:.2f}%\n")
            f.write(f"\nPerformance comparison:\n")
            f.write(f"Validation score: {best_map:.4f}\n")
            f.write(f"Score difference: {score_diff:.4f}\n")
        
        print(f"\n📝 Test results saved to: {test_results_path}")
        print("="*80)
    
    print("\n✨ All done! ✨")


if __name__ == "__main__":
    main()