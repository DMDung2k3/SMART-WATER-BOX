#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import logging
import subprocess
import yaml
import numpy as np
import cv2
import glob
import json
import torch.utils.data as data
from pytorch_nndct.apis import torch_quantizer, Inspector

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Architecture
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1015625)
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
        for _ in range(num_repeats):
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
            x = x + layer(x) if self.use_residual else layer(x)
        return x

class DPUCompatibleScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DPUCompatibleScalePrediction, self).__init__()
        self.num_classes = num_classes
        self.common = CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.anchor1_pred = CNNBlock(2 * in_channels, num_classes + 5, bn_act=False, kernel_size=1)
        self.anchor2_pred = CNNBlock(2 * in_channels, num_classes + 5, bn_act=False, kernel_size=1)
        self.anchor3_pred = CNNBlock(2 * in_channels, num_classes + 5, bn_act=False, kernel_size=1)

    def forward(self, x):
        features = self.common(x)
        pred1 = self.anchor1_pred(features)
        pred2 = self.anchor2_pred(features)
        pred3 = self.anchor3_pred(features)
        return (pred1, pred2, pred3)

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=11):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
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
            ["B", 4],  # Darknet-53
            (512, 1, 1),
            (1024, 3, 1),
            "S",  # Scale 13x13
            (256, 1, 1),
            "U",
            (256, 1, 1),
            (512, 3, 1),
            "S",  # Scale 26x26
            (128, 1, 1),
            "U",
            (128, 1, 1),
            (256, 3, 1),
            "S",  # Scale 52x52
        ]
        
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, DPUCompatibleScalePrediction):
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
                        DPUCompatibleScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers

class CalibDataset(data.Dataset):
    def __init__(self, image_files, img_size):
        self.image_files = image_files
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()

def load_water_meter_model(checkpoint_path, num_classes=11):
    model = YOLOv3(num_classes=num_classes).to('cpu')
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model from checkpoint state_dict")
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            logger.info("Loaded model from Ignite checkpoint")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model directly from weights file")
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_calib_data(calib_dir, batch_size, img_size):
    image_files = glob.glob(os.path.join(calib_dir, "*.jpg")) + \
                  glob.glob(os.path.join(calib_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(calib_dir, "*.png"))
    if not image_files:
        logger.error(f"No images found in {calib_dir}")
        return None
    logger.info(f"Found {len(image_files)} images for calibration")
    dataset = CalibDataset(image_files, img_size)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def inspect_model_compatibility(model, input_shape, arch_json):
    """Kiểm tra tính tương thích với DPU bằng fingerprint từ arch.json"""
    try:
        with open(arch_json, 'r') as f:
            arch_data = json.load(f)
            dpu_fingerprint = arch_data.get('fingerprint', '0x101000056010407')  # Mặc định cho KV260
        inspector = Inspector(dpu_fingerprint)
        dummy_input = torch.randn(input_shape)
        inspector.inspect(model, (dummy_input,))
        logger.info("Model compatibility inspection completed.")
    except Exception as e:
        logger.warning(f"Inspector failed: {str(e)}. Skipping compatibility check.")

def quantize_and_compile(model_path, calib_dir, arch_json, config_yaml=None, img_size=416, batch_size=1):
    try:
        for path, name in [(model_path, "Model file"), (calib_dir, "Calibration directory"), (arch_json, "DPU arch file")]:
            if not os.path.exists(path):
                logger.error(f"{name} {path} not found!")
                return

        num_classes = 11
        if config_yaml and os.path.exists(config_yaml):
            with open(config_yaml, 'r') as file:
                yaml_data = yaml.safe_load(file)
                num_classes = yaml_data.get('nc', 11)
                logger.info(f"Loaded configuration with {num_classes} classes")

        quant_output_dir = os.path.join(os.path.dirname(model_path), 'quantized_model')
        os.makedirs(quant_output_dir, exist_ok=True)

        logger.info(f"Loading model from {model_path}...")
        pytorch_model = load_water_meter_model(model_path, num_classes=num_classes)

        input_shape = (batch_size, 3, img_size, img_size)
        logger.info("Checking model output structure...")
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            outputs = pytorch_model(dummy_input)
            logger.info(f"Model outputs: {len(outputs)} scales")
            for i, scale in enumerate(outputs):
                logger.info(f"Scale {i}: {[pred.shape for pred in scale]}")

        logger.info("Inspecting model compatibility with DPU...")
        inspect_model_compatibility(pytorch_model, input_shape, arch_json)

        dataloader = load_calib_data(calib_dir, batch_size, img_size)
        if dataloader is None:
            return

        logger.info("Running calibration...")
        quantizer = torch_quantizer(
            quant_mode='calib',
            module=pytorch_model,
            input_args=(torch.randn(input_shape),),
            output_dir=quant_output_dir,
            bitwidth=8,
            device=torch.device('cpu')
        )
        quant_model = quantizer.quant_model

        calib_batches = min(1000, len(dataloader))
        for i, batch in enumerate(dataloader):
            if i >= calib_batches:
                break
            logger.info(f"Calibrating batch {i+1}/{calib_batches}")
            with torch.no_grad():
                quant_model(batch)
        quantizer.export_quant_config()
        logger.info(f"Calibration completed. Config saved to {quant_output_dir}/quant_info.json")

        logger.info("Running test and exporting intermediate xmodel...")
        quantizer = torch_quantizer(
            quant_mode='test',
            module=pytorch_model,
            input_args=(torch.randn(input_shape),),
            output_dir=quant_output_dir,
            bitwidth=8,
            device=torch.device('cpu')
        )
        quant_model = quantizer.quant_model

        for i, batch in enumerate(dataloader):
            if i >= 1:
                break
            logger.info("Testing batch 1")
            with torch.no_grad():
                quant_model(batch)

        # Fix: Remove the 'multi_output' parameter from export_xmodel call
        # Note: Vitis AI saves the xmodel to "quantize_result/YOLOv3_int.xmodel"
        quantizer.export_xmodel(deploy_check=False)
        
        # Check for the xmodel in the expected locations
        possible_paths = [
            os.path.join(quant_output_dir, "WaterMeterDetectionModel_int.xmodel"),  # Original expected path
            os.path.join(quant_output_dir, "YOLOv3_int.xmodel"),                   # Alternative name
            os.path.join(quant_output_dir, "quantize_result", "YOLOv3_int.xmodel"), # Vitis AI default path
            os.path.join("quantize_result", "YOLOv3_int.xmodel")                   # Relative path
        ]
        
        # Try to find the xmodel file
        intermediate_xmodel = None
        for path in possible_paths:
            if os.path.exists(path):
                intermediate_xmodel = path
                logger.info(f"Found xmodel at: {intermediate_xmodel}")
                break
                
        # If still not found, try a more general glob search
        if intermediate_xmodel is None:
            # Search in the output directory and its subdirectories
            search_patterns = [
                os.path.join(quant_output_dir, "*.xmodel"),
                os.path.join(quant_output_dir, "*", "*.xmodel")
            ]
            
            for pattern in search_patterns:
                xmodel_files = glob.glob(pattern)
                if xmodel_files:
                    intermediate_xmodel = xmodel_files[0]
                    logger.info(f"Found xmodel using pattern search at: {intermediate_xmodel}")
                    break
            
        # Check if we found the file
        if intermediate_xmodel is None:
            logger.error("Could not find any xmodel file. Checking current working directory...")
            # Try the current working directory as a last resort
            cwd_files = glob.glob(os.path.join("quantize_result", "*.xmodel"))
            if cwd_files:
                intermediate_xmodel = cwd_files[0]
                logger.info(f"Found xmodel in current working directory: {intermediate_xmodel}")
            else:
                logger.error("Could not find any xmodel file anywhere")
                return

        logger.info("Compiling xmodel for KV260...")
        final_xmodel = os.path.join(quant_output_dir, "yolov3_water_meter_11_classes.xmodel")
        
        # Create compilation output directory if it doesn't exist
        os.makedirs(quant_output_dir, exist_ok=True)
        
        # Absolute path to the intermediate_xmodel
        if not os.path.isabs(intermediate_xmodel):
            # If it's a relative path, convert to absolute
            intermediate_xmodel = os.path.abspath(intermediate_xmodel)
            
        logger.info(f"Using intermediate xmodel: {intermediate_xmodel}")
        
        compile_cmd = [
            "vai_c_xir",
            "-x", intermediate_xmodel,
            "-a", arch_json,
            "-o", quant_output_dir,
            "-n", "yolov3_water_meter_11_classes"
        ]

        logger.info(f"Running command: {' '.join(compile_cmd)}")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Compilation failed: {result.stderr}")
            with open(os.path.join(quant_output_dir, "compile_error.log"), "w") as f:
                f.write(result.stderr)
            return

        logger.info(f"Successfully compiled xmodel to {final_xmodel}")
        logger.info("Compilation log:")
        logger.info(result.stdout)
        with open(os.path.join(quant_output_dir, "compile_log.txt"), "w") as f:
            f.write(result.stdout)

        logger.info("Verifying compiled xmodel...")
        subgraph_cmd = ["xir", "subgraph", final_xmodel]
        subgraph_result = subprocess.run(subgraph_cmd, capture_output=True, text=True)
        with open(os.path.join(quant_output_dir, "subgraph_output.txt"), "w") as f:
            f.write(subgraph_result.stdout)
        logger.info("Subgraph structure saved to subgraph_output.txt")
        logger.info("Full subgraph output:")
        logger.info(subgraph_result.stdout)

        logger.info("Quantization and compilation process completed successfully!")

    except Exception as e:
        logger.error(f"Error during process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Print debugging information about working directories
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Output directory existence: {os.path.exists(quant_output_dir)}")
        
        # Check for any xmodel files in the workspace
        for root, dirs, files in os.walk(os.getcwd()):
            xmodels = [os.path.join(root, file) for file in files if file.endswith('.xmodel')]
            if xmodels:
                logger.error(f"Found xmodel files: {xmodels}")

def main():
    model_path = "/app/models/yolov3_water_meter_best_map0.8860.pth"
    config_yaml = "/app/water_meter.yaml"
    calib_dir = "/app/water_meter_detection_darknet/train/images"
    arch_json = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
    img_size = 416
    batch_size = 4
    
    quantize_and_compile(model_path, calib_dir, arch_json, config_yaml, img_size, batch_size)

if __name__ == "__main__":
    main()