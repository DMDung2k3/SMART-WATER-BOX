import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.ops import nms

# Model Definition
def conv_bn_leaky(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    )

class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class YOLOv3TinyWithCNN(nn.Module):
    def __init__(self, num_classes_yolo=1, num_classes_cnn=10):
        super(YOLOv3TinyWithCNN, self).__init__()
        self.num_classes_yolo = num_classes_yolo
        self.num_det_filters = 3 * (num_classes_yolo + 5)
        self.num_classes_cnn = num_classes_cnn

        self.conv1 = conv_bn_leaky(3, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_bn_leaky(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv_bn_leaky(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = conv_bn_leaky(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = conv_bn_leaky(128, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = conv_bn_leaky(256, 512, 3, 1, 1)
        self.pool6 = nn.MaxPool2d(2, 1)
        self.conv7 = conv_bn_leaky(512, 1024, 3, 1, 1)

        self.conv_reduction1 = conv_bn_leaky(1024, 256, 1, 1, 0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_medium = conv_bn_leaky(256 + 512, 256, 3, 1, 1)
        self.det_medium = nn.Conv2d(256, self.num_det_filters, 1, 1, 0)

        self.conv_reduction2 = conv_bn_leaky(self.num_det_filters, 128, 1, 1, 0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_small = conv_bn_leaky(128 + 256, 128, 3, 1, 1)
        self.det_small = nn.Conv2d(128, self.num_det_filters, 1, 1, 0)

        self.conv_large = conv_bn_leaky(1024, 512, 3, 1, 1)
        self.det_large = nn.Conv2d(512, self.num_det_filters, 1, 1, 0)

        self.cnn = DigitCNN(num_classes=num_classes_cnn)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        route2 = self.conv5(x)
        x = self.pool5(route2)
        route1 = self.conv6(x)
        x = self.pool6(route1)
        x = self.conv7(x)

        det_large = self.conv_large(x)
        det_large = self.det_large(det_large)

        x_reduced1 = self.conv_reduction1(x)
        x_upsampled1 = self.upsample1(x_reduced1)
        x_upsampled1 = torch.nn.functional.interpolate(x_upsampled1, size=route1.shape[2:], mode='bilinear', align_corners=False)
        x_concat1 = torch.cat([x_upsampled1, route1], dim=1)
        det_medium = self.conv_medium(x_concat1)
        det_medium = self.det_medium(det_medium)

        x_reduced2 = self.conv_reduction2(det_medium)
        x_upsampled2 = self.upsample2(x_reduced2)
        x_upsampled2 = torch.nn.functional.interpolate(x_upsampled2, size=route2.shape[2:], mode='bilinear', align_corners=False)
        x_concat2 = torch.cat([x_upsampled2, route2], dim=1)
        det_small = self.conv_small(x_concat2)
        det_small = self.det_small(det_small)

        return det_small, det_medium, det_large, self.cnn

    def extract_and_classify(self, x, det_small, det_medium, det_large, conf_thres=0.3, nms_thres=0.4):
        detections = []
        batch_size = x.size(0)
        img_size = x.size(2)

        for pred in [det_small, det_medium, det_large]:
            grid_size = pred.size(2)
            stride = img_size // grid_size
            pred = pred.view(batch_size, 3, 5 + self.num_classes_yolo, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            pred_xy = torch.sigmoid(pred[..., 0:2])
            pred_wh = pred[..., 2:4]
            pred_obj = torch.sigmoid(pred[..., 4])
            pred_cls = torch.sigmoid(pred[..., 5:])

            for b in range(batch_size):
                for i in range(grid_size):
                    for j in range(grid_size):
                        for a in range(3):
                            if pred_obj[b, a, i, j] > conf_thres:
                                cx = (j + pred_xy[b, a, i, j, 0]) * stride
                                cy = (i + pred_xy[b, a, i, j, 1]) * stride
                                w = torch.exp(pred_wh[b, a, i, j, 0]) * stride
                                h = torch.exp(pred_wh[b, a, i, j, 1]) * stride
                                x_min = cx - w / 2
                                y_min = cy - h / 2
                                cls_id = torch.argmax(pred_cls[b, a, i, j]).item()
                                score = pred_obj[b, a, i, j].item()
                                detections.append([b, cls_id, x_min, y_min, x_min + w, y_min + h, score])

        detections = torch.tensor(detections, dtype=torch.float32) if detections else torch.zeros((0, 7))
        if len(detections) == 0:
            return [], []

        boxes = detections[:, 2:6]
        scores = detections[:, 6]
        keep = nms(boxes, scores, iou_threshold=nms_thres)
        detections = detections[keep]

        digit_preds = []
        for det in detections:
            b, cls_id, x_min, y_min, x_max, y_max, _ = det
            if int(cls_id) == 0:
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_size, x_max)
                y_max = min(img_size, y_max)
                if x_max > x_min and y_max > y_min:
                    roi = x[int(b), :, y_min:y_max, x_min:x_max]
                    roi = torch.nn.functional.interpolate(roi.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False)
                    digit_pred = self.cnn(roi)
                    digit_preds.append([int(b), torch.argmax(digit_pred, dim=1).item(), x_min, y_min, x_max, y_max])

        return detections, digit_preds

# Dataset Class
class WaterMeterDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None, img_size=420):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.classes = ['border_water_meter_number']
        self.digit_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].rsplit('.', 1)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        yolo_boxes = []
        digit_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        if int(class_id) < 10:
                            digit_labels.append([class_id, x_min, y_min, x_min + width, y_min + height])
                            yolo_boxes.append([0, x_min, y_min, x_min + width, y_min + height])
                        else:
                            yolo_boxes.append([0, x_min, y_min, x_min + width, y_min + height])
                    except ValueError:
                        print(f"Warning: Invalid label format in {label_path}, skipping line")

        if h != self.img_size or w != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
            scale = self.img_size / max(h, w)
            if yolo_boxes:
                yolo_boxes = np.array(yolo_boxes)
                yolo_boxes[:, 1:] = yolo_boxes[:, 1:] * scale

                yolo_boxes[:, 1:] = np.clip(yolo_boxes[:, 1:], 0, self.img_size)
            if digit_labels:
                digit_labels = np.array(digit_labels)
                digit_labels[:, 1:] = digit_labels[:, 1:] * scale
                digit_labels[:, 1:] = np.clip(digit_labels[:, 1:], 0, self.img_size)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        yolo_boxes_tensor = torch.tensor(yolo_boxes, dtype=torch.float32) if len(yolo_boxes) > 0 else torch.zeros((0, 5))
        digit_labels_tensor = torch.tensor(digit_labels, dtype=torch.float32) if len(digit_labels) > 0 else torch.zeros((0, 5))
        return img, yolo_boxes_tensor, digit_labels_tensor

# Custom Collate Function
def custom_collate_fn(batch):
    images = []
    yolo_boxes = []
    digit_labels = []
    for img, yolo_box, digit_label in batch:
        images.append(img)
        yolo_boxes.append(yolo_box)
        digit_labels.append(digit_label)
    images = torch.stack(images, dim=0)
    return images, yolo_boxes, digit_labels

# YOLO Loss
class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors, img_size=420):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = [torch.tensor(anchor, dtype=torch.float32) for anchor in anchors]
        self.num_anchors = len(anchors[0])
        self.img_size = img_size
        self.obj_scale = 1.0
        self.noobj_scale = 50.0
        self.box_scale = 5.0
        self.cls_scale = 1.0
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        batch_size = predictions[0].size(0)
        total_loss = 0.0

        anchors = [anchor.to(predictions[0].device) for anchor in self.anchors]

        for pred, anchor_set in zip(predictions, anchors):
            grid_size = pred.size(2)
            stride = self.img_size // grid_size
            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            pred_xy = torch.sigmoid(pred[..., 0:2])
            pred_wh = pred[..., 2:4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            obj_loss = 0.0
            noobj_loss = 0.0
            box_loss = 0.0
            cls_loss = 0.0

            for b in range(batch_size):
                if targets[b].size(0) == 0:
                    noobj_loss += self.bce_loss(pred_obj[b], torch.zeros_like(pred_obj[b]))
                    continue

                target_boxes = targets[b].clone()
                target_boxes[:, 1:] = target_boxes[:, 1:] / stride
                target_xy = target_boxes[:, 1:3]
                target_wh = target_boxes[:, 3:5] - target_boxes[:, 1:3]
                target_cls = target_boxes[:, 0].long()

                for t in range(target_boxes.size(0)):
                    gx, gy = target_xy[t]
                    gw, gh = target_wh[t]
                    grid_x, grid_y = int(gx), int(gy)

                    if not (0 <= grid_x < grid_size and 0 <= grid_y < grid_size):
                        continue

                    anchor_wh = anchor_set / stride
                    inter_w = torch.min(gw.unsqueeze(0), anchor_wh[:, 0])
                    inter_h = torch.min(gh.unsqueeze(0), anchor_wh[:, 1])
                    inter_area = inter_w * inter_h
                    union_area = (gw * gh) + (anchor_wh[:, 0] * anchor_wh[:, 1]) - inter_area
                    iou = inter_area / (union_area + 1e-16)
                    best_anchor = torch.argmax(iou)

                    gt_obj = torch.tensor(1.0).to(pred.device)
                    gt_xy = torch.tensor([gx - grid_x, gy - grid_y]).to(pred.device)
                    gt_wh = torch.log(target_wh[t] / (anchor_set[best_anchor] / stride + 1e-16)).to(pred.device)
                    gt_cls = torch.zeros(self.num_classes).to(pred.device)
                    gt_cls[target_cls[t]] = 1.0

                    pred_xy_t = pred_xy[b, best_anchor, grid_y, grid_x]
                    pred_wh_t = pred_wh[b, best_anchor, grid_y, grid_x]
                    pred_obj_t = pred_obj[b, best_anchor, grid_y, grid_x]
                    pred_cls_t = pred_cls[b, best_anchor, grid_y, grid_x]

                    box_loss += self.mse_loss(pred_xy_t, gt_xy)
                    box_loss += self.mse_loss(pred_wh_t, gt_wh)
                    obj_loss += self.bce_loss(pred_obj_t, gt_obj)
                    cls_loss += self.bce_loss(pred_cls_t, gt_cls)

                    noobj_mask = torch.ones_like(pred_obj[b])
                    noobj_mask[best_anchor, grid_y, grid_x] = 0
                    noobj_loss += self.bce_loss(pred_obj[b] * noobj_mask, torch.zeros_like(pred_obj[b]))

            total_loss += (self.box_scale * box_loss + self.obj_scale * obj_loss +
                          self.noobj_scale * noobj_loss + self.cls_scale * cls_loss)

        return total_loss / batch_size

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, num_classes_yolo, num_classes_cnn, anchors, img_size=420):
        super(CombinedLoss, self).__init__()
        self.yolo_loss = YOLOLoss(num_classes_yolo, anchors, img_size)
        self.cnn_loss = nn.CrossEntropyLoss()
        self.cnn_loss_weight = 5.0

    def forward(self, det_small, det_medium, det_large, cnn, images, targets_yolo, targets_cnn, anchors):
        yolo_loss = self.yolo_loss([det_small, det_medium, det_large], targets_yolo)

        cnn_loss = 0.0
        for b in range(images.size(0)):
            if targets_cnn[b].size(0) > 0:
                for t in targets_cnn[b]:
                    cls_id, x_min, y_min, x_max, y_max = t
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(images.size(2), x_max)
                    y_max = min(images.size(3), y_max)
                    if x_max > x_min and y_max > y_min:
                        roi = images[b:b+1, :, y_min:y_max, x_min:x_max]
                        roi = torch.nn.functional.interpolate(roi, size=(28, 28), mode='bilinear', align_corners=False)
                        pred = cnn(roi)
                        target = torch.tensor([int(cls_id)], dtype=torch.long).to(images.device)
                        cnn_loss += self.cnn_loss(pred, target)

        total_loss = yolo_loss + self.cnn_loss_weight * cnn_loss
        return total_loss