from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt")
net = model.model

x = torch.randn(1, 3, 640, 640)

print(f"Input Shape: {x.shape}")

x = net.model[0](x)

print(f"After layer 0: {x.shape}")

x = net.model[1](x)

print(f"After layer 1: {x.shape}")

x = net.model[2](x)

print(f"After layer 2: {x.shape}")