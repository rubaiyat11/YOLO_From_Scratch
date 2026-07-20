from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo11n.pt")
net = model.model

weights = net.model[0].conv.weight.detach()

print("Weight Tensor Shape:", weights.shape)

print("\nFirst filter weights:\n")

print(weights[0])