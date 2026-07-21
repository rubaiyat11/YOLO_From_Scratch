from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo11n.pt")
net = model.model

weights = net.model[0].conv.weight.detach()

print("Weight Tensor Shape:", weights.shape)

print("\nFirst filter weights:\n")

print(weights[0])


print("\n========== Backbone Convolution Layers ==========\n")

for idx, layer in enumerate(net.model):

    if hasattr(layer, "conv"):
        weight = layer.conv.weight

        print(f"Layer {idx}")
        print(f"Weight Shape : {tuple(weight.shape)}")
        print(f"Kernel Size : {weight.shape[2]}x{weight.shape[3]}")
        print(f"Input Channels : {weight.shape[1]}")
        print(f"Output Channels: {weight.shape[0]}")
        print("-" * 40)



print("\n========== Backbone Summary ==========\n")

for idx, layer in enumerate(net.model):

    if hasattr(layer, "conv"):
        w = layer.conv.weight

        print(
            f"Conv {idx:02d}: "
            f"{w.shape[1]} -> {w.shape[0]} channels | "
            f"{w.shape[2]}x{w.shape[3]} kernel"
        )