from ultralytics import YOLO


model = YOLO("yolo11n.pt")


net = model.model

print(type(net))

print("\nFirst module:\n")
print(net.model[0])

print("\nSecond module:\n")
print(net.model[1])

print("\nThird module:\n")
print(net.model[2])