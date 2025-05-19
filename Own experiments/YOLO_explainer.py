# Prerequisite - install easy_explain package using pip
# This code was copied from easy_explain examples
# Link:https://github.com/stavrostheocharis/easy_explain/blob/main/examples/lrp/easy_explain_lrp_for_Yolov8_use_as_package_pipeline.ipynb
from io import BytesIO
from PIL import Image
import torchvision
from ultralytics import YOLO
from easy_explain import YOLOv8LRP
import requests
import urllib.request

model = YOLO('ultralyticsplus/yolov8s')
response = requests.get("https://images.unsplash.com/photo-1584649525122-8d6895492a5d?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
# response = requests.get("https://d2tk9av7ph0ga6.cloudfront.net/image/catalog/1522919683-35709-700xauto.png")
desired_size = (512, 640)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(desired_size),
    torchvision.transforms.ToTensor(),
])
# image = Image.open(BytesIO(response.content))
urllib.request.urlretrieve('https://d2tk9av7ph0ga6.cloudfront.net/image/catalog/1522919683-35709-700xauto.png', "crochet_hook.png")
image = Image.open("crochet_hook.png").convert('RGB')
image.show()

image = transform(image)

lrp = YOLOv8LRP(model, power=2, eps=1, device='gpu')
explanation_lrp = lrp.explain(image, cls='toothbrush', contrastive=False).cpu()

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=True, cmap='seismic', title='Explanation for Class "traffic light"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='seismic', title='Explanation for Class "traffic light"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='Reds', title='Explanation for Class "traffic light"')

