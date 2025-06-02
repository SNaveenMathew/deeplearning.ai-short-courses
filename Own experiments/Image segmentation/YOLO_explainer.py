# Recommendation: Create a new python/conda environment and run the following command:
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

########################################################################################################
# This section of code takes an image as input and 'predicts' different segments included in the image #
########################################################################################################

# This code was copied from Ultralytics website
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
results = model("https://static.wikia.nocookie.net/eighteleven/images/6/65/ZSprite_Vittorinoc.png")
# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    result.save_crop("try")

# There's only one prediction - scissors

#####################################################################################################
# This section of code explains the 'prediction' of the model using Layerwise Relevance Propagation #
#####################################################################################################

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
# response = requests.get("https://images.unsplash.com/photo-1584649525122-8d6895492a5d?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
response = requests.get("https://static.wikia.nocookie.net/eighteleven/images/6/65/ZSprite_Vittorinoc.png")
desired_size = (512, 640)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(desired_size),
    torchvision.transforms.ToTensor(),
])
# image = Image.open(BytesIO(response.content))
urllib.request.urlretrieve('https://static.wikia.nocookie.net/eighteleven/images/6/65/ZSprite_Vittorinoc.png', "character_sprite.png")
image = Image.open("character_sprite.png").convert('RGB')
image.show()

image = transform(image)

lrp = YOLOv8LRP(model, power=2, eps=1, device='gpu')

# Explaining the predicted 'class'

explanation_lrp = lrp.explain(image, cls='scissors', contrastive=False).cpu()

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=True, cmap='seismic', title='Explanation for Class "scissors"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='seismic', title='Explanation for Class "scissors"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='Reds', title='Explanation for Class "scissors"')

# I think it's visually closest to a toothbrush

explanation_lrp = lrp.explain(image, cls='toothbrush', contrastive=False).cpu()

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=True, cmap='seismic', title='Explanation for Class "toothbrush"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='seismic', title='Explanation for Class "toothbrush"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='Reds', title='Explanation for Class "toothbrush"')

# Here's something interesting: try changing cls from 'toothbrush' to 'traffic light' or 'person' or even 'laptop', the results don't change significantly - at least not visually

explanation_lrp = lrp.explain(image, cls='laptop', contrastive=False).cpu()

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=True, cmap='seismic', title='Explanation for Class "laptop"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='seismic', title='Explanation for Class "laptop"')

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=False, cmap='Reds', title='Explanation for Class "laptop"')

# Entering a wrong class name throws an error. But independent of the class used in the explanation the outline looks like a person