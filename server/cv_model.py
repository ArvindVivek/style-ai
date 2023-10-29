from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image, ImageDraw
import colorgram
from io import BytesIO
import os
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import supervision as sv
from typing import Tuple
from lavis.models import load_model_and_preprocess
import torch

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Initialize Roboflow
# Model Credits: https://universe.roboflow.com/roboflow-jvuqo/fashion-assistant-segmentation
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("fashion-assistant-segmentation")
model = project.version(5).model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT = "What is the style of this clothing?"

model2, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_vqa", model_type="vqav2"
)

prompt_processed = txt_processors["eval"](PROMPT)

images_numpy = []
predictions = []


def load_image(image_path: str) -> Tuple[PIL.Image.Image, np.ndarray]:
    image_pil = Image.open(image_path).convert("RGB")
    image_numpy = np.asarray(image_pil)
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    return image_pil, image_numpy


# Function to extract ROI
def extract_roi(data, img_path):
    # Get the points for the ROI
    points = data["predictions"][0]["points"]

    # Create an empty mask
    mask = Image.new(
        "L",
        (int(data["predictions"][0]["width"]), int(data["predictions"][0]["height"])),
        0,
    )
    draw = ImageDraw.Draw(mask)

    # Convert points to tuples
    points = [(int(point["x"]), int(point["y"])) for point in points]

    # Draw polygon on the mask
    draw.polygon(points, fill=255)

    # Get the bounding box of the mask
    bbox = mask.getbbox()

    # Crop the original image using the bounding box
    roi = Image.open(img_path)  # Replace with the path to your original image
    roi = roi.crop(bbox)

    return roi


# Function to get dominant color
def get_dominant_color(image):
    colors = colorgram.extract(image, 1)
    dominant_color = colors[0].rgb
    return dominant_color


# Infer on local images and iterate through data folder
for image in os.listdir("data/subset"):
    image_path = f"data/subset/{image}"
    prediction = model.predict(f"data/subset/{image}").json()
    class_value = prediction["predictions"][0]["class"]

    roi = extract_roi(prediction, f"data/subset/{image}")

    # Get the dominant color
    dominant_color = get_dominant_color(roi)

    # Combine class and RGB color
    result = f"Class: {class_value}, Color: RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"

    image_pil, image_numpy = load_image(image_path=str(image_path))
    image_processed = vis_processors["eval"](image_pil).unsqueeze(0).to(DEVICE)
    prediction2 = model2.predict_answers(
        samples={"image": image_processed, "text_input": prompt_processed},
        inference_method="generate",
    )[0]
    images_numpy.append(image_numpy)
    predictions.append(prediction2)

    print(result, prediction2)
