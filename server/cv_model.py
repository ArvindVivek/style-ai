from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image, ImageDraw
import colorgram
from io import BytesIO
import os
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


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


# Initialize Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("fashion-assistant-segmentation")
model = project.version(5).model

# Infer on local images and iterate through data folder
for image in os.listdir("data/subset"):
    prediction = model.predict(f"data/subset/{image}").json()
    class_value = prediction["predictions"][0]["class"]

    roi = extract_roi(prediction, f"data/subset/{image}")

    # Get the dominant color
    dominant_color = get_dominant_color(roi)

    # Combine class and RGB color
    result = f"Class: {class_value}, Dominant Color: RGB({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"

    print(result)
