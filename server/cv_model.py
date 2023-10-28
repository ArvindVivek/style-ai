from roboflow import Roboflow
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


def find_dominant_color(image, num_clusters=3):
    # Flatten the image to a list of pixels
    pixels = image.reshape(-1, 3)

    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(pixels)
    cluster_centers = kmeans.cluster_centers_

    # Find the dominant color (cluster center) in RGB format
    dominant_color_rgb = cluster_centers[
        np.argmax(kmeans.labels_ == kmeans.predict([cluster_centers.mean(axis=0)]))
    ]

    # Convert dominant color to BGR format (OpenCV convention)
    dominant_color_bgr = dominant_color_rgb[::-1]

    return dominant_color_bgr.astype(int)


# Initialize Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("fashion-assistant-segmentation")
model = project.version(5).model

# Infer on local images and iterate through data folder
for image in os.listdir("data/subset"):
    prediction = model.predict(f"data/subset/{image}").json()
    class_value = prediction["predictions"][0]["class"]

    # Load the image for dominant color detection
    image_path = f"data/subset/{image}"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find the dominant color for the entire image
    try:
        dominant_color = find_dominant_color(image)

        print(f"Class Value: {class_value}")
        print(f"Dominant Color: {dominant_color}")
    except Exception as e:
        print(f"Error processing image {image}: {e}")
