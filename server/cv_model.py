from roboflow import Roboflow
import os

from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("fashion-assistant-segmentation")
model = project.version(5).model

# infer on a local image
print(model.predict("your_image.jpg").json())

# infer on an image hosted elsewhere
print(model.predict("URL_OF_YOUR_IMAGE").json())

# save an image annotated with your predictions
model.predict("your_image.jpg").save("prediction.jpg")
