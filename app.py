import torch
import streamlit as st
from PIL import Image
import numpy as np
from utils.transformations import get_transforms
from models.resnet34 import get_resnet34

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet34(3)
model.load_state_dict(torch.load("weights/resnet34.pth", map_location=device))
model.to(device)
model.eval()

# Get custom transformations for input images
transform = get_transforms()


# Function to make predictions
def predict(image):
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu()


# Create the Streamlit interface
st.title("Lung Disease Classification")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout with columns: image on the left, chart on the right
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        probabilities = predict(image)
        probabilities = probabilities.numpy()[0]

        # Create a dictionary for labels and probabilities
        labels = ["Normal", "Lung Opacity", "Viral Pneumonia"]
        probability_dict = dict(zip(labels, probabilities))

        # Determine the most probable class
        predicted_class = labels[np.argmax(probabilities)]

        with col2:
            # Displaying the probabilities with a bar chart
            st.bar_chart(probability_dict)
            st.write(f"Predicted Condition: {predicted_class}")
