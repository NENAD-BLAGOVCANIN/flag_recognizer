import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from model import ImageClassifier

# Load the model and its state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)

# Load the saved model state
model.load_state_dict(torch.load('model_state.pt', map_location=device))
model.eval()  # Set the model to evaluation mode

# Prepare the image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((20, 30)),   # Resize image to the same size as during training
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

# Load and preprocess the image
image_path = 'testing/test.jpg'
image = Image.open(image_path).convert('RGB')  # Open the image and ensure it's in RGB mode
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Make a prediction
with torch.no_grad():  # No need to track gradients for inference
    output = model(image)  # Get model output
    _, predicted_class = torch.max(output, 1)  # Get the index of the class with highest probability

# Print the predicted class
print(f"Predicted class: {predicted_class.item()}")
