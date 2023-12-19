# app.py
import base64
import io
import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 

num_diseases = 38

# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# base class for the model
class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# resnet architecture
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim: 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim: 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim: 512 x 4 x 4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        # Corrected: Use self.classifier here
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)  # Use self.classifier

        return out

# Load the ResNet9 model
model = ResNet9(in_channels=3, num_diseases=38)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the state dictionary into the model
# Load the state dictionary into the model
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device
model = model.to(device)
model.eval()

print(model.state_dict().keys())
print(model)


# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_image(image):
    # Preprocess the image
    img = transform(image).unsqueeze(0)

    # Perform image classification
    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output[0], dim=0)

    # Get the top 6 predictions
    top6_probabilities, top6_classes = torch.topk(probabilities, 6)

    # Convert class indices to class names (adjust this based on your model)
    class_names = [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_apple_rust",
    "Apple_healthy",
    "Background_without_leaves",
    "Blueberry_healthy",
    "Cherry_powdery_mildew",
    "Cherry_healthy",
    "Corn_gray_leaf_spot",
    "Corn_common_rust",
    "Corn_northern_leaf_blight",
    "Corn_healthy",
    "Grape_black_rot",
    "Grape_black_measles",
    "Grape_leaf_blight",
    "Grape_healthy",
    "Orange_haunglongbing",
    "Peach_bacterial_spot",
    "Peach_healthy",
    "Pepper_bacterial_spot",
    "Pepper_healthy",
    "Potato_early_blight",
    "Potato_healthy",
    "Potato_late_blight",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_powdery_mildew",
    "Strawberry_healthy",
    "Strawberry_leaf_scorch",
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_healthy",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot",
    "Tomato_mosaic_virus",
    "Tomato_yellow_leaf_curl_virus"
]

    top6_classes = [class_names[idx] for idx in top6_classes]

    return top6_probabilities, top6_classes

# Function to rotate an image by a specified angle
def rotate_image(image, angle):
    return image.rotate(angle)

# Function to convert an image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to create a rotating image in Streamlit
def rotating_image(image, rotation_angle, rotation_speed):
    # Convert the image to base64
    image_base64 = image_to_base64(image)

    # Display the rotating image using HTML and JavaScript
    st.markdown(
        f"""
        <div style="display: inline-block; overflow: hidden;">
            <img id="rotating-image" src="data:image/png;base64,{image_base64}" style="transform-origin: center center; transition: transform {rotation_speed}s linear;">
            <script>
                function rotateImage() {{
                    var image = document.getElementById('rotating-image');
                    image.style.transform = 'rotate(' + {rotation_angle} + 'deg)';
                    setTimeout(rotateImage, {rotation_speed * 1000});
                }}
                rotateImage();
            </script>
        </div>
        """,
        unsafe_allow_html=True
    )


# Streamlit app layout
def main():\

 # Load your logo image from the root directory
    logo_image = Image.open("logo.png")  # Replace "logo.png" with the actual filename of your logo
    
    # Convert the logo image to base64
    buffered = io.BytesIO()
    logo_image.save(buffered, format="PNG")
    logo_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Display the title and rotating logo image in the same row
    st.markdown(
        '<div style="display: flex; align-items: center;">'
        f'<img id="rotating-image" src="data:image/png;base64,{logo_base64}" style="width: 100px; height: 100px;">'
        '<h1 style="margin-left: 20px;">Plant Disease Classification Web App</h1>'
        '</div>'
        '<script>'
        'function rotateImage() {'
        'const rotatingImage = document.getElementById("rotating-image");'
        'rotatingImage.style.transform = "rotate(" + (new Date().getTime() / 1000) + "deg)";'
        'requestAnimationFrame(rotateImage);'
        '}'
        'rotateImage();'
        '</script>',
        unsafe_allow_html=True
    )

    # Introductory message
    st.markdown(
        """
        Welcome to our Plant Disease Detection website powered by artificial intelligence.
        Upload an image of a plant leaf, and our advanced AI model will analyze it to identify 
        any signs of common diseases. This technology enables early detection, prevention, and 
        effective treatment of plant diseases, contributing to healthier crops and increased 
        agricultural productivity.
        """
    )

    # Thin green line separator
    st.markdown('<hr style="border-top: 2px solid #13bf4a;">', unsafe_allow_html=True)

    # Image upload section
    st.title("Upload Image")
    st.markdown(
        """
        Please upload an image of the affected plant leaf here.

        Our AI model will analyze the image and accurately identify any signs of disease.
        """
    )
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")

        # Predict button
        if st.button("Predict"):
            # Perform image classification
            top6_probabilities, top6_classes = classify_image(image)

            # Display top prediction
            top_prediction = top6_classes[0]
            st.subheader("Top Prediction:")
            st.write(f"The top prediction is: {top_prediction}")

            # Display top 6 predictions in a DataFrame
            df = pd.DataFrame({"Class": top6_classes, "Probability": top6_probabilities.numpy()})
            
            st.subheader("Top 6 Predictions:")
            st.dataframe(df)

            # Plot the predictions on a bar chart
            st.subheader("Top 6 Predictions (Bar Chart):")
            st.bar_chart(df.set_index("Class"))

if __name__ == "__main__":
    main()