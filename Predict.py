import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# =========================
# ⚙️ Configuration
# =========================
CLASS_NAMES = ["Normal", "Bleeding", "Ischemia"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./ct_memmap_cache/best_resnet18.pt"


def load_model(weights_path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def run_prediction(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    return CLASS_NAMES[predicted_idx], confidence.item(), image


# =========================
# 🚀 Interactive Terminal Loop
# =========================
if __name__ == "__main__":
    print("--- Brain Stroke CT Classifier: Inference Mode ---")

    # Load model once at the start
    try:
        net = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

    # Loop allows you to check multiple images without restarting the script
    while True:
        print("Enter the path to a CT scan image (or type 'exit' to quit):")
        user_input = input("👉 Path: ").strip()

        # Remove quotes if the user dragged and dropped the file into the terminal
        user_input = user_input.replace('"', '').replace("'", "")

        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        if not os.path.exists(user_input):
            print(f"❌ File not found: {user_input}. Please try again.\n")
            continue

        try:
            label, score, img = run_prediction(user_input, net)

            print("\n" + "="*30)
            print(f"PREDICTION : {label}")
            print(f"CONFIDENCE : {score*100:.2f}%")
            print("="*30 + "\n")

            # Show the image
            plt.imshow(img)
            plt.title(f"Result: {label} ({score*100:.1f}%)")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"❌ An error occurred during processing: {e}\n")
