import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import get_model

# 1. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_trained_model():
    model = get_model("resnet50")
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_trained_model()

# 2. Pre-processing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])


# 3. Predict
def predict(image):
    if image is None: return None
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs[0], dim=0)

    return {classes[i]: float(probabilities[i]) for i in range(10)}


# 4. Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload CIFAR-10 Image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Classes"),
    title="🚢 CIFAR-10 Image Classifier (ResNet50)",
    description="Tải lên một tấm ảnh (máy bay, ô tô, mèo, chó...) để model ResNet50 dự đoán",
)

if __name__ == "__main__":
    demo.launch(share=True)
