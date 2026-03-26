import torch
import timm

def get_model(model_name, num_classes=10, device="cuda"):
    # Load pretrained từ ImageNet-1K
    # ========================
    # MODELS
    # ========================
    if model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
    elif model_name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Model not supported")

    return model.to(device)