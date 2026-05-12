
import torch
import cv2
import numpy as np
import gradio as gr
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = smp.Unet(
    encoder_name    = 'resnet34',
    encoder_weights = None,
    in_channels     = 3,
    classes         = 1,
    activation      = None,
)
model.load_state_dict(torch.load('best_salt_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

def predict_salt(input_image):
    # Preprocess
    img_gray = np.array(input_image.convert("L"))
    img3ch   = np.stack([img_gray, img_gray, img_gray], axis=-1)
    aug      = transform(image=img3ch)
    tensor   = aug["image"].unsqueeze(0).float().to(DEVICE)

    # Predict
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

    # Resize back
    h, w   = img_gray.shape
    prob_r = cv2.resize(prob, (w, h))
    mask   = (prob_r > 0.5).astype(np.uint8)

    # Overlay blue = salt
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[mask == 1] = [0, 100, 255]
    blended  = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    coverage = mask.mean() * 100

    return (
        Image.fromarray(blended),
        Image.fromarray(mask * 255),
        f"🧂 Salt Coverage: {coverage:.1f}%"
    )

# Gradio UI
with gr.Blocks(title="Salt Deposit Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧂 Salt Deposit Identification
    **Upload a seismic image to detect salt deposits using Deep Learning**
    > Model: U-Net + ResNet34  |  Best IoU: 0.7852  |  By: Kumpatla Gayathri
    """)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="📤 Upload Seismic Image")
            btn = gr.Button("🔍 Detect Salt", variant="primary")
        with gr.Column():
            out_overlay = gr.Image(label="🔵 Prediction Overlay (Blue = Salt)")
            out_mask    = gr.Image(label="⬜ Binary Mask (White = Salt)")
            out_label   = gr.Textbox(label="📊 Result")

    btn.click(
        fn      = predict_salt,
        inputs  = inp,
        outputs = [out_overlay, out_mask, out_label]
    )

    gr.Markdown("""
    ### 📌 How to use:
    1. Upload any seismic image (PNG/JPG)
    2. Click **Detect Salt**
    3. View overlay, binary mask and salt coverage %
    """)

demo.launch()
