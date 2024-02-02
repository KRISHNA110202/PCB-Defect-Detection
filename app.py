import os
import cv2
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from flask import Flask, render_template, request
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

defect_names = {
    1: "Open Circuit",
    2: "Short Circuit",
    3: "Mouse Bite",
    4: "Spur",
    5: "Copper Trace Cut",
    6: "Missing Hole"
}

colors = ['yellow', 'red', 'blue', 'cyan', 'magenta', 'purple']

model = None


def load_model():
    global model
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
    model.load_state_dict(torch.load('92pcbdetection.pt', map_location=device))
    model.eval()
    model.to(device)


def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    image = transform(image).to(device)
    image = image.float()  # Convert the tensor to Float type
    image = image.unsqueeze(0)
    return image


def predict_defect(image):
    outputs = model(image)
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()

    # Count the occurrences of each defect label
    defect_counts = {}
    for label in labels:
        if label in defect_counts:
            defect_counts[label] += 1
        else:
            defect_counts[label] = 1

    return boxes, labels, defect_counts


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = preprocess_image(image)

            # Call the load_model() function here
            load_model()

            # Call the predict_defect() function after the model is loaded
            boxes, labels, defect_counts = predict_defect(image)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=colors[label - 1], facecolor='none')
                ax.add_patch(rect)

            plt.axis('off')
            result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
            plt.savefig(result_path, bbox_inches='tight')

            unique_labels = set(labels)
            unique_defects = [{'name': defect_names[label], 'color': colors[label - 1], 'count': defect_counts[label]}
                              for label in unique_labels]

            # Generate a unique filename for the uploaded image
            uploaded_image_filename = f"uploaded_{time.time()}.png"
            uploaded_image_path = os.path.join(app.config['RESULT_FOLDER'], uploaded_image_filename)
            cv2.imwrite(uploaded_image_path, cv2.cvtColor(image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0, cv2.COLOR_RGB2BGR))

            # Check if labels array is empty or not
            result = 'Defect found' if len(labels) > 0 else 'No defects'

            return render_template('result.html',
                                   uploaded_image=os.path.join(app.config['RESULT_FOLDER'], uploaded_image_filename),
                                   result_image='result.png', unique_defects=unique_defects)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
