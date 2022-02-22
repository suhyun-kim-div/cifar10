# app.py
from flask import Flask, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import os 

import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, datasets
from cnn import CNN


app = Flask(__name__)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

new_model_path = "/home/jovyan/cifar10-practice/models/cifar10-65.20.pt"
idx_to_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

model_reloaded = torch.load(new_model_path)
loaded_model = CNN().to(DEVICE)
loaded_model.load_state_dict(model_reloaded)

# curl -X POST "127.0.0.1:5000/api/v1/predict" -H "Content-Type: image/*" --data-binary "@00003.png"

@app.route('/api/v1/info', methods=['GET'])
def get_model_info():
    info = "CIFAR10 65.20 MODEL"
    return info


def predict(file_streams):
    img_to_tensor = transforms.ToTensor()
    test_img = Image.open(io.BytesIO(file_streams))
    # test_img = Image.open(test_img_path)
    test_img_tensor = img_to_tensor(test_img)
    test_img_stack = torch.stack((test_img_tensor,)).to(DEVICE)
    
    proba_logsoftmax = loaded_model(test_img_stack).tolist()[0]
    proba = [np.exp(val) for val in proba_logsoftmax]
    proba_max = max(proba)
    proba_max_idx = np.argmax(proba)
    pred_class = idx_to_class[proba_max_idx]
    print(pred_class, proba_max)
    return pred_class

@app.route("/api/v1/predict",methods=["POST"])
def predict_input_data():
    file_streams = request.get_data()
    result = predict(file_streams)
    print(f"predict result: (result)")
    return result


if __name__ == '__main__':
    app.run(debug=True)