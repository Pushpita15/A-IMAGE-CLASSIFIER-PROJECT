# Imports here


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import json

# Load the JSON file
with open("cat_to_name.json", 'r') as f:
    cat_to_name = json.load(f)

model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(1024,512)),
                            ('Relu',nn.ReLU()),
                            ('dropout',nn.Dropout(0.2)),
                            ('fc2',nn.Linear(512,256)),
                            ('Relu',nn.ReLU()),
                            ('dropout',nn.Dropout(0.2)),
                            ('fc3',nn.Linear(256,102)),
                            ('output',nn.LogSoftmax(dim=1))]))
model.classifier = classifier
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)

# TODO: Write a function that loads a checkpoint and rebuilds the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=device)
    input_size = checkpoint['input_size']
    output_size=checkpoint['output_size']
    epoch = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['learning_rate']
    model.load_state_dict(checkpoint['state_dict'])

    return model,class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    MAX_SIZE=(256,256)
    image.thumbnail(MAX_SIZE)

    #center crop to 224X224
    left_margin = (image.width - 224) / 2
    top_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    bottom_margin = top_margin + 224

    image = image.crop((left_margin, top_margin, right_margin, bottom_margin))

    #converts 0-255 to 0-1
    np_image = np.array(image)

    np_image = np_image/255.0

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    np_image = (np_image-means)/std

    formatted_img = np_image.transpose((2,0,1))
    tensor_image  = torch.from_numpy(formatted_img)
    return tensor_image





from flask import Flask, jsonify, request, render_template
import io

app = Flask(__name__)

model,class_to_idx= load_checkpoint(r"D:\My Projects\A-IMAGE-CLASSIFIER-PROJECT\checkpoint2.pth")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route('/', methods=['GET'])
def homepage():
    return render_template("frontend.html")



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get file from POST request and convert to image
        topk=1
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        input = process_image(image)
        input = input.unsqueeze(0)
        input = input.float()




        # Get prediction
        with torch.no_grad():
            input = input.to(device)
            logps = model(input)
        ps = torch.exp(logps)

        top_p,top_indices = ps.topk(topk,dim=1)
    # Convert indices to actual classes
        idx_to_class = {value: key for key, value in class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices[0].tolist()]
    # Convert class integers to actual flower names
        flowers = [cat_to_name[str(index)] for index in top_classes]
    # Return result as JSON
        print(flowers)
        return jsonify({'flower_name': flowers})
        #return jsonify({'class_id': top_classes, 'flower_name': flowers})

if __name__ == '__main__':
    app.run(debug=True)
