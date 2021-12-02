from torchvision import models, transforms
import torch
from PIL import Image
import torch.nn as nn

device = torch.device('cpu')
pretrained=True
num_classes = 5
num_epochs = 60

def predict(image_path):
    model = models.resnet34(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    model.load_state_dict(torch.load('./output/resnet18-model.pt', map_location=device))

    model.fc = nn.Sequential(nn.Linear(512,256),
                         nn.ReLU(inplace=True),
                         nn.Linear(256,128),
                         nn.ReLU(inplace=True),
                         nn.Linear(128,64),
                         nn.ReLU(inplace=True),
                         nn.Linear(64,5),    
                    )

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(model.fc(img), 0)

    model.eval()
    out = model(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
