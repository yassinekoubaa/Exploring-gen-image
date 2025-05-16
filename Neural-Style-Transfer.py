# Common Setup - Install All Necessary Libraries
!pip install torch torchvision torchaudio matplotlib pillow numpy
!pip install diffusers transformers accelerate
# Building Neural Style Transfer (NST) Code

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256 if torch.cuda.is_available() else 128  # Use smaller size if no GPU

# Paths to your images (Upload these to Colab or change paths)
content_image_path = "input.jpg" # Make sure this file exists
style_image_path = "target.jpg"   # Make sure this file exists

# --- Helper Functions ---
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB') # Ensure image is RGB
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # Resize to a square image
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # We clone the tensor to not do changes on it
    image = image.squeeze(0)      # Remove the fake batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # Pause a bit so that plots are updated

# --- Loss Definitions ---
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach() # Detach the target from graph

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d) # normalize

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# --- Model Loading and Modification ---
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# VGG19 normalization (from PyTorch docs)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Define content and style layers
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0  # Increment every time we see a conv layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False) # Non-inplace ReLU for VGG
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d): # Not in VGG19 features, but good to have
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

# --- Style Transfer Execution ---
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=5000000, content_weight=4):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input image, not model parameters
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0] # Mutable to be updated in closure
    while run[0] <= num_steps:
        def closure():
            # Correct the values of updated input image
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score

        optimizer.step(closure)

    # A final correction...
    input_img.data.clamp_(0, 1)
    return input_img

# --- Load images and Run ---
try:
    content_img = image_loader(content_image_path)
    style_img = image_loader(style_image_path)

    # Ensure images loaded successfully
    assert content_img is not None and style_img is not None, \
        "Content or style image not found. Please check paths."

    input_img = content_img.clone() # Initialize with content image
    # Or, for white noise initialization:
    # input_img = torch.randn(content_img.data.size(), device=device)

    plt.figure(figsize=(12, 4))
    imshow(style_img, title='Style Image')
    plt.figure(figsize=(12, 4))
    imshow(content_img, title='Content Image')

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=1000)

    plt.figure(figsize=(12, 4))
    imshow(output, title='Output Image')
    plt.ioff()
    plt.show()

except FileNotFoundError:
    print(f"Error: Make sure '{content_image_path}' and '{style_image_path}' exist in your Colab environment.")
except Exception as e:
    print(f"An error occurred: {e}")

# This NST example loads a content and a style image, then iteratively optimizes
# a target image to match the content of one and the style of the other.
# Make sure to upload 'content.jpg' and 'style.jpg' or update the paths.
