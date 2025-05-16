# Common Setup - Install All Necessary Libraries
!pip install torch torchvision torchaudio matplotlib pillow numpy
!pip install diffusers transformers accelerate
## Part A (Smileys): Generating the Synthetic Smiley Face Dataset
# At the TOP of your VAE Smiley Data Generation cell:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw # ImageFont is not used in this basic version

# --- Configuration for Smiley Data Generation ---
IMAGE_SIZE_SMILEYS = 64 # As per the example code [2]
EXPRESSIONS_LIST = ['happy', 'sad', 'surprised', 'wink', 'neutral'] # Added neutral
NUM_SAMPLES_PER_EXPRESSION_TRAIN = 400 # Number of training samples per expression
NUM_SAMPLES_PER_EXPRESSION_TEST = 80   # Number of testing samples per expression
TOTAL_SMILEY_TRAIN_SAMPLES = len(EXPRESSIONS_LIST) * NUM_SAMPLES_PER_EXPRESSION_TRAIN
TOTAL_SMILEY_TEST_SAMPLES = len(EXPRESSIONS_LIST) * NUM_SAMPLES_PER_EXPRESSION_TEST

# --- Function to Generate a Smiley Face (from Search Result [2], slightly adapted) ---
def generate_smiley_image(expression='happy', image_size=IMAGE_SIZE_SMILEYS):
    img = Image.new('L', (image_size, image_size), color=0)  # Grayscale image
    draw = ImageDraw.Draw(img)

    # Face circle (white face on black background)
    face_radius = image_size // 2 - 5
    center_x = image_size // 2
    center_y = image_size // 2
    # To draw a circle with a center and radius, you define a bounding box
    # Top-left: (center_x - radius, center_y - radius)
    # Bottom-right: (center_x + radius, center_y + radius)
    draw.ellipse([center_x - face_radius, center_y - face_radius,
                  center_x + face_radius, center_y + face_radius], fill=255) # White face

    # Eyes (black eyes)
    eye_radius = image_size // 10 # Adjusted eye radius slightly
    eye_offset_y = image_size // 6
    eye_offset_x = image_size // 4

    eye_y_pos = center_y - eye_offset_y
    left_eye_x_pos = center_x - eye_offset_x
    right_eye_x_pos = center_x + eye_offset_x

    if expression == 'wink':
        # Left eye closed (line)
        draw.line([left_eye_x_pos - eye_radius, eye_y_pos,
                   left_eye_x_pos + eye_radius, eye_y_pos], fill=0, width=image_size//20)
        # Right eye open (circle)
        draw.ellipse([right_eye_x_pos - eye_radius, eye_y_pos - eye_radius,
                      right_eye_x_pos + eye_radius, eye_y_pos + eye_radius], fill=0)
    else:
        # Both eyes open (circle)
        draw.ellipse([left_eye_x_pos - eye_radius, eye_y_pos - eye_radius,
                      left_eye_x_pos + eye_radius, eye_y_pos + eye_radius], fill=0)
        draw.ellipse([right_eye_x_pos - eye_radius, eye_y_pos - eye_radius,
                      right_eye_x_pos + eye_radius, eye_y_pos + eye_radius], fill=0)

    # Mouth (black mouth)
    mouth_width_factor = 0.4 # Proportion of face diameter
    mouth_height_factor = 0.25
    mouth_center_y_offset = image_size // 5

    mouth_width = int(2 * face_radius * mouth_width_factor)
    mouth_height = int(2 * face_radius * mouth_height_factor)
    mouth_center_y = center_y + mouth_center_y_offset

    if expression == 'happy' or expression == 'wink':
        # Smile (arc)
        draw.arc([center_x - mouth_width // 2, mouth_center_y - mouth_height // 2,
                  center_x + mouth_width // 2, mouth_center_y + mouth_height // 2],
                 start=0, end=180, fill=0, width=image_size//20) # Start at 0 for upward curve
    elif expression == 'sad':
        # Sad mouth (arc, inverted)
        draw.arc([center_x - mouth_width // 2, mouth_center_y, # Adjusted y for downward curve
                  center_x + mouth_width // 2, mouth_center_y + mouth_height],
                 start=180, end=360, fill=0, width=image_size//20)
    elif expression == 'surprised':
        # Surprised mouth (ellipse/circle)
        surprised_mouth_radius_w = mouth_width // 3
        surprised_mouth_radius_h = mouth_height // 1.5
        draw.ellipse([center_x - surprised_mouth_radius_w, mouth_center_y - surprised_mouth_radius_h // 2,
                      center_x + surprised_mouth_radius_w, mouth_center_y + surprised_mouth_radius_h // 2 ], fill=0)
    elif expression == 'neutral':
        # Neutral mouth (line)
        draw.line([center_x - mouth_width // 2, mouth_center_y,
                   center_x + mouth_width // 2, mouth_center_y], fill=0, width=image_size//20)
    # Add more expressions if desired

    return np.array(img)

# --- Generate Training and Testing Datasets for Smileys ---
def create_smiley_dataset(num_samples_per_expr, expressions_list):
    images_data = []
    labels_data = [] # Integer labels for expressions
    for label_idx, expr_name in enumerate(expressions_list):
        for _ in range(num_samples_per_expr):
            images_data.append(generate_smiley_image(expr_name))
            labels_data.append(label_idx)
    return np.array(images_data), np.array(labels_data)

print(f"Generating {TOTAL_SMILEY_TRAIN_SAMPLES} training smiley images...")
train_smileys_np, train_smiley_labels_np = create_smiley_dataset(NUM_SAMPLES_PER_EXPRESSION_TRAIN, EXPRESSIONS_LIST)
print(f"Generating {TOTAL_SMILEY_TEST_SAMPLES} testing smiley images...")
test_smileys_np, test_smiley_labels_np = create_smiley_dataset(NUM_SAMPLES_PER_EXPRESSION_TEST, EXPRESSIONS_LIST)

print(f"Training smileys shape: {train_smileys_np.shape}")
print(f"Training smiley labels shape: {train_smiley_labels_np.shape}")
print(f"Test smileys shape: {test_smileys_np.shape}")
print(f"Test smiley labels shape: {test_smiley_labels_np.shape}")

# --- Display some sample generated smiley images ---
print("\nSample Training Smiley Images:")
fig, axs = plt.subplots(len(EXPRESSIONS_LIST), 5, figsize=(10, len(EXPRESSIONS_LIST) * 2))
for i, expr_name in enumerate(EXPRESSIONS_LIST):
    # Find indices for the current expression
    expr_indices = np.where(train_smiley_labels_np == i)[0]
    for j in range(5): # Show 5 samples per expression
        if j < len(expr_indices):
            ax = axs[i, j]
            idx_to_show = expr_indices[j]
            ax.imshow(train_smileys_np[idx_to_show], cmap='gray')
            if j == 0: # Add title only to the first image of each row
                 ax.set_title(expr_name.capitalize(), loc='left', fontsize=10, y=0.9) # Position title to the left
            ax.axis('off')
plt.tight_layout()
plt.show()

# The variables train_smileys_np, train_smiley_labels_np, etc. will be used in the next cell.


## Part B (Smileys): VAE Model, Training, and Visualization for Smileys


# At the TOP of your VAE Model & Training cell:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np # Should be available if previous cell ran
import matplotlib.pyplot as plt

# --- Custom PyTorch Dataset for our Smileys ---
class SmileysDataset(Dataset): # Renamed for clarity
    def __init__(self, images_np_array, labels_np_array, transform=None):
        self.images = images_np_array
        self.labels = labels_np_array
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, torch.tensor(label, dtype=torch.long)


# --- VAE Model Definition (Convolutional VAE - Can reuse from shapes example) ---
# The ConvVAE architecture used for geometric shapes should work well here too,
# as it's designed to be somewhat flexible with image_size.
# We need to ensure IMAGE_SIZE_SMILEYS is used.
class ConvVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=20, image_size=IMAGE_SIZE_SMILEYS): # Using smiley image size
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder
        # Input: (batch_size, 1, image_size, image_size) e.g. (1, 64, 64)
        self.enc_conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2) # 64->32
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)          # 32->16
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)          # 16->8
        self.enc_bn3 = nn.BatchNorm2d(64)
        
        # Calculate the flattened size dynamically
        self.final_conv_size = self.image_size // 8 # After 3 stride-2 convs
        self.flattened_size = 64 * self.final_conv_size * self.final_conv_size

        self.enc_fc1 = nn.Linear(self.flattened_size, 256) # Increased FC layer size
        self.enc_mu = nn.Linear(256, latent_dim)
        self.enc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, self.flattened_size)
        
        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1) # 8->16
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1) # 16->32
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_deconv3 = nn.ConvTranspose2d(16, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1) # 32->64


    def encode(self, x):
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc1(z))
        x = F.relu(self.dec_fc2(x))
        x = x.view(-1, 64, self.final_conv_size, self.final_conv_size) # Reshape to match conv input
        x = F.relu(self.dec_bn1(self.dec_deconv1(x)))
        x = F.relu(self.dec_bn2(self.dec_deconv2(x)))
        x_reconstructed = torch.sigmoid(self.dec_deconv3(x))
        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- VAE Loss Function (Same as before) ---
def vae_loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Configuration for Smiley VAE Training ---
device_smiley_vae = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smiley_vae_learning_rate = 1e-3
smiley_vae_batch_size = 64
smiley_vae_num_epochs = 50 # Increased epochs for potentially better results on smileys
smiley_vae_latent_dim = 16 # Can experiment

# --- Prepare DataLoaders for Smileys ---
# Ensure train_smileys_np and test_smileys_np are available from the previous cell
train_smileys_dataset_obj = SmileysDataset(train_smileys_np, train_smiley_labels_np)
test_smileys_dataset_obj = SmileysDataset(test_smileys_np, test_smiley_labels_np)

train_smileys_loader = DataLoader(train_smileys_dataset_obj, batch_size=smiley_vae_batch_size, shuffle=True)
test_smileys_loader = DataLoader(test_smileys_dataset_obj, batch_size=smiley_vae_batch_size, shuffle=False)

# --- Initialize Model and Optimizer ---
model_smiley_conv_vae = ConvVAE(latent_dim=smiley_vae_latent_dim, image_size=IMAGE_SIZE_SMILEYS).to(device_smiley_vae)
optimizer_smiley_conv_vae = optim.Adam(model_smiley_conv_vae.parameters(), lr=smiley_vae_learning_rate)

print(f"Training Smiley VAE on {device_smiley_vae} with {len(train_smileys_dataset_obj)} training samples...")

# --- Smiley VAE Training Loop ---
for epoch in range(smiley_vae_num_epochs):
    model_smiley_conv_vae.train()
    epoch_train_loss = 0
    for batch_idx, (data_batch, _) in enumerate(train_smileys_loader): # Labels are ignored
        data_batch = data_batch.to(device_smiley_vae)
        optimizer_smiley_conv_vae.zero_grad()
        recon_batch, mu, logvar = model_smiley_conv_vae(data_batch)
        loss = vae_loss_fn(recon_batch, data_batch, mu, logvar)
        loss.backward()
        epoch_train_loss += loss.item()
        optimizer_smiley_conv_vae.step()

    avg_epoch_loss = epoch_train_loss / len(train_smileys_loader.dataset)
    print(f'Epoch [{epoch+1}/{smiley_vae_num_epochs}], Average Loss: {avg_epoch_loss:.4f}')

print("Smiley VAE Training Finished.")

# --- Smiley VAE Results Visualization ---
model_smiley_conv_vae.eval()
with torch.no_grad():
    # 1. Show Reconstructed Smiley Images
    sample_data_batch, _ = next(iter(test_smileys_loader))
    sample_data_batch = sample_data_batch.to(device_smiley_vae)
    reconstructed_batch, _, _ = model_smiley_conv_vae(sample_data_batch)

    num_images_to_show = min(8, smiley_vae_batch_size)
    plt.figure(figsize=(16, 4))
    plt.suptitle("Original vs. Reconstructed Smileys", fontsize=16)
    for i in range(num_images_to_show):
        ax = plt.subplot(2, num_images_to_show, i + 1)
        plt.imshow(sample_data_batch[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Orig {i+1}")
        ax.axis('off')
        ax = plt.subplot(2, num_images_to_show, i + 1 + num_images_to_show)
        plt.imshow(reconstructed_batch[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Recon {i+1}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 2. Show Generated Smiley Images from Random Latent Vectors
    num_generated_samples = 10
    z_samples = torch.randn(num_generated_samples, smiley_vae_latent_dim).to(device_smiley_vae)
    generated_smileys = model_smiley_conv_vae.decode(z_samples).cpu()

    plt.figure(figsize=(num_generated_samples * 1.5, 3))
    plt.suptitle("Generated Smileys from Latent Space", fontsize=16)
    for i in range(num_generated_samples):
        ax = plt.subplot(1, num_generated_samples, i + 1)
        plt.imshow(generated_smileys[i].squeeze(), cmap='gray')
        ax.set_title(f"Gen {i+1}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

