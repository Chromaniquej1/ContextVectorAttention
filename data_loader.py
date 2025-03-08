import numpy as np
import scipy.io
from sklearn.preprocessing import LabelEncoder

def load_pavia_university(image_file, gt_file):
    """Load the Pavia University dataset."""
    print("Loading Pavia University dataset...")
    image_data = scipy.io.loadmat(image_file)['paviaU']  # Adjust key if necessary
    ground_truth = scipy.io.loadmat(gt_file)['paviaU_gt']  # Adjust key if necessary
    print(f"Image data shape: {image_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    return image_data, ground_truth

def preprocess_data(image_data, ground_truth, window_size=5):
    """Preprocess the data by normalizing and adding spatial context."""
    # Normalize the image data
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Add spatial information via padding
    padded_image = np.pad(image_data, ((window_size//2, window_size//2),
                                       (window_size//2, window_size//2),
                                       (0, 0)), mode='reflect')
    spatial_spectral_data = np.zeros((image_data.shape[0], image_data.shape[1],
                                      window_size, window_size, image_data.shape[2]))
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            spatial_spectral_data[i, j] = padded_image[i:i+window_size, j:j+window_size, :]

    spatial_spectral_data = spatial_spectral_data.reshape(-1, window_size, window_size, image_data.shape[2])
    y = ground_truth.flatten()
    mask = y != 0
    spatial_spectral_data = spatial_spectral_data[mask]
    y = y[mask]

    # Encode labels from 0 to num_classes-1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return spatial_spectral_data, y, label_encoder