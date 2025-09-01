
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
import zipfile

def read_images_from_dataframe(df, target_size=256):
    images = []
    masks = []
    image_fnames = []
    mask_fnames = []

    for idx, row in df.iterrows():
        image_path = os.path.join(row['Image_Directory'], row['Image_Filename'])
        mask_path = os.path.join(row['Mask_Directory'], row['Mask_Filename'])

        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=1, expand_animations=False)
        image = tf.image.resize(image, [target_size, target_size])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        images.append(image)
        image_fnames.append(image_path)

        # Load and decode mask
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, [target_size, target_size])
        mask = tf.cast(mask, tf.float32) / 255.0  # Normalize to [0,1]
        masks.append(mask)
        mask_fnames.append(mask_path)

    images_tensor = tf.stack(images)
    masks_tensor = tf.stack(masks)

    return images_tensor, masks_tensor, image_fnames, mask_fnames

def load_image_mask(image_path, mask_path, target_size=256):
    image = tf.io.read_file(image_path)
    # Add expand_animations=False to get a fully shaped tensor
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [target_size, target_size])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [target_size, target_size])
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask


def create_dataset(image_paths, mask_paths, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(lambda img_p, mask_p: load_image_mask(img_p, mask_p),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def plot_random_prediction(model, base_dir, category='Normal', input_size=256, save_path=None):
    """
    Plots a random image, its true mask overlay, and the predicted mask using the given model.

    Args:
        model: Trained segmentation model.
        base_dir: Base directory containing category folders with 'images' and 'masks' subfolders.
        category: Subfolder name like 'Normal', 'COVID', etc.
        input_size: Size to resize images and masks (int).
    """

    images_dir = os.path.join(base_dir, category, 'images')
    masks_dir = os.path.join(base_dir, category, 'masks')

    # Select random index
    all_images = sorted(os.listdir(images_dir))
    idx = random.randint(0, len(all_images) - 1)

    image_path = os.path.join(images_dir, all_images[idx])
    mask_path = os.path.join(masks_dir, sorted(os.listdir(masks_dir))[idx])
    print(f'Image Path: {image_path}')
    # Helper function to process an image or mask path into a normalized tensor
    def process_path(path, is_mask=False):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, [input_size, input_size])
        img = tf.cast(img, tf.float32) / 255.0
        if is_mask:
            img = tf.where(img > 0.5, 1.0, 0.0)  # Binarize mask
        return img

    image = process_path(image_path)
    mask = process_path(mask_path, is_mask=True)

    # Prepare batch dimension for prediction
    input_image = tf.expand_dims(image, 0)

    predicted_mask = model.predict(input_image)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)  # Binarize predicted mask

    # Plotting
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Original Image with True Mask")
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.imshow(tf.squeeze(mask), cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')

    plt.show()


def plot_specific_prediction(model, image_dir, mask_dir, image_filename=None, mask_filename=None, input_size=256, save_path=None):
    """
    Plots a specific image, its true mask overlay, and the predicted mask using the given model.
    Args:
        model: Trained segmentation model.
        image_dir: Directory containing images.
        mask_dir: Directory containing masks.
        image_filename: Filename of the image to load.
        mask_filename: Filename of the mask to load.
        input_size: Size to resize images and masks (int).
        save_path: Path to save the plotted figure (optional).
    """
    if image_filename is None or mask_filename is None:
        raise ValueError("Must provide both image_filename and mask_filename.")
    image_path = os.path.join(image_dir, image_filename)
    mask_path = os.path.join(mask_dir, mask_filename)
    print(f"Image Path: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
    
    def process_path(path, is_mask=False):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, [input_size, input_size])
        img = tf.cast(img, tf.float32) / 255.0
        if is_mask:
            img = tf.where(img > 0.5, 1.0, 0.0)  # Binarize mask
        return img
    
    image = process_path(image_path)
    mask = process_path(mask_path, is_mask=True)
    input_image = tf.expand_dims(image, 0)
    predicted_mask = model.predict(input_image)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)
    
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Original Image with True Mask")
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.imshow(tf.squeeze(mask), cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def retrieve_and_process_data(zip_filename='COVID-19_Radiography_Dataset.zip'):
    """
    Retrieves files from a zip file in Google Drive, and extracts all its content.
    It handles potential errors during the process and prints informative messages.

    Args:
        zip_filename (str, optional): The name of the zip file in Google Drive.
            Defaults to 'birdclef-2025.zip'.


    Returns:
        None:  The function extracts files to a directory.
    """
    try:
        # Mount Google Drive
        drive.mount('/content/drive')

        # Construct the path to the zip file in Google Drive.  This assumes the zip
        # file is in the root of your Drive.  You might need to adjust the path.
        zip_filepath = f'/content/drive/My Drive/Lung_Segmentation/{zip_filename}'

        # Check if the zip file exists
        if not os.path.exists(zip_filepath):
            print(f"Error: Zip file not found at {zip_filepath}.  Please ensure the file is in your Google Drive.")
            return None

        # Extract the  file
        extraction_path = '/content/data'  # Directory to extract to
        os.makedirs(extraction_path, exist_ok=True) #make the directory
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            try:
                zf.extractall(extraction_path)  # Extract *all* files
                print(f"Successfully extracted all files from {zip_filename} to {extraction_path}")
            except Exception as e:
                print(f"Error extracting files: {e}")
                return None



    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
