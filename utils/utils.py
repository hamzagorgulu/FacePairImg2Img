import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def get_first_n_pairs(directory, n):
    """
    Get the first n sets of beard, no_beard, and mask images from the directory.
    Returns a list of tuples (beard_path, no_beard_path, mask_path)
    """
    # Get all files and sort them
    all_files = sorted(os.listdir(directory))

    pairs = []
    for i in range(n):
        # Format the index with leading zeros
        idx = f"{i:03d}"  # This will give '000', '001', etc.

        # Create the expected filenames
        beard_name = f"{idx}_beard.png"
        no_beard_name = f"{idx}_no_beard.png"
        #mask_name = f"{idx}_mask.png"

        # Create full paths
        beard_path = os.path.join(directory, beard_name)
        no_beard_path = os.path.join(directory, no_beard_name)
        #mask_path = os.path.join(directory, mask_name)

        # Verify files exist
        if all(name in all_files for name in [beard_name, no_beard_name]):
            pairs.append((beard_path, no_beard_path))
        else:
            print(f"Warning: Couldn't find complete set for index {idx}")
            break

    return pairs

def visualize_dataset_pairs(pairs, num_samples=5, figsize=(12, 4)):
    """
    Display the dataset pairs in a grid format.
    Each row shows: original beard image and no beard image
    """
    num_samples = min(num_samples, len(pairs))

    # Create figure with subplots for each sample
    fig, axes = plt.subplots(num_samples, 2, figsize=(figsize[0], figsize[1] * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Set title for columns
    axes[0, 0].set_title('Original (With Beard)', pad=10)
    axes[0, 1].set_title('Generated (No Beard)', pad=10)

    for idx, (beard_path, no_beard_path) in enumerate(pairs[:num_samples]):
        # Load images
        beard_img = Image.open(beard_path)
        no_beard_img = Image.open(no_beard_path)

        # Display images
        axes[idx, 0].imshow(beard_img)
        axes[idx, 1].imshow(no_beard_img)

        # Remove axes ticks
        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add row number
        axes[idx, 0].set_ylabel(f'Sample {idx}', rotation=0, labelpad=40, va='center')

    plt.tight_layout()
    return fig

def generate_and_visualize_epochs(epoch_max, step=1, sample_nums=[0], base_dir="samples"):
    """
    Generate file paths for specified samples and epochs, then visualize the images.

    Args:
        epoch_max (int): Maximum number of epochs.
        step (int): Step size for skipping epochs.
        sample_nums (list of int): List of sample numbers to visualize.
        base_dir (str): Base directory containing the images.

    Returns:
        list: List of generated file paths.
    """
    filenames_lst = []

    # Generate file paths
    for epoch in range(0, epoch_max, step):
        for sample_num in sample_nums:
            filenames_lst.append(os.path.join(base_dir, f"epoch_{epoch}_sample_{sample_num}.png"))

    # Visualize images
    visualize_images(filenames_lst, epoch_max, step, sample_nums)

    return filenames_lst

def visualize_images(image_paths, epoch_max, step, sample_nums):
    """
    Display images in a grid format to visualize performance over epochs.

    Args:
        image_paths (list of str): List of file paths to the images.
        epoch_max (int): Maximum number of epochs.
        step (int): Step size for skipping epochs.
        sample_nums (list of int): List of sample numbers visualized.

    Returns:
        None
    """
    # Calculate grid size
    num_epochs = len(range(0, epoch_max, step))
    num_samples = len(sample_nums)

    fig, axes = plt.subplots(num_samples, num_epochs, figsize=(num_epochs * 4, num_samples * 4))

    # Handle single row or column cases
    if num_samples == 1:
        axes = [axes]
    if num_epochs == 1:
        axes = [[ax] for ax in axes]

    # Loop over image paths and display them
    idx = 0
    for i, sample_num in enumerate(sample_nums):
        for j, epoch in enumerate(range(0, epoch_max, step)):
            ax = axes[i][j]
            if idx < len(image_paths):
                img_path = image_paths[idx]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f'Epoch {epoch}, Sample {sample_num}', fontsize=12, pad=10)
                except Exception as e:
                    ax.axis('off')
                    ax.set_title(f'Missing Image\nEpoch {epoch}, Sample {sample_num}', fontsize=10, pad=10)
                    print(f"Error loading {img_path}: {e}")
            idx += 1

    # Adjust layout for better spacing
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.show()

def visualize_test_samples(folder_path, step=1):
    """
    Concatenates images from a folder vertically and displays the result.

    Args:
        folder_path (str): Path to the folder containing images.
        step (int): Step size for skipping samples.

    Returns:
        None: Displays the concatenated image.
    """
    # List all files in the folder
    all_files = sorted(os.listdir(folder_path))

    # Filter PNG files and sort by numerical order
    image_files = [f for f in all_files if f.endswith('.png')]

    # Select files based on the step
    selected_files = image_files[::step]

    # Open images
    images = [Image.open(os.path.join(folder_path, img)) for img in selected_files]

    # Calculate the width and total height of the resulting image
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    # Create a blank canvas for the concatenated image
    concatenated_image = Image.new('RGB', (max_width, total_height))

    # Paste images onto the canvas
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Display the concatenated image using matplotlib
    plt.figure(figsize=(8, len(images) * 3))  # Adjust size dynamically
    plt.imshow(concatenated_image)
    plt.axis("off")
    plt.show()

def show_epoch_progress(epoch_max, sample_num, step=1):
    """
    Display the progression of generated images across epochs in a vertical layout with epoch numbers on the y-axis.
    Allows skipping epochs based on the specified step.

    Args:
        epoch_max (int): Maximum number of epochs.
        sample_num (int): The specific sample number to visualize.
        step (int): Step size to skip epochs (e.g., 1 for all, 2 for every other epoch).

    Returns:
        None: Displays the concatenated images with annotations.
    """

    # Create the list of image paths, skipping steps
    image_paths = [
        os.path.join("samples", f"epoch_{epoch}_sample_{sample_num}.png")
        for epoch in range(0, epoch_max, step)
    ]

    # Load images
    images = []
    epochs = []  # Keep track of the epochs being used
    for epoch, img_path in zip(range(0, epoch_max, step), image_paths):
        try:
            img = Image.open(img_path)
            images.append(img)
            epochs.append(epoch)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # Create the plot with epoch numbers as y-axis labels
    fig, ax = plt.subplots(len(images), 1, figsize=(6, len(images) * 4))

    if len(images) == 1:
        ax = [ax]  # Ensure ax is iterable if there's only one image

    for i, (img, epoch) in enumerate(zip(images, epochs)):
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(f"Epoch {epoch}", fontsize=12, loc="left")

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()