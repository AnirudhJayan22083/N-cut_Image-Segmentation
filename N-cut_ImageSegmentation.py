import os
import cv2
import numpy as np
from skimage import segmentation, color, graph
import matplotlib.pyplot as plt

dataset_dir = "C:\\Users\\aniru\\OneDrive\\Pictures\\Screenshots"
images = {}

# Load one image (choose the first one or any specific image)
selected_image_filename = "ss5.png"  

# Load the specific image in grayscale
img = cv2.imread(os.path.join(dataset_dir, selected_image_filename), cv2.IMREAD_GRAYSCALE)

if img is not None:
    images[selected_image_filename] = img
    print(f"Loaded image: {selected_image_filename}")
else:
    print("Error loading the image.")

# Step 2: Preprocess the image
def preprocess_image(img):
    """Preprocess the image to make it easier for segmentation"""
    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    
    # Apply binary thresholding to separate letters from background (assuming white background)
    _, thresh_img = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    
    return thresh_img

# Apply preprocessing to the selected image
processed_images = {filename: preprocess_image(img) for filename, img in images.items()}

# Step 3: Function to perform segmentation using Normalized Cuts (Ncut)
def segment_image_with_ncut(img):
    """Segment the image using SLIC and Normalized Cuts"""
    
    # Check if the image is grayscale (single channel)
    if len(img.shape) == 2:  # grayscale image
        img_rgb = np.stack([img] * 3, axis=-1)  # Convert to 3-channel RGB image
    else:
        img_rgb = img  # already an RGB image
    
    # Apply SLIC to generate superpixels
    labels_slic = segmentation.slic(img_rgb, compactness=30, n_segments=100, channel_axis=-1)  
    
    # Build Region Adjacency Graph (RAG)
    g = graph.rag_mean_color(img_rgb, labels_slic, mode='similarity')
    #plt.plot(g)
    
    # Apply Normalized Cuts
    ncuts_labels = graph.cut_normalized(labels_slic, g)
    plt.plot(ncuts_labels)
    return ncuts_labels, img_rgb  

# Step 4: Function to refine Ncut segmentation
def refine_ncut_segmentation(ncuts_labels, img):
    """Refine the Ncut segmented image by removing small regions and focusing on important areas"""
    # Convert to grayscale for further processing
    labels_unique = np.unique(ncuts_labels)
    
    # Remove small segments by area (keep only large ones)
    min_area = 500  # Minimum area threshold 
    refined_labels = np.zeros_like(ncuts_labels, dtype=np.int32) 
    
    for label in labels_unique:
        # Create a binary mask for the current label
        label_mask = np.uint8(ncuts_labels == label)
        label_area = np.sum(label_mask)

        # If the segment is large enough, keep it
        if label_area >= min_area:
            refined_labels[label_mask == 1] = label
    
    # Ensure the image is in RGB format if it is grayscale
    if len(img.shape) == 2:  # Grayscale image
        img_rgb = np.stack([img] * 3, axis=-1)  # Convert to 3-channel RGB image
    else:
        img_rgb = img  # already an RGB image
    
    # Map the labels to unique colors
    num_labels = np.max(refined_labels) + 1  # Find the number of unique labels
    colormap = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)  # Random colors for each label

    # Create a blank RGB image to store the final result
    refined_segmented_img = np.zeros_like(img_rgb)

    # Iterate over all labels and colorize the regions
    for label in range(num_labels):
        mask = refined_labels == label
        refined_segmented_img[mask] = colormap[label]

    # Return the refined segmented image
    return refined_segmented_img

# Step 5: Apply Ncut segmentation and refinement to the preprocessed image
segmented_images = {filename: segment_image_with_ncut(img) for filename, img in processed_images.items()}
ncuts_labels_images = {filename: segmented_img[0] for filename, segmented_img in segmented_images.items()}  # Extracting labels
images = {filename: segmented_img[1] for filename, segmented_img in segmented_images.items()}  # Extracting original images

# Now refine the Ncut segmented images
refined_ncuts_images = {filename: refine_ncut_segmentation(ncuts_labels_images[filename], images[filename])
                        for filename in ncuts_labels_images}


def display_images(original_images, ncut_images, refined_ncut_images, ncols=2):
    """Display the original, Ncut segmented, and refined Ncut segmented images side by side"""
    nrows = 1  # Only 1 row since we're displaying just one image

    plt.figure(figsize=(12, nrows * 4))  # Adjusting height for more images
    for idx, (filename, img) in enumerate(original_images.items()):
        # Plot original image (ensure grayscale display for grayscale images)
        plt.subplot(nrows, ncols, 3 * idx + 1)
        plt.imshow(img, cmap='gray')  # Grayscale display
        plt.title(f"Original: {filename}")
        plt.axis('off')
        
        # Extract the segmented label image from the tuple
        segmented_img = ncut_images[filename][0]  # Extract the ncuts_labels part of the tuple
        
        # If the segmented image is a label array, convert it to a displayable image
        if len(segmented_img.shape) == 2:  # If it's a label array (2D)
            plt.imshow(segmented_img, cmap='nipy_spectral')  # Use a colormap for label visualization
        else:
            plt.imshow(segmented_img)  # For RGB images, just plot them as they are
        plt.title(f"Ncut Segmented: {filename}")
        plt.axis('off')

        # Extract the refined segmented image
        refined_img = refined_ncut_images[filename]
        
        # If it's a label array, convert to an RGB image
        if len(refined_img.shape) == 2:  # If it's a label array (2D)
            plt.imshow(refined_img, cmap='nipy_spectral')  # Use a colormap for label visualization
        else:
            plt.imshow(refined_img)  # For RGB images, just plot them as they are
        
        plt.title(f"Refined Ncut Segmented: {filename}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()



display_images(processed_images, segmented_images, refined_ncuts_images)
