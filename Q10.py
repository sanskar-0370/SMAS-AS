import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and set up center coordinates
img = cv2.imread('raviQ10.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# Define 2D transformation matrices
A1 = np.array([[2, 0], [0, 0.5]])       # Scaling
A2 = np.array([[0, -1], [1, 0]])       # 90° CCW Rotation
A3 = np.array([[1, 1], [0, 1]])        # Horizontal Shear
A4 = np.array([[-1, 0], [0, 1]])       # Reflection in y-axis
A5_vis = np.array([[1, 0], [0, 0.01]]) # X-axis projection (scaled for visualization)

matrices = [A1, A2, A3, A4, A5_vis]
titles = [
    'A1: Scaling', 
    'A2: 90° Rotation', 
    'A3: Horizontal Shear', 
    'A4: Y-Axis Reflection', 
    'A5: X-Axis Projection'
]

def transform_and_fit(image, A):
    h, w = image.shape[:2]
    
    # 1. Define corners relative to image center (0,0)
    corners = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2]
    ])
    
    # 2. Transform corner points to find dynamic canvas limits
    transformed_corners = (A @ corners.T).T
    
    x_min, x_max = transformed_corners[:, 0].min(), transformed_corners[:, 0].max()
    y_min, y_max = transformed_corners[:, 1].min(), transformed_corners[:, 1].max()
    
    new_w = int(np.ceil(x_max - x_min)) + 20
    new_h = int(np.ceil(y_max - y_min)) + 20
    
    # 3. Build Affine Matrix: Shift Center -> Apply A -> Shift to New Canvas Center
    T_to_center = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]], dtype=float)
    
    T_affine = np.eye(3, dtype=float)
    T_affine[0:2, 0:2] = A
    
    T_from_center = np.array([[1, 0, new_w/2], [0, 1, new_h/2], [0, 0, 1]], dtype=float)
    
    # Combine transformations
    M = T_from_center @ T_affine @ T_to_center
    affine_2x3 = M[0:2, :]
    
    # 4. Warp image with calculated canvas size
    transformed_img = cv2.warpAffine(
        image, 
        affine_2x3, 
        (new_w, new_h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255) # White padding
    )
    return transformed_img

# Plotting the original and transformed images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Original Image
axes[0].imshow(img)
axes[0].set_title('Original Image', fontsize=12)
axes[0].axis('on') # Display pixel axes

# Transformed Images
for i, (matrix, title) in enumerate(zip(matrices, titles)):
    out_img = transform_and_fit(img, matrix)
    axes[i+1].imshow(out_img)
    axes[i+1].set_title(title, fontsize=12)
    axes[i+1].axis('on') # Display pixel axes

plt.tight_layout()
plt.show()
