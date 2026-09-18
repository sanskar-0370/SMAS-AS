```markdown
# Linear Transformations on Images | Assignment 3
question 10

**Course:** System Modelling, Analysis and Stories–1 (SMAS–1)  
*

This repository implements 2D linear geometric transformations on image pixel vectors defined relative to the image center. Using linear algebra and Python, each pixel coordinate is treated as a 2D vector $\begin{bmatrix} x \\ y \end{bmatrix}$ and transformed using standard 2x2 transformation matrices.

---

## Technical Overview

To perform transformations around the image center rather than the default top-left origin $(0,0)$, each operation follows a three-step affine mapping pipeline:

1. **Center Shift:** Translate the image center $(c_x, c_y)$ to $(0,0)$.
2. **Linear Transformation:** Multiply 2D coordinates by matrix $A_i$.
3. **Canvas Fit & Recenter:** Translate transformed coordinates onto a dynamically calculated canvas to prevent border clipping or trimming.

---



*Note: For visualization of $A_5$, a minor vertical scaling factor ($y = 0.01$) is used so the collapsed image remains visible on the plot.*

---

## Project Structure

```text
.
├── sample_photo.jpg       # Sample input image
├── transformation_tool.py # Python script / Jupyter notebook
└── README.md              # Assignment documentation

```

--- question 11

## Setup & Execution

### Prerequisites

Ensure Python 3.8+ is installed along with the required libraries:

```bash
pip install opencv-python numpy matplotlib

```

### Running the Script

Run the script directly via terminal or inside Jupyter Notebook:

```bash
python transformation_tool.py

```

---

## Python Code (`transformation_tool.py`)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load sample image
img = cv2.imread('sample_photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# Define 2D Transformation Matrices
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
axes[0].axis('on')

# Transformed Images
for i, (matrix, title) in enumerate(zip(matrices, titles)):
    out_img = transform_and_fit(img, matrix)
    axes[i+1].imshow(out_img)
    axes[i+1].set_title(title, fontsize=12)
    axes[i+1].axis('on')

plt.tight_layout()
plt.show()

```

```

```
   ```markdown
# Interactive Image Transformation Toolbox (VS Code)

An interactive Python application for desktop environments (VS Code, PyCharm, or local terminal) that allows users to pick an image using a native file selector and apply geometric transformations in real-time.

---

## Features & Operations
* **Native File Picker:** Select any `.jpg`, `.jpeg`, or `.png` file directly from your computer.
* **Live Interactive Display:** Real-time plot rendering using Matplotlib's interactive mode (`plt.ion()`).
* **Transformations Supported:**
  1. **Rotate:** Rotate by any angle (in degrees) around the image center.
  2. **Resize:** Scale image dynamically by a scale factor.
  3. **Flip:** Mirror horizontally, vertically, or along both axes.
  4. **Shear:** Apply horizontal and vertical shear factors with auto-adjusting canvas bounds.
  5. **Custom Matrix:** Apply any custom $2 \times 3$ Affine matrix.
  6. **Reset:** Revert image to its original uploaded state at any point.

---

## Prerequisites & Installation

Install the required dependencies via terminal:

```bash
pip install opencv-python numpy matplotlib

```

*Note: `tkinter` comes pre-installed with standard Python distributions on Windows and macOS. On Linux (Ubuntu/Debian), install it via `sudo apt-get install python3-tk` if missing.*

---

## How to Run

1. Open the project folder in VS Code.
2. Run the script from the VS Code terminal:
```bash
python main.py

```


3. A file browser window will pop up. Select an image.
4. Interact with the terminal menu to transform the image live in the Matplotlib display window.

---

## Line-by-Line Code Explanation

### 1. Imports and File Selection Helper

```python
import cv2  # OpenCV library for image processing and geometric transformations
import numpy as np  # NumPy for matrix calculations and linear algebra operations
import matplotlib.pyplot as plt  # Matplotlib for rendering real-time image updates
import tkinter as tk  # Tkinter standard library for GUI dialogs
from tkinter import filedialog  # File dialog module to select local files
import sys  # System module for script termination
import os  # OS module to verify file path existence


def select_image():
    """Opens a native file dialog to 'upload' an image locally."""
    root = tk.Tk()  # Initializes the Tkinter engine
    root.withdraw()  # Hides the empty main Tkinter root window
    root.attributes("-topmost", True)  # Forces file dialog window to top of all desktop windows
    file_path = filedialog.askopenfilename(  # Triggers file picker dialog
        title="Select a JPG/PNG Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return file_path  # Returns chosen file path string (or empty string if canceled)

```

---

### 2. Main Setup & Interactive Window Initialization

```python
def main():
    print("=== Interactive Image Transformation Toolbox ===")
    print("Waiting for file selection...")

    # 1. Upload / Select File
    file_path = select_image()  # Calls file picker dialog
    if not file_path or not os.path.exists(file_path):  # Checks if user canceled or file is missing
        print("No image selected. Exiting.")
        sys.exit()  # Exits program safely

    # Load image
    original_image = cv2.imread(file_path)  # Reads image into NumPy array in BGR format
    if original_image is None:  # Validates image loading
        print("Error: Could not read the image file.")
        sys.exit()

    current_image = original_image.copy()  # Holds active working copy of image

    # 2. Setup Matplotlib Interactive Mode
    plt.ion()  # Enables interactive plot mode (non-blocking display update)
    fig, ax = plt.subplots(figsize=(6, 6))  # Creates 6x6 inch display window frame
    fig.canvas.manager.set_window_title("Image Transformation Output")  # Sets window title
    ax.axis("off")  # Hides plot axis ticks and labels for clean viewing

    # Display initial image
    img_display = ax.imshow(
        cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    )  # Converts BGR to RGB for correct colors
    plt.show()  # Displays non-blocking plot window

    def update_display(img):
        """Updates the matplotlib window with the new image."""
        img_display.set_data(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )  # Updates raw pixel data of active plot
        ax.set_xlim(0, img.shape[1])  # Adjusts X-axis bounds to match new width
        ax.set_ylim(img.shape[0], 0)  # Adjusts Y-axis bounds (inverted for image coordinates)
        fig.canvas.draw()  # Redraws figure canvas
        fig.canvas.flush_events()  # Flushes GUI events to refresh screen immediately

```

---

### 3. Interactive Menu Loop & Operations

```python
    # 3. Interactive Terminal Menu
    while True:
        update_display(current_image)  # Refreshes GUI plot window with active image state

        print("\n" + "=" * 40)
        print("Select an operation:")
        print("1. Rotate")
        print("2. Resize")
        print("3. Flip")
        print("4. Shear")
        print("5. Custom Matrix (2x3)")
        print("6. Reset (Back to original)")
        print("7. Exit")

        choice = input("\nEnter your choice (1-7): ").strip()  # Gets user selection input

        try:
            h, w = current_image.shape[:2]  # Unpacks current height and width

            if choice == "1":
                # ROTATE
                angle = float(input("Enter rotation angle (in degrees): "))  # Reads angle
                center = (w // 2, h // 2)  # Calculates geometric center point
                M = cv2.getRotationMatrix2D(
                    center, angle, 1.0
                )  # Constructs 2x3 rotation matrix
                current_image = cv2.warpAffine(
                    current_image, M, (w, h), borderValue=(255, 255, 255)
                )  # Applies rotation warp

            elif choice == "2":
                # RESIZE
                scale = float(
                    input("Enter resize factor (e.g., 0.5 for half, 2.0 for double): ")
                )  # Reads scale
                if scale <= 0:
                    print("Scale must be greater than 0.")
                    continue
                current_image = cv2.resize(
                    current_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
                )  # Resizes image matrix

            elif choice == "3":
                # FLIP
                print("Options: 0 (Vertical), 1 (Horizontal), -1 (Both)")
                flip_code = int(input("Enter flip code: "))  # Reads flip direction code
                if flip_code in [0, 1, -1]:
                    current_image = cv2.flip(
                        current_image, flip_code
                    )  # Flips image matrix across target axis
                else:
                    print("Invalid flip code.")

            elif choice == "4":
                # SHEAR
                shear_x = float(
                    input("Enter shear factor for X-axis (e.g., 0.2): ")
                )  # Reads horizontal shear
                shear_y = float(
                    input("Enter shear factor for Y-axis (e.g., 0.0): ")
                )  # Reads vertical shear
                M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])  # Defines shear matrix
                new_w = w + int(
                    abs(shear_x) * h
                )  # Expands width bound to prevent clipping
                new_h = h + int(
                    abs(shear_y) * w
                )  # Expands height bound to prevent clipping
                current_image = cv2.warpAffine(
                    current_image, M, (new_w, new_h), borderValue=(255, 255, 255)
                )  # Applies shear warp

            elif choice == "5":
                # CUSTOM MATRIX
                print("Enter 6 space-separated values for a 2x3 affine matrix.")
                print("Format: m11 m12 tx m21 m22 ty")
                print("Example (Shift right 50, down 30): 1 0 50 0 1 30")
                vals = input("Matrix values: ").split()  # Splits user input string into list
                if len(vals) == 6:
                    M = np.float32([float(v) for v in vals]).reshape(
                        2, 3
                    )  # Reshapes array into 2x3 matrix
                    current_image = cv2.warpAffine(
                        current_image, M, (w, h), borderValue=(255, 255, 255)
                    )  # Applies affine transformation
                else:
                    print("Error: You must enter exactly 6 values.")

            elif choice == "6":
                # RESET
                current_image = original_image.copy()  # Reverts current image back to original state
                print("Image reset to original.")

            elif choice == "7":
                # EXIT
                print("Exiting Toolbox. Goodbye!")
                break  # Exits while loop

            else:
                print("Invalid choice. Please enter a number between 1 and 7.")

        except ValueError:
            print("Invalid input! Please enter numerical values where requested.")
        except Exception as e:
            print(f"An error occurred: {e}")

    plt.ioff()  # Disables Matplotlib interactive mode
    plt.close()  # Closes display plot window


if __name__ == "__main__":
    main()  # Entry point execution

```

```

```
