import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
import os

def select_image():
    """Opens a native file dialog to 'upload' an image locally."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True) # Bring dialog to the front
    file_path = filedialog.askopenfilename(
        title="Select a JPG/PNG Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return file_path

def main():
    print("=== Interactive Image Transformation Toolbox ===")
    print("Waiting for file selection...")
    
    # 1. Upload / Select File
    file_path = select_image()
    if not file_path or not os.path.exists(file_path):
        print("No image selected. Exiting.")
        sys.exit()

    # Load image
    original_image = cv2.imread(file_path)
    if original_image is None:
        print("Error: Could not read the image file.")
        sys.exit()

    current_image = original_image.copy()

    # 2. Setup Matplotlib Interactive Mode
    # This allows the image window to stay open and update while the terminal waits for input
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title('Image Transformation Output')
    ax.axis('off')
    
    # Display initial image
    img_display = ax.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    plt.show()

    def update_display(img):
        """Updates the matplotlib window with the new image."""
        img_display.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Update axes limits in case image size changed (e.g., resize or shear)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # 3. Interactive Terminal Menu
    while True:
        update_display(current_image)
        
        print("\n" + "="*40)
        print("Select an operation:")
        print("1. Rotate")
        print("2. Resize")
        print("3. Flip")
        print("4. Shear")
        print("5. Custom Matrix (2x3)")
        print("6. Reset (Back to original)")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        try:
            h, w = current_image.shape[:2]
            
            if choice == '1':
                # ROTATE
                angle = float(input("Enter rotation angle (in degrees): "))
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                current_image = cv2.warpAffine(current_image, M, (w, h), borderValue=(255, 255, 255))
                
            elif choice == '2':
                # RESIZE
                scale = float(input("Enter resize factor (e.g., 0.5 for half, 2.0 for double): "))
                if scale <= 0:
                    print("Scale must be greater than 0.")
                    continue
                current_image = cv2.resize(current_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                
            elif choice == '3':
                # FLIP
                print("Options: 0 (Vertical), 1 (Horizontal), -1 (Both)")
                flip_code = int(input("Enter flip code: "))
                if flip_code in [0, 1, -1]:
                    current_image = cv2.flip(current_image, flip_code)
                else:
                    print("Invalid flip code.")
                    
            elif choice == '4':
                # SHEAR
                shear_x = float(input("Enter shear factor for X-axis (e.g., 0.2): "))
                shear_y = float(input("Enter shear factor for Y-axis (e.g., 0.0): "))
                M = np.float32([
                    [1, shear_x, 0],
                    [shear_y, 1, 0]
                ])
                # Expand bounds to prevent severe cropping
                new_w = w + int(abs(shear_x) * h)
                new_h = h + int(abs(shear_y) * w)
                current_image = cv2.warpAffine(current_image, M, (new_w, new_h), borderValue=(255, 255, 255))
                
            elif choice == '5':
                # CUSTOM MATRIX
                print("Enter 6 space-separated values for a 2x3 affine matrix.")
                print("Format: m11 m12 tx m21 m22 ty")
                print("Example (Shift right 50, down 30): 1 0 50 0 1 30")
                vals = input("Matrix values: ").split()
                if len(vals) == 6:
                    M = np.float32([float(v) for v in vals]).reshape(2, 3)
                    current_image = cv2.warpAffine(current_image, M, (w, h), borderValue=(255, 255, 255))
                else:
                    print("Error: You must enter exactly 6 values.")
                    
            elif choice == '6':
                # RESET
                current_image = original_image.copy()
                print("Image reset to original.")
                
            elif choice == '7':
                # EXIT
                print("Exiting Toolbox. Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")
                
        except ValueError:
            print("Invalid input! Please enter numerical values where requested.")
        except Exception as e:
            print(f"An error occurred: {e}")

    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()