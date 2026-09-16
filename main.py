import cv2
import argparse
import os
import sys

def apply_animation_style(image):
    """Applies a stylized, animation-like aesthetic to the image."""
    # Extract edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # Smooth colors
    color = cv2.bilateralFilter(image, 9, 300, 300)
    
    # Combine edges and smoothed colors
    stylized = cv2.bitwise_and(color, color, mask=edges)
    return stylized

def apply_bw_portrait(image):
    """Converts the image to a high-contrast realistic black and white."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    high_contrast_bw = clahe.apply(gray)
    return high_contrast_bw

def main():
    parser = argparse.ArgumentParser(description="AestheticVision CLI: Image Stylization Engine")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the processed output")
    parser.add_argument("-m", "--mode", required=True, choices=['animation', 'bw'], 
                        help="Style mode to apply: 'animation' or 'bw'")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Loading image from {args.input}...")
    img = cv2.imread(args.input)

    if img is None:
        print("Error: Could not decode the image. Ensure it is a valid format (e.g., .jpg, .png).")
        sys.exit(1)

    if args.mode == 'animation':
        print("Applying stylized animation aesthetic...")
        result = apply_animation_style(img)
    elif args.mode == 'bw':
        print("Applying high-contrast black and white filter...")
        result = apply_bw_portrait(img)

    cv2.imwrite(args.output, result)
    print(f"Success! Processed image saved to {args.output}")

if __name__ == "__main__":
    main()
