import cv2
import numpy as np


def divide_into_patches(image_path, num_patches, show_patches=False):
    patches = []
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]
    patch_h = h // num_patches
    patch_w = w // num_patches
    for i in range(num_patches):
        for j in range(num_patches):
            y0 = i * patch_h
            y1 = (i + 1) * patch_h if i < num_patches - 1 else h
            x0 = j * patch_w
            x1 = (j + 1) * patch_w if j < num_patches - 1 else w
            patch = image[y0:y1, x0:x1]
            patches.append((patch, (y0, x0)))
            if show_patches:
                cv2.imshow(f'Patch ({i}, {j})', patch)
                cv2.waitKey(0)  # Wait for a key press to show the next patch
                cv2.destroyAllWindows()  # Close the window displaying the patch
    return patches


def combine_patches(patches, image_shape):
    h, w = image_shape[:2]
    combined_image = np.zeros((h, w, 3), dtype=np.uint8)
    for patch, (y0, x0) in patches:
        y1 = y0 + patch.shape[0]
        x1 = x0 + patch.shape[1]
        combined_image[y0:y1, x0:x1] = patch
    return combined_image


def resize_to_match(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))


def validate_image_shape(original_image, result_image):
    if original_image.shape != result_image.shape:
        raise ValueError(
            f"Shape mismatch: original image shape {original_image.shape}, result image shape {result_image.shape}")


def main(image_path, num_patches, show_patches=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = image.shape[:2]
    if num_patches <= 0 or num_patches > min(h, w):
        raise ValueError(
            f"Invalid number of patches: {num_patches}. It must be a positive integer and less than or equal to the smaller dimension of the image ({min(h, w)}).")

    patches = divide_into_patches(image, num_patches, show_patches)
    result_image = combine_patches(patches, image.shape)

    # Ensure the combined image matches the original image dimensions
    if result_image.shape != image.shape:
        result_image = resize_to_match(result_image, image.shape)

    # Validate if the original image and the result image have the same shape
    validate_image_shape(image, result_image)

    return result_image


if __name__ == "__main__":
    result_image = main('input/im_3.png', 3, show_patches=False)
    # Example of how you might display the result image (optional)
    # cv2.imshow('Result Image', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
