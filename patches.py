import cv2
import numpy as np


def divide_into_patches(image_path, num_patches, show_patches=False):
    patches = []
    image = cv2.imread(image_path)
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
            patches.append(patch)
            if show_patches:
                cv2.imshow(f'Patch ({i}, {j})', patch)
                cv2.waitKey(0)  # Wait for a key press to show the next patch
                cv2.destroyAllWindows()  # Close the window displaying the patch
    return patches


def combine_patches(patches, image_path, factor=2):
    image_shape = cv2.imread(image_path).shape*factor
    h, w = image_shape[:2]
    num_patches = int(np.sqrt(len(patches)))
    patch_h = h // num_patches
    patch_w = w // num_patches

    combined_image = np.zeros((h, w, 3), dtype=np.uint8)
    idx = 0
    for i in range(num_patches):
        for j in range(num_patches):
            y0 = i * patch_h
            y1 = (i + 1) * patch_h if i < num_patches - 1 else h
            x0 = j * patch_w
            x1 = (j + 1) * patch_w if j < num_patches - 1 else w
            combined_image[y0:y1, x0:x1] = patches[idx]
            idx += 1

    if combined_image.shape != image_shape*factor:
        combined_image = resize_to_match(combined_image, image_shape, factor)
    return combined_image


def resize_to_match(image, target_shape, factor=2):
    return cv2.resize(image, (target_shape[1]*factor, target_shape[0]*factor))


def validate_image_shape(original_image, result_image, factor=2):
    if original_image.shape*factor != result_image.shape:
        raise ValueError(
            f"Shape mismatch: original image shape {original_image.shape}, result image shape {result_image.shape}")


def main(image_path, num_patches, show_patches=False):
    patches = divide_into_patches(image_path, num_patches, show_patches)
    result_image = combine_patches(patches, image_path)

    # Validate if the original image and the result image have the same shape
    image = cv2.imread(image_path)
    validate_image_shape(image, result_image, 1)

    return result_image


if __name__ == "__main__":
    result_image = main('input/im_3.png', 3, show_patches=False)
    # Example of how you might display the result image (optional)
    # cv2.imshow('Result Image', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
