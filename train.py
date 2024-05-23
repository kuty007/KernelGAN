import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner

import cv2
import numpy as np


def divide_into_patches(image, num_patches):
    patches = []
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
    return patches


def combine_patches(patches, image_shape):
    h, w = image_shape[:2]
    combined_image = np.zeros((h, w, 3), dtype=np.uint8)
    num_patches = int(np.sqrt(len(patches)))
    patch_h = h // num_patches
    patch_w = w // num_patches
    for i in range(num_patches):
        for j in range(num_patches):
            patch = patches[i * num_patches + j]
            y0, y1 = i * patch_h, (i + 1) * patch_h
            x0, x1 = j * patch_w, (j + 1) * patch_w
            combined_image[y0:y1, x0:x1] = patch
    return combined_image


def train_on_patch(patch, conf, gan, learner, data):
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        g_in, d_in = data.__getitem__(iteration)  # Assuming batch size of 1 for simplicity
        gan.train(g_in, d_in)
        learner.update(iteration, gan)

    gan.finish()


def main():
    import argparse
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    prog.add_argument('--num_patches', type=int, default=3, help='Number of patches to divide the image into.')
    args = prog.parse_args()

    for filename in os.listdir(os.path.abspath(args.input_dir)):
        input_image_path = os.path.join(args.input_dir, filename)
        output_dir_path = os.path.abspath(args.output_dir)

        # Load the input image
        input_image = cv2.imread(input_image_path)
        if input_image is None:
            print(f"Error: Could not read image {input_image_path}")
            continue

        # Divide the input image into patches
        patches = divide_into_patches(input_image, args.num_patches)

        # Initialize the KernelGAN, Learner, and DataGenerator instances
        conf = Config().parse(create_params(filename, args))
        gan = KernelGAN(conf)
        learner = Learner()
        data = DataGenerator(conf, gan)

        # Run the KernelGAN algorithm on each patch
        for patch in patches:
            train_on_patch(patch, conf, gan, learner, data)

            # Placeholder; replace with actual result patch obtained after training the KernelGAN on each patch
            result_patch = patch

        # Combine the results from all patches into a single output image
        output_image = combine_patches(patches, input_image.shape)

        # Save the output image
        output_image_path = os.path.join(output_dir_path, f"result_{filename}")
        cv2.imwrite(output_image_path, output_image)

    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    return params


if __name__ == '__main__':
    main()
