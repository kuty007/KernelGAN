import os

import cv2
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from patches import divide_into_patches, combine_patches


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    gan.finish()


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        image_path = os.path.join(args.input_dir, filename)
        patch_imgs = divide_into_patches(image_path, num_patches=3)  # Adjust num_patches as needed
        # create dir from the img name
        img_dir = os.path.join(args.output_dir, os.path.splitext(filename)[0])  # create dir from the img name
        os.makedirs(img_dir, exist_ok=True)
        for img in patch_imgs:
            # save the patch
            patch_path = os.path.join(img_dir, img)
            # Save the patch image
            img.save(patch_path)
        res_patches = []
        for patch_name in os.listdir(img_dir):
            conf = Config().parse(create_params(patch_name, args))
            train(conf)
            res_patches.append(patch_name)
        # Combine patches
        combined_image_path = os.path.join(args.output_dir, filename)
        patchs = []
        for p in res_patches:
            patchs.append(cv2.imread(p))
        combine_patches(patchs, combined_image_path)
        # Optionally remove all patch images
        if args.remove_patches:
            for patch_name in res_patches:
                os.remove(os.path.join(img_dir, patch_name))
        print(f"Combined patches saved at: {combined_image_path}")
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