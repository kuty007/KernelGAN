import os
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
    import argparse
    import shutil
    import cv2
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            patch_imgs = divide_into_patches(image_path, num_patches=3) # 3X3 patches = 9
            img_dir = os.path.join(input_dir, os.path.splitext(filename)[0])
            os.makedirs(img_dir, exist_ok=True)

            patch_paths = []
            for idx, img in enumerate(patch_imgs):
                patch_path = os.path.join(img_dir, f"patch_{idx}.png")
                cv2.imwrite(patch_path, img)
                patch_paths.append(patch_path)

                conf = Config().parse(create_params(patch_path, args))
                train(conf)

            # Combine patches from the output directory
            output_img_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            combined_image_path = os.path.join(output_dir, filename)
            output_patch_paths = [os.path.join(output_img_dir, f"patch_{idx}.png") for idx in range(len(patch_imgs))]
            patch_imgs = [cv2.imread(p) for p in output_patch_paths]  # Read the patch images
            combine_patches(patch_imgs, image_path)

            # Remove intermediate patch images and directories from both input and output directories
            for patch_path in patch_paths:
                os.remove(patch_path)
            shutil.rmtree(img_dir)  # Remove the input directory and its contents

            for patch_path in output_patch_paths:
                os.remove(patch_path)
            shutil.rmtree(output_img_dir)  # Remove the output directory and its contents

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