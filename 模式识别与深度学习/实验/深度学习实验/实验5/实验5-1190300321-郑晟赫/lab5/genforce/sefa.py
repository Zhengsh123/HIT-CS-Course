# python3.7
"""A simple tool to synthesize images with pre-trained models."""

import os
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch

from models import MODEL_ZOO
from models import build_generator
from utils.misc import bool_parser
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    images = images.detach().cpu().numpy()
    images = (images + 1) * 255 / 2
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images

def save_video(visuals, path):
    gifs = []
    import torch
    import imageio
    from torchvision.utils import make_grid
    for img in visuals:
        x = (img + 1) / 2
        x = x.clamp_(0, 1)
        grid = make_grid(x.data.cpu(), nrow=1, padding=0, pad_value=0,
                         normalize=False, range=None, scale_each=None)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        ndarr = ndarr.transpose((1, 2, 0))
        gifs.append(ndarr)
    imageio.mimwrite(path, gifs)

def save_frame(visuals, path):
    from torchvision.utils import save_image as th_save
    x = (visuals + 1) / 2
    x = x.clamp_(0, 1)
    th_save(x.data.cpu(), path, nrow=1, padding=0)

def resize_image(img, size):
    return torch.nn.functional.interpolate(img, (size, size), mode='bilinear')

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize images with pre-trained models.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/synthesis/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--generate_html', type=bool_parser, default=False,
                        help='Whether to use HTML page to visualize the '
                             'synthesized results. (default: %(default)s)')
    parser.add_argument('--save_raw_synthesis', type=bool_parser, default=True,
                        help='Whether to save raw synthesis. '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-2.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=2.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=21,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--cuda', type=bool_parser, default=True,
                        help='Whether to use cuda.')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if not args.save_raw_synthesis and not args.generate_html:
        return

    num_sam = args.num_samples
    num_sem = args.num_semantics

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in '
                         f'`models/model_zoo.py`!')

    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('work_dirs', 'synthesis')
    os.makedirs(work_dir, exist_ok=True)

    prefix = (f'{args.model_name}_'
              f'N{num_sam}_K{num_sem}_seed{args.seed}')
    job_dir = os.path.join(work_dir, prefix)
    os.makedirs(job_dir, exist_ok=True)
    frame_dir = os.path.join(job_dir, 'frames')
    if args.save_raw_synthesis:
        os.makedirs(frame_dir, exist_ok=True)


    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    if args.cuda:
        generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')

    directions = [torch.randn(num_sam, generator.z_space_dim) for _ in range(num_sem)]
    weight = generator.__getattr__('layer0').weight
    weight = weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1)

    ################
    # TODO factorize the weight of layer0 to get the directions
    # run: python sefa.py pggan_celebahq1024 --cuda false/true
    # 将weight转化为numpy格式
    weight=weight.cpu().detach().numpy()
    # 归一化
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    # 求特征值和特征向量
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    directions = torch.tensor(eigen_vectors.T)
    ################

    if args.cuda:
        directions = [d.cuda() for d in directions]

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Sample and synthesize.
    code = torch.randn(num_sam, generator.z_space_dim)
    if args.cuda:
        code = code.cuda()
    distances = np.linspace(args.start_distance, args.end_distance, args.step)
    visual_list = [[] for _ in range(num_sem)]
    with torch.no_grad():
        video_list = []
        for s in tqdm(range(args.step)):
            row_list = []
            images = generator(code, **synthesis_kwargs)['image']
            images = resize_image(images, args.viz_size)
            row_list.append(images.cpu())
            for i, dr in enumerate(directions[:num_sem]):
                temp_code = code.clone()
                temp_code += dr * distances[s]
                images = generator(temp_code, **synthesis_kwargs)['image']
                images = resize_image(images, args.viz_size)
                row_list.append(images.cpu())
                visual_list[i].append(images.cpu())
            video_list.append(torch.cat(row_list, dim=3))
        save_video(video_list, os.path.join(job_dir, prefix + '.mov'))
        for i, v in enumerate(visual_list):
            save_frame(torch.cat(v, dim=3), os.path.join(frame_dir, f'frame{i}.png'))
    print(f'Finish synthesizing.')


if __name__ == '__main__':
    main()
