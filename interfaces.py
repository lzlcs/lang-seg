import argparse
import os

import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms as transforms
from encoding.models.sseg import BaseNet
from PIL import Image
from tqdm import tqdm

from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule


def load_model():
    """
    Copied from lang-seg/lseg_app.py
    """

    class Options:
        def __init__(self):
            parser = argparse.ArgumentParser(description="PyTorch Segmentation")
            # model and dataset
            parser.add_argument(
                "--model",
                type=str,
                default="encnet",
                help="model name (default: encnet)",
            )
            parser.add_argument(
                "--backbone",
                type=str,
                default="clip_vitl16_384",
                help="backbone name (default: resnet50)",
            )
            parser.add_argument(
                "--dataset",
                type=str,
                default="ade20k",
                help="dataset name (default: pascal12)",
            )
            parser.add_argument(
                "--workers",
                type=int,
                default=16,
                metavar="N",
                help="dataloader threads",
            )
            parser.add_argument(
                "--base-size", type=int, default=520, help="base image size"
            )
            parser.add_argument(
                "--crop-size", type=int, default=480, help="crop image size"
            )
            parser.add_argument(
                "--train-split",
                type=str,
                default="train",
                help="dataset train split (default: train)",
            )
            parser.add_argument(
                "--aux", action="store_true", default=False, help="Auxilary Loss"
            )
            parser.add_argument(
                "--se-loss",
                action="store_true",
                default=False,
                help="Semantic Encoding Loss SE-loss",
            )
            parser.add_argument(
                "--se-weight",
                type=float,
                default=0.2,
                help="SE-loss weight (default: 0.2)",
            )
            parser.add_argument(
                "--batch-size",
                type=int,
                default=16,
                metavar="N",
                help="input batch size for \
                                training (default: auto)",
            )
            parser.add_argument(
                "--test-batch-size",
                type=int,
                default=16,
                metavar="N",
                help="input batch size for \
                                testing (default: same as batch size)",
            )
            # cuda, seed and logging
            parser.add_argument(
                "--no-cuda",
                action="store_true",
                default=False,
                help="disables CUDA training",
            )
            parser.add_argument(
                "--seed",
                type=int,
                default=1,
                metavar="S",
                help="random seed (default: 1)",
            )
            # checking point
            parser.add_argument(
                "--weights", type=str, default="", help="checkpoint to test"
            )
            # evaluation option
            parser.add_argument(
                "--eval", action="store_true", default=False, help="evaluating mIoU"
            )
            parser.add_argument(
                "--export",
                type=str,
                default=None,
                help="put the path to resuming file if needed",
            )
            parser.add_argument(
                "--acc-bn",
                action="store_true",
                default=False,
                help="Re-accumulate BN statistics",
            )
            parser.add_argument(
                "--test-val",
                action="store_true",
                default=False,
                help="generate masks on val set",
            )
            parser.add_argument(
                "--no-val",
                action="store_true",
                default=False,
                help="skip validation during training",
            )

            parser.add_argument(
                "--module",
                default="lseg",
                help="select model definition",
            )

            # test option
            parser.add_argument(
                "--data-path",
                type=str,
                default="../datasets/",
                help="path to test image folder",
            )

            parser.add_argument(
                "--no-scaleinv",
                dest="scale_inv",
                default=True,
                action="store_false",
                help="turn off scaleinv layers",
            )

            parser.add_argument(
                "--widehead",
                default=False,
                action="store_true",
                help="wider output head",
            )

            parser.add_argument(
                "--widehead_hr",
                default=False,
                action="store_true",
                help="wider output head",
            )
            parser.add_argument(
                "--ignore_index",
                type=int,
                default=-1,
                help="numeric value of ignore label in gt",
            )

            parser.add_argument(
                "--label_src",
                type=str,
                default="default",
                help="how to get the labels",
            )

            parser.add_argument(
                "--arch_option",
                type=int,
                default=0,
                help="which kind of architecture to be used",
            )

            parser.add_argument(
                "--block_depth",
                type=int,
                default=0,
                help="how many blocks should be used",
            )

            parser.add_argument(
                "--activation",
                choices=["lrelu", "tanh"],
                default="lrelu",
                help="use which activation to activate the block",
            )

            self.parser = parser

        def parse(self):
            args = self.parser.parse_args(args=[])
            args.cuda = not args.no_cuda and torch.cuda.is_available()
            print(args)
            return args

    args = Options().parse()

    torch.manual_seed(args.seed)
    args.test_batch_size = 1
    alpha = 0.5

    args.scale_inv = False
    args.widehead = True
    args.dataset = "ade20k"
    args.backbone = "clip_vitl16_384"
    args.weights = "checkpoints/demo_e200.ckpt"
    args.ignore_index = 255

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )

    # model
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()
    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]
    evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
    evaluator.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    return evaluator, transform


def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches
        
class LSegInterface:
    
    def __init__(self, colored_mask=False) -> None:
        """load Lseg model"""
        self.lseg_model, self.lseg_transform = load_model()
        self.colored_mask = colored_mask
        
    def get_mask(self, image: torch.Tensor, prompt: str) -> Image.Image:
        """Get image mask through prompt
    
        Args:
            image: A tensor with a shape of [1, H, W, C]
            prompt: A string containing keywords separated by commas
            
        Returns:
            mask: A Image whose mode is 'L' or 'RGB'
        """
        
        labels = []
        print("Input prompt: {}".format(prompt))
        lines = prompt.split(",")
        for line in lines:
            label = line
            labels.append(label)
        
        with torch.no_grad():
            outputs = self.lseg_model.parallel_forward(image, labels)
            # outputs = model(image,labels)
            predicts = [torch.max(output, 1)[1].cpu().numpy() for output in outputs]
            
        
        if self.colored_mask:
            new_palette = get_new_pallete(len(labels))
            mask, _ = get_new_mask_pallete(
                predicts[0], new_palette, out_label_flag=True, labels=labels
            )
            mask = mask.convert("RGB")
        else:
            mask = predicts[0].astype(np.uint8)
            print(mask.shape)
            mask = Image.fromarray(mask[0])
            
        return mask
        
    def lseg(self, image_path: str, prompt: str, output_path: str) -> None:
        
        os.makedirs(output_path, exist_ok=True)
        if os.path.isdir(image_path):
            file_list = os.listdir(image_path)
            for i in tqdm(range(file_list)):
                file = file_list[i]
                image = Image.open(os.path.join(image_path, file))
                image = self.lseg_transform(image).unsqueeze(0)
                mask = self.get_mask(image, prompt)
                mask.save(os.path.join(output_path, file))
        else:
            image = Image.open(image_path)
            image = self.lseg_transform(image).unsqueeze(0)
            mask = self.get_mask(image, prompt)
            mask.save(os.path.join(output_path, os.path.basename(image_path)))
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Lseg images')
    
    parser.add_argument('--input_path', type=str, help='input image or folder')
    parser.add_argument('--prompt', type=str, help='str containing keywords separated by commas')
    parser.add_argument('--output_path', type=str, help='output image or folder')
    parser.add_argument("--colorful", action="store_true")
    args = parser.parse_args()
    
    model = LSegInterface(args.colorful)
    model.lseg(args.input_path, args.prompt, args.output_path)
    
