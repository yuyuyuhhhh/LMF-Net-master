import os
import argparse
import numpy as np
from skimage import io
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.moon_dataset import MoonDataset
from nets.LMF_Net import LMFNet
from nets.brn import BoundaryFusionNet

MEAN, STD = 0.521705004231865, 0.2203818810958583
norm = T.Normalize([MEAN], [STD])
denorm = lambda x: x * STD + MEAN

def t2np_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.squeeze().cpu().numpy()
    t = (t * STD + MEAN) * 255.0
    return t.clip(0, 255).astype('uint8')

def main():
    parser = argparse.ArgumentParser(description="Test LMFNet on test dataset")
    parser.add_argument('--test_dir', type=str, default='/home/lz/yh/mff/Moon_split/real_ce8')
    parser.add_argument('--output_dir', type=str, default='/home/lz/yh/mff/LMF-Net-master/result/new/real_ce8_1')
    parser.add_argument('--lmf_ckpt', type=str, default='/home/lz/yh/mff/LMF-Net-master/chinkpoint/CMEA.pkl')
    parser.add_argument('--bfn_ckpt', type=str, default='/home/lz/yh/mff/LMF-Net-master/chinkpoint/bfn_best.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_ds = MoonDataset(args.test_dir,
                          transform=norm,
                          need_crop=False,
                          need_augment=False)
    test_loader = DataLoader(test_ds,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)
    print(f"测试集大小: {len(test_ds)}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    lmf = LMFNet()
    ckpt_s = torch.load(args.lmf_ckpt, map_location='cpu')
    lmf.model.load_state_dict(ckpt_s)
    lmf.model.to(device).eval()
    for p in lmf.model.parameters():
        p.requires_grad = False

    bfn = BoundaryFusionNet(in_channels=1, num_blocks=6).to(device)
    bfn.load_state_dict(torch.load(args.bfn_ckpt, map_location=device))
    bfn.eval()

    subfolders = ['fused', 'band', 'pred']
    for sf in subfolders:
        os.makedirs(os.path.join(args.output_dir, sf), exist_ok=True)

    for idx, sample in enumerate(tqdm(test_loader, desc="Testing")):
        near_t = sample['near'].to(device)
        far_t = sample['far'].to(device)

        with torch.no_grad():
            near_np, far_np = t2np_uint8(near_t[0]), t2np_uint8(far_t[0])
            fused_np, band_np = lmf.fuse(near_np, far_np)

        fused_t = torch.from_numpy(fused_np.astype(np.float32) / 255.0)
        fused_t = fused_t.unsqueeze(0).unsqueeze(1).to(device)
        mask_t = torch.from_numpy(band_np.astype(np.float32))
        mask_t = mask_t.unsqueeze(0).unsqueeze(1).to(device)

        with torch.no_grad():
            x_boundary = fused_t * mask_t
            refined    = bfn(x_boundary)
            pred = fused_t * (1.0 - mask_t) + refined * mask_t

        idx_str = f"{idx+1:04d}"
        io.imsave(os.path.join(args.output_dir, 'fused', idx_str + '.png'), fused_np)
        io.imsave(os.path.join(args.output_dir, 'band', idx_str + '.png'), (band_np*255).astype(np.uint8))
        pred_vis = (pred.squeeze().cpu().numpy()*255.0).clip(0,255).astype(np.uint8)
        io.imsave(os.path.join(args.output_dir, 'pred', idx_str + '.png'), pred_vis)

    print("测试完成，结果保存在:", args.output_dir)

if __name__ == '__main__':
    main()