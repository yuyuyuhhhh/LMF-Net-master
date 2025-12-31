import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def rgb_to_y(pil_img):
    y, _, _ = pil_img.convert('YCbCr').split()
    return y


class MoonDataset(Dataset):
    def __init__(self,
                 root_dir,
                 phase='train',
                 transform=None,
                 need_crop=False,
                 need_augment=False,
                 crop_size=None,
                 return_target=True):
        self.phase = phase
        self.transform = transform
        self.need_crop = need_crop
        self.need_augment = need_augment
        self.crop_size = crop_size
        self.return_target = return_target

        def _guess(path, p):
            cand = os.path.join(path, p)
            return cand if os.path.isdir(cand) else path

        root_dir = _guess(root_dir, phase)

        self.near_dir = os.path.join(root_dir, 'near')
        self.far_dir = os.path.join(root_dir, 'far')
        self.target_dir = os.path.join(root_dir, 'full_clear')

        self.near_list = sorted(os.listdir(self.near_dir))
        self.far_list = sorted(os.listdir(self.far_dir))
        self.target_list = sorted(os.listdir(self.target_dir))

        assert len(self.near_list) == len(self.far_list) == len(self.target_list), \
            "near / far / full_clear 数量必须一致"

        if self.need_crop and self.crop_size is None:
            raise ValueError("need_crop=True 时必须指定 crop_size")

    def __len__(self):
        return len(self.near_list)

    def __getitem__(self, idx):
        n_path = os.path.join(self.near_dir, self.near_list[idx])
        f_path = os.path.join(self.far_dir, self.far_list[idx])
        t_path = os.path.join(self.target_dir, self.target_list[idx])

        near_y = rgb_to_y(Image.open(n_path).convert('RGB'))
        far_y = rgb_to_y(Image.open(f_path).convert('RGB'))
        target_y = rgb_to_y(Image.open(t_path).convert('RGB')) if self.return_target else None

        if self.need_augment and self.phase == 'train':
            if random.random() < 0.5:
                near_y = TF.hflip(near_y)
                far_y = TF.hflip(far_y)
                if target_y is not None: target_y = TF.hflip(target_y)
            if random.random() < 0.5:
                near_y = TF.vflip(near_y)
                far_y = TF.vflip(far_y)
                if target_y is not None: target_y = TF.vflip(target_y)

        if self.need_crop:
            if isinstance(self.crop_size, int):
                ch, cw = self.crop_size, self.crop_size
            else:
                ch, cw = self.crop_size

            if self.phase == 'train':
                i, j, h, w = transforms.RandomCrop.get_params(near_y, (ch, cw))
            else:
                W, H = near_y.size
                i = max(0, (H - ch) // 2)
                j = max(0, (W - cw) // 2)
                h, w = ch, cw

            near_y = TF.crop(near_y, i, j, h, w)
            far_y = TF.crop(far_y, i, j, h, w)
            if target_y is not None:
                target_y = TF.crop(target_y, i, j, h, w)

        near_y_t = TF.to_tensor(near_y)
        far_y_t = TF.to_tensor(far_y)
        if target_y is not None:
            tgt_y_t = TF.to_tensor(target_y)

        if self.transform is not None:
            near_y_t = self.transform(near_y_t)
            far_y_t = self.transform(far_y_t)
            if target_y is not None:
                tgt_y_t = self.transform(tgt_y_t)

        sample = {
            'near': near_y_t,
            'far': far_y_t,
            'name': self.near_list[idx]
        }
        if self.return_target:
            sample['full_clear'] = tgt_y_t

        return sample

    def get_with_target(self, idx):
        bak = self.return_target
        self.return_target = True
        out = self.__getitem__(idx)
        self.return_target = bak
        return out
