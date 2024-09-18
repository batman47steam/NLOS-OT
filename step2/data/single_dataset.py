import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageOps
import re
import pdb

class SingleDataset(BaseDataset): # dataset部分还是挺简单的，就是两张图片对，也没什么其他的数据增强
    def initialize(self, opt):
        if opt.phase=='train':
            self.opt = opt
            self.root = opt.datarootTarget
            self.dir_B = os.path.join(opt.datarootTarget)
            self.dir_A = os.path.join(opt.datarootData)
            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)
            self.A_paths = sorted(self.A_paths) # 反正都用相同的规则进行排序了，所以就能构成这些数据对
            self.B_paths = sorted(self.B_paths)


            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list) # data/custom_dataset_data_loader.py(21)CreateDataset
        elif opt.phase == 'val':
            self.opt = opt
            self.root = opt.datarootTarget
            self.dir_B = os.path.join(opt.datarootValTarget)
            self.dir_A = os.path.join(opt.datarootValData)
            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)
            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)


            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list) # data/custom_dataset_data_loader.py(21)CreateDataset
        elif opt.phase == 'test':
            self.opt = opt
            self.root = opt.datarootData
            self.dir_A = os.path.join(opt.datarootData)
            self.A_paths = make_dataset(self.dir_A)
            self.A_paths = sorted(self.A_paths)

            self.dir_B = os.path.join(opt.datarootTarget)
            self.B_paths = make_dataset(self.dir_B)
            self.B_paths = sorted(self.B_paths)
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        u1, u2 = self.find_unmatched_paths(self.A_paths, self.B_paths)

    def __getitem__(self, index):
        # load input images
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((256, 256), Image.BICUBIC)
        A_img = self.transform(A_img)

        # load gt
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        B_img = B_img.resize((256, 256), Image.BICUBIC)
        B_img = self.transform(B_img)

        return {'A': A_img, 'A_paths': A_path,'B': B_img, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'

    def find_unmatched_paths(self, paths1, paths2):
        # 获取路径列表中的文件名
        filenames1 = [os.path.basename(path) for path in paths1]
        filenames2 = [os.path.basename(path) for path in paths2]

        # 找到两个列表中不同的部分
        unmatched1 = set(filenames1) - set(filenames2)
        unmatched2 = set(filenames2) - set(filenames1)

        print(unmatched1, unmatched2)

        return unmatched1, unmatched2
