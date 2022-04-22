import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy

from .randaug import RandAugment

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label



# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     root = "E:/Project/Datasets/CUB_200_2011/CUB_200_2011/datas/"
#     train_root = root + "train/"
#     test_root = root + "test/"
#     train_set = GridDataset(train_root, 224, 32, 8, 8)
#     train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=4, num_workers=2)
#     test_set = GridDataset(test_root, 224, 32, 8, 8)
#     test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=4, num_workers=2)

#     for datas, sub_datas, labels, sub_boundarys in train_loader:
#         print("datas.size(): ", datas.size())
#         print("labels.size() :", labels.size())
#         _, _, c, h, w =  sub_datas.size()
#         sub_datas = sub_datas.view(-1, c, h, w)
#         print("sub_datas.size(): ", sub_datas.size())
#         sub_labels = labels.unsqueeze(0).repeat(64, 1)
#         sub_labels = sub_labels.transpose(0, 1).flatten().view(-1, 1)
#         print("sub_labels.size() :", sub_labels.size())

#         # show entir image
#         plt.imshow(datas[0].permute(1, 2, 0).numpy()[:, :, ::-1].astype(np.float32))
#         plt.show()

#         rows = None
#         for i in range(8):
#             cols = None
#             for j in range(8):
#                 sub_arr = sub_datas[i*8 + j].permute(1, 2, 0).numpy()[:, :, ::-1].astype(np.float32)
#                 if cols is None:
#                     cols = copy.deepcopy(sub_arr)
#                 else:
#                     cols = np.concatenate((cols, copy.deepcopy(sub_arr)), axis=1)

#             if rows is None:
#                 rows = cols
#             else:
#                 rows = np.concatenate((rows, copy.deepcopy(cols)), axis=0)

#         plt.imshow(rows)
#         plt.show()
#         break