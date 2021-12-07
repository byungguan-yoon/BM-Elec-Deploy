import albumentations as A
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(1024, 1024),
            # A.RandomCrop(256, 256),
            A.CLAHE(always_apply=True, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(1024, 1024),
            A.CLAHE(always_apply=True, p=1),
            ToTensorV2(),
        ])

    elif data == 'None':
        return A.Compose([
            A.Resize(1024, 1024),
            ToTensorV2(),
        ])

    elif data == 'test':
        return A.Compose([
            A.CenterCrop(1900, 1900),
            A.Resize(1024, 1024),
            A.CLAHE(always_apply=True, p=1),
            ToTensorV2(),
        ])

if __name__ == '__main__':
    trans = get_transforms(data='test')
    print(trans)