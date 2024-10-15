import glob
import random
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import tqdm

class ImageDataset(Dataset):
    def __init__(self,root,transforms_=None, unaligned=False,ct='ct',pet='pet'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, ct) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, pet) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
class FromToDataset(Dataset):
    def __init__(self,root,transforms_=None, unaligned=False,to_a='pet',from_b='ct',config=None):
        from back.config import Config
        self.config = config if config is not None else Config()
        self.load_in_memoty = config.opt.load_in_memory if config is not None else 1
        self.cmd_bytes = config.cmd_bytes if config is not None else 160
        self.root = root
        self.to_a = to_a
        self.from_b = from_b
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.unaligned = unaligned
        self.files_A = sorted([f for f in glob.glob(os.path.join(root, to_a) + '/*.*') if not f.endswith('Thumbs.db')])
        self.files_B = sorted([f for f in glob.glob(os.path.join(root, from_b) + '/*.*') if not f.endswith('Thumbs.db')])
        self.length = max(len(self.files_A), len(self.files_B))
        self.data_A = []
        self.data_B = []
        if self.load_in_memoty != 0:
            
            # to_a_data = sorted(glob.glob(os.path.join(root, to_a) + '/*.*'))
            # from_b_data = sorted(glob.glob(os.path.join(root, from_b) + '/*.*'))
            path= '\\'+root+'\\'+to_a
            to_a_loop=tqdm.tqdm(enumerate(self.files_A),total=len(self.files_A),colour='green',ncols=self.cmd_bytes)
            for idx,file in to_a_loop:
                
                # 尝试打开图像文件
                try:
                    with Image.open(file) as img:
                        # print(f"Processing {file}")
                        self.data_A.append(self.transform(img.convert('RGB')))
                except IOError:
                    print(f"Skipping non-image file: {file}")
                to_a_loop.set_postfix_str(f" Loading all images of `{path:>20}` into memory. {idx+1:08d}/{self.length:08d}")
            
            path= '\\'+root+'\\'+from_b
            from_b_loop=tqdm.tqdm(enumerate(self.files_B),total=len(self.files_B),colour='yellow',ncols=self.cmd_bytes)
            for idx,file in from_b_loop:
                # 尝试打开图像文件
                try:
                    with Image.open(file) as img:
                        # print(f"Processing {file}")
                        self.data_B.append(self.transform(img.convert('RGB')))
                except IOError:
                    print(f"Skipping non-image file: {file}")
                from_b_loop.set_postfix_str(f"Loading all images of `{path:>20}` into memory. {idx+1:08d}/{self.length:08d}")
            # self.data_A = [self.transform(Image.open(file).convert('RGB')) for file in sorted(glob.glob(os.path.join(root, to_a) + '/*.*'))]
            # self.data_B = [self.transform(Image.open(file).convert('RGB')) for file in sorted(glob.glob(os.path.join(root, from_b) + '/*.*'))]
            

    def __getitem__(self, index):
        if self.load_in_memoty != 0:
            item_A = self.data_A[index % len(self.data_A)]
            if self.unaligned:
                item_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
            else:
                item_B = self.data_B[index % len(self.data_B)]
            
            return {'A': item_A, 'B': item_B}
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
            if self.unaligned:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
            return max(len(self.files_A), len(self.files_B))
        
    def save_to_path(self, batch_size=8, save_dir='ds_output'):
        if self.load_in_memoty == 0:
            raise ValueError("This method is only implemented for datasets loaded into memory.")
        
        opt = self.config.opt
        save_dir = os.path.join(save_dir, opt.time_stamp)
        save_dir = os.path.join("samples/", save_dir)
        save_dir_A = os.path.join(save_dir, f"{self.to_a}/")
        save_dir_B = os.path.join(save_dir, f"{self.from_b}/")
        path_A = '\\'+self.root+'\\'+self.to_a
        path_B = '\\'+self.root+'\\'+self.from_b
        if not os.path.exists(save_dir):
            os.makedirs(save_dir,exist_ok=True)
            os.makedirs(save_dir_A,exist_ok=True)
            os.makedirs(save_dir_B,exist_ok=True)


        from torchvision.utils import save_image, make_grid
        
        # Iterate through the dataset in batches
        num_batches = len(self.data_A) // batch_size + (1 if len(self.data_A) % batch_size > 0 else 0)
        for i in range(num_batches):
            batch_A = self.data_A[i * batch_size:(i + 1) * batch_size]
            batch_B = self.data_B[i * batch_size:(i + 1) * batch_size]
            
            # Convert lists of tensors to a single tensor
            batch_A = torch.stack(batch_A)
            batch_B = torch.stack(batch_B)
            
            from Base_Method_py import de_norm
            batch_A = de_norm(batch_A)
            batch_B = de_norm(batch_B)
            
            # Create a grid of images
            grid_A = make_grid(batch_A, nrow=batch_size)
            grid_B = make_grid(batch_B, nrow=batch_size)
            
            # Save the grids to files
            save_image(grid_A, os.path.join(save_dir_A, f"A_{i + 1:08d}.png"))
            save_image(grid_B, os.path.join(save_dir_B, f"B_{i + 1:08d}.png"))
        print((f"Already save all images of `{path_A:>10}` into disk `{save_dir_A}`."))
        print((f"Already save all images of `{path_B:>10}` into disk `{save_dir_B}`."))

def FromToDataset_Entrance_1():
    # Configure dataloaders
    import torch
    import torchvision
    
    transforms = torchvision.transforms
    transforms_ = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    # ds_train = FromToDataset('dataset',transforms_=transforms_,to_a='ct-train',from_b='pet-train')
    ds_test = FromToDataset('dataset',transforms_=transforms_,to_a='ct-test',from_b='pet-test')
    
    ds_test.save_to_path()
    pass

class FromToDatasetInTensor(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, to_a='pet', from_b='ct'):
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.unaligned = unaligned

        # Load all images into memory
        print(f"Load all images of {from_b} and {to_a} of `{root}` into memory. Please wait ...")
        self.data_A = [self.transform(Image.open(file).convert('RGB')) for file in sorted(glob.glob(os.path.join(root, to_a) + '/*.*'))]
        self.data_B = [self.transform(Image.open(file).convert('RGB')) for file in sorted(glob.glob(os.path.join(root, from_b) + '/*.*'))]

    def __getitem__(self, index):
        item_A = self.data_A[index % len(self.data_A)]
        
        if self.unaligned:
            item_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            item_B = self.data_B[index % len(self.data_B)]
        
        return {'A': item_A, 'B': item_B}
        

    def __len__(self):
        return max(len(self.data_A), len(self.data_B))


class ImageDataset1(Dataset):
    def __init__(self,root,transforms_=None, unaligned=False,ct='ct',pet='pet'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, ct) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, pet) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'A_path':self.files_A[index % len(self.files_A)],'B': item_B, 'B_path':self.files_B[index % len(self.files_B)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDatasetV2(Dataset):
    def __init__(self,root,transforms_=None, unaligned=False,ct='ct',pet='pet',ctzz='ctzz'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, ct) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, pet) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, ctzz) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_C = self.transform(Image.open(self.files_C[index % len(self.files_A)]).convert('RGB'))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'B': item_B,'C':item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDatasetV3(Dataset):
    def __init__(self,root,transforms_=None, unaligned=False,ct='ct'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, ct) + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        return item_A

    def __len__(self):
        return len(self.files_A)

class ImageDatasetV4(Dataset):
    def __init__(self,root,transforms_=None, unaligned=False,ct='ct'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, ct) + '/*.*'))
        i=0
        for name in self.files_A:
            print(name)
            directory=name.split("\\")[0]+"\\"+name.split("\\")[1]
            new_file_path = os.path.join(directory, "pet_%d.png" %i)
            print(new_file_path)
            i=i+1
            os.rename(name, new_file_path)

if __name__ == "__main__":
    FromToDataset_Entrance_1()
    pass