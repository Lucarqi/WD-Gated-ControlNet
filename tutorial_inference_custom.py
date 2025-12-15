import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import torch.multiprocessing as mp 
import albumentations


from share import * 
from cldm.model import create_model, load_state_dict


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

pl.seed_everything(42, workers=True)


class MedicalInferenceDataset(Dataset):
    def __init__(self, root, size=384, classes=3):
        self.root = root
        self.size = size
        self.classes = classes
        self.max_label = classes - 1
        
        self.images_dir = os.path.join(root, 'imagesTr')
        self.labels_dir = os.path.join(root, 'labelsTr')
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.nii.gz')])
        self.label_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith('.nii.gz')])
        
        assert len(self.image_files) == len(self.label_files)


        self.transform = albumentations.Compose([
            albumentations.Resize(height=self.size, width=self.size)
        ])

    def __len__(self):
        return len(self.image_files)

    def _direct_to_grayscale(self, mask_2d):
       
        if self.max_label == 0:
            gray_mask = np.zeros_like(mask_2d, dtype=np.uint8)
        else:
          
            gray_mask = (mask_2d / self.max_label * 255).astype(np.uint8)
        
        return np.stack([gray_mask] * 3, axis=-1)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.label_files[idx]
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.labels_dir, mask_name)

       
        img_itk = sitk.ReadImage(img_path)
        mask_itk = sitk.ReadImage(mask_path)
        
        img_array = sitk.GetArrayFromImage(img_itk)
        mask_array = sitk.GetArrayFromImage(mask_itk)
        
        
        if img_array.ndim == 3: img_2d = img_array[0, :, :]
        else: img_2d = img_array
            
        if mask_array.ndim == 3: mask_2d = mask_array[0, :, :].astype(np.int32)
        else: mask_2d = mask_array.astype(np.int32)

        
        img_min, img_max = img_2d.min(), img_2d.max()
        if img_max > img_min:
            
            img_2d_norm = ((img_2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_2d_norm = np.zeros_like(img_2d, dtype=np.uint8)
        
        
        img_rgb = np.stack([img_2d_norm] * 3, axis=-1)

       
        mask_processed = self._direct_to_grayscale(mask_2d)

       
        preprocess = self.transform(image=img_rgb, mask=mask_processed)
        target_np = preprocess['image'] # Image
        source_np = preprocess['mask']  # Mask

       
        target_np = target_np.astype(np.float32) / 127.5 - 1.0
        source_np = source_np.astype(np.float32) / 255.0

      
        jpg_tensor = torch.from_numpy(target_np).float() 
        hint_tensor = torch.from_numpy(source_np).float()

        prompt = "A photo of cardiac MRI"

        return {
            "jpg": jpg_tensor,
            "hint": hint_tensor,
            "txt": prompt,
            "meta_path": mask_path,
            "original_index": idx 
        }
        

def save_physically_aligned(images, meta_path, result_dir, counter, target_size, prefix):
    mask_dir = os.path.join(result_dir, "masks")
    image_dir = os.path.join(result_dir, "images")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    key = "samples_cfg_scale_9.00_mask"
    if key not in images: return
    
    # [1, C, H, W]
    gen_tensor = images[key].detach().cpu()
    
    gen_tensor = torch.mean(gen_tensor, dim=1, keepdim=True) 

   
    template = sitk.ReadImage(meta_path)
    
    original_w, original_h = template.GetSize()[:2]
    
    mask_out_name = f"{prefix}_{counter}.nii.gz"
    sitk.WriteImage(template, os.path.join(mask_dir, mask_out_name))

  
    gen_resized = F.interpolate(
        gen_tensor, 
        size=(original_h, original_w), 
        mode='bicubic', 
        align_corners=False
    )
    
  
    gen_np = (gen_resized.squeeze().numpy() + 1.0) / 2.0 * 255.0

    gen_np = np.clip(gen_np, 0, 255).astype(np.float32)
    

    gen_itk = sitk.GetImageFromArray(gen_np[np.newaxis, :, :])
    

    gen_itk.SetOrigin(template.GetOrigin())
    gen_itk.SetDirection(template.GetDirection())
    gen_itk.SetSpacing(template.GetSpacing())
    
    img_out_name = f"{prefix}_{counter}_0000.nii.gz"
    sitk.WriteImage(gen_itk, os.path.join(image_dir, img_out_name))


def inference_worker(rank, world_size, args):
   
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(args.ckpt_path, location='cpu'), strict=False)
    model.to(device)
    model.eval()

   
    full_dataset = MedicalInferenceDataset(args.data_root, size=args.model_size, classes=args.classes)
    total_len = len(full_dataset)
    
  
    weights = [float(w) for w in args.gpu_weights.split(',')]
    assert len(weights) == world_size, f"权重数量 ({len(weights)}) 必须等于 GPU 数量 ({world_size})"
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    split_points = (np.cumsum(weights) * total_len).astype(int)
    split_points[-1] = total_len
    split_points = np.insert(split_points, 0, 0)
    
    start_idx = split_points[rank]
    end_idx = split_points[rank+1]
    
    my_indices = list(range(start_idx, end_idx))
    
    count = len(my_indices)
    print(f"[Rank {rank} | {device}] Weight: {weights[rank]:.2f} -> Assigned {count} images (Index {start_idx} to {end_idx-1})")

    if count == 0:
        print(f"[Rank {rank}] No images assigned, exiting.")
        return

    subset = Subset(full_dataset, my_indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)

    
    with torch.no_grad():
        with model.ema_scope():
            for i, batch in enumerate(dataloader):
                original_idx = batch["original_index"].item()
                global_counter = 1001 + original_idx

                model_feed = {
                    "jpg": batch["jpg"].to(device),
                    "hint": batch["hint"].to(device),
                    "txt": batch["txt"]
                }
                
                meta_path = batch["meta_path"][0]

                images = model.log_images(model_feed, N=1, ddim_steps=50)

                save_physically_aligned(
                    images=images,
                    meta_path=meta_path,
                    result_dir=args.result_dir,
                    counter=global_counter, 
                    target_size=args.target_size,
                    prefix=args.prefix
                )

    print(f"[Rank {rank}] Finished processing {count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/Dataset501_RENJISLICE')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--result_dir', type=str, default="./infer_results")
    parser.add_argument('--prefix', type=str, default="lge")
    parser.add_argument('--target_size', type=int, default=192)
    parser.add_argument('--model_size', type=int, default=384)
    parser.add_argument('--gpus', type=int, default=3)
    
    parser.add_argument('--gpu_weights', type=str, default="1,1,1", 
                        help='Relative weights for each GPU, separated by comma. E.g., "1,1.5,1.2"')
    parser.add_argument('--classes', type=int, default=3, help='Number of classes for inference, including background')
    
    args = parser.parse_args()

    world_size = args.gpus
    
    print(f"Starting inference on {world_size} GPUs with weights: {args.gpu_weights}...")
    
    mp.spawn(
        inference_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
    

    print("All processes completed.")
