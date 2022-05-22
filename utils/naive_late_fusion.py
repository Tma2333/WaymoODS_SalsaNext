from tasks import load_model
from data import H5Dataset
from tasks.utils import log_spherical_image_helper
from torchmetrics import JaccardIndex
from tqdm import tqdm
import torch


def get_trained_model (checkpt_lidar_path, checkpt_img_path, device):
    model_lidar = load_model(checkpt_lidar_path)
    model_img = load_model(checkpt_img_path)    
    model_lidar = model_lidar['spherical_model']
    model_img = model_img['spherical_model']
    model_lidar.eval().to(device)
    model_img.eval().to(device)
    return model_lidar, model_img


def fuse_outputs(model_img, model_lidar, x, device, weight1=0.6, weight2=0.4):
    x_lidar = x['range_image'].to(device)
    x_img = x['proj_pixel'].to(device)
    
    y_lidar = model_lidar(x_lidar)
    y_img = model_img(x_img)

    y = y_lidar*weight1 + y_img*weight2

    mask = torch.all(x_img == 0, dim=1, keepdim=True)
    masked_output = torch.where(mask, y_lidar, y)

    return masked_output, y_lidar


def late_fusion_naive_eval (checkpt_lidar_path, checkpt_img_path, device, h5_path, batch_size=30, weight_lidar=0.5, weight_img=0.5):
    model_lidar, model_img = get_trained_model(checkpt_lidar_path, checkpt_img_path, device)

    key2load = ['range_image', 'proj_pixel', 'ri1_label']
    dset = H5Dataset(h5_path, key2load)
    data_loader = torch.utils.data.DataLoader(dset, shuffle=False, batch_size=batch_size, num_workers=8)

    jaccard = JaccardIndex(num_classes=23, ignore_index=0, absent_score=1.0, reduction='elementwise_mean').to(device)

    total_mIoU_comb = 0
    total_mIoU_lidar = 0
    total_mIoU_diff = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            y, y_lidar = fuse_outputs(model_img, model_lidar, batch, device, weight_lidar, weight_img)
            label = batch["ri1_label"][:, 1, :].long().to(device)

            pred = y.argmax(dim=1)
            pred_only_lidar = y_lidar.argmax(dim=1)

            mIoU_comb = jaccard(pred, label)
            total_mIoU_comb += mIoU_comb

            mIoU_lidar = jaccard(pred_only_lidar, label)
            total_mIoU_lidar += mIoU_lidar

            total_mIoU_diff +=  mIoU_comb-mIoU_lidar
        
        mean_mIoU_comb = total_mIoU_comb/len(data_loader)
        mean_mIoU_lidar = total_mIoU_lidar/len(data_loader)
        mean_mIoU_diff = total_mIoU_diff/len(data_loader)

    return mean_mIoU_comb, mean_mIoU_lidar, mean_mIoU_diff