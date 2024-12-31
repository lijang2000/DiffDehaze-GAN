import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def psnr(img1, img2, data_range=1.0):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
        data_range (float): The dynamic range of the image. For example, 255 for 8-bit images.

    Returns:
        float: PSNR value.
    """
    # Ensure images are in float format
    img1 = img1.float()
    img2 = img2.float()

    # Compute the Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2)

    # Compute PSNR
    psnr_value = 10 * torch.log10((data_range ** 2) / mse)

    return psnr_value.item()


def compute_ssim(img1, img2, data_range=1.0):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Parameters:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
        data_range (float): The dynamic range of the image. For example, 255 for 8-bit images.

    Returns:
        float: SSIM value.
    """
    # Convert tensors to numpy arrays
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)  # [3, H, W] -> [H, W, 3]
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)  # [3, H, W] -> [H, W, 3]

    ssim_values = []
    for i in range(img1_np.shape[2]):  # Loop over channels
        ssim_channel, _ = ssim(img1_np[:, :, i], img2_np[:, :, i], data_range=data_range, full=True)
        ssim_values.append(ssim_channel)

    # Average SSIM values across channels
    ssim_value = np.mean(ssim_values)

    return ssim_value


def calculate_psnr_ssim(x_c, fake_sample, data_range=1.0):
    """
    Calculate PSNR and SSIM between two images.

    Parameters:
        x_c (torch.Tensor): Original image tensor.
        fake_sample (torch.Tensor): Generated image tensor.
        data_range (float): The dynamic range of the image.

    Returns:
        tuple: PSNR and SSIM values.
    """
    psnr_value = psnr(x_c, fake_sample, data_range)
    ssim_value = compute_ssim(x_c, fake_sample, data_range)

    return psnr_value, ssim_value


# 示例用法
if __name__ == "__main__":
    # 示例图像张量
    x_c = torch.rand(1, 1, 256, 256)  # Example original image tensor
    fake_sample = torch.rand(1, 1, 256, 256)  # Example generated image tensor

    psnr_value, ssim_value = calculate_psnr_ssim(x_c, fake_sample, data_range=1.0)
    print(f"PSNR: {psnr_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")
