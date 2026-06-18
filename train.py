import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time

from dataset import ElectrodeDataset
from model import DS_MIN
from loss import BoundaryAwareJointLoss
from tqdm import tqdm


def train_one_fold(train_files, data_dir, image_dir, save_path,
                   num_epochs=100, batch_size=1, learning_rate=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_wall_start = time.time()

    # 传入当前折的训练文件
    train_dataset = ElectrodeDataset(data_dir, image_dir, is_train=True)
    train_dataset.file_list = train_files

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = DS_MIN(num_classes=5, d_k=64, num_layers=3).to(device)
    criterion = BoundaryAwareJointLoss(lambda_weight=0.5, sigma=2.0, gamma=2.0, tau=5.0).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float("inf")
    peak_mem_gb = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"第 {epoch + 1}/{num_epochs} 轮", leave=False)

        for batch in pbar:
            points = batch['xyz'].to(device)
            targets = batch['labels'].to(device)
            dist_field = batch['distance_field'].to(device)
            images = batch['images'].to(device)
            uv_indices = batch['uv_indices'].to(device)

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad()
            logits = model(points, images, uv_indices)
            loss_total, _, _ = criterion(logits, targets, dist_field)
            loss_total.backward()
            optimizer.step()

            if device.type == 'cuda':
                peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated() / 1024**3)

            epoch_loss += loss_total.item()
            pbar.set_postfix({"损失": f"{loss_total.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    t_wall_end = time.time()
    wall_time_s = t_wall_end - t_wall_start
    print(f"[计时] 单折训练耗时: {wall_time_s:.1f}s ({wall_time_s/60:.1f}min), "
          f"峰值显存: {peak_mem_gb:.2f} GB")
    return save_path


if __name__ == "__main__":
    print("单折全数据训练模式启动")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "train_data_v2")
    image_dir = os.path.join(script_dir, "train_images_v2")

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    save_path = os.path.join(script_dir, "checkpoints", "debug_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_one_fold(
        train_files=all_files,
        data_dir=data_dir,
        image_dir=image_dir,
        save_path=save_path,
        num_epochs=100
    )
