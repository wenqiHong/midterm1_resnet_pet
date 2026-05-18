import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import types

# --- 1. 核心修复：定义独立的 Transform 包装器 ---
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# --- 2. 参数配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 37
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR_BACKBONE = 1e-5  # 预训练层极小学习率
LR_HEAD = 1e-3      # 新分类层学习率

# --- 3. 数据集准备 (训练/验证/测试三路分离) ---
def get_data_loaders():
    # 基础 Transform
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载原始数据
    raw_trainval = datasets.OxfordIIITPet(root='./data', split='trainval', download=True)
    raw_test = datasets.OxfordIIITPet(root='./data', split='test', download=True)

    # 严格 8:2 划分验证集
    train_idx, val_idx = train_test_split(
        range(len(raw_trainval)), test_size=0.2, stratify=raw_trainval._labels, random_state=42
    )

    # 使用包装器分配独立 Transform，解决 PIL Image 报错
    train_set = TransformedDataset(Subset(raw_trainval, train_idx), transform=train_tf)
    val_set = TransformedDataset(Subset(raw_trainval, val_idx), transform=val_tf)
    test_set = TransformedDataset(raw_test, transform=val_tf)

    params = {'batch_size': BATCH_SIZE, 'num_workers': 2, 'pin_memory': True}
    
    return (DataLoader(train_set, shuffle=True, **params),
            DataLoader(val_set, shuffle=False, **params),
            DataLoader(test_set, shuffle=False, **params))

# --- 4. 标准 SE 注意力模块 ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def inject_se_standard(model):
    """ 在 ResNet 的 BasicBlock 中间插入 SE 模块 """
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            # 1. 添加 SE 层
            channels = block.bn2.num_features
            block.add_module("se", SEBlock(channels))
            
            # 2. 覆盖 forward 函数，将 SE 插入 Add 之前
            def forward_with_se(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.se(out)  # <--- 标准插入点
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
                return self.relu(out)
            
            block.forward = types.MethodType(forward_with_se, block)
    return model
def run_train(model, name, is_fine_tune, loaders):
    train_ld, val_ld, test_ld = loaders
    wandb.init(project="CV_HW2_Final", name=name)
    model = model.to(DEVICE)
    
    # 严格分层学习率
    if is_fine_tune:
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': LR_BACKBONE},
            {'params': model.fc.parameters(), 'lr': LR_HEAD}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR_HEAD)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # ====================== 保存最优模型 ======================
    best_acc = 0.0
    best_model_path = f"best_model_{name}.pth"
    # ==========================================================

    for epoch in range(NUM_EPOCHS):
        # ====================== 训练 ======================
        model.train()
        total_train_loss = 0.0  # 新增：记录训练 loss
        pbar = tqdm(train_ld, desc=f"{name} Ep {epoch+1}")
        
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * imgs.size(0)  # 累计 loss
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        
        avg_train_loss = total_train_loss / len(train_ld.dataset)  # 平均训练 loss
        scheduler.step()

        # ====================== 验证 ======================
        model.eval()
        total_val_loss = 0.0  # 新增：记录验证 loss
        correct, total = 0, 0
        
        with torch.no_grad():
            for imgs, lbls in val_ld:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                total_val_loss += loss.item() * imgs.size(0)
                
                preds = outputs.argmax(1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        
        avg_val_loss = total_val_loss / len(val_ld.dataset)  # 平均验证 loss
        val_acc = correct / total

        # ====================== wandb 画图（核心修改） ======================
        wandb.log({
            "Train Loss": avg_train_loss,       # 训练 loss
            "Val Loss": avg_val_loss,           # 验证 loss
            "Val Acc": val_acc,                # 验证精度
            "LR": optimizer.param_groups[0]['lr']
        })
        # ==================================================================

        print(f" -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" ✅ 最优模型已保存：{best_model_path}")

    # 最终测试
    model.eval()
    t_correct = 0
    with torch.no_grad():
        for imgs, lbls in test_ld:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            t_correct += (model(imgs).argmax(1) == lbls).sum().item()
    print(f"== {name} Final Test Acc: {t_correct/len(test_ld.dataset):.4f} ==")
    
    wandb.finish()
# def run_train(model, name, is_fine_tune, loaders):
#     train_ld, val_ld, test_ld = loaders
#     wandb.init(project="CV_HW2_Final", name=name)
#     model = model.to(DEVICE)
    
#     # 严格分层学习率
#     if is_fine_tune:
#         optimizer = optim.Adam([
#             {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': LR_BACKBONE},
#             {'params': model.fc.parameters(), 'lr': LR_HEAD}
#         ])
#     else:
#         optimizer = optim.Adam(model.parameters(), lr=LR_HEAD)
    
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     criterion = nn.CrossEntropyLoss()

#     # ====================== 新增：保存最优模型 ======================
#     best_acc = 0.0  # 记录最好的验证集精度
#     best_model_path = f"best_model_{name}.pth"
#     # ===============================================================

#     for epoch in range(NUM_EPOCHS):
#         model.train()
#         pbar = tqdm(train_ld, desc=f"{name} Ep {epoch+1}")
#         for imgs, lbls in pbar:
#             imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#             optimizer.zero_grad()
#             loss = criterion(model(imgs), lbls)
#             loss.backward()
#             optimizer.step()
#             pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        
#         scheduler.step()
        
#         # 验证
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for imgs, lbls in val_ld:
#                 imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#                 preds = model(imgs).argmax(1)
#                 correct += (preds == lbls).sum().item()
#                 total += lbls.size(0)
        
#         val_acc = correct / total
#         wandb.log({"Val_Acc": val_acc, "LR": optimizer.param_groups[0]['lr']})
#         print(f" -> Val Acc: {val_acc:.4f}")

#         # ====================== 新增：保存最优模型 ======================
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f" ✅ 最优模型已保存：{best_model_path} (最佳精度: {best_acc:.4f})")
#         # ===============================================================

#     # 最终测试
#     model.eval()
#     t_correct = 0
#     with torch.no_grad():
#         for imgs, lbls in test_ld:
#             imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#             t_correct += (model(imgs).argmax(1) == lbls).sum().item()
#     print(f"== {name} Final Test Acc: {t_correct/len(test_ld.dataset):.4f} ==")
    
#     wandb.finish()
if __name__ == '__main__':
    loaders = get_data_loaders()
    
    # 实验一: Baseline (预训练 + 分层 LR)
    print("\n--- 启动实验 1: Baseline ---")
    m1 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m1.fc = nn.Linear(512, NUM_CLASSES)
    run_train(m1, "Baseline", True, loaders)

    # 实验二: 消融 (从零开始)
    print("\n--- 启动实验 2: Ablation ---")
    m2 = models.resnet18(weights=None)
    m2.fc = nn.Linear(512, NUM_CLASSES)
    run_train(m2, "Ablation", False, loaders)

    # 实验三: 注意力 (标准 SE + 分层 LR)
    print("\n--- 启动实验 3: SE-Attention ---")
    m3 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m3 = inject_se_standard(m3)
    m3.fc = nn.Linear(512, NUM_CLASSES)
    run_train(m3, "SE-Attention", True, loaders)