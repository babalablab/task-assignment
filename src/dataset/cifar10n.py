# import torch
# import torchvision.transforms as transforms
# from timm import create_model
# from torch.utils.data import Dataset
# from tqdm import tqdm
# 
# 
# class Cifar10nDataset(Dataset):
#     def __init__(self, data):
#         self.data = []
#         self.device = self._device()
#         self.backbone = create_model("resnet18", pretrained=True).to(self.device)
#         self.transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#         self.preprocess(data)
# 
#     def _device(self):
#         device = ""
#         if torch.cuda.is_available():
#             device = "cuda"
#         elif torch.backends.mps.is_available():
#             device = "mps"
#         else:
#             device = "cpu"
#         return device
# 
#     def preprocess(self, data):
#         with torch.no_grad():
#             self.backbone.eval()
#             for d in tqdm(data, desc="Preprocessing images"):
#                 img = d["img"]
#                 if len(img.shape) == 1:
#                     continue
#                 assert len(img.shape) == 3, (
#                     f"expected image shape is 3, but got {img.shape}"
#                 )
#                 img = self.transform(img).unsqueeze(0)
#                 img = img.to(self.device)
#                 img_f = self.backbone(img).squeeze(0)
# 
#                 d["img"] = img_f.to("cpu").detach().numpy()
#                 self.data.append(d)
#         for i in range(len(self.data)):
#             d = self.data[i]
#             self.data[i]["system_validations"] = torch.tensor(
#                 [v for k, v in d.items() if "pred_system" in k]
#             )
#             self.data[i]["system_predictions"] = torch.tensor(
#                 [v for k, v in d.items() if "pred_label" in k]
#             )
# 
#     def __getitem__(self, index):
#         d = self.data[index]
# 
#         img = d["img"]
#         img = torch.from_numpy(img)
#         correct_label = torch.tensor([d["label"]])
#         pred_validation = d["system_validations"]
#         correct_validation = d["correct_validation"]
# 
#         n = (correct_label == pred_validation).sum()
#         # img = torch.concat((img, pred_label))
#         return {
#             "img": img,
#             "correct_label": correct_label,
#             "correct_validation": correct_validation,
#             "pred_validation": pred_validation,
#             "index": index,
#             "n": n,
#         }
# 
#     def __len__(self):
#         return len(self.data)
