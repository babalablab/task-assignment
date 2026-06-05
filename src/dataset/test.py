# from cifar import CIFAR10
# from cifar10n import Cifar10nDataset
# 
# 
# def main():
#     system_noise = "worse_label"
#     crowd_noise = "random_label2"
#     noise_path = "./data/CIFAR-10_human.pt"
# 
#     train_dataset = CIFAR10(
#         root="~/data/",
#         download=True,
#         train=True,
#         system_noise=system_noise,
#         crowd_noise=crowd_noise,
#         noise_path=noise_path,
#     )
#     train, test = train_dataset()
#     train_dataset = Cifar10nDataset(data=train)
#     print(train_dataset.__getitem__(0))
# 
# 
# if __name__ == "__main__":
#     main()
