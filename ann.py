import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 定义网络
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# 数据加载
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=128, shuffle=True)

# 测试函数
def evaluate(test_data, net, device):
    net.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.view(-1, 28*28).to(device), y.to(device)
            outputs = net(x)
            preds = outputs.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
    net.train()
    return n_correct / n_total

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)

    print("Initial accuracy:", evaluate(test_data, net, device))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 记录训练开始时间
    train_start = time.time()

    for epoch in range(2):
        for x, y in train_data:
            x, y = x.view(-1, 28*28).to(device), y.to(device)
            optimizer.zero_grad()
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, accuracy:", evaluate(test_data, net, device))

    # 记录训练结束时间
    train_end = time.time()
    print(f"Training time: {train_end - train_start:.2f} seconds")
    print(f"Training time: {train_end - train_start:.2f} seconds")

    # 记录识别开始时间
    infer_start = time.time()

    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        x = x.to(device)
        predict = torch.argmax(net(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].cpu().view(28, 28))
        plt.title("prediction: " + str(int(predict)))

    infer_end = time.time()
    print(f"Inference (recognition) time: {infer_end - infer_start:.2f} seconds")

    plt.show()

if __name__ == "__main__":
    main()
