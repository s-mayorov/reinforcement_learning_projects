import torch
import torch.nn as nn


class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, drop_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, num_inputs*2),
            nn.ReLU(),
            nn.Linear(num_inputs*2, num_inputs*4),
            nn.ReLU(),
            nn.Linear(num_inputs*4, num_classes),
            nn.Dropout(p=drop_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    net = OurModule(2, 3).to("cuda")
    data = torch.tensor([[2.0, 3.0]], device="cuda")
    result = net(data)
    print(net)
    print(result)
