import torch
import torch.nn.functional as F


# class torch_vlcs_model(torch.nn.Module):
#     def __init__(self, input_dim, n_class):
#         super(torch_vlcs_model, self).__init__()
#         self.fc1 = torch.nn.Linear(input_dim, 1024)
#         self.fc2 = torch.nn.Linear(1024, 128)
#         self.fc3 = torch.nn.Linear(128, n_class)
#
#         self.feature_extractor = [self.fc1, self.fc2]
#         self.classifier = [self.fc3]
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x


class torch_vlcs_model(torch.nn.Module):
    def __init__(self, input_dim, n_class):
        super(torch_vlcs_model, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, n_class)
        self.feature_extractor = [self.fc1, self.fc2]
        self.classifier = [self.fc3]
        self.w = self.fc3.weight

    def forward(self, x, weights=None):
        x = self.linear(x, self.fc1.weight, self.fc1.bias, weights=weights, pos='fc1')
        x = F.relu(x)
        x = self.linear(x, self.fc2.weight, self.fc2.bias, weights=weights, pos='fc2')
        x = F.relu(x)
        x = self.linear(x, self.fc3.weight, self.fc3.bias, weights=weights, pos='fc3')
        return x

    def linear(self, x, weight, bias, weights=None, pos='fc1'):
        if weights is None:
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, weights[pos+'.weight'], weights[pos+'.bias'])

