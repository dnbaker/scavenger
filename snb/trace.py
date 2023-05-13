import torch

model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.Softmax())
traced = torch.compile(model)

traced(torch.randn([4, 128]))
