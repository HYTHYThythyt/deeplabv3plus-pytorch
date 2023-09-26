import torch

from net.backbone import build_backbone

net = build_backbone('xception')
torch.save(net.state_dict(), '/home/yude/project/pretrained/netmodel.pth')
