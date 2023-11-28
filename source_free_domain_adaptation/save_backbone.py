from collections import OrderedDict

import torch


# ckpt_path = 'log/iresnet100_MFR1/FDA-129/epoch=8_step=0.00_loss=0.279.ckpt'
# backbone_path = 'pretrained/backbone/unmasked_backbone.pth'
# state_dict = OrderedDict()
# fc = OrderedDict()
# for k, v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
#     if 'backbone' in k:
#         state_dict['.'.join(k.split('.')[1:])] = v
#     elif 'fc' in k:
#         print(k)
#         fc['.'.join(k.split('.')[1:])] = v
#
# with open(backbone_path, 'wb') as f:
#     torch.save(state_dict, f)

# with open(fc_path, 'wb') as f:
#     torch.save(fc, f)

state_dict = OrderedDict()
for k, v in torch.load('pretrained/transda_typ.pt', map_location='cpu').items():
    print(k, v.shape)
    if 'feature' in k:
        state_dict['.'.join(k.split('.')[1:])] = v


with open('pretrained/backbone/transda_typ.pth', 'wb') as f:
    torch.save(state_dict, f)

