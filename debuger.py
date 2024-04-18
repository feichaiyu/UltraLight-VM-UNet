import torch
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from thop import profile, clever_format
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    module_params = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        module_name = name.split('.')[0]
        param = parameter.numel()
        module_params[module_name] = module_params.get(module_name, 0) + param
        total_params += param

    for module_name, params in module_params.items():
        table.add_row([module_name, params])

    print(table)
    print(f"Total Params:{total_params}")
    return total_params

model_cfg = {
    'num_classes': 1,
    'input_channels': 3,
    'c_list': [8,16,24,32,48,64],
    'split_att': 'fc',
    'bridge': True,
}

model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                            input_channels=model_cfg['input_channels'], 
                            c_list=model_cfg['c_list'], 
                            split_att=model_cfg['split_att'], 
                            bridge=model_cfg['bridge'],).cuda()
model.eval()


sample = torch.randn(1, 3, 256, 256).cuda()
output = model(sample)

count_parameters(model)
flops, params = profile(model, inputs=(sample,))
flops, params = clever_format([flops, params], "%.3f")
# 打印flops和params
print('flops: ', flops)
print('params: ', params)



pass