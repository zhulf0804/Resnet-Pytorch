import torch
import torchvision.models as models

from resnet import resnet56


if __name__ == '__main__':
    model = resnet56()
    if torch.cuda.is_available():
        model = model.cuda()

    restore_path = 'checkpoint/resnet_180.pth'
    if torch.cuda.is_available():
        net_dict = model.load_state_dict(torch.load(restore_path))
    else:
        net_dict = model.load_state_dict(torch.load(restore_path, map_location='cpu'))
    model.eval()
    
    print('start to transform pytorch model to onnx')
    dummy_input = torch.randn(1, 3, 32, 32)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model, dummy_input, 'checkpoint/model.onnx', export_params=True, 
                      opset_version=10, do_constant_folding=True, input_names=['input_x'],
                      output_names=['output_x'], dynamic_axes={'input_x': {0: 'batch_size'}, 'output_x': {0: 'batch_size'}})
    print('finished')