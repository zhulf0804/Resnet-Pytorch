import numpy as np
import onnx
import onnxruntime
import time
import torch
import torchvision
import torchvision.transforms as transforms

from resnet import resnet56


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    ## 1.1 onnx check and onnx load
    # 我们可以使用异常处理的方法进行检验
    try:
        # 当我们的模型不可用时，将会报出异常
        onnx.checker.check_model('checkpoint/model.onnx')
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        # 模型可用时，将不会报出异常，并会输出“The model is valid!”
        print("The model is valid!")

    # sess = onnxruntime.InferenceSession('checkpoint/model.onnx')
    print('onnx version: ', onnxruntime.get_device())
    sess = onnxruntime.InferenceSession('checkpoint/model.onnx', providers=['CUDAExecutionProvider'])
    print(sess.get_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_inputs()[0].name


    ## 1.2 pytorch model load
    model = resnet56()
    if torch.cuda.is_available():
        model = model.cuda()

    restore_path = 'checkpoint/resnet_180.pth'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(restore_path))
    else:
        model.load_state_dict(torch.load(restore_path, map_location='cpu'))
    model.eval()


    ## 2. data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    ## 3. evaluation
    correct_pytorch, correct_onnx = 0, 0
    total = 0
    pytorch_time, onnx_time = 0, 0
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # pytorch inference
        tic_pytorch = time.time()
        outputs_pytoch = model(images)
        toc_pytorch = time.time()
        pytorch_time += (toc_pytorch - tic_pytorch)
        _, predicted_pytorch = torch.max(outputs_pytoch.data, 1)
        total += labels.size(0)
        correct_pytorch += (predicted_pytorch == labels).sum()

        # onnx inference
        tic_onnx = time.time()
        input_data = {input_name: to_numpy(images)}
        output_data = sess.run(None, input_data)
        toc_onnx = time.time()
        onnx_time += (toc_onnx - tic_onnx)
        correct_onnx += (np.argmax(output_data[0][0]) == to_numpy(labels)[0])
        # print(type(output_data), len(output_data), output_data[0].shape)


    pytorch_accuracy = correct_pytorch.double() * 1.0 / total
    pytorch_time = pytorch_time * 1.0 / total
    onnx_accuracy = correct_onnx * 1.0 / total
    onnx_time = onnx_time * 1.0 / total
    print('total number: ', total)
    print('pytorch: accuracy - {:.6f}, time - {:.6f}s'.format(pytorch_accuracy, pytorch_time))
    print('onnx: accuracy - {:.6f}, time - {:.6f}s'.format(onnx_accuracy, onnx_time))

if __name__ == '__main__':
    main()
