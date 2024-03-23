#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        // Suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// 简化错误处理
#define CHECK(status) \
    if (status != 0) \
    { \
        std::cerr << "Cuda failure: " << status << std::endl; \
        abort(); \
    }

// 读取引擎文件
std::vector<char> readEngineFile(const std::string& enginePath)
{
    // 使用 std::ifstream 打开一个文件流，以二进制模式（std::ios::binary）和读取位置指向文件末尾（std::ios::ate）的方式打开文件。
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    // 通过 file.good() 检查文件流状态是否正常，如果不正常（即文件打开失败或其他错误），则抛出一个 std::runtime_error 异常。
    if (!file.good())
    {
        throw std::runtime_error("Error reading engine file");
    }

    // 使用 file.tellg() 获取当前文件指针位置，由于文件是以 std::ios::ate 模式打开的，所以这将给出文件的大小。
    std::streamsize size = file.tellg();

    // 通过 file.seekg(0, std::ios::beg) 将文件指针重新定位到文件的开始位置。
    file.seekg(0, std::ios::beg);

    // 创建一个足够大的 std::vector<char> 来存储文件内容，其大小由步骤3中获取的文件大小确定。
    std::vector<char> buffer(size);

    // 使用 file.read(buffer.data(), size) 将文件内容读取到之前创建的向量中。buffer.data() 提供了向量内存的指针，size 是要读取的字节数。
    // 如果读取失败（即 file.read() 返回 false），则抛出一个 std::runtime_error 异常。
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Error reading engine file");
    }

    return buffer;
}

std::vector<std::string> split(const std::string &str, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, delim)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}


int main(){
    const std::string engineFilePath = "../checkpoint/model.trt";  // 替换为你的 .trt 文件路径

    // 读取序列化的引擎
    auto engineData = readEngineFile(engineFilePath);
    
    // 创建运行时和引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 分配设备内存
    void* buffers[2]; // 输入和输出缓冲区
    const int inputIndex = engine->getBindingIndex("input_x"); // 替换为你的输入层名称
    const int outputIndex = engine->getBindingIndex("output_x"); // 替换为你的输出层名称

    int inputSize = 3 * 32 * 32, outputSize = 10;
    CHECK(cudaMalloc(&buffers[inputIndex], sizeof(float) * inputSize)); // 替换 inputSize 为实际输入大小
    CHECK(cudaMalloc(&buffers[outputIndex], sizeof(float) * outputSize)); // 替换 outputSize 为实际输出大小

    // 创建 CUDA 流
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::vector<std::string> test_data_list = {
        "../data/0_3.tiff",
        "../data/1_8.tiff",
        "../data/2_8.tiff",
        "../data/3_0.tiff",
        "../data/4_6.tiff",
        "../data/5_6.tiff",
        "../data/6_1.tiff",
        "../data/7_6.tiff",
        "../data/8_3.tiff",
        "../data/9_1.tiff"
    };
    int test_number = test_data_list.size();
    for(int i = 0; i < test_number; i++){
        // 准备输入数据
        std::vector<float> input(inputSize); // 替换 inputSize 为实际输入大小

        // 使用OpenCV的cv::imread函数读取TIFF图像
        cv::Mat image = cv::imread(test_data_list[i], cv::IMREAD_UNCHANGED);
        // 检查是否成功读取图像
        if(image.empty()) {
            std::cerr << "Error: Unable to open image file." << std::endl;
            return -1;
        }
        cv::Size imgSize = image.size();
        int channels = image.channels();
        int width = image.cols;
        int height = image.rows;
        // // 拷贝图像数据到浮点数组  (h, w, c)
        // if (image.isContinuous()) {
        //     input.assign((float*)image.datastart, (float*)image.dataend);
        // } else {
        //     for (int i = 0; i < height; ++i) {
        //         std::copy(image.ptr<float>(i), image.ptr<float>(i) + width * channels, &input[i * width * channels]);
        //     }
        // }
        // 重新排列图像数据  (c, h, w)
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    input[c * (width * height) + h * width + w] = image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        // 复制输入数据到设备
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), sizeof(float) * inputSize, cudaMemcpyHostToDevice, stream));

        // 执行推理
        context->enqueue(1, buffers, stream, nullptr);

        // 复制输出数据到主机
        std::vector<float> output(outputSize); // 替换 outputSize 为实际输出大小
        CHECK(cudaMemcpyAsync(output.data(), buffers[outputIndex], sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream));


        // 同步 CUDA 流以等待任务完成
        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "dur: " << elapsed.count() * 1000 << " ms" << std::endl;

        // Find the iterator to the max element
        auto max_it = std::max_element(output.begin(), output.end());
        // Calculate the index of the max element
        int max_index = std::distance(output.begin(), max_it);
        
        std::string gt = split(split(test_data_list[i], '_')[1], '.')[0];
        // 输出结果
        std::cout << i << " ";
        for (auto val : output)
        {
            std::cout << val << " ";
        }
        std::cout << max_index << " " << gt;
        std::cout << std::endl;
    }

    // 释放资源
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}