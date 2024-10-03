



// 1. Read model
torch::jit::script::Module readModel(std::string fileName) 
{

    torch::jit::script::Module model = torch::jit::load(fileName);

    torch::DeviceType cpuDeviceType = torch::kCPU;
    torch::Device cpuDevice(cpuDeviceType);

    model.to(cpuDevice);
    

    return model;
}