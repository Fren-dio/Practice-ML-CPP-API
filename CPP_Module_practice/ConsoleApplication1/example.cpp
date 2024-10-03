
#include <iostream>
#include <string.h>


#include "utils.cpp"


int main(int argc, char* argv[]) 
{
	try {
		std::string fileName = "model.ckpt";

		//1. load ckpt NNs model
		//read and return module 
		torch::jit::script::Module model = readModel(fileName);

		//2. convert to torch tensor
		torch::DeviceType cpu_device_type = torch::kCPU;
		torch::Device cpu_device(cpu_device_type);
		model.to(cpu_device);

		// 3. Execute the model and turn its output into a tensor.
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor);
		torch::NoGradGuard no_grad;
		torch::Tensor output = model.forward(inputs).toTensor();

		torch::DeviceType cpu_device_type = torch::kCPU;
		torch::Device cpu_device(cpu_device_type);
		output = output.to(cpu_device);

		// 4. print tensor's content
		double tensorSize = torch::elementSize(output);
		double tensorElementSize = output.element_size()
		void* ptr = output.data_ptr();
		for (size_t i = 0; i < tensorSize/tensorElementSize; i++) {
			std::cout << *((float*)(ptr + i)) << std::endl;
		}

	}
	catch (const char* err) {
		std::cout << err << std::endl;
	}

	return 0;
}