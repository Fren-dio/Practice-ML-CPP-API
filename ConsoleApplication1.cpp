﻿


#include <iostream>
#include <string.h>
#include <vector>
#include <torch/script.h>




bool is_file_exist(std::string fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}




int main(int argc, char* argv[])
{
	try {

		if (argc != 2) {
			std::cerr << "usage: predict <path-to-exported-script-module> > \n";
			return -1;
		}

		std::string model_path = argv[1];

		//for CPP part of code need .pt file's format
		if (is_file_exist(model_path))
			std::cout << "file wasn't found\n";
		else {
			std::cout << "file was successfull found\n";
			//1. load ckpt NNs model
			//read model from ckpt file 
		
			torch::jit::script::Module model;
			model = torch::jit::load(model_path);
			torch::DeviceType cpuDeviceType = torch::kCPU;
			torch::Device cpuDevice(cpuDeviceType);
			model.to(cpuDevice);
			std::cout << "torch::jit::script::Module element was created\n";
		
			//2. convert to torch tensor
			torch::DeviceType cpu_device_type = torch::kCPU;
			torch::Device cpu_device(cpu_device_type);
			model.to(cpu_device);
			std::cout << "model was convert to torch tensor\n";
		

			// 3. create input data for testing loaded model
			// Create a vector of inputs.
			std::vector<torch::jit::IValue> inputs;
			// now it is just random values, but in future must be input data from other file
			inputs.push_back(torch::ones({ 1, 4, 45, 223 })); 

			// Execute the model and turn its output into a tensor.
			// start pre-learning model, which was loaded before
			at::Tensor output = module.forward(inputs).toTensor();
			std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
		}		

	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	catch (const char* err) {
		std::cout << err << std::endl;
	}

	return 0;
}






//#include <torch/torch.h>
//#include "torch.h"
//#include <iostream>

/*template <typename T>
void pretty_print(const std::string& info, T&& data) {
    std::cout << info << std::endl;
    std::cout << data << std::endl << std::endl;
}
*/

//int main() {
    //std::cout << "hello";
    // Create an eye tensor
    //torch::Tensor tensor = torch::eye(3);
    //pretty_print("Eye tensor: ", tensor);

    // Tensor view is like reshape in numpy, which changes the dimension representation of the tensor
    // without touching its underlying memory structure.
    //tensor = torch::range(1, 9, 1);
    //pretty_print("Tensor range 1x9: ", tensor);
    //pretty_print("Tensor view 3x3: ", tensor.view({ 3, 3 }));
    //pretty_print("Tensor view 3x3 with D0 and D1 transposed: ", tensor.view({ 3, 3 }).transpose(0, 1));
    //tensor = torch::range(1, 27, 1);
    //pretty_print("Tensor range 1x27: ", tensor);
    //pretty_print("Tensor view 3x3x3: ", tensor.view({ 3, 3, 3 }));
    //pretty_print("Tensor view 3x3x3 with D0 and D1 transposed: ",
    //    tensor.view({ 3, 3, 3 }).transpose(0, 1));
    //pretty_print("Tensor view 3x1x9: ", tensor.view({ 3, 1, -1 }));
//}