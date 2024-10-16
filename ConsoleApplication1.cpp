// read_nns_model.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <torch/script.h>

#include <iostream>

#include "npy.hpp"



bool is_file_exist(std::string fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}


int main(int argc, char* argv[])
{
	try {

		std::string model_path = "model.pt";  // here it's a argv[1]
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
		int SIZE_ARRAY = 1024;
		double* dataArray = (double*)calloc(SIZE_ARRAY, sizeof(double));

		//  read input data from 'data.npy' file and initialize array, which will use for create tensor
		// created tensor will used to start NNs on this data
		// using libnpy
		const std::string path{ "data.npy" };
		std::vector<double> data = (npy::read_npy<double>(path)).data;
		//convert vector<> to double*
		dataArray = &data[0];

		at::Tensor inputDataTensor = torch::tensor(dataArray, { torch::kFloat64 });

		inputs.push_back(inputDataTensor);
		std::cout << "input data was loaded\n";




		// 4. Execute the model and turn its output into a tensor.
		// start pre-learning model, which was loaded before
		at::Tensor output = model.forward(inputs).toTensor();
		std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	catch (const char* err) {
		std::cout << err << std::endl;
	}

}
