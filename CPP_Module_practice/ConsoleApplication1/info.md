# About

This file contains information that was used in the process of writing the code in the file main.cpp
Libraries that can be used:
- libtorch
- boost
- tensorflow
- Projects of VS format .NET Core (.NET Network) - but it's more about learning

At the end of the Links header, there are links to a number of sources that were used in the search and collection of information about ML libraries from the API to C++
. Part of the links are installation tutorials

# Some errors
There are some problems installing libtorch right now... But I hope to solve everything in the near future
The key problem in all this is that all ML libraries with the C++ API are poorly adapted to work under Windows. And setting all this up on Windows is like dancing with a tambourine and Russian roulette.
Therefore, in the coming days I will try to put all this in the VirtualBox VM under Linux, or if this also fails, I will run the container and configure everything there.
The most extreme option is to allocate and install an additional one (Linux) in parallel with the main OS and install and configure the required libraries there.

# Backup option
There is a tensorflow library with an API in C++
https://stackoverflow.com/questions/41070330/is-it-possible-to-use-tensorflow-c-api-on-windows
 - these are the installation instructions


# Code snippets
 ## TensorFlow
 https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305
 ```
 #COMMENT
 C++ part (inference)
Note that checkpointPath isn't a path to any of the existing files, just their common prefix. 
If you mistakenly put there path to the .index file, TF won't tell you that was wrong, 
but it will die during inference due to uninitialized variables.
 ```

 ```
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace tensorflow;

...
// set up your input paths
// это не путь до файлов - только до директории с файлами чекпоинта
const string pathToGraph = "models/my-model.meta"
const string checkpointPath = "models/my-model";
...

auto session = NewSession(SessionOptions());
if (session == nullptr) {
    throw runtime_error("Could not create Tensorflow session.");
}

Status status;

// Read in the protobuf graph we exported
MetaGraphDef graph_def;
status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
if (!status.ok()) {
    throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
}

// Add the graph to the session
status = session->Create(graph_def.graph_def());
if (!status.ok()) {
    throw runtime_error("Error creating graph: " + status.ToString());
}

// Read weights from the saved checkpoint
Tensor checkpointPathTensor(DT_STRING, TensorShape());
checkpointPathTensor.scalar<std::string>()() = checkpointPath;
status = session->Run(
        {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
        {},
        {graph_def.saver_def().restore_op_name()},
        nullptr);
if (!status.ok()) {
    throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
}

// and run the inference to your liking
auto feedDict = ...
auto outputOps = ...
std::vector<tensorflow::Tensor> outputTensors;
status = session->Run(feedDict, outputOps, {}, &outputTensors);
 ```


 Code example with interest + repository (you can clone and climb in more detail if necessary)
 https://github.com/shijungg/tensorflow-cpp-inference/tree/master

 ```
 #include <vector>
#include "tensorflow/core/public/session.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace tensorflow;
using namespace cv;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;

void convertMat2Tensor(Mat img, Tensor* output_tensor, int height, int width) {
  resize(img, img, cv::Size(height, width));
  float *p = output_tensor->flat<float>().data();
  Mat output_Mat(height, width, CV_32FC1, p);
  img.convertTo(output_Mat, CV_32FC1);
}

int main(int argc, char** argv )
{
  //Read image
  string image_path = "../digit_one.jpg";
  int height = 28;
  int width = 28;
  Mat img = imread(image_path);
  if (img.empty()) {
      cout << "ERROR: Opening image failed successfully !" << endl;
      return -1;
  }
  
  //Create input tensor and output tensor
  Tensor input_tensor(DT_FLOAT, TensorShape({1, height, width, 1}));
  convertMat2Tensor(img, &input_tensor, height, width);
  vector<Tensor> output_tensors;
  string input_node_name = "inputs";
  string output_node_name = "softmax";

  //Load model
  string model_path = "../model.pb";
  GraphDef graphdef;
  Status status_model_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
  if (!status_model_load.ok()) {
      cout << "ERROR: Loading model to graph failed successfully!" << endl; 
      cout << status_model_load.ToString() << endl;
      return -1;
  }
  cout << "INFO: Loading model to graph successfully !" << endl;

  //Create graph in session
  Session* session;
  NewSession(SessionOptions(), &session);
  Status status_create = session->Create(graphdef);
  if (!status_create.ok()) {
     cout << "ERROR: Creating graph in session failed successfully!" << endl;
     cout << status_create.ToString() << endl;
     return -1;  
  }
  cout << "INFO: Session successfully created!"<< endl;
  
  Status status_run = session->Run({{input_node_name, input_tensor}}, {output_node_name}, {}, &output_tensors);
  if (!status_run.ok()) {
      cout << "ERROR: Session run failed successfully !" << endl;
      cout << status_run.ToString() << endl;
      return -1;      
  }
  int output_dim = output_tensors[0].shape().dim_size(1);
  for (int i = 0; i < output_dim; i++) {
      cout << "Class " << i << " Prob: " << output_tensors[0].tensor<float, 2>()(0, i) << endl;

  }
  return 0;
}
 ```
OpenCV is superfluous for us in principle. Machine vision can be removed from the code. (full version on gita)
 Therefore, there will be fragments without OpenCV code that can be inserted into the program without significant changes. 
 + suitable for analyzing and studying the methods that are used for standard operations

 Loading the model (the only thing here is the .pb format in the example)
 ```
  string model_path = "../model.pb";
  GraphDef graphdef;
  Status status_model_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
  if (!status_model_load.ok()) {
      cout << "ERROR: Loading model to graph failed successfully!" << endl; 
      cout << status_model_load.ToString() << endl;
      return -1;
  }
  cout << "INFO: Loading model to graph successfully !" << endl;
 ```

  //Create graph in session
 ```
 //Create graph in session
  Session* session;
  NewSession(SessionOptions(), &session);
  Status status_create = session->Create(graphdef);
  if (!status_create.ok()) {
     cout << "ERROR: Creating graph in session failed successfully!" << endl;
     cout << status_create.ToString() << endl;
     return -1;  
  }
  cout << "INFO: Session successfully created!"<< endl;
 ```

Checking the correctness of the session
 output_tensors is filled in - it is used in data output
, In fact, the functionality is only the first line, the rest is handling exceptional situations
 ```
  Status status_run = session->Run({{input_node_name, input_tensor}}, {output_node_name}, {}, &output_tensors);
  if (!status_run.ok()) {
      cout << "ERROR: Session run failed successfully !" << endl;
      cout << status_run.ToString() << endl;
      return -1;      
  }
 ```

 ```
 int output_dim = output_tensors[0].shape().dim_size(1);
  for (int i = 0; i < output_dim; i++) {
      cout << "Class " << i << " Prob: " << output_tensors[0].tensor<float, 2>()(0, i) << endl;
  }
 ```




 ## libtorch (Torch)
### Load .ckpt model

```

torch::jit::script::Module module;

std::String filename = "model.ckpt";

try {
  // load NNs model
  module = torch::jit::load(filename);
}
catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return -1;
}

```



```

// Trying to load the previously saved model
  torch::serialize::InputArchive archive;
  std::string file("test_model.pt");
  archive.load_from(file);
  torch::nn::Sequential savedSeq;
  savedSeq->load(archive);

```


### Save data to file

https://caffe2.ai/doxygen-c/html/classes.html


```

// Trying to save the model.
        std::string model_path = "test_model.pt";
        torch::serialize::OutputArchive output_archive;
        seqConvLayer->save(output_archive);
        output_archive.save_to(model_path);

```





# Links

https://discuss.pytorch.org/t/is-there-a-way-to-read-pt-files-in-c/34392

Is there a way to read the files.pt in C++ without a python shell?

ANSWER: This is supported for PyTorch 1.0, and in this guide https://pytorch.org/tutorials/advanced/cpp_export.html the best explanation is how to load the PyTorch model in C++.

 - here the first 2 steps have already been completed on the python side 

- watch from step 3





To work with tensorflow, you need the libtorch library:

Install vcpkg according to the instructions: https://learn.microsoft.com/ru-ru/vcpkg/get_started/get-started-vs?pivots=shell-cmd

Install libtorch: vcpkg install libtorch:x64-windows


https://habr.com/ru/companies/ods/articles/480328/


Code examples:
https://discuss.pytorch.org/t/libtorch-how-to-save-model-in-mnist-cpp-example/34234/5






# Formats

Checkpoint (.ckpt) is a binary format that is used to save and restore TensorFlow models.

Checkpoint files store the values of all model parameters, as well as the state of the optimizer, so that training can be resumed from the same point.

Checkpoint files can be quite large, so they are not suitable for saving and loading models for deployment.