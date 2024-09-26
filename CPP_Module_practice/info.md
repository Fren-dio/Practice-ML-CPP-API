# About

Этот файл содержит информацию, которая использовалась в процессе написания кода в файле main.cpp
Библиотеки, которые можно использовать:
- libtorch
- boost
- tensorflow
- Проекты VS формата .NET Core (.NET Network) - но это больше именно про обучение

В конце заголовок Links - там ссылки на ряд источников, которые использваолись при поиске и сборе информации о ML библиотеках с API на С++
часть ссылок - туториалы по установке

# Some errors
Сейчас есть некоторые проблемы при установке libtorch... Но в ближайшее время надеюсь все решить
Ключевая проблема во всем этом - все библиотеки ML с API на С++ слабо приспособлен для работы под Windows. И настройка всего этого на винде напоминает танцы с бубном и русскую рулетку.
Поэтому в ближайшие дни буду пытаться ставить все это в VM Virtual Box под Linux-ом или, если это тоже не увенчачется успехом - буду запускать конетйнер и настраивать все там.
Самый крайний вариант - выделение и установка параллельно с основной ОС дополнительной (Linux) и установка и наладка требуемых библиотек там.


# Запасной вариант
Есть библиотека tensorflow с API на С++
https://stackoverflow.com/questions/41070330/is-it-possible-to-use-tensorflow-c-api-on-windows
 - это инструкции по установке


 # Фрагменты кода
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


 Пример кода с инференсом + репозиторий (можно склонировать и более подробно полазить, если будет нужно)
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
 OpenCv - для нас вприниципе лишнее. Машинное зрение можно из кода убрать. (полный вариант на гите)
 Поэтому дальше будут фрагменты без кода без OpenCv, которые можно вставить в программу без значительных изменений 
 + подойдут для разбора и изучения методов, которые используются для стандартных операций

 Загрузка модели (единственное, здесь .pb формат в примере)
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

 Проверка корректности сессии
 Заполняется output_tensors - он используется в выводе данных
 По сути функционал - только первая строка, остальное обработка исключительных ситуаций
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

Есть ли способ прочитать файлы .pt на C ++ без оболочки python?

ANSWER: Это поддерживается для PyTorch 1.0, и в этом руководстве https://pytorch.org/tutorials/advanced/cpp_export.html лучше всего объясняется, как загрузить модель PyTorch на C ++.

 - здесь первые 2 шага уже выполнены на стороне python 

- смотреть с шага 3





Для работы с tensorflow нужна библиотека libtorch:

Установить vcpkg по инструкции: https://learn.microsoft.com/ru-ru/vcpkg/get_started/get-started-vs?pivots=shell-cmd

Установить libtorch:  vcpkg install libtorch:x64-windows


https://habr.com/ru/companies/ods/articles/480328/


Примеры кода:
https://discuss.pytorch.org/t/libtorch-how-to-save-model-in-mnist-cpp-example/34234/5






# Formats

Checkpoint (.ckpt) — это двоичный формат, который используется для сохранения и восстановления моделей TensorFlow.

Файлы checkpoint сохраняют значения всех параметров модели, а также состояние оптимизатора, чтобы можно было возобновить обучение с той же точки.

Checkpoint-файлы могут быть довольно большими, поэтому они не подходят для сохранения и загрузки моделей для развёртывания.