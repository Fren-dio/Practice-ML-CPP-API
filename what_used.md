
PyTorch CPP API docs: https://pytorch.org/cppdocs/



create mode object and load file to object
.ckpt format file get error
maybe need .pt format
```
	torch::jit::script::Module model;
	model = torch::jit::load(model_path);
```


try to convert modek.ckpt to .pt format by python

```
https://github.com/CBICA/BrainMaGe/blob/master/BrainMaGe/utils/convert_ckpt_to_pt.py
```
or just fine in some repos