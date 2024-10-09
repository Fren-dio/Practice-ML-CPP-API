Connecting the libtorch library in Visual Studio:
1. Install libtorch https://pytorch.org/get-started/locally /

![alt text](2024-10-09_13-39-03.png)

I use a debug version

2. Open Visual Studio, create a Console Application

3. Next, go to the project settings: Project -> Console Application Properties

	3.1. C/C++ -> general -> additional directories - specify the path to the libtorch/include directory of the library installed earlier

	3.2. C/C++ -> general -> language - check that the ISO C++17 Standard is installed (/std:c++17)

	3.3. Linker -> general -> additional library directories - specify the path to libarch/lib

	3.4. Linker -> input -> additional dependencies - go to libarch/lib, specify all files with the extension in additional dependencies.lib 

	3.5. Go to the lib torch/bin directory - copy the file from there to the project directory.



This completes the pre-setup.

In the project file (I have this ConsoleApplication1.cpp ) include libtorch

```
#include <torch/script.h>
```


4. We are building a project

5. We catch a huge number of warnings (ignore them)

6. Launch the debugger.

7. We get errors (We read the error description (I had them all like "I couldn't find such and such .dll file. Try reinstalling..."))

	(or we don't get it if all the files were found)

	Of course, we will not reinstall anything.

	Go to libtorch/lib, find the file that was not found during debugging (and it is there), copy it and add it to the project directory

8. Run the debugger again

	Now everything is running smoothly.