# deepsim

This project is a prototype implementation of the paper **DeepSim: Deep Learning Code Functional Similarity** that will appear in FSE 2018. 

## Setup

*We haven't created a setup script yet. At this moment you can install those dependency packages manually.*

We tested it on Ubuntu 16.04.3 64bit. The dependency packages are listed below:
- Python 2.7
- Tensorflow 1.3 (higher version should be fine)
- Keras 2.x
- All the other packages required by the above packages

This can be easily installed through [Conda](https://anaconda.org/) and instructions on [Tensorflow](https://www.tensorflow.org). For the encoding part, we already included the WALA jar package.

## How To Run it

In order to run the tool, you need first to generate encoded matrices from Java bytecode files. We already provided the executable jar file *encoding.jar* in the folder *encoding*. Once you complied your Java source code files into a jar package, you can run the below command to generate the matrices:

```bash
./encoding.jar your-bytecode-jar-path.jar
```
The generated matrices will be stored in the folder *data* under your current working directory.

*NOTE: the default matrix size is 128. If you want to change this, just change the value of the variable fixedSize in the Encoder.java source code file.*

If you just want to have a quick try of the tool, we also provided the matrices we generated for the GCJ dataset used in our paper. They are in the dataset folder. In particular, we already stored the matrices using numpy's dump function. So you can directly read it using the below code:

```Python
file_path = "path-to-the-datafile/g4_128.npy"
dataset = np.load(open(file_path, 'r'))
X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
```
Each sample here is a in a spart format. For each 88d feature vector, we only store the indices on which the value is 1. If you want to visualize a sample (as the one in our paper), just convert it back to the normal matrix.

After getting the matrices, you can run the classification.py to train the model. By default we are running a 10-fold cross-validation experiment. *You may need to change some paths to your desired folders, since we haven't cleaned the code yet*. Feel free to tweak those super-parameters (batch size, learning rate, layer size, class weights, etc.)



If you you have any questions concerning running this tool, just post an issue here. If you think this tool is useful in your own work, please cite our paper (a bibtex style will be given later):

DeepSim: Deep Learning Code Functional Similarity. In Proceedings of the 26th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering.
