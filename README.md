# DeepSIM

This project is a prototype implementation of DeepSim, a deep learning-based approach to measure code functional similarity. If you find the tool useful in your work, please cite our FSE 2018 paper:

**"DeepSim: Deep Learning Code Functional Similarity"**. In Proceedings of the 26th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2018.

## Setup

*We haven't created a setup script yet. At this moment you can install those dependency packages manually.*

We tested it on Ubuntu 16.04.3 64bit. Our hardware environment includes:
- Intel i7 6700K, 4.0GHz
- NVIDIA GTX 1080, 8GB
- DDR4 3000MHz, 48GB

The dependency packages are listed below:
- Python 2.7
- Tensorflow 1.3 (higher version should be fine)
- Keras 2.x
- All the other packages required by the above packages

This can be easily installed through [Conda](https://anaconda.org/) and instructions on [Tensorflow](https://www.tensorflow.org). For the encoding part, we already included the WALA jar package.

## How To Run it

In order to run the tool, you need first to generate encoded matrices from Java bytecode files. We already provided the executable jar file `encoding.jar` in the folder `encoding`. Once you complied your Java source code files into a jar package, you can run the below command to generate the matrices:

```bash
./encoding.jar your-bytecode-jar-path.jar
```
The generated matrices will be stored in the folder *data* under your current working directory. (We already test this tool on a set of Java projects and it works well. If you find any crashes/errors, please post an issue here.)

*NOTE: the default matrix size is 128. If you want to change this, just change the value of the variable fixedSize in the Encoder.java source code file.*

If you just want to have a quick try of the tool, we also provided the matrices we generated for the GCJ dataset used in our paper. They are in the `dataset` folder. In particular, we already stored the matrices using numpy's dump function. So you can directly read it using the below code:

```Python
file_path = "path-to-the-datafile/g4_128.npy"
dataset = np.load(open(file_path, 'r'))
X, y = np.array(dataset['X']), np.array(dataset['y'], dtype=np.int)
```
Each sample here is a in a spart format. For each 88d feature vector, we only store the indices on which the value is 1. If you want to visualize a sample (as the one in our paper), just convert it back to the normal matrix.

After getting the matrices, you can run `classification.py` to train the model. By default we are running a 10-fold cross-validation experiment. *You may need to change some paths to your desired folders, since we haven't cleaned the code yet*. Feel free to tweak those super-parameters (batch size, learning rate, layer size, class weights, etc.)

On our environment, each run of the 10-fold takes nearly 3.75 hours. If you are running it using a weaker GPU, please expect longer time to finish. If you use larger batch size, please make sure that you have enough large memory, since each sample contains 128*128*88 elements. If the result you get are different from what reported in the paper, just change the super-parameters to the values presented in the paper (if you are running on the GCJ dataset), or you can write a simple script to find your best parameter setting on your dataset.

Running the rest two baseline models are similar.

## NOTE

We are working on a set of improved models, some of them are trying to address
the limitations of this work. Hope we can finish and release them soon.

In addition, we probably will include a simple web project in this repo for collecting larger
and more comprehensive training samples (though we will not hold it on our server).