# ISYOUCAT - FINAL REPORT

# Model training

## Dataset

Certainly we needed a lot of cat pictures to our cat detector. Both tested methods used cats from the *[Cats vs Dogs](https://www.kaggle.com/karakaggle/kaggle-cat-vs-dog-dataset)* dataset (12.5K images). For one class training that was enough. For naive one class training we needed pictures of negative class (all non-cat objects). We approximate that with *Caltech-101* (9,146 images) and *Caltech-256* datasets (30,607 images). During the first weeks of our work on models, we assumed that there are no cat-looking animals in those datasets. After getting mediocre results on real-life tests, we decided to manually clear pictures of catlike animals/objects.

## Models

### HarrCascades

These pretrained models are what we have chosen for the baseline. The results were not satisfying, but we have not tried to improve them (ie. track why they were so bad), because it was not what our projects was meant to be.

### One Class Training

Reference: [https://arxiv.org/pdf/1901.08688.pdf](https://arxiv.org/pdf/1901.08688.pdf)
The first method we tried was One Class Training. This model consists of two following parts: a feature extractor and a multi-layer perceptron. After feature extraction, the data samples are ‘corrupted’ by adding some zero-centered gaussian noise. These modified samples are then concatenated batchwise with their original pairs. The result of this approach is that we have a duplicated batch of images formed by original samples (class 1) and corrupted ones (class 0). The pretrained models we used for feature extractor were: VGG, ResNet, Google Inception, MobileNetV3, due to unknown reason the model were able to train only with VGG. 

![](https://i.imgur.com/Zx9rRf5.png)

### "Naive" One Class Training

After failing to make use of one-class approach we decided to follow more naive one ie. training using data set with positive and negative class samples instead of positive class ones only. We trained our models using transfer learning.

From the very beggining we got pretty lovely results. Thus we tested smaller and smaller models for use on our device.

Models we tried were:
- Torch-MnasNet1.0
- Torch-MobileNetV2
- Torch-MobileNetV3-large
- Torch-MobileNetV3-small
- Torch-MobileNetV3-small (2 linear layers)
- TensorFlow-MobilenetV3-small (substantially different from PyTorch one)

### Results
|                   Model                   | Accuracy |
|:-----------------------------------------:| -------- |
|             HarrCascade-large             | 67.4%    |
|             HarrCascade-small             | 65.9%    |
|                  VGG-OCT                  | 91.95%   |
|             Torch-MnasNet1.0              | 94.78%   |
|             Torch-MobileNetV2             | 96.51%   |
|          Torch-MobileNetV3-large          | 96.44%   |
|          Torch-MobileNetV3-small          | 95.34%   |
| Torch-MobileNetV3-small (2 linear layers) | 98.23%   |
|          TensorFlow-MobilenetV2           | 94.82%   |


# Model performance optimisation

## Picking suitable model
At this point we have benchmarked models in PyTorch, and the horrible performance made us choose the smallest model, MobileNetV3. We have also decided to heavliy cut the last fully connected layers.

## Pruning and Quantization
We have briefly experimented with pruning and quantization.
- Pruning. This has been covered in the class, so we'll assume brief knowledge on what pruning is, and how it works. In PyTorch, there are two options for pruning. There is:
    - Unstructured Pruning. This option is the simplest and implemented the best since you can just say how many network connections you want to have pruned out. We managed to remove up to 50% of the network* while keeping error rates reasonable. However, pruning your network requires you to switch to sparse matrix representations that absolutely kill any performance. 
    * When one is dealing with convolutional layers, 50% of the network doesn't actually mean 50% of computation. PyTorch doesn't support pruning by computation amount, so we don't know what speedup would we get if MAD (multiple-accumulate (add) operation) to MAD performance on sparse matrixes was the same as on dense ones. Pruning would make sense if we started with a network that was poorly optimised. However, at this point we were set on MobileNetV3. And that network is already heavily optimised, so we really shouldn't have expected much.
    - Structural Pruning. This is an approach where one would remove entire neurons, omitting the problems with sparse matrix representations. However, the PyTorch implementation was absolutely terrible (no option to prune entire network at once) so we gave up on it as well. We were already pretty dissappointed in this branch of optimisation, and decided to focus on other areas.

- Quantization. This is a method of converting network from `fp32` (32-bit floating point) representation to `int8` one. While this might seem like a reasonable solution at first, it's only beneficial when one needs to heavily optimize model size (not our case since we have 4GiB of ram) or is running the model on a processor without an FPU or with very little FP power (again, not our case). As the last nail to the coffin, the `int8` computation increases operation count, and while it should be faster than emulation of `fp32`, it makes absolutelly zero sense when one is using a processor with FPU.

## Model exporting
There are two proper ways of exporting a model. 
- for PyTorch, one would use torch JIT and export the model into a `.pt` file
- for TensorFlow, one would use `TFLiteConverter()` and export into a `.tflite` file

|                   Model                   | FPS    |
|:-----------------------------------------:| ------ |
|             HarrCascade-large             | 27.1  |
|             HarrCascade-small             | 27.2  |
|                  VGG-OCT                  | trash |
|             Torch-MnasNet1.0              | 0.71 |
|             Torch-MobileNetV2             | 0.76 |
|          Torch-MobileNetV3-large          | 3.10 |
|          Torch-MobileNetV3-small          | 1.08 |
|       TensorFlow-MobilenetV2 (ONNX)       | 12.7   |
|      Torch-MobileNetV3-small (ONNX)       | 26.18  |
| Torch-MobileNetV3-small (2 linear layers) (ONNX) | 25.97  |

This however, exposed a peculiar difference. MobilenetV3 implementation differed between PyTorch and TensorFlow! The PyTorch implementation x had less parameters, yet run x slower than the TensorFlow implementation. This posed a question: "could we somehow get the PyTorch model to run under a faster executor"  

The simplest approach that we considered, would be to rewrite the network from scratch in TensorFlow, and then transplant the weights from PyTorch to TensorFlow. However, at this time we have found ONNX, a tool that will let you convert a PyTorch network into a `.onnx` file. While looking at benchmarks we decided to give it a try. There were some hurdles, that required us to go into development branches of both onnx and PyTorch, but we managed to export the PyTorch model into `onnx` file.  

Also at this time, we have learned about the OpenCV/dnn module. Said module can import `.onnx` files and was the library that we have already chosen for interfacing the camera. It can also do image resizing, cropping, and conversion of `BGR2RGB` in a single function. It all looked promising, however, the DNN module had failed when importing our model - due to the presence of the `HardSigmoid` operator.  

While we have considered just giving up on this, the decision was made to patch the OpenCV library with our custom implementation of HardSigmoid. Conceptually, it's not tough. HardSigmoid is just two ReLUs stacked on top of each other. The problem was digging around in OpenCV to patch it. If one would look for the "embedded engineers are sewage divers of computer science" thesis, this would be it. While this part was really exciting, and we can't justice in a single paragraph, we decide to cut the explanation short, since it's really a minor part in the grand scheme of things.

To our suprise, the damn thing actually worked. And after launching the benchmark, we got the news we wanted.  
With our model running 25FPS, we decided that it was the time to leave the model alone. Since even without further optimisation, we have beaten our goal, almost twofold.
***
In retrospect we now know that the OpenCV DNN module allows adding a custom operator inside user client code. While the workload looks more or less the same, it's probably advised to go that route, since one doesn't need to recompile entire OpenCV if the operator is somehow broken.

## Tricks for performance boosting
At this point, we had a model that would run comfortably with 25 FPS. That's a lot, and actually way more than we hoped for initially. However, there is still a case to be made for increasing the model performance even further. While the model got 25 FPS, it did so while pushing the Raspberry Pi CPU with 400% util. This is obviously undesirable. At this point, the easiest way out of this would be to push the model even further (30 FPS) and just run it every other frame. While this is a neat solution, there are neater ones, that will run the model only if the frame changes.
- [Phase Correlation](https://docs.OpenCV.org/master/d7/df3/group__imgproc__motion.html#ga552420a2ace9ef3fb053cd630fdb4952)
  To keep this report reasonably long we won't go into detail on how frequency domain <something something> FFT bs thing works. What matters, is that it's an already implemented part of OpenCV that will basically tell you if the image has moved. Once the movement exceeds a treshold, you run the network once again.
- Network precomputation.
  We can take some prefix of the network, and run it every frame. Then, we calculate the mean square deviation between the prefices activations in order to decide if the image has changed. This way, we spend only roughly 30% of computation every frame and run the last 70% of the network if we decide that the frame changed based on model prefix activations.
  This however, proses a rather large implementation problem. OpenCV in DNN module only contains a `forward(layer)` function, that will run your network up to a given point. However, in the latest release the option to run a model from a given layer to the end has been removed. (and also, each `follow()` will remove computations done beforehand). We have debated on modifying the OpenCV source once again, however, the `follow()` function is so complicated that we have given up due to the time constraints. A less elegant solution was developed that uses a mode pre-split to two parts and exported as two ONNX files. We loose some performance since we need to perform an additional `memcpy` in the code, but it's miniscule compared to overall model complexity.

## Hardware

To run our real-time cat detection app we use Raspberry Pi 4 put in 3D printed case.

## Software

### System

We used 64-bit Raspberry Pi OS which is kind of unofficial beta at the time of writing. ARM64 system and binaries should be a little bit faster than 32-bit `armhf` arch on our Raspberry Pi 4. What's more, all ARM64 processors support vectorisation, so you do need to think whether your code or library will build with or without vectorization support, because compilers introduce them implicitly.

### Building for Raspberry

We were to dumb to build our app for Raspberry Pi on our desktop using toolchain, as one should do. Instead we prepared base ARM64 system and emulated it on desktop using QEMU. Then we used this emulated system to build for the Raspberry.

## Closing remarks

Never again. We have poured so many hours and cash into a meme that's not even remotely funny.

## Cat pictures


