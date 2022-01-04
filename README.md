# Monocular depth estimation on Android

A simple monocular depth estimation project using [MiDaS](https://github.com/isl-org/MiDaS) with the backend from [ncnn](https://github.com/Tencent/ncnn). The work from https://github.com/nihui/ncnn-android-nanodet served as a backbone for real-time processing with ncnn.

MiDaS uses .... to .... 

Ncnn is used as the backend for deployment on android. The original model (66.3MB) was optimized using the [guide](https://ncnn.docsforge.com/master/how-to-use-and-faq/quantized-int8-inference/). First, the layers were fused. Secondly post-quantization was employed with a calibration dataset to quantize the model to int8. The optimized model has a size of 16.9MB.

Ncnn offers support for both CPU and GPU. The GPU utilizes a Vulkan backend. However, for lower to mid range phones the benefits of GPU can be minimal or worse compared to the CPU.

<p float="left">
  <img src="demo.gif", width=400>
</p>
Figure 1. A simple example from my room. The model seems to work better from longer distances and the depth map is significantly worse after quantization. The fps is still very limited due to the relatively large model size despite quantization.</br>
</br>

#### References

https://github.com/Tencent/ncnn

Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

https://github.com/isl-org/MiDaS


https://github.com/nihui/ncnn-android-nanodet



