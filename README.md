# README

These are a few codes used in demo of Chapel language features and how they apply to AI/ML workloads.

1. `loops.chpl`: A short and sweet introduction to parallel execution in Chapel. The code can be compiled and run as `chpl --fast loops.chpl && ./loops`.
2. `softmax.chpl`: A simple implementation of the softmax function in Chapel. This is a common function used for AI/ML workloads. The implementation is not optimized for performance, but showcases how CPU and GPU code can be written in Chapel. The code can be compiled and run as `chpl --fast sofrtmax.chpl && ./softmax`.
3. `ChAI/`: Contains a snapshot of the [ChAI](https://github.com/Iainmon/ChAI) library. This is a library that provides a high-level interface for writing AI/ML workloads in Chapel. See `ChAI/demo/vgg/README.md` for more details on how to run the VGG demo.
