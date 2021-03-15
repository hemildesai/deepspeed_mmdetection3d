# SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds

## Introduction
> NOTE: This config is modified from https://github.com/open-mmlab/mmdetection3d/tree/master/configs/ssn. The README introduction is the same.

MMDetection3d implements PointPillars with Shape-aware grouping heads used in the SSN and provide the results and checkpoints on the nuScenes and Lyft dataset.

```
@inproceedings{zhu2020ssn,
  title={SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds},
  author={Zhu, Xinge and Ma, Yuexin and Wang, Tai and Xu, Yan and Shi, Jianping and Lin, Dahua},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}

```


Note:

The main difference of the shape-aware grouping heads with the original SECOND FPN heads is that the former groups objects with similar sizes and shapes together, and design shape-specific heads for each group. Heavier heads (with more convolutions and large strides) are designed for large objects while smaller heads for small objects. Note that there may appear different feature map sizes in the outputs, so an anchor generator tailored to these feature maps is also needed in the implementation.

Users could try other settings in terms of the head design. Here they basically refer to the implementation [HERE](https://github.com/xinge008/SSN).