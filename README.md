# Tiramisu implementation on Camvid dataset

## Notes
* Implementation of Tiramisu - 56, 67, 103
* The image dimension used to train the model is 480x384
* The original size of the image/mask from camvid dataset is 960x720 which was resized to 480x360 and zeropadded, resulting in 480x384
* 12 custom classes used
* The original implementation was trained on lower resolution which used batch normalization, since the batch size used to train the model is 2 batch normalization is omitted in the implementation

## Main idea
* Use densenet idea, concatenated convolutional layers to build a 100 layer network for semantic segmentation

## To do
- [x] TiramisuNet
- [ ] Compute metrics
- [ ] Sample output

## Reference
* [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
* [The One Hundred Layers Tiramisu - Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
* [Camvid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
