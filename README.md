# Tiramisu implementation on Camvid dataset

## Notes
* Implementation of Tiramisu - 56, 67, 103
* The image dimension used to train the model is 480x384
* The original size of the image/mask from camvid dataset is 960x720 which is resized to 480x360 and **zero\_padded**, resulting in 480x384
* 12 custom classes used
* The original implementation was trained on lower resolution samples which used batch normalization, since **batch\_size = 2** is used to train the model, batch normalization is not used in the implementation

## Main idea
* Using dense\_net idea, concatenated convolutional layers within a dense block to build a 100 layer network for semantic segmentation

## Intructions to run
> To run training use - **python3 tiramisu\_train.py --help**
>
> To run inference use - **python3 tiramisu\_infer.py --help**
>
> This lists all possible commandline arguments

## Visualization of results
* [Tiramisu-103](https://youtu.be/UIGzFRB4hs0)

## Reference
* [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
* [The One Hundred Layers Tiramisu - Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
* [Camvid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
