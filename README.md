# FitVid Video Prediction Model

Implementation of [FitVid][website] video prediction model in JAX/Flax.

If you find this code useful, please cite it in your paper:
```
@article{babaeizadeh2021fitvid,
  title={FitVid: Overfitting in Pixel-Level Video Prediction},
  author= {Babaeizadeh, Mohammad and Saffar, Mohammad Taghi and Nair, Suraj 
  and Levine, Sergey and Finn, Chelsea and Erhan, Dumitru},
  journal={arXiv preprint arXiv:2106.13195},
  year={2020}
}
```

[website]: https://sites.google.com/view/fitvidpaper

## Method

FitVid is a new architecture for conditional variational video prediction. 
It has ~300 million parameters and can be trained with minimal training tricks.

![Architecture](https://i.imgur.com/ym8uOxB.png)

## Sample Videos

| Human3.6M             |  RoboNet |
:-------------------------:|:-------------------------:
![Humans1](https://i.imgur.com/y621cvE.gif)  |  ![RoboNet1](https://i.imgur.com/KsZDnh0.gif)
![Humans2](https://i.imgur.com/yMHkqoh.gif)  |  ![RoboNet2](https://i.imgur.com/fOYPNMx.gif)

For more samples please visit [FitVid][website].
[website]: https://sites.google.com/view/fitvidpaper

## Instructions

Get dependencies:

```sh
pip3 install --user tensorflow
pip3 install --user tensorflow_addons
pip3 install --user flax
pip3 install --user ffmpeg
```

Train on RoboNet:
```sh
python -m fitvid.train  --output_dir /tmp/output
```

Disclaimer: Not an official Google product.

