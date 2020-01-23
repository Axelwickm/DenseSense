# DenseSense
WIP: Integration of Facebook's DensePose algorithm with person tracking and clothing recognition

This project includes:
* A DensePose (detectron2) wrapper, which does some post processing to merge all output layers into one.
* A sanitizer which merges and masks DensePose output to more accurately detect individual people.
* A Kalman Filter tracker to correlate detected individuals across multiple frames.
* An UV-mapper which projects each person Segmentation onto a standardized flat image.
* WIP: a descriptionExtractor which identifies the kind of clothing and what color that people are wearing.
* WIP: a pose detector which classifies what each person is doing.

Included are a web cam example which all above features utilized, 
and a training script for regenerating the models which can be trained.
There are however trained models included under ./models.

## To install:
Install [PyTorch](https://pytorch.org/)

Install all the dependencies:
```bash
python -m pip install -r ./DenseSense/requirements.txt
```
Install [detectron2](https://github.com/facebookresearch/detectron2) from cloned github repository:
```bash
git clone https://github.com/facebookresearch/detectron2/
cd detectron2 && python -m pip install -e .
```

Add DensePose to the PYTHONPATH by locating it using output of:
```bash
python -c "import detectron2, os; print(os.path.dirname(os.path.dirname(detectron2.__file__))+'/projects/DensePose')"
```

Also add DenseSense to PYTHONPATH.
<br/><br/>

Get the DensePose model configuration file and put in under ./models. Be sure to also get the BaseConfig:
https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose/configs

Also get DensePose model and put it under ./models:
https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/MODEL_ZOO.md

Currently DenseSense is configured for densepose_rcnn_R_50_FPN_s1x.yaml and R_50_FPN_s1x.pkl,
but this can be changed by modifying the paths in the beginning of `./DenseSense/algorithms/DensePoseWrapper.py`.
Adding a configuration file for DenseSense is TODO.


## Getting the datasets for training
TODO

Sanitizer: COCO (script included for downloading just the training instances with people in)

DescriptionExtractor: DeepFashion2

## Example debug output
TODO
     