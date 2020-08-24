# GDumb
 
This repository contains simplified code for the paper:

**GDumb: A Simple Approach that Questions Our Progress in Continual Learning, ECCV 2020 (Oral: Top 2%)**  
[Ameya Prabhu](https://drimpossible.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Puneet Dokania](https://puneetkdokania.github.io)

[[Arxiv](https://arxiv.org/abs/)]
[[PDF](https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)]
[[Slides](https://github.com/drimpossible/drimpossible.github.io/blob/master/documents/gdumb_slides.pdf)]
[[Bibtex](https://github.com/drimpossible/GDumb/#citation)]

<a href="url"><img src="https://github.com/drimpossible/GDumb/blob/master/Model.png" height="300" width="381" ></a>

## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.x by:
 ```	
# First, activate a new virtual environment
$ pip3 install -r requirements.txt
 ```
 
* Create two additional folders in the repository `data/` and `logs/` which will store the datasets and logs of experiments. Point `--data_dir` and `--log_dir` in `src/opts.py` to locations of these folders.

 * Select `Imagenet100` from Imagenet using [this link](https://github.com/wuyuebupt/LargeScaleIncrementalLearning/tree/master/dataImageNet100) and TinyImagenet from [this link](https://tiny-imagenet.herokuapp.com/) and convert them to `ImageFolder` format with `train` and `test` splits.  
 
## Usage

* To run the GDumb model you can simply specify conditions from argument, an example command below:
```
$ python main.py --dataset CIFAR100 --num_classes_per_task 5 --num_tasks 20 --memory_size 500 --num_passes 256 --regularization cutmix --model ResNet --depth 32 --exp_name my_experiment_name
```
Arguments you can freely tweak given a dataset and model: 
  - Number of classes per task (`--num_classes_per_task`)
  - Number of tasks (`--num_tasks`)
  - Maximum memory size (`--memory_size`)
  - Number of classes to pretrain a dumb model (`--num_pretrain_classes`)
  - Number of passes through the memory for learning the dumb model and pretraining (`--num_passes` and `--num_pretrain_passes`) 

To add your favorite dataset: 
  - Convert it to ImageFolder format (as in imagenet) with `train` and `test` folders 
  - Add the dataset folder name exactly to `src/opts.py`
  - Add dataset details to `get_statistics()` function in `src/dataloader.py`
  - Run you model with `--dataset your_fav_dataset`! 

Additional details and default hyperparameters can be found in `src/opts.py` 
  
 * To replicate the complete set of experiments, copy `scripts/replicate.sh` to `src/` and run with substituting `$SEED` with {0,1,2}:
```
$ bash replicate.sh $SEED
```
Similarly, other scripts can replicate results for specific formulations.

### Results

After running `replicate.sh` you should get results somewhat like these:

| GDumb Model | Mem (k) | Table | Accuracy |
|---|---|---|---|
| MNIST-MLP-100 | 300 | 3,8 | 89.1 ± 0.4 |
| MNIST-MLP-100 | 500 | 3 | 90.2 ± 0.4 |
| MNIST-MLP-400 | 500 | 4 | 91.9 ± 0.5 |
| MNIST-MLP-400 | 4400 | 5,6 | 97.8 ± 0.1 |
| SVHN-ResNet18 | 4400 | 3 | 93.4 ± 0.1 |
| CIFAR10-ResNet18 | 200 | 3 | 35.0 ± 0.4 |
| CIFAR10-ResNet18 | 500 | 3,4,8 | 45.4 ± 1.9 |
| CIFAR10-ResNet18 | 1000 | 3,4 | 61.2 ± 1.0 |
| CIFAR100-ResNet32 | 2000 | 5 | 24.3 ± 0.4 |
| TinyImageNet-DenseNet-100-12-BC | 9000 | 6 | 57.32 (best of 3) |

## Extensibility to other setups

- Settings can be tweaked by adjusting the above parameters. Additionally, GDumb can be used in a wide-variety of settings beyond current CL formulations:
  - GDumb is sortof robust against drastic variations in sample orders given the same/similar set of samples land in memory, hence this implementation abstracts the sampling process out.
  - Masking is to be used to handle dynamic variations to likely subset of classes, adding class-priors, handling scenarios like cost-sensitive classification

##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.


## Citation

We hope GDumb is a strong baseline and comparison, and the sampler or masking introduced are useful for your cool CL formulation! To cite our work:

```
@inproceedings{prabhu2020greedy,
  title={GDumb: A Simple Approach that Questions Our Progress in Continual Learning},
  author={Prabhu, Ameya and Torr, Philip and Dokania, Puneet},
  booktitle={The European Conference on Computer Vision (ECCV)},
  month={August},
  year={2020}
}
```
