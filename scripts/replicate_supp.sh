SEED=$1

# MNIST: Full Benchmark (Formulations B1)
python main.py --dataset MNIST --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 250 --num_passes 512 --regularization none --model NIN --exp_name MNIST_NIN_M250_t5_nc2_512epochs_seed$SEED
python main.py --dataset MNIST --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 250 --num_passes 512 --regularization cutmix --model NIN --exp_name MNIST_NIN_M250_t5_nc2_512epochs_cutmix_seed$SEED

# SVHN: Full Benchmark  (Formulations B1)
python main.py --dataset SVHN --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 2000 --num_passes 512 --regularization cutmix --model NIN --exp_name SVHN_NIN_M2000_t5_nc2_512epochs_cutmix_seed$SEED
python main.py --dataset SVHN --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 2000 --num_passes 512 --regularization none --model NIN --exp_name SVHN_NIN_M2000_t5_nc2_512epochs_seed$SEED

# CIFAR10: Full Benchmark (Formulations A1/A2/A3)
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 200 --num_passes 512 --regularization cutmix --model DenseNet --depth 100 --width 12 --exp_name CIFAR10_DenseNetBC-100-12_M200_t5_nc2_512epochs_cutmix_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 500 --num_passes 512 --regularization cutmix --model DenseNet --depth 100 --width 12 --exp_name CIFAR10_DenseNetBC-100-12_M500_t5_nc2_512epochs_cutmix_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 1000 --num_passes 512 --regularization cutmix --model DenseNet --depth 100 --width 12 --exp_name CIFAR10_DenseNetBC-100-12_M1000_t5_nc2_512epochs_cutmix_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 200 --num_passes 512 --regularization none --model DenseNet --depth 100 --width 12 --exp_name CIFAR10_DenseNetBC-100-12_M200_t5_nc2_512epochs_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 500 --num_passes 512 --regularization none --model DenseNet --depth 100 --width 12 --exp_name CIFAR10_DenseNetBC-100-12_M500_t5_nc2_512epochs_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 1000 --num_passes 512 --regularization none --model DenseNet --depth 100 --width 12 --exp_name CIFAR10_DenseNetBC-100-12_M1000_t5_nc2_512epochs_seed$SEED

# CIFAR100: Full Benchmark (except Formulation D for which adjustment to ResNet is needed before running)
python main.py --dataset CIFAR100 --num_classes_per_task 5 --num_tasks 20 --seed $SEED --memory_size 2000 --num_passes 512 --regularization cutmix --model DenseNet --depth 100 --width 12  --exp_name CIFAR100_DenseNetBC-100-12_M2000_t20_nc5_512epochs_cutmix_seed$SEED
python main.py --dataset CIFAR100 --num_classes_per_task 5 --num_tasks 20 --seed $SEED --memory_size 2000 --num_passes 512 --regularization none --model DenseNet --depth 100 --width 12  --exp_name CIFAR100_DenseNetBC-100-12_M2000_t20_nc5_512epochs_seed$SEED
