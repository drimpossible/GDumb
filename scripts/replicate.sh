SEED=$1

# MNIST: Full Benchmark (Formulations B1, A1, A2, A3, E) and optimization in architecture+memory
python main.py --dataset MNIST --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 4400 --num_passes 128 --regularization none --model MLP --width 400 --exp_name MNIST_MLP400_M4400_t5_nc2_128epochs_seed$SEED
python main.py --dataset MNIST --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 300 --num_passes 128 --regularization none --model MLP --width 100 --exp_name MNIST_MLP100_M300_t5_nc2_128epochs_seed$SEED
python main.py --dataset MNIST --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 500 --num_passes 128 --regularization none --model MLP --width 100 --exp_name MNIST_MLP100_M500_t5_nc2_128epochs_seed$SEED
python main.py --dataset MNIST --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 500 --num_passes 128 --regularization none --model MLP --width 400 --exp_name MNIST_MLP400_M500_t5_nc2_128epochs_seed$SEED

# SVHN: Full Benchmark (B1)
python main.py --dataset SVHN --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 4400 --num_passes 256 --regularization cutmix --model ResNet --depth 18 --exp_name SVHN_ResNet18_M4400_t5_nc2_256epochs_cutmix_seed$SEED

# CIFAR10: Full Benchmark (Formulations A1, A2, A3, E)
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 200 --num_passes 256 --regularization cutmix --model ResNet --depth 18 --exp_name CIFAR10_ResNet18_M200_t5_nc2_256epochs_cutmix_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 500 --num_passes 256 --regularization cutmix --model ResNet --depth 18  --exp_name CIFAR10_ResNet18_M500_t5_nc2_256epochs_cutmix_seed$SEED
python main.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed $SEED --memory_size 1000 --num_passes 256 --regularization cutmix --model ResNet --depth 18  --exp_name CIFAR10_ResNet18_M1000_t5_nc2_256epochs_cutmix_seed$SEED

# CIFAR100: Full Benchmark (Formulation B2. For Setup D, which adjustment to ResNet & main before running -- see in code for instructions)
python main.py --dataset CIFAR100 --num_classes_per_task 5 --num_tasks 20 --seed $SEED --memory_size 2000 --num_passes 256 --regularization cutmix --model ResNet --depth 32 --exp_name CIFAR100_ResNet32_M20_t20_nc5_256epochs_cutmix_seed$SEED

# TinyImagenet: Full Benchmark (Formulation C2)
python main.py --dataset TinyImagenet --num_classes_per_task 20 --num_tasks 10 --seed $SEED --memory_size 4500 --num_passes 128 --regularization cutmix --model DenseNet --depth 100 --width 12  --exp_name TinyImagenet_DenseNetBC-100-12_M4500_t20_nc10_128epochs_cutmix_seed$SEED
python main.py --dataset TinyImagenet --num_classes_per_task 20 --num_tasks 10 --seed $SEED --memory_size 9000 --num_passes 64 --regularization cutmix --model DenseNet --depth 100 --width 12  --exp_name TinyImagenet_DenseNetBC-100-12_M9000_t20_nc10_64epochs_cutmix_seed$SEED
