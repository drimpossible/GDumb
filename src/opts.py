import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    # Changing options -- Apart from these arguments, we do not mess with other arguments
    parser.add_argument('--data_dir', type=str, default='/media/anarchicorganizer/Qiqi/', help='Directory where all datasets are stored')
    parser.add_argument('--log_dir', type=str, default='../logs/', help='Directory where all logs are stored')
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'TinyImagenet', 'ImageNet100', 'ImageNet'])
    parser.add_argument('--num_classes_per_task', type=int, required=True, help='Number of classes per task')
    parser.add_argument('--num_tasks', type=int, required=True, help='Number of tasks')
    parser.add_argument('--num_pretrain_classes', type=int, default=0, help='Number of pretraining classes')
    parser.add_argument('--memory_size', type=int, required=True, help='Total slots available in the storage for the experiment')
    parser.add_argument('--num_passes', type=int, required=True, help='Number of passes to train over the storage')
    parser.add_argument('--num_pretrain_passes', type=int, default=0, help='Number of passes to train over the storage')
    parser.add_argument('--regularization', type=str, default='none', choices=['none', 'cutmix'], help='Regularization types')
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'ResNet', 'DenseNet', 'NIN'], help='Model architecture')
    parser.add_argument('--depth', type=int, default=0, help='Depth of the model')
    parser.add_argument('--width', type=int, default=0, help='Width of a model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility of class-setting etc')
    parser.add_argument('--exp_name', type=str, default='test', help='Experiment name')
    parser.add_argument('--old_exp_name', type=str, default='test', help='Name of experiment to take pretrained model from')

    # Default experiment options
    parser.add_argument('--maxlr', type=float, default=0.05, help='Starting Learning rate')
    parser.add_argument('--minlr', type=float, default=0.0005, help='Ending Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size to be used in training')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Cutmix alpha parameter')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Cutmix probability')
    parser.add_argument('--clip', type=float, default=10.0, help='Gradient Clipped if val >= clip')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')
    
    # Default model options
    parser.add_argument('--activetype', default='ReLU', choices=['ReLU6', 'LeakyReLU', 'PReLU', 'ReLU', 'ELU', 'Softplus', 'SELU', 'None'], help='Activation types')
    parser.add_argument('--pooltype', type=str, default='MaxPool2d', choices=['MaxPool2d', 'AvgPool2d', 'adaptive_max_pool2d', 'adaptive_avg_pool2d'], help='Pooling types')    
    parser.add_argument('--normtype', type=str, default='BatchNorm', choices=['BatchNorm', 'InstanceNorm'], help='Batch normalization types')
    parser.add_argument('--preact', action="store_true", help='Places norms and activations before linear/conv layer. Set to False by default')
    parser.add_argument('--bn', action="store_false", help='Apply Batchnorm. Set to True by default')
    parser.add_argument('--affine_bn', action="store_false", help='Apply affine transform in BN. Set to True by default')
    parser.add_argument('--bn_eps', type=float, default=1e-6, help='Affine transform for batch norm')    
    parser.add_argument('--compression', type=float, default=0.5, help='DenseNet BC hyperparam')
    opt = parser.parse_args()
    return opt
