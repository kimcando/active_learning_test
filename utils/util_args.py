import argparse

def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")

    # logging, path, randomness
    parser.add_argument('--needs_tb_writer', type=bool, default=True)
    parser.add_argument('--needs_logger', type=bool, default=True)
    parser.add_argument('--needs_confusion_matrix', type=bool, default=True)
    parser.add_argument('--logging_path', type=str, default='logger')
    parser.add_argument('--logger_name', type=str, default='cifar_random_test')
    parser.add_argument('--tb_path', type=str, default='log/')
    parser.add_argument('--xlsx_path', type=str, default='workbook')
    parser.add_argument('--file_name', type=str, default='cifar_random_test')
    parser.add_argument('--random_seed', type=int, default=2222)
    parser.add_argument('--shuffle', type=bool, default=True)

    # imbalance
    parser.add_argument('--imbal_mode', type=bool, default=True) #
    parser.add_argument('--split_order', type=str, default='0')
    parser.add_argument('--ratio', nargs="+", default=["1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0"])

    # learning mode
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--active', type=bool, default=False)

    # config file
    parser.add_argument('--strategy', type=str, default='rand')
    parser.add_argument(
        '--config', '-c', default='configs/cifar10_active_baseline_test.yaml'
    )
    parser.add_argument(
        '--episode', '-e', default='episodes/mnist-split-online.yaml'
    )
    # model
    parser.add_argument('--model_name', type=str, default='vgg')

    # dataset
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='/home/ncl/ADD/data')
    parser.add_argument('--num_datasplit', type=int, default=2) # usually, num_datasplit = frac*total_clients
    
    # data
    parser.add_argument('--tr_batch_size', type=int, default=128)
    parser.add_argument('--te_batch_size', type=int, default=128)
    parser.add_argument('--img_c', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)

    # active learning
    parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0001) #1e-4
    parser.add_argument('--momentum', help='momentum for adam', type=float, default=0.3)  # 1e-4
    parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
    parser.add_argument('--nEnd', help='total number of points to query', type=int, default=50000)
    parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
    
    # learning
    parser.add_argument('--opt_name', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr_sgd', type=float, default=0.01)
    parser.add_argument('--lr_adam', type=float, default=0.05)
    parser.add_argument('--momentum_sgd', type=float, default=0.3)

    args = parser.parse_args()
    return args


def test_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")

    # logging, path, randomness
    parser.add_argument('--needs_tb_writer', type=bool, default=True)
    parser.add_argument('--needs_logger', type=bool, default=True)
    parser.add_argument('--logging_path', type=str, default='logger')
    parser.add_argument('--logger_name', type=str, default='img_test')
    parser.add_argument('--tb_path', type=str, default='log/')
    parser.add_argument('--xlsx_path', type=str, default='workbook')
    parser.add_argument('--file_name', type=str, default='img_test')
    parser.add_argument('--random_seed', type=int, default=2222)

    # config file
    parser.add_argument('--strategy', type=str, default='rand')
    parser.add_argument(
        '--config', '-c', default='configs/cifar10_active_baseline_test.yaml'
    )
    parser.add_argument(
        '--episode', '-e', default='episodes/mnist-split-online.yaml'
    )
    # model
    parser.add_argument('--model_name', type=str, default='lenet')

    # dataset
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='/home/ncl/ADD/data')
    parser.add_argument('--num_datasplit', type=int, default=2)  # usually, num_datasplit = frac*total_clients

    # data
    parser.add_argument('--tr_batch_size', type=int, default=128)
    parser.add_argument('--te_batch_size', type=int, default=1000)
    parser.add_argument('--img_c', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)

    # active learning
    parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)  # 1e-4
    parser.add_argument('--momentum', help='momentum for adam', type=float, default=0.3)  # 1e-4
    parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
    parser.add_argument('--nEnd', help='total number of points to query', type=int, default=50000)
    parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)

    # learning
    parser.add_argument('--opt_name', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr_sgd', type=float, default=0.01)
    parser.add_argument('--lr_adam', type=float, default=0.001)
    parser.add_argument('--momentum_sgd', type=float, default=0.9)

    args = parser.parse_args()
    return args