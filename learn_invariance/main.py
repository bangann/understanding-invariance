from argparse import ArgumentParser
import os
import cox.store
from torchvision import transforms
from robustness.datasets import DATASETS
from robustness.defaults import check_and_fill_args
from robustness import defaults
from make_models import make_and_restore_model
from train_inv import train_model, eval_model
import warnings

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
parser.add_argument('--no-store', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--inv-method', choices=['aug', 'reg', 'none'], default='aug')
parser.add_argument('--inv-method-beta', type=float, default=1.0)
parser.add_argument('--loss', choices=['std', 'adv'], default='std')
parser.add_argument('--loss-aux', choices=['avg', 'adv'], default='avg')
parser.add_argument('--transforms',
                    choices=['none', 'crop', 'flip', 'jitter', 'rotate', 'view', 'cutout']
                    , nargs='+')
parser.add_argument('--metainfo-path', default=None)
parser.add_argument('--n-per-class', type=str,choices=['100', '1000', 'all'], default='all')

args = parser.parse_args()


def main(args):
    store = None if args.no_store else setup_store_with_metadata(args)
    if args.debug:
        args.workers = 0

    if args.dataset == 'r2n2':
        transform_dict = {
            'crop': transforms.RandomCrop(32, padding=4),
            'flip': transforms.RandomHorizontalFlip(),
            'jitter': transforms.ColorJitter(.25, .25, .25),
            'rotate': transforms.RandomRotation(30),
            'cutout': transforms.RandomErasing(value=0.5, scale=(0.05, 0.05), ratio=(1,1))
        }
        if 'view' in args.transforms:
            args.transform_view = True
        args.transforms = [_ for _ in args.transforms
                           if _ in transform_dict.keys()]

        transform_train = transforms.Compose(
            [transforms.Resize(32)] +
            [transform_dict[t] for t in args.transforms if t != 'cutout'] +
            [transforms.ToTensor()] +
            [transform_dict[t] for t in args.transforms if t == 'cutout'])

        transform_test = transforms.Compose(
            [transforms.Resize(32)] +
            [transforms.ToTensor()]
        )
        custom_class_args = {'metainfo_path': args.metainfo_path}
        kwargs = {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'custom_class_args': custom_class_args
        }
        data_path = os.path.expandvars(args.data)
        dataset = DATASETS[args.dataset](data_path, **kwargs)

        train_loader, val_loader = dataset.make_loaders(args.workers,
                                                        args.batch_size, data_aug=bool(args.data_aug))

        loaders = (train_loader, val_loader)

    elif args.dataset == 'cifar':

        transform_dict = {
            'crop': transforms.RandomCrop(32, padding=4),
            'flip': transforms.RandomHorizontalFlip(),
            'jitter': transforms.ColorJitter(.25, .25, .25),
            'rotate': transforms.RandomRotation(30),
            'cutout': transforms.RandomErasing(p=0.5, value=0.5, scale=(0.05, 0.05), ratio=(1, 1))
        }

        args.transforms = [_ for _ in args.transforms
                           if _ in transform_dict.keys()]

        transform_train = transforms.Compose(
            [transform_dict[t] for t in args.transforms if t != 'cutout'] +
            [transforms.ToTensor()] +
            [transform_dict[t] for t in args.transforms if t == 'cutout'])

        transform_test = transforms.Compose(
            [transforms.ToTensor()])
        kwargs = {
            'transform_train': transform_train,
            'transform_test': transform_test}
        data_path = os.path.expandvars(args.data)
        dataset = DATASETS[args.dataset](data_path, **kwargs)

        if args.n_per_class == 'all':
            train_loader, val_loader = dataset.make_loaders(args.workers,
                                                            args.batch_size, data_aug=bool(args.data_aug))
        elif args.n_per_class == '1000':
            train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug),
                                                            subset=10000, subset_start=0, subset_type='rand',
                                                            subset_seed=0)
        elif args.n_per_class == '100':
            train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug),
                                                            subset=1000, subset_start=0, subset_type='rand',
                                                            subset_seed=0)

        loaders = (train_loader, val_loader)

    else:
        raise Exception("Invalid dataset")

    attacker_model, checkpoint = make_and_restore_model(arch=args.arch,
                                                        dataset=dataset, resume_path=args.resume)
    if args.eval_only:
        eval_model(args, attacker_model, val_loader, store=store)
    else:
        train_model(args, attacker_model, loaders,
                    checkpoint=checkpoint, store=store)


# adapted from https://github.com/MadryLab/robustness
def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    args.adv_train = (args.classifier_loss == 'robust') or \
                     (args.estimator_loss == 'worst')
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
        "Must provide a resume path if only evaluating"
    return args


def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)
    return store


if __name__ == "__main__":
    args = cox.utils.Parameters(args.__dict__)
    args = setup_args(args)

    args.workers = 0 if args.debug else args.workers
    print(args)

    main(args)
