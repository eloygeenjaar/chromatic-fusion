import argparse
import importlib
import torch
import pandas as pd
import multiprocessing

from pathlib import Path
from catalyst.utils.misc import set_global_seed


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    parser = argparse.ArgumentParser(description='Chromatic fusion')
    parser.add_argument(
        '-idf', '--info-df', type=Path,
        default=Path('./info_df_icasmri.csv'))
    parser.add_argument(
        '-npr', '--numpy-root', type=Path,
        default=Path(
            '/data/users1/egeenjaar/chromatic-fusion/Data/fBIRNICAsMRI'))
    parser.add_argument(
        '-ds', '--dataset', type=str, default='fBIRNICAsMRI',
        help='Dataset to be used: ICAsMRI, FNCsMRI, FAsMRI')
    parser.add_argument(
        '-cf', '--current-fold', type=int,
        help='Fold to train on', required=True)
    parser.add_argument(
        '-nl', '--num-layers', type=int, default=4)
    parser.add_argument(
        '-sd', '--shared-dim', type=int, default=32,
        help='Latent dimensionality after encoder')
    parser.add_argument(
        '-pd', '--private-dim', type=int, default=32,
        help='Latent dimensionality after encoder')
    parser.add_argument(
        '-m', '--model', type=str,
        default='VAE', help='The model to be used')
    parser.add_argument(
        '-e1', '--encoder1', type=str,
        default='ResNet3D', help='Encoder architecture for modality 1')
    parser.add_argument(
        '-e2', '--encoder2', type=str,
        default=None, help='Encoder architecture for modality 2')
    parser.add_argument('-d1', '--decoder1', type=str,
                        default='Conv3DDecoder',
                        help='Decoder architecture for modality 1')
    parser.add_argument(
        '-d2', '--decoder2', type=str,
        default=None,
        help='Decoder architecture for modality 2')
    parser.add_argument(
        '-nf', '--num-features', type=int,
        default=16, help='Number of features, '
                         'doubled with each layer')
    parser.add_argument(
        '-ep', '--epochs', type=int,
        default=500, help='Number of epochs')
    parser.add_argument(
        '-postd', '--post-dist', type=str,
        default='Normal', help='Posterior distribution')
    parser.add_argument(
        '-p', '--prior-dist', type=str,
        default='Normal', help='Prior distribution')
    parser.add_argument(
        '-pp', '--prior-params', type=str,
        default='0,1', help='Prior parameters')
    parser.add_argument(
        '-lld', '--likelihood-dist', help='Likelihood dist',
        default='Bernoulli', type=str)
    parser.add_argument(
        '-lr', '--learning-rate', type=float,
        default=0.0001, help='Learning rate')
    parser.add_argument(
        '-bs', '--batch-size', type=int,
        default=5, help='Batch size')
    parser.add_argument(
        '-b', '--beta', type=float,
        default=1.0, help='Beta parameter')
    parser.add_argument(
        '-c', '--criterion', type=str, default='ELBO',
        help='Criterion to use')
    parser.add_argument(
        '-s', '--seed', type=int,
        default=42, help='Seed')
    parser.add_argument(
        '-kf', '--k-folds', help='Number of K-folds',
        default=10, type=int)
    parser.add_argument(
        '-mt', '--main-target', type=str, default='sz')

    args = parser.parse_args()

    set_global_seed(args.seed)

    datasets_module = importlib.import_module('chromatic.datasets')
    models_module = importlib.import_module('chromatic.models')
    architectures_module = importlib.import_module(
        'chromatic.architectures')
    tasks_module = importlib.import_module('chromatic.tasks')
    distributions_module = importlib.import_module(
        'torch.distributions'
    )
    criterions_module = importlib.import_module('chromatic.criterions')

    dataset_class = getattr(datasets_module, args.dataset)
    info_df = pd.read_csv(args.info_df)
    ds_generator = dataset_class(
        info_df=info_df,
        target_names=[args.main_target, 'age', 'sex'],
        main_target=args.main_target, numpy_root=args.numpy_root,
        preprocess=True, seed=args.seed,
        num_folds=args.k_folds,
        batch_size=args.batch_size)
    tasks = ds_generator.tasks
    input_shape = ds_generator.data_shape

    encoder1_class = getattr(architectures_module, args.encoder1)
    encoder1 = encoder1_class(
        input_shape=input_shape,
        priv_size=args.private_dim,
        shared_size=args.shared_dim,
        num_features=args.num_features)
    decoder1_class = getattr(architectures_module, args.decoder1)
    decoder1 = decoder1_class(
        input_shape=input_shape,
        private_dim=args.private_dim,
        shared_dim=args.shared_dim,
        num_features=args.num_features)
    model_class = getattr(models_module, args.model)
    post_dist_class = getattr(distributions_module, args.post_dist)
    ll_dist_class = getattr(distributions_module, args.likelihood_dist)

    if args.encoder2 is not None and args.decoder2 is not None:
        encoder2_class = getattr(architectures_module, args.encoder2)
        encoder2 = encoder2_class(
            input_shape=input_shape,
            priv_size=args.private_dim,
            shared_size=args.shared_dim,
            num_features=args.num_features)
        decoder2_class = getattr(architectures_module, args.decoder2)
        decoder2 = decoder2_class(
            input_shape=input_shape,
            private_dim=args.private_dim,
            shared_dim=args.shared_dim,
            num_features=args.num_features)
        model = model_class(
            input_shape=input_shape,
            encoder1=encoder1,
            encoder2=encoder2,
            decoder1=decoder1,
            decoder2=decoder2,
            post_dist=post_dist_class,
            likelihood_dist=ll_dist_class,
            dataset=args.dataset)
    elif args.encoder2 is None and args.decoder2 is not None:
        decoder2_class = getattr(architectures_module, args.decoder2)
        decoder2 = decoder2_class(
            input_shape=input_shape,
            private_dim=args.private_dim,
            shared_dim=args.shared_dim,
            num_features=args.num_features)
        model = model_class(
            input_shape=input_shape,
            encoder1=encoder1,
            encoder2=encoder1,
            decoder1=decoder1,
            decoder2=decoder2,
            post_dist=post_dist_class,
            likelihood_dist=ll_dist_class,
            dataset=args.dataset)
    elif args.encoder2 is not None and args.decoder2 is None:
        encoder2_class = getattr(architectures_module, args.encoder2)
        encoder2 = encoder2_class(
            input_shape=input_shape,
            priv_size=args.private_dim,
            shared_size=args.shared_dim,
            num_features=args.num_features)
        model = model_class(
            input_shape=input_shape,
            encoder1=encoder1,
            encoder2=encoder2,
            decoder1=decoder1,
            decoder2=decoder1,
            post_dist=post_dist_class,
            likelihood_dist=ll_dist_class,
            dataset=args.dataset)
    elif args.encoder2 is None and args.decoder2 is None:
        model = model_class(
            input_shape=input_shape,
            encoder1=encoder1,
            encoder2=encoder1,
            decoder1=decoder1,
            decoder2=decoder1,
            post_dist=post_dist_class,
            likelihood_dist=ll_dist_class,
            dataset=args.dataset)

    prior_dist_class = getattr(distributions_module, args.prior_dist)
    prior_params = tuple(map(float, args.prior_params.split(',')))
    prior_dist = prior_dist_class(*prior_params)

    criterion_class = getattr(criterions_module, args.criterion)
    criterion = criterion_class(
        dataset=args.dataset,
        beta=args.beta,
        prior_dist=prior_dist)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logdir = Path(f'logs/{args.dataset}_{args.num_features}_'
                  f'{args.private_dim}_{args.shared_dim}_'
                  f'{args.beta}_{args.model}_'
                  f'{args.learning_rate}_{args.encoder1}_{args.encoder2}'
                  f'{args.decoder1}_{args.decoder2}_{args.criterion}'
                  f'_{args.batch_size}_{args.seed}_{args.epochs}_new')
    print(f'Model: {logdir}')
    if not logdir.is_dir():
        logdir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_class = getattr(tasks_module, task)
        task = task_class(
            model=model,
            batch_size=args.batch_size,
            dataset_generator=ds_generator,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            criterion=criterion,
            checkpoint_path=None,
            num_folds=args.k_folds,
            device=device,
            logdir=logdir,
            current_fold=args.current_fold)
        task.run()
