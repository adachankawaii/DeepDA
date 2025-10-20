# -*- coding: utf-8 -*-
import logging
import os

import cfg
import numpy as np
import torch
import torch.nn as nn
from methods import DigitsDA
from models import Classifier1, SVHN_generator, USPS_generator, weights_init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import BalancedBatchSampler
import time

torch.multiprocessing.set_sharing_strategy("file_system")


def get_dataset_size28(dataset="mnist", data_dir="./data"):
    trans = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    if dataset == "mnist":
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=trans)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    elif dataset == "usps":
        train_ds = datasets.USPS(data_dir, train=True, download=True, transform=trans)
        test_ds = datasets.USPS(data_dir, train=False, download=True, transform=trans)

    return train_ds, test_ds


def get_dataset_size32(dataset="mnist", data_dir="./data"):
    if dataset == "mnist":
        trans = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=trans)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    elif dataset == "svhn":
        trans = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_ds = datasets.SVHN(data_dir, split="train", download=True, transform=trans)
        test_ds = datasets.SVHN(data_dir, split="test", download=True, transform=trans)

    return train_ds, test_ds


def get_dataloader(source, target, data_dir, batch_size, num_workers=0):
    if source == "svhn" or target == "svhn":
        source_train_ds, source_test_ds = get_dataset_size32(source, data_dir)
        target_train_ds, target_test_ds = get_dataset_size32(target, data_dir)
    elif source == "usps" or target == "usps":
        source_train_ds, source_test_ds = get_dataset_size28(source, data_dir)
        target_train_ds, target_test_ds = get_dataset_size28(target, data_dir)

    # Lấy nhãn trực tiếp, không enumerate (nhanh hơn rất nhiều)
    if hasattr(source_train_ds, "labels"):
        source_labels = torch.as_tensor(source_train_ds.labels)
    else:
        source_labels = torch.as_tensor(source_train_ds.targets)

    source_train_sampler = BalancedBatchSampler(source_labels, batch_size=batch_size)
    source_train_dl = DataLoader(
        source_train_ds,
        batch_sampler=source_train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    target_train_dl = DataLoader(
        target_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    source_test_dl = DataLoader(
        source_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    target_test_dl = DataLoader(
        target_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return source_train_dl, target_train_dl, source_test_dl, target_test_dl


def main():
    args = cfg.parse_args()
    
    # Thêm debug này
    print(f"GPU ID argument: {args.gpu_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(int(args.gpu_id))}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Logging config
    if args.use_bomb:
        prefix = "b"
    else:
        prefix = ""
    description = f"{prefix}{args.method}_{args.source_ds}_to_{args.target_ds}"
    description += f"_k{args.k}_m{args.mbsize}_lr{args.lr}_epsilon{args.epsilon}_be{args.batch_epsilon}_mass{args.mass}_tau{args.tau}"
    base_dir = "snapshot/"
    out_dir = os.path.join(base_dir, description)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.info(args)

    # Set up parameters
    batch_size = args.k * args.mbsize
    n_epoch = args.n_epochs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpus = args.gpu_id.split(",")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get dataloaders
    source_train_dl, target_train_dl, source_test_dl, target_test_dl = get_dataloader(
        args.source_ds, args.target_ds, args.data_dir, batch_size, args.num_workers
    )

    # Train
    if args.source_ds == "svhn" or args.target_ds == "svhn":
        model_g = SVHN_generator().cuda().apply(weights_init)
    elif args.source_ds == "usps" or args.target_ds == "usps":
        model_g = USPS_generator().cuda().apply(weights_init)
    model_f = Classifier1(nclass=args.nclass).cuda().apply(weights_init)
    if len(gpus) > 1:
        model_g = nn.DataParallel(model_g, device_ids=[int(i) for i in gpus])
        model_f = nn.DataParallel(model_f, device_ids=[int(i) for i in gpus])
    model_g.train()
    model_f.train()
    model_da = DigitsDA(
        model_g,
        model_f,
        n_class=args.nclass,
        logger=logger,
        out_dir=out_dir,
        eta1=args.eta1,
        eta2=args.eta2,
        epsilon=args.epsilon,
        batch_epsilon=args.batch_epsilon,
        mass=args.mass,
        tau=args.tau,
        test_interval=args.test_interval,
    )
    model_da.source_only(source_train_dl, lr=args.lr)  # Train on source domain only
    t0 = time.time()
    if args.use_bomb:
        # Use the stable version because the training loss is small
        model_da.fit_bomb2(
            source_train_dl,
            target_train_dl,
            target_test_dl,
            n_epochs=n_epoch,
            lr=args.lr,
            k=args.k,
            batch_size=batch_size,
            method=args.method,
        )
    else:
        model_da.fit(
            source_train_dl,
            target_train_dl,
            target_test_dl,
            n_epochs=n_epoch,
            lr=args.lr,
            k=args.k,
            batch_size=batch_size,
            method=args.method,
        )
    elapsed = time.time() - t0
    logger.info(f"WALL_CLOCK_TRAIN_SEC={elapsed:.3f}")
    with open(os.path.join(out_dir, "time.csv"), "w") as f:
        f.write(f"wall_clock_sec,{elapsed:.6f}\n")
        
    # Evaluate
    source_acc = model_da.evaluate(source_test_dl)
    target_acc = model_da.evaluate(target_test_dl)
    logger.info("source_acc={}, target_acc={}".format(source_acc, target_acc))


if __name__ == "__main__":
    main()
