import os
import torch
import datetime
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from eval_cluster_utils import *
from main_args import set_default_args
from augs.augs import IMAGE_AUGMENTATIONS, EMBED_AUGMENTATIONS, AugWrapper
import generic_loader as loaders
import timm
from torchvision import models as torchvision_models, transforms
from feature_extractor import FeatureExtractor
import h5py
import argparse
import json
import train_main as tm
import utils
import torch.backends.cudnn as cudnn
import losses
import h5py

def get_features(args, model_name, images):
    extractor = FeatureExtractor(model_name)
    features = extractor.extract_features(images)
    tensor = torch.tensor(features, dtype=torch.float32) 
    torch.save(tensor, args.datapath + '/feature_vectors.pt')
    return tensor

def get_args_parser():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--precomputed', action='store_true')
    parser.add_argument('--arch', type=str, default='clip_ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--use_fp16', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--max_momentum_teacher', type=float, default=0.996)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default="./experiments/TEMI-output-test")
    parser.add_argument('--dataset', type=str, default="mh17")
    parser.add_argument('--knn', type=int, default=50)
    parser.add_argument('--out_dim', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--loss', type=str, default="TEMI")
    parser.add_argument('--loss-args', type=json.loads, default='{"beta":0.6}')
    parser.add_argument('--disable_ddp', type=lambda x: (str(x).lower() == 'true'), default=True)
    return parser

def make_out_dir(args):
    if args.new_run:
        n = 1
        dir_name = args.output_dir
        while Path(args.output_dir).is_dir():
            n += 1
            args.output_dir = f"{dir_name}{n}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=not args.new_run)
    knn_path = Path(args.embedding_path)
    if not knn_path.exists():
        knn_path.mkdir(parents=True, exist_ok=False)

@torch.no_grad()
def compute_neighbors(embedding, k):
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8 * 1e4:
        dists = embedding @ embedding.permute(1, 0)
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1)
    else:
        topk_knn_ids = []
        topk_knn_dists = []
        print("Chunk-wise implementation of k-nn in GPU")
        step_size = 64
        embedding = embedding.cuda()
        for idx in range(0, num_embeds, step_size):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx:idx_next_chunk, :]
            dists_chunk = torch.mm(features, embedding.T).cpu()
            dists_chunk.fill_diagonal_(-torch.inf)
            max_dists, indices = dists_chunk.topk(k, dim=-1)
            topk_knn_ids.append(indices)
            topk_knn_dists.append(max_dists)
        return torch.cat(topk_knn_dists), torch.cat(topk_knn_ids)

def train_dino(args, writer):
    args.batch_size_per_gpu = 512
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    student, _, normalize = tm.load_model(args, split_preprocess=True)
    teacher, _ = tm.load_model(args)
    if not args.precomputed:
        aug = IMAGE_AUGMENTATIONS[args.image_aug](num_augs=args.num_augs, **args.aug_args)
        transform = AugWrapper(
            vit_image_size=args.vit_image_size,
            aug_image_size=args.aug_image_size,
            global_augs=aug,
            normalize=normalize,
            image_size=args.image_size
        )
    else:
        aug = EMBED_AUGMENTATIONS[args.embed_aug](num_augs=args.num_augs, **args.aug_args)
        transform = AugWrapper(global_augs=aug)
    dataset = getattr(loaders, args.loader)(
        dataset=args.dataset,
        knn_filename='knn.pt',
        datapath=args.datapath,
        k=args.knn,
        transform=transform,
        precompute_arch=args.arch if args.precomputed else None,
        **args.loader_args)
    sampler = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"In-distribution Data loaded: there are {len(dataset)} images.")
    print("len dataloader", len(data_loader))
    student_teacher_model = tm.TeacherStudentCombo(teacher=teacher, student=student, args=args)
    student_teacher_model = student_teacher_model.cuda()
    loss_class = getattr(losses, args.loss)
    dino_loss_args = dict(
        out_dim=args.out_dim,
        batchsize=args.batch_size_per_gpu,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        num_heads=16,
        **args.loss_args)
    if losses.is_multihead(loss_class):
        dino_loss = loss_class(**dino_loss_args).cuda()
    elif args.num_heads == 1:
        dino_loss = loss_class(**dino_loss_args).cuda()
    else:
        dino_loss = torch.nn.ModuleList([loss_class(**dino_loss_args) for _ in range(args.num_heads)]).cuda()
    params_groups = utils.get_params_groups(student_teacher_model.module.trainable_student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    bs_factor = (args.batch_size_per_gpu * utils.get_world_size()) / 256.
    lr_schedule = utils.cosine_scheduler(
        args.lr * bs_factor,
        args.min_lr * bs_factor,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, args.max_momentum_teacher,
                                               args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student_teacher_model.module.student,
        teacher=student_teacher_model.module.teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        train_stats = tm.train_one_epoch(student_teacher_model, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, writer)
        save_dict = {
            'student': student_teacher_model.module.student_dict(),
            'teacher': student_teacher_model.module.teacher_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        try:       
            torch.set_printoptions(profile="full")
            if epoch % 10 == 0:
                d_loss = dino_loss[0] if hasattr(dino_loss, "__getitem__") else dino_loss
                print("highest probs:", torch.topk(d_loss.probs_pos * 100, 50)[0])
                print("lowest probs:", torch.topk(d_loss.probs_pos * 100, 50, largest=False)[0])
            torch.set_printoptions(profile="default")
        except:
            print(" ")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def run_temi_pipeline(dataset_name, out_dim, images):
    # Parse args and override with desired defaults
    parser = get_args_parser()
    args = parser.parse_args([])  # ignore CLI args
    args.disable_ddp = True
    args.arch = 'custom'
    args.knn = 50
    args.knn_path = './data/embeddings/' + dataset_name + '/knn.pt'
    args.embedding_path = './data/embeddings/' + dataset_name
    args.num_heads = 16
    args.precomputed = True
    args.embed_dim = 1536
    args.dataset = dataset_name
    args.out_dim = out_dim
    id_ = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    args.output_dir = "./experiments/" + dataset_name + str(args.out_dim) + id_
    args.new_run = True
    args.seed = 0
    args.embed_aug = 'none'
    args.num_augs = 1
    args.aug_args = {}
    args.loader = 'EmbedNN'
    args.datapath = './data'
    args.loader_args = {}
    args.train_backbone = False
    args.warmup_teacher_temp = 0.04
    args.teacher_temp = 0.04
    args.warmup_teacher_temp_epochs = 30
    args.optimizer = 'adamw'
    args.weight_decay = 0.0001
    args.weight_decay_end = 0.0005
    args.momentum_teacher = 0.9995
    args.clip_grad = 0
    args.freeze_last_layer = 1
    args.saveckp_freq = 20

    make_out_dir(args)

    feature_file = os.path.join(args.embedding_path, 'feature_vectors.pt')
    knn_file = os.path.join(args.embedding_path, 'knn.pt')
    knn_dist_file = os.path.join(args.embedding_path, 'knn_dist.pt')

    # Check if feature extraction is needed
    if not os.path.exists(feature_file):
        get_features(args, 'swinv2_large_window12to24_192to384', images)
    else:
        print("Feature file exists, skipping feature extraction.")

    # Check if neighbor computation is needed
    if not (os.path.exists(knn_file) and os.path.exists(knn_dist_file)):
        tensor = torch.load(feature_file)
        k = 25
        nn_dists, neighbors = compute_neighbors(tensor, k)
        torch.save(nn_dists, knn_dist_file)
        torch.save(neighbors, knn_file)
    else:
        print("KNN files exist, skipping neighbor computation.")

    writer = None
    if utils.is_main_process():
        writer = SummaryWriter(args.output_dir)

    train_dino(args, writer)

    return args.output_dir
