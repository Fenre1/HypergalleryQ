import numpy as np
import os
import torch
import glob
from eval_cluster_utils import *
# from temi.train_main import make_out_dir
import datetime
import time
from main_args import set_default_args
from augs.augs import IMAGE_AUGMENTATIONS, EMBED_AUGMENTATIONS, AugWrapper
import generic_loader as loaders
import timm
from torchvision import models as torchvision_models, transforms
from feature_extractor import FeatureExtractor
import h5py
import argparse
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import train_main as tm
import utils
import torch.backends.cudnn as cudnn
import losses

def get_features(args, model_name, images):
    extractor = FeatureExtractor(model_name)
    features = extractor.extract_features(images)
    
    tensor = torch.tensor(features, dtype=torch.float32) 
    torch.save(tensor,args.knn_path + '/feature_vectors.pt') 
    return tensor

# def generate_hypergraph(all_indices,t):
#     hypergraph = np.where(all_indices < t, 0, 1)
#     for i in range(hypergraph.shape[0]):
#         if np.all(hypergraph[i] == 0):  # If all values in the row are 0
#             hypergraph[i, np.argmax(all_indices[i])] = 1  # Set the max value of that row to 1
#     return hypergraph

def get_args_parser():
    parser = argparse.ArgumentParser(description='Your script description')
    # Add all arguments here as per your script's requirements
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
    parser.add_argument('--out_dim', type=int, default=10)  # clusters variable
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
    
    knn_path = Path(args.knn_path)
    if not knn_path.exists():
        knn_path.mkdir(parents=True, exist_ok=False)  # exist_ok=False ensures it only creates if missing


# def get_custom_dataset(dataset, datapath='./data', train=True, precompute_arch=None):    
#     return PrecomputedEmbeddingDataset(
#         dataset=dataset,
#         arch=precompute_arch,
#         datapath="data",  # assumes embeddings are saved in the ./data folder
#         train=train)
    

# def get_embeddings_for_temi(args):
#     config = set_default_args(args)
    
#     preprocess = None
#     if config.precomputed:
#         backbone = config.arch
#     elif "timm" in config.arch:  # timm models
#         arch = config.arch.replace("timm_", "")
#         arch = arch.replace("timm-", "")
#         backbone = timm.create_model(arch, pretrained=True, in_chans=3, num_classes=0)
#     elif "swag" in config.arch:
#         arch = config.arch.replace("swag_", "")
#         backbone = torch.hub.load("facebookresearch/swag", model=arch)
#         backbone.head = None
#     elif "dino" in config.arch:  # dino pretrained models on IN
#         # dino_vitb16, dino_vits16
#         arch = config.arch.replace("-", "_")
#         backbone = torch.hub.load('facebookresearch/dino:main', arch)
#     elif "clip" in config.arch: # load clip vit models from openai
#         arch = config.arch.replace("clip_", "")
#         arch = arch.replace("clip-", "")
#         assert arch in clip.available_models()
#         clip_model, preprocess = clip.load(arch)
#         backbone = clip_model.visual
#     elif "mae" in config.arch or "msn" in config.arch or "mocov3" in config.arch:
#         backbone = build_arch(config.arch)
#     elif "convnext" in config.arch:
#         backbone = getattr(torchvision_models, config.arch)(pretrained=True)
#         backbone.classifier = torch.nn.Flatten(start_dim=1, end_dim=-1)
    
#     elif config.arch in torchvision_models.__dict__.keys(): # torchvision models
#         backbone = torchvision_models.__dict__[config.arch](num_classes=0)
#     else:
#         print(f"Architecture {config.arch} non supported")
#         sys.exit(1)
#     if not config.precomputed:
#         print(f"Backbone {config.arch} loaded.")
#     else:
#         print("No backbone loaded, using precomputed embeddings from", config.arch)
#     from PIL import Image
    
#     model_name = 'msn_vit_large'
#     model = build_arch(model_name)
#     model = model.cuda()
#     mean = [0.4511, 0.4537, 0.4268]
#     std = [0.2723, 0.2740, 0.2929]
#     transform = transforms.Compose([
#                 transforms.Resize((224,224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ])
    
#     def get_vector(image_name, transform, model):
#         img = Image.open(image_name)
#         if img.mode == 'RGB':
#             try:
#                 t_img = transform(img).unsqueeze(0)
#             except OSError:
#                 t_img = transform(img).unsqueeze(0)
#             t_img = t_img.cuda()
#             my_embeddingz = model(t_img)
#             my_embeddingz = my_embeddingz.cpu()
#             my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
#             return my_embeddingz
    
#     embeds = []
#     for item in im_list:
#         embeds.append(get_vector(item.strip(),transform,model))
    
#     embedz = []
#     for item in embeds:
#         try:
#             if item == None:
#                 embedz.append(np.zeros((1024),dtype='float32'))
#         except ValueError:
#             embedz.append(item)
    
            
#     tensor = torch.tensor(np.stack(embedz))
#     torch.save(tensor, args.knn_path + 'feature_vectors.pt')



@torch.no_grad()
def compute_neighbors(embedding, k):
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8*1e4:
        dists = embedding @ embedding.permute(1, 0)
        # exclude self-similarity
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1)   
    else:
        topk_knn_ids = []
        topk_knn_dists = []
        print("Chunk-wise implementation of k-nn in GPU")
        # num_chunks = 12000 
        step_size = 64 # num_embeds // num_chunks
        embedding = embedding.cuda()
        for idx in tqdm(range(0, num_embeds, step_size)):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx : idx_next_chunk, :]
            # calculate the dot product dist
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
        transform = AugWrapper(
            global_augs=aug
        )

    dataset = getattr(loaders, args.loader)(
        dataset=args.dataset,
        knn_filename='knn.pt',  # or use args.knn_filename if you want to pass it dynamically
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
    # move networks to gpu
    student_teacher_model = student_teacher_model.cuda()


    # ============ preparing loss ... ============
    loss_class = getattr(losses, args.loss)
    dino_loss_args = dict(
            out_dim=args.out_dim,
            batchsize=args.batch_size_per_gpu,
            warmup_teacher_temp=args.warmup_teacher_temp,
            teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
            nepochs=args.epochs,
            # beta=0.6,
            num_heads=16,
            **args.loss_args)
    if losses.is_multihead(loss_class):
        # dino_loss_args.update(num_heads=args.num_heads)
        dino_loss = loss_class(**dino_loss_args).cuda()
    elif args.num_heads == 1:
        dino_loss = loss_class(**dino_loss_args).cuda()
    else:
        dino_loss = nn.ModuleList([loss_class(**dino_loss_args) for _ in range(args.num_heads)]).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student_teacher_model.module.trainable_student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    bs_factor = (args.batch_size_per_gpu * utils.get_world_size()) / 256.
    lr_schedule = utils.cosine_scheduler(
        args.lr * bs_factor,  # linear scaling rule
        args.min_lr * bs_factor,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, args.max_momentum_teacher,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
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
        # if not args.disable_ddp:
        #     data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = tm.train_one_epoch(student_teacher_model, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, writer)

        # ============ writing logs ... ============
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

dataset_name = 'mh17'
id_ = '_20250302_'
output_dir = "./experiments/" + dataset_name + id_
out_dim = 80
parser = get_args_parser()
args = parser.parse_args()
args.disable_ddp=True
args.arch = 'custom'
args.knn = 50
args.knn_path = './data/embeddings/' + dataset_name + '/knn.pt'
args.embedding_path = './data/embeddings/' + dataset_name
args.num_heads=16
args.precomputed = True
args.embed_dim = 1536
args.dataset= dataset_name
args.out_dim = out_dim 
args.output_dir = output_dir
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


with h5py.File('F:\PhD\experiments\IS\hdf5\data_mh.h5', 'r') as hdf:
    # features = np.array(hdf['features_swin'])
    allcat = np.array(hdf['allcat'])
    allcat_bin = np.array(hdf['allcat_bin'])
    im_list = [str(name, 'utf-8') for name in hdf['file_list']]  # Convert back to list of strings

make_out_dir(args)

images = []
# for item in im_list:
#     images.append(item.strip())

k = 25
get_features(args, 'swinv2_large_window12to24_192to384', images)

tensor = torch.load(args.embedding_path +'/feature_vectors.pt')
nn_dists, neighbors = compute_neighbors(tensor, k)

torch.save(nn_dists, args.embedding_path + '/knn_dist.pt')
torch.save(neighbors, args.embedding_path + '/knn.pt')





writer = None
if utils.is_main_process():
    writer = SummaryWriter(output_dir)
train_dino(args, writer)

#### eval...#### 


# dataset = dataset_name
# cudnn.deterministic = True
# arch = 'msn_vit_large'
# num_workers = 0
# precomputed = True
# num_heads = 16
# embed_dim = 1000

# ckpt_folder = output_dir
# checkpoint_list = glob.glob(os.path.join(ckpt_folder, "*.pth"))

# ckpt = checkpoint_list[0]
# epoch = torch.load(ckpt, map_location='cpu')['epoch'] - 1
# # epochs.append(epoch)
# extractor = None

# if extractor is None or args.no_cache:
#     extractor = FeatureExtractionPipeline(args, cache_backbone=not args.no_cache, datapath=args.datapath)

# train_features, test_features, train_labels, val_labels = \
#     extractor.get_features(ckpt)

# ( _ , max_indices) = torch.max(test_features, dim=1)
# max_indices = max_indices.cpu().numpy()

# all_indices = test_features.cpu().numpy()
