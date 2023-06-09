from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
from torch.utils.data import DataLoader
from util import parallel_apply, get_logger
from dataloaders.dataloader_CC3M import CC3M_TrainDataLoader
from dataloaders.dataloader_WebVid import WebVid_TrainDataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
#from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from torch.nn import functional as F
import copy
torch.distributed.init_process_group(backend="nccl")
# from light import light_init
global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--cc3m_json', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--webvid_json', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='mask ratio')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.output_dir, "ckpt")):
        os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu



    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    
    global_step = 0
    restore_path = os.path.join(args.output_dir, 'restore.bin')
    if args.init_model and not os.path.exists(restore_path):
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        #clip_16 = torch.load("/apdcephfs/share_1324356/zifengchai/smart/data/zhw/pytorch_model.bin.702")
        if 'model_state_dict' in model_state_dict:
            print('loading model from its model_state_dict')
            #for key in  clip_16['model_state_dict']:
               # model_state_dict['model_state_dict'][key] = clip_16['model_state_dict'][key]
            model_state_dict = model_state_dict['model_state_dict']
    elif os.path.exists(restore_path):
        print(f'find previous checkpoint, try to resume training from {restore_path}')
        t = torch.load(restore_path, map_location='cpu')
        
        model_state_dict = t['model_state_dict']
        global_step = t['global_step']
        assert model_state_dict is not None, "the model is not correctly loaded"
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args,restore_path=restore_path)

    model.to(device)
    

    return model, global_step

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if ("clip." in n and "fusion" not in n)]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]
    decay_clip_param_tp_fuse = [(n, p) for n, p in decay_param_tp if ("clip." in n and "fusion" in n)]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if ("clip." in n and "fusion" not in n)]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]
    no_decay_clip_param_tp_fuse = [(n, p) for n, p in no_decay_param_tp if ("clip." in n and "fusion" in n)]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in decay_clip_param_tp_fuse], 'weight_decay': weight_decay, 'lr': 1e-5},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0},
        {'params': [p for n, p in no_decay_clip_param_tp_fuse], 'weight_decay': 0.0, 'lr': 1e-5}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    resume = os.path.join(args.output_dir, "restore.bin")
    if os.path.exists(resume):
        print(f'prepare optimizer from {resume}')
        t = torch.load(resume, map_location=device)
        assert t['optim_state_dict'] is not None, "reloading optimizer is not correct"
        optimizer.load_state_dict(t['optim_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_CC3M_train(args, tokenizer):
    CC3M_dataset = CC3M_TrainDataLoader(
        json_path=args.cc3m_json,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(CC3M_dataset)
    dataloader = DataLoader(
        CC3M_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(CC3M_dataset), train_sampler

def dataloader_webvid_train(args, tokenizer):
    webvid_dataset = WebVid_TrainDataLoader(
        json_path=args.webvid_json,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(webvid_dataset)
    dataloader = DataLoader(
        webvid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(webvid_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path='/apdcephfs/share_1324356/zifengchai/smart/data/data_CLIP4clip/MSRVTT_JSFUSION_test.csv',
        features_path='/apdcephfs/share_1324356/zifengchai/smart/data/data_CLIP4clip/MSRVTT_Videos',
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)




def save_model(epoch, args, model, type_name="", optimizer=None, step=None):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "ckpt", "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", step))
    
    resume = os.path.join(args.output_dir, "restore.bin")
    if os.path.exists(resume):
        os.remove(resume)
    assert optimizer is not None, 'optimizer is invalid'
    assert step is not None, 'step must be not None'
    checkpoint = {'global_step': step,
                  'model_state_dict': model_to_save.state_dict(),
                  'optim_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, output_model_file)
    # torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(checkpoint, resume)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_video, batch_text = [], []
        batch_video_mask, batch_text_mask = [], []
        batch_v_sep_token, batch_t_sep_token = [],[]
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            #input_mask = input_mask.float()
            #input_mask[input_mask==0] = float('-inf')
            #input_mask[input_mask==1] = 0
            #input_mask = input_mask.repeat(1,input_mask.shape[-1],1)


            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output,text_tokens = model.get_sequence_output(input_ids, segment_ids, None)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output, text_tokens = model.get_sequence_visual_output(input_ids, segment_ids, None, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

            max_video_len = video_mask.shape[-1]
            sep_idx = torch.cat((video_mask.sum(dim=-1),max_video_len+8+input_mask.sum(dim=-1)),dim=-1)
            input_mask = input_mask.float()
            input_mask[input_mask==0] = float('-inf')
            input_mask[input_mask==1] = 0
            # T x 3 x H x W
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            # mean pooling
            video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
            video_tokens = visual_output * video_mask_un
            video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
            video_mask_un_sum[video_mask_un_sum == 0.] = 1.
            v_sep_token = (torch.sum(video_tokens, dim=1) / video_mask_un_sum).unsqueeze(1)
            t_sep_token = torch.cat([text_tokens[i:i+1,sep_idx[i][1]-max_video_len-9:sep_idx[i][1]-max_video_len-8] for i in range(text_tokens.shape[0])],dim=0)

            batch_v_sep_token.append(v_sep_token)
            batch_t_sep_token.append(t_sep_token)

            video_tokens = torch.cat((visual_output,v_sep_token),dim=-2)

        
            video_mask = torch.cat((video_mask[:,0:1],video_mask,video_mask[:,0:8]),dim=-1).float()
            video_mask[video_mask==0] = float('-inf')
            video_mask[video_mask==1] = 0
            video_mask = video_mask.unsqueeze(1)
            
            
            batch_video.append(video_tokens)
            batch_text.append(text_tokens)
            batch_video_mask.append(video_mask)
            batch_text_mask.append(input_mask)


        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

        
        t2v_matrix = torch.from_numpy(sim_matrix).to(input_ids.device)
        #t2v_matrix = torch.softmax(t2v_matrix,dim=-1)
        v2t_matrix = torch.from_numpy(sim_matrix.T).to(input_ids.device)
        #v2t_matrix = torch.softmax(v2t_matrix,dim=-1)
        
        t2v_candidates = torch.topk(t2v_matrix, 32, dim=-1)[1]
        v2t_candidates = torch.topk(v2t_matrix, 32, dim=-1)[1]
        all_video_outputs = torch.cat(batch_video,dim=0)
        all_text_outputs = torch.cat(batch_text,dim=0)
        all_video_mask = torch.cat(batch_video_mask,dim=0)
        all_text_mask = torch.cat(batch_text_mask,dim=0)
        all_v_sep_token = torch.cat(batch_v_sep_token,dim=0)
        all_t_sep_token = torch.cat(batch_t_sep_token,dim=0)

        for tid in range(all_text_outputs.shape[0]):
            candidate_video_outputs = all_video_outputs[t2v_candidates[tid]]
            candidate_video_masks = all_video_mask[t2v_candidates[tid]]
            t2v_hard_v_sep = all_v_sep_token[t2v_candidates[tid]]
            t2v_hard_t_sep = all_t_sep_token[tid].repeat(t2v_hard_v_sep.shape[0],1,1)
            t2v_query_tokens = model.clip.fusion_ln(torch.cat((t2v_hard_v_sep,t2v_hard_t_sep),dim=-1))
            t2v_query_tokens = t2v_query_tokens.repeat(1,8,1)
            t2v_query_tokens = torch.cat([(t2v_query_tokens[:,i,:] @ model.clip.fusion_query[i,:,:]).unsqueeze(1) for i in range(8)],dim=1)
            t2v_query_weight = F.softmax((t2v_query_tokens.view(-1,t2v_query_tokens.shape[-1]) @ model.clip.fusion_memory.T),dim=-1)
            t2v_fusion_tokens = (t2v_query_weight @ model.clip.fusion_memory).view(-1,8,model.clip.fusion_memory.shape[-1])
            vtm_inputs = torch.cat((candidate_video_outputs,t2v_fusion_tokens,all_text_outputs[tid:tid+1].repeat(candidate_video_outputs.shape[0],1,1)),dim=1)
            
            vtm_masks = torch.cat((candidate_video_masks,all_text_mask[tid:tid+1].repeat(candidate_video_masks.shape[0],1,1)),dim=-1)
            vtm_masks = vtm_masks.repeat(1,vtm_masks.shape[-1],1)

            fusion_masks = vtm_masks.clone()
            fusion_masks[:,:max_video_len+1,max_video_len+9:] = torch.full(fusion_masks[:,:max_video_len+1,max_video_len+9:].shape, float('-inf')).to(vtm_masks.device)
            fusion_masks[:,max_video_len+9:,:max_video_len+1] = torch.full(fusion_masks[:,max_video_len+9:,:max_video_len+1].shape, float('-inf')).to(vtm_masks.device)

            fusion_input = vtm_inputs.clone()
            fusion_labels = vtm_masks.clone()
            fusion_labels[fusion_labels==0] = 1
            fusion_labels[fusion_labels==float('-inf')] = 0
            fusion_labels = fusion_labels.long()
            sep_idx = torch.cat((fusion_labels[:,0:1,:max_video_len+1].sum(dim=-1),max_video_len+fusion_labels[:,0:1,max_video_len+1:].sum(dim=-1)),dim=-1)

            pos_emd = model.clip.fusion_positional_embedding[:vtm_inputs.size(1), :]
            vtm_inputs = vtm_inputs + pos_emd

            #去除SEP的Positional Embedding
            for i in range(vtm_inputs.shape[0]):
                vtm_inputs[i,sep_idx[i,0]] = fusion_input[i,sep_idx[i,0]]
                vtm_inputs[i,sep_idx[i,1]] = fusion_input[i,sep_idx[i,1]]

            vtm_inputs = vtm_inputs.permute(1, 0, 2)  # NLD -> LND
            vtm_outputs = model.clip.fusion_transformer(vtm_inputs.half(),mask=fusion_masks.half(),task='fusion')
            vtm_outputs = vtm_outputs.permute(1, 0, 2) # LND -> NLD

            vtm_v_sep = vtm_outputs[:,max_video_len,:]
            vtm_t_sep = torch.cat([vtm_outputs[i][vtm_masks[i][0]==0][-1:] for i in range(vtm_outputs.shape[0])],dim=0)
            match_scores = model.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ model.clip.fusion_match_matrix.half() @ model.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))
            unmatch_scores = model.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ model.clip.fusion_unmatch_matrix.half() @ model.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))
            diag_mask = torch.ones(match_scores.shape[0])
            diag_mask = torch.diag(diag_mask)
            match = match_scores[diag_mask==1].view(-1,1)
            unmatch = unmatch_scores[diag_mask==1].view(-1,1)
            vtm_predicted = torch.cat((unmatch,match),dim=-1).float()
            
            #t2v_matrix[tid][t2v_candidates[tid]] = t2v_matrix[tid][t2v_candidates[tid]] + vtm_predicted[:,1].float()
            vtm_predicted = torch.softmax(vtm_predicted,dim=-1) * 1000. 
            t2v_matrix[tid][t2v_candidates[tid]] =  torch.softmax(t2v_matrix[tid][t2v_candidates[tid]],dim=-1) * vtm_predicted[:,1] + 1000.
            
            
            

        for vid in range(all_video_outputs.shape[0]):
            candidate_text_outputs = all_text_outputs[v2t_candidates[vid]]
            candidate_text_masks = all_text_mask[v2t_candidates[vid]]
            v2t_hard_t_sep = all_t_sep_token[v2t_candidates[vid]]
            v2t_hard_v_sep = all_v_sep_token[vid].repeat(v2t_hard_t_sep.shape[0],1,1)
            v2t_query_tokens = model.clip.fusion_ln(torch.cat((v2t_hard_v_sep,v2t_hard_t_sep),dim=-1))
            v2t_query_tokens = v2t_query_tokens.repeat(1,8,1)
            v2t_query_tokens = torch.cat([(v2t_query_tokens[:,i,:] @ model.clip.fusion_query[i,:,:]).unsqueeze(1) for i in range(8)],dim=1)
            v2t_query_weight = F.softmax((v2t_query_tokens.view(-1,v2t_query_tokens.shape[-1]) @ model.clip.fusion_memory.T),dim=-1)
            v2t_fusion_tokens = (v2t_query_weight @ model.clip.fusion_memory).view(-1,8,model.clip.fusion_memory.shape[-1])
            vtm_inputs = torch.cat((all_video_outputs[vid:vid+1].repeat(candidate_text_outputs.shape[0],1,1),v2t_fusion_tokens,candidate_text_outputs),dim=1)
            vtm_masks = torch.cat((all_video_mask[vid:vid+1].repeat(candidate_text_masks.shape[0],1,1),candidate_text_masks),dim=-1)
            vtm_masks = vtm_masks.repeat(1,vtm_masks.shape[-1],1)

            fusion_masks = vtm_masks.clone()
            fusion_masks[:,:max_video_len+1,max_video_len+9:] = torch.full(fusion_masks[:,:max_video_len+1,max_video_len+9:].shape, float('-inf')).to(vtm_masks.device)
            fusion_masks[:,max_video_len+9:,:max_video_len+1] = torch.full(fusion_masks[:,max_video_len+9:,:max_video_len+1].shape, float('-inf')).to(vtm_masks.device)

            fusion_input = vtm_inputs.clone()
            fusion_labels = vtm_masks.clone()
            fusion_labels[fusion_labels==0] = 1
            fusion_labels[fusion_labels==float('-inf')] = 0
            fusion_labels = fusion_labels.long()
            sep_idx = torch.cat((fusion_labels[:,0:1,:max_video_len+1].sum(dim=-1),max_video_len+fusion_labels[:,0:1,max_video_len+1:].sum(dim=-1)),dim=-1)
            pos_emd = model.clip.fusion_positional_embedding[:vtm_inputs.size(1), :]
            vtm_inputs = vtm_inputs + pos_emd
   
            #去除SEP的Positional Embedding
            for i in range(vtm_inputs.shape[0]):
                vtm_inputs[i,sep_idx[i,0]] = fusion_input[i,sep_idx[i,0]]
                vtm_inputs[i,sep_idx[i,1]] = fusion_input[i,sep_idx[i,1]]

            vtm_inputs = vtm_inputs.permute(1, 0, 2)  # NLD -> LND
            vtm_outputs = model.clip.fusion_transformer(vtm_inputs.half(),mask=fusion_masks.half(),task='fusion')
            vtm_outputs = vtm_outputs.permute(1, 0, 2) # LND -> NLD
            
            vtm_v_sep = vtm_outputs[:,max_video_len,:]
            vtm_t_sep = torch.cat([vtm_outputs[i][vtm_masks[i][0]==0][-1:] for i in range(vtm_outputs.shape[0])],dim=0)
            match_scores = model.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ model.clip.fusion_match_matrix.half() @ model.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))
            unmatch_scores = model.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ model.clip.fusion_unmatch_matrix.half() @ model.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))
            diag_mask = torch.ones(match_scores.shape[0])
            diag_mask = torch.diag(diag_mask)
            match = match_scores[diag_mask==1].view(-1,1)
            unmatch = unmatch_scores[diag_mask==1].view(-1,1)
            vtm_predicted = torch.cat((unmatch,match),dim=-1).float()
            #vtm_predicted = model.clip.fusion_vtm_linear(vtm_cls_sep)
            #v2t_matrix[vid][v2t_candidates[vid]] = v2t_matrix[vid][v2t_candidates[vid]] + vtm_predicted[:,1].float()
            vtm_predicted = torch.softmax(vtm_predicted,dim=-1) * 1000.
            v2t_matrix[vid][v2t_candidates[vid]] =  torch.softmax(v2t_matrix[vid][v2t_candidates[vid]],dim=-1) * vtm_predicted[:,1] + 1000.
            
            
        #t2v_matrix = torch.softmax(t2v_matrix,dim=-1)
        #v2t_matrix = torch.softmax(v2t_matrix,dim=-1)


    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        #tv_metrics = compute_metrics(sim_matrix)
        tv_metrics = compute_metrics(t2v_matrix.cpu().numpy())
        #vt_metrics = compute_metrics(sim_matrix.T)
        vt_metrics = compute_metrics(v2t_matrix.cpu().numpy())
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    model.train()
    R1 = tv_metrics['R1']
    return R1



def train_epoch(epoch, args, model, cc3m_train_dataloader, webvid_train_dataloader, test_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    cc3m_iter = iter(cc3m_train_dataloader)
    webvid_iter = iter(webvid_train_dataloader)

    sample_distribution = torch.Tensor([len(cc3m_train_dataloader),len(webvid_train_dataloader)])
    for step in range(len(webvid_train_dataloader)):
        
        batch = next(webvid_iter)
        
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask,fusion_labels = batch
        vtm_loss,mlm_loss,plm_loss = model(input_ids, segment_ids, input_mask, video, video_mask,fusion_labels)

        # if n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu.

        if args.rank == 0:
            print(f'step is {step}, vtm_loss is',vtm_loss.item(),'mlm_loss is',mlm_loss.item(),'plm_loss is',plm_loss.item())

        loss =  vtm_loss + mlm_loss + plm_loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps


        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % 1000 ==0 and args.local_rank == 0:
                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
            if global_step % 2000 ==0 and args.rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="", optimizer=optimizer, step=global_step)
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(webvid_train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(webvid_train_dataloader)
    return total_loss, global_step


DATALOADER_DICT = {}
DATALOADER_DICT["cc3m"] = {"train":dataloader_CC3M_train, "val":None, "test":dataloader_msrvtt_test}
DATALOADER_DICT["webvid"] = {"train":dataloader_webvid_train, "val":None, "test":dataloader_msrvtt_test}
DATALOADER_DICT["msrvtt"] = {"val":dataloader_msrvtt_test, "test":dataloader_msrvtt_test}

#@light_init(params={"training_framework": "pytorch_ddp"})
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model, global_step = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                if  'fusion' in name:
                    continue
                print(name, ' is freezed')
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    
    test_dataloader, test_length = DATALOADER_DICT['msrvtt']["test"](args, tokenizer)



    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
 
    if args.do_train:
        cc3m_train_dataloader, cc3m_train_length, cc3m_train_sampler = DATALOADER_DICT['cc3m']["train"](args, tokenizer)
        webvid_train_dataloader, webvid_train_length, webvid_train_sampler = DATALOADER_DICT['webvid']["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(webvid_train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", webvid_train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
        for epoch in range(args.epochs):
            cc3m_train_sampler.set_epoch(epoch)
            webvid_train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, cc3m_train_dataloader, webvid_train_dataloader,test_dataloader,device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            # break
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = None
                ## Uncomment if want to save checkpoint
                if args.rank == 0:
                    output_model_file = save_model(epoch, args, model, type_name="", optimizer=optimizer, step=global_step)

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)

                # R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)



if __name__ == "__main__":
    main()