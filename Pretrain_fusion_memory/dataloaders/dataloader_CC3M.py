from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from random import sample
from dataloaders.rawvideo_util import RawVideoExtractor
from PIL import Image

class CC3M_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0

        train_video_ids = self.data.keys()
        self.sentences_dict = {}
        for vid in self.data:
            if vid in train_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.data[vid])
        self.sample_len = len(self.sentences_dict)



        self.sample_len = len(self.data)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, str(video_id))

            #if os.path.exists(video_path) is False:
               # video_path = video_path.replace(".mp4", ".webm")
            
            #raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            #raw_video_data = raw_video_data['video']

            try:
                img = self.rawVideoExtractor.transform(Image.open(video_path).convert("RGB"))
            except:
                img = Image.new('RGB', (224,224), (0, 0, 0))
                img = self.rawVideoExtractor.transform(img)
                print('Error image',video_path)
            raw_video_data = torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0)
            
            if len(raw_video_data.shape) > 3:
                # L x T x 3 x H x W
                
                raw_video_slice = raw_video_data
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def get_pretrain_mask(self, pairs_text, pairs_mask, pairs_segment, video, video_mask):
        max_video_len = video_mask.shape[-1]
        max_text_len = pairs_mask.shape[-1]
        fusion_labels = np.concatenate((video_mask[:,0:1],video_mask,video_mask[:,0:8],pairs_mask),axis=-1)[0]
        sep_idx = np.expand_dims(np.concatenate((video_mask.sum(axis=-1),max_video_len+8+pairs_mask.sum(axis=-1)),axis=-1),axis=0)
        
       #仅MLM任务
            
        mlm_mask = np.array([i for i in range(max_video_len+10,sep_idx[0][1])])
        mlm_idx = np.random.binomial(1, 0.15, len(mlm_mask))

        mask = mlm_mask[mlm_idx==1]

        
        try:
            if len(mask) == 0:
                mask = sample(mlm_mask.tolist(),1)
            fusion_labels[mask] = -1
        except:
            fusion_labels[max_video_len+10] = -1

        return fusion_labels

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        fusion_labels = self.get_pretrain_mask(pairs_text, pairs_mask, pairs_segment, video, video_mask)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels
