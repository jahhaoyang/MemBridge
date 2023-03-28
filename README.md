# MemBridge: Video-Language Pre-training with Memory-Augmented Inter-Modality Bridge

The implementation of paper [**MemBridge: Video-Language Pre-training with Memory-Augmented Inter-Modality Bridge**].

Video-language pre-training has attracted considerable attention recently for its promising performance on various downstream tasks. Most existing methods utilize the
modality-specific or modality-joint representation architectures for the cross-modality pre-training. Different from previous methods, this paper presents a novel architecture named Memory augmented Inter-Modality Bridge (MemBridge), which uses the learnable intermediate modality representations as the bridge for the interaction between videos and language. Specifically, in the transformer-based cross-modality encoder, we introduce the learnable bridge tokens as the interaction approach, which means the video and language tokens can only perceive information from bridge tokens and themselves. Moreover, a memory bank is proposed to store abundant modality interaction information for adaptively generating bridge tokens according to different cases, enhancing the capacity and robustness of the inter-modality bridge. Through pre-training, MemBridge explicitly models the representations for more sufficient inter-modality interaction. Comprehensive experiments show state-of-the-art performance on various downstream tasks including video-text retrieval, video captioning, and video question answering on multiple datasets, demonstrating the effectiveness of the proposed method.


## Requirement
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip3 install ftfy regex tqdm
pip3 install opencv-python boto3 requests pandas
pip3 install tensorflow==2.4.0
pip3 install java java-1.8.0-openjdk-devel
```

## Trained Models

The code and pre-trained models will be open sourced soon.
