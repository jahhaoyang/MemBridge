unset OMPI_MCA_plm_rsh_agent
DATA_PATH=/apdcephfs/share_1324356/zifengchai/smart/data/data/CC3M
python3 -m torch.distributed.launch \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=2 --n_display=50 \
--cc3m_json ${DATA_PATH}/CC3M_Train.json \
--webvid_json /apdcephfs/share_1324356/zifengchai/smart/data/data_CLIP4clip/frozen_train_labels.json \
--val_csv /apdcephfs/share_1324356/zifengchai/smart/data/data_CLIP4clip/MSRVTT_JSFUSION_test.csv \
--features_path ${DATA_PATH}/training \
--output_dir Webvid_Pretrain_VTM_fusion_memory \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 8 \
--datatype webvid --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model   /apdcephfs/share_1324356/zifengchai/smart/data/zhw/Webvid_Pretrain_fusion_memory/ckpt/pytorch_model.bin.14000 \
--n_gpu 1