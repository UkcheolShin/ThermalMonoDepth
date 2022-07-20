#!/bin/bash
# run script : bash run_vivid_inference.sh

NAMES=("T_vivid_resnet18_indoor")  

DATA_ROOT=/HDD/Dataset_processed/VIVID_256/
RESNET=18
IMG_H=256
IMG_W=320
GPU_ID=0

for NAME in ${NAMES[@]}; do
	echo "Name : ${NAME}"
	RESULTS_DIR=results/${NAME}
	POSE_NET=checkpoints/${NAME}/exp_pose_disp_model_best.pth.tar
	DISP_NET=checkpoints/${NAME}/dispnet_disp_model_best.pth.tar

	SEQS=("indoor_robust_dark"	"indoor_robust_varying" "indoor_aggresive_dark"	
		"indoor_aggresive_local" "indoor_unstable_dark" "indoor_robust_varying_well_lit")

	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"

		CUDA_VISIBLE_DEVICES=${GPU_ID} python run_inference.py ${DATA_ROOT} --resnet-layers $RESNET \
		--output-dir $RESULTS_DIR --sequence ${SEQ} --scene_type indoor --pretrained-model $DISP_NET 
	done

	SEQS=("outdoor_robust_night1" "outdoor_robust_night2")

	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"

		CUDA_VISIBLE_DEVICES=${GPU_ID} python run_inference.py ${DATA_ROOT} --resnet-layers $RESNET \
		--output-dir $RESULTS_DIR --sequence ${SEQ} --scene_type outdoor --pretrained-model $DISP_NET 
	done
done
