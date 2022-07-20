#!/bin/bash
# run script : bash 

NAMES=("T_vivid_resnet18_indoor")
RESULTS_DIR=results

for NAME in ${NAMES[@]}; do
	echo "NAME : ${NAME}"

	# depth
	echo "Indoor Depth : well-lit results"
	SEQS=("indoor_aggresive_local" "indoor_robust_varying_well_lit")
	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"
		cat ${RESULTS_DIR}/${NAME}/Depth/${SEQ}/eval_depth.txt 
	done

	echo "Indoor Depth : low-light results"
	SEQS=("indoor_robust_dark" "indoor_unstable_dark" "indoor_aggresive_dark" )
	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"
		cat ${RESULTS_DIR}/${NAME}/Depth/${SEQ}/eval_depth.txt 
	done

	# pose
	SEQS=("indoor_aggresive_local" "indoor_robust_varying" "indoor_robust_dark" "indoor_unstable_dark" "indoor_aggresive_dark" )
	echo "Indoor Pose : All results"
	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"
		cat ${RESULTS_DIR}/${NAME}/POSE/${SEQ}/eval_pose.txt 
	done
done

NAMES=("T_vivid_resnet18_outdoor")
for NAME in ${NAMES[@]}; do
	echo "NAME : ${NAME}"

	# depth
	echo "Outdoor Depth : night-time results"
	SEQS=("outdoor_robust_night1" "outdoor_robust_night2" )
	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"
		cat ${RESULTS_DIR}/${NAME}/Depth/${SEQ}/eval_depth.txt 
	done

	# pose
	SEQS=("outdoor_robust_night1" "outdoor_robust_night2" )
	echo "Outdoor Pose : night-time results"
	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"
		cat ${RESULTS_DIR}/${NAME}/POSE/${SEQ}/eval_pose.txt 
	done
done

