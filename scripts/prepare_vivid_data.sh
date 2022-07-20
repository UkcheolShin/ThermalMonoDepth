# for VIVID rgb-thermal dataset
DATASET=/HDD/Dataset/KAIST_VIVID
TRAIN_SET=/HDD/Dataset_processed/VIVID_256/
mkdir -p $TRAIN_SET
python common/data_prepare/prepare_train_data_VIVID.py $DATASET --dump-root $TRAIN_SET --width 320  --height 256 --num-threads 16 --with-depth  --with-pose
