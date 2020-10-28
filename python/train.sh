DATASET_DIR=./dataset
TRAIN_DATA_DIR=${DATASET_DIR}/data
LABEL_FILE_PATH=${DATASET_DIR}/labels.txt
python3 train.py --data_dir ${TRAIN_DATA_DIR} --label_file_path ${LABEL_FILE_PATH}
