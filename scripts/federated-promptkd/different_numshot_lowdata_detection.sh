#!/bin/bash


# custom config
DATA="/media/his/7B2F59427DA77DE4/wzw/second_paper_code/cross_modal_adaptation/data"
TRAINER=PromptSRC

DATASET=$1
SEED=1
LOADEP=5
#CFG=fed_vit_b16_different_num_shot
CFG=fed_vit_b16_different_num_shot_light_colour_pollution

SHOTS=1

experiment='different_num_shot_light_colour_pollution'

DIR=output/feb_data_detection/${experiment}/${DATASET}/${TRAINER}/${experment}/seed${SEED}
COMMON_DIR=fed_vit_b16_${experiment}/seed${SEED}
MODEL_DIR=output/${DATASET}/${TRAINER}/${COMMON_DIR}


python fed_data_detection.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base



