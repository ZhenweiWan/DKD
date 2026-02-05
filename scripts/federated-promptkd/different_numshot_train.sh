#!/bin/bash


# custom config
DATA="/media/his/7B2F59427DA77DE4/wzw/second_paper_code/cross_modal_adaptation/data"
TRAINER=PromptSRC

DATASET=$1
SEED=1

CFG=fed_vit_b16_different_num_shot_pollution_ablation
SHOTS=1


DIR=output/${DATASET}/${TRAINER}/${CFG}/seed${SEED}

python fed_train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base



