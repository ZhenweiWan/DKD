#!/bin/bash

# custom config
DATA="/media/yht/37aa263c-dee5-4f68-ac01-07813ff4a404/wzw/fed-promptsrc/fed-promptsrc-data"
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

#CFG=fed_vit_b16_different_num_shot_light_colour_pollution
CFG=fed_train_only
SHOTS=0


DIR=output_fed_zzh/fed_train_only/${DATASET}/${TRAINER}/${CFG}/seed${SEED}

python fed_train_DKD-FL.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT 1000.0 \
#    --dirichlet False \



