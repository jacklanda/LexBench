#!/bin/bash

pie_dataset_url="https://raw.githubusercontent.com/zhjjn/MWE_PIE/main/data_cleaned.csv"

asilm_ref_dataset_url="https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskA/EN/ContextIncluded_IdiomIncluded/test.csv"
asilm_idiom_dataset_url="https://raw.githubusercontent.com/H-TayyarMadabushi/AStitchInLanguageModels/main/Dataset/Task1/SubTaskB/EN/ContextIncluded/test.csv"

id10m_dataset_url="https://raw.githubusercontent.com/Babelscape/ID10M/master/resources/bio_format/english/dev_english.tsv"

# Add help for dataset downloading
if [ "$1" == "" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./download_dataset.sh [dataset_name]"
    echo -e "Available dataset_name: [asilm, id10m, pie]"
    exit 0
fi

# Leverage the first arg to select the dataset

# ASILM task1 for [Idiom Detection]
if [ "$1" == "asilm" ]; then
    echo "[Downloading] ASILM task1 dataset"
    mkdir -p "dataset/idiom_detection/"
    wget -O "dataset/idiom_detection/reference_data.csv" $asilm_ref_dataset_url
    wget -O "dataset/idiom_detection/idiom_data.csv" $asilm_idiom_dataset_url
    exit 0
fi

# ID10OM for [Idiom Extraction]
if [ "$1" == "id10m" ]; then
    echo "[Downloading] ID10M dataset"
    mkdir -p "dataset/idiom_extraction/"
    wget -P "dataset/idiom_extraction/" $id10m_dataset_url
    exit 0
fi

# PIE for [Idiom Paraphrase]
if [ "$1" == "pie" ]; then
    echo "[Downloading] PIE dataset"
    mkdir -p "dataset/idiom_paraphrase/"
    wget -P "dataset/idiom_paraphrase/" $pie_dataset_url
    exit 0
fi
