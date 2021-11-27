
# Towards More Reliable Text Classification on Edge Devices via a Human-in-the-Loop

Requires Python 3.8.8

## Content

File/Folder|  Describtion
--- | --- 
`./eval` | Scripts to run the experiments
`./eval/encoding` | Scripts to create encodings
`./src` | Source code of the models / preprocessing steps

## Required External Files (save in root directory)
- IMDB Dataset: 

`wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`

`tar -zxvf aclImdb_v1.tar.gz`

- App Store Dataset (is already included)

`wget -qO dataset.csv "https://drive.google.com/uc?export=download&id=19tJyVyUo2IE8B8rJ8S9n8mJ4Cxsidz4c"`

- Hate Speech Dataset (is already included)

`wget -qO labeled_data.csv "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"`

## Run Experiments
### Create Encodings
- `cd eval/encodings`
- `python run_encode_[*dataset*].py`

### Run Models
- `cd eval`
- `python run_[*dataset*].py >> output_file.txt`

*dataset*: appstore, hatespeech, imdb, reuters
