#!/bin/bash

#download LjSpeech
echo "download LjSpeech"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir "$1"
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 "$1/LJSpeech-1.1"

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt "$1"

#download Waveglow
echo "download Waveglow"
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mv waveglow_256channels_ljs_v2.pt "$2/waveglow_256channels.pt"

#download mels
echo "download mels"
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz >> /dev/null
echo $(ls mels | wc -l)
mv mels "$1/mels"

#download alignments
echo "download alignments"
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip -o /dev/null
unzip alignments.zip >> /dev/null
mv alignments "$1/alignments"

#download checkpoint
gdown https://drive.google.com/u/0/uc?id=1VVyVwQxDtKyWCvpMgI069V5tZJRIuKAO
mv final_model.pth.tar "$2"

#download results
gdown https://drive.google.com/u/0/uc?id=1YusPal6ao80Ou8sRSNZSEAljOA7gmMSc
unzip final_model.zip >> /dev/null
mv final_model "$3/final_model"
