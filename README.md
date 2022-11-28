# Homework 2 (TTS)

TTS homework repository of HSE DLA course. The goal of the project is to implement FastSpeech 2 model and train it on LJ-Speech dataset.

### Installation

To install necessary python packages run the command:

`pip install -r requirements.txt`

### Resources

Download all needed resources (data, checkpoints & inference examples) with

`python3 bin/download.py`

If you use Yandex DataSphere, specify the config

`python3 bin/download.py -c datasphere`

**One may use** `datasphere.ipynb` **notebook that contains all necessary commands to reproduce the results of the project.**

### Training

Once the resources are downloaded, start the training with

`python3 train.py`

or

`python3 train.py -c datasphere`

for DataSphere `g1.1` configuration.

### Generating test audio

Use

`python3 inference.py final_model`

or

`python3 inference.py final_model -c datasphere`

to generate audio files with the following configurations:

* usual generated audio
* audio with +20%/-20% for pitch/speed/energy
* audio with +20/-20% for pitch, speed and energy together

The audio is stored at `resources/final_model`.

### General usage

Run `python3 some_script.py -h` for help.
