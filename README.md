# SongDreamer

This is a GPT inspired song writing assistant. 

## Get started

Setup the environment using the following commands.

```
conda create -n songdreamer
conda activate songdreamer
conda install -c anaconda python=3.10 pip
pip install -r requirements.txt
```

## Train

Run the following command to train

```
(songdreamer) $ python train.py
```

This will download the necessary datasets and put the models in the models folder. You might want to back this models on to S3 for serving on another machine.

## Serving

To generate some samples check out `server.py`

```
(songdreamer) $ python server.py
```

The server starts on port 4000 and is setup for CORS.