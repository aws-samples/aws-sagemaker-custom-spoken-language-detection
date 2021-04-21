"""

  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0
 
"""

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback
import flask
import pandas as pd
from collections import Counter
import torch

from data import create_audio_transform, load_waveform, sliding_window_partition

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

def _get_device():
    if torch.cuda.is_available():  
        device = torch.device("cuda:0")
    else:  
        device = torch.device("cpu")
    return device

class Predictor(object):
    model = None 
    transform = None
    hparams = None
    index_to_name = None

    @classmethod
    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""    
        device = _get_device()

        if self.model == None:
            self.model = torch.jit.load(os.path.join(model_path, 'model.pt')).to(device)

        if self.hparams == None:
            self.hparams = json.load(open(os.path.join(model_path, 'hparams.json'), 'r'))
        
        if self.transform == None:
            self.transform = create_audio_transform(
                sample_rate=self.hparams['sample_rate'], n_samples=self.hparams['n_samples'], random_pitch_shift=False, 
                random_crop=False, feature_type=self.hparams['feature_type'], resize=self.hparams['resize'], 
                normalize=self.hparams['normalize'], standardize=self.hparams['standardize'], 
                standardize_mean=self.hparams['standardize_mean'], standardize_std=self.hparams['standardize_std'], 
                frequency_masking=False, time_masking=False
            )

        if self.index_to_name == None:
            self.index_to_name = json.load(open(os.path.join(model_path, 'index_to_name.json'), 'r'))

    @classmethod
    def predict(self, input_data):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        
        self.get_model()

        waveform = load_waveform(input_data, sample_rate=self.hparams['sample_rate'])

        overlap_size = 2*self.hparams['sample_rate'] # 2 seconds
        min_size = 4*self.hparams['sample_rate'] # 4 seconds

        partitions = sliding_window_partition(
            waveform, window_size=self.hparams['n_samples'], overlap_size=overlap_size, min_size=min_size, dim=-1
        )

        self.model.eval()
        with torch.no_grad():
            preds, scores = [], []
            for x in partitions:
                features = self.transform(x).unsqueeze(0)
                outputs = self.model(features)
                scores.append(outputs.squeeze())
                preds.append(outputs.argmax(-1).item())

        prediction = max(set(preds), key=preds.count)

        if self.index_to_name:
            preds = dict(Counter([self.index_to_name[str(x)] for x in preds]))
            prediction = self.index_to_name[str(prediction)]
        return prediction
    
# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    Predictor.get_model()
    health = (
        (Predictor.model != None) and
        (Predictor.hparams != None) and
        (Predictor.transform != None) and
        (Predictor.index_to_name != None)
    )

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    if flask.request.content_type == 'application/json':
        data = flask.request.get_json()
    else:
        return flask.Response(response='This predictor only supports application/json data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(len(data)))

    # Do the prediction
    predictions = []
    for f in data:
        predictions.append(Predictor.predict(f))

    return json.dumps(predictions), 200, {'content-type':'application/json'}