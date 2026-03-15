import json, os
import numpy as np
from tensorflow import keras
print('Model yukleniyor...')
model = keras.models.load_model('mnist_model.h5')
all_weights, weight_specs = [], []
for layer in model.layers:
    for w_array, w_var in zip(layer.get_weights(), layer.weights):
        w = w_array.astype(np.float32)
        all_weights.append(w.flatten())
        weight_specs.append({'name': w_var.name, 'shape': list(w.shape), 'dtype': 'float32'})
os.makedirs('model', exist_ok=True)
weight_data = np.concatenate(all_weights).astype(np.float32)
with open('model/group1-shard1of1.bin', 'wb') as f2:
    f2.write(weight_data.tobytes())
tfjs = {'format': 'layers-model', 'modelTopology': json.loads(model.to_json()), 'weightsManifest': [{'paths': ['group1-shard1of1.bin'], 'weights': weight_specs}]}
with open('model/model.json', 'w') as f2:
    json.dump(tfjs, f2)
print('Tamam! model/ klasoru olusturuldu')
