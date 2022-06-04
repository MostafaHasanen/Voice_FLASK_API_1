from flask import Flask, request, jsonify
import numpy as np
import keras
import librosa
import joblib

app = Flask(__name__)
model = keras.models.load_model('Level_2_full_model.h5')
encoder_Y = joblib.load("one_hot_encoder_Y.joblib")
scaler_X = joblib.load("scaler_X.joblib")

@app.route('/')

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
  
    return result  

def index_page():
	return jsonify({ 'meaasge': 'This is the api which returns the type of Basic Emotion.'
				 ,'Parameters required':'path' })

@app.route('/predict', methods=['POST'])
def predict():
    query= request.args
    path = query.get('path') # it pass string
    data, sample_rate = librosa.load(path,sr=8025) # downsampling sr=8025
    result = extract_features(data,sample_rate)
    result = scaler_X.transform(result.reshape(1, -1))
    result = np.expand_dims(result, axis=2)
    prediciton = model.predict(result)
    pred_name = encoder_Y.inverse_transform(prediciton)
    return jsonify({'prediction' : pred_name[0][0]})
if __name__ == "__main__":
	app.run(debug=True,host='0.0.0.0',port=5000)
