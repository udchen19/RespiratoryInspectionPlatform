# setup
import torch
from torch import nn, optim
import numpy as np
from os import listdir, path, environ
import librosa, csv, random
environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision = 5, sci_mode = False)
#device = "cpu"

# data fetching
diseases = ['Healthy', 'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'LRTI', 'Pneumonia', 'URTI']
positions = ['trachea', 'anterior left', 'anterior right', 'posterior left', 'posterior right', 'lateral left', 'lateral right']
mfcc_len = 864
mfcc_wave_len = 22050 * 20
wave_len = 100000
spectrals = []
waves = []

def mfcc_clip(file_dir, cut):
    y, sr = librosa.load(file_dir)
    audio_len = y.shape[0] / sr
    y, sr = librosa.load(file_dir, sr = mfcc_wave_len / audio_len, offset = cut, duration = audio_len - 2 * cut)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 48)
    pad_len = mfcc_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width = ((0, 0), (pad_len // 2, pad_len - pad_len // 2)), mode='constant')
    return mfcc

def clip(file_dir, cut):
    y, sr = librosa.load(file_dir)
    audio_len = y.shape[0] / sr
    y, sr = librosa.load(file_dir, sr = wave_len / audio_len, offset = cut, duration = audio_len - 2 * cut)
    if y.shape[0] >= wave_len:
        y = y[: wave_len]
    else:
        pad_len = wave_len - y.shape[0]
        y = np.pad(y, pad_width = ((pad_len // 2, pad_len - pad_len // 2)), mode='constant')
    return y

def normalize(audio):
    mean = np.mean(audio)
    sigma = np.std(audio)
    return (audio - mean) / sigma

for i in range(7):
    spectrals.append(mfcc_clip(str(i + 1) + '.wav', 4))
    wave = normalize(clip(str(i + 1) + '.wav', 4))
    waves.append(wave)

spectrals = torch.FloatTensor(np.array(spectrals)).unsqueeze(1).to(device)
waves = torch.FloatTensor(np.array(waves)).unsqueeze(1).to(device)

# models
def conv_layer(in_sz, out_sz, k_sz, padding, dropout):
    return nn.Sequential(
        nn.Conv2d(in_sz, out_sz, k_sz, padding = padding),
        nn.BatchNorm2d(out_sz),
        nn.ReLU(),
        nn.Dropout(dropout)
    )

def fc_layer(in_sz, out_sz, dropout):
    return nn.Sequential(
        nn.Linear(in_sz, out_sz),
        nn.ReLU(),
        nn.Dropout(dropout)
    )

class CNN1(nn.Module):
    def __init__(self, cv_sz, fc_sz, dropout):
        super().__init__()
        self.conv1 = conv_layer(1, cv_sz, (1, 3), (0, 1), dropout)
        self.conv2 = nn.Sequential(
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout), 
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout)
        )
        self.conv3 = nn.Sequential(
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout), 
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout)
        )
        self.conv4 = nn.Sequential(
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout), 
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout)
        )
        self.conv5 = nn.Sequential(
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout), 
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout)
        )
        self.conv6 = nn.Sequential(
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout), 
            conv_layer(cv_sz, cv_sz, (1, 3), (0, 1), dropout)
        )
        self.pool1 = nn.MaxPool2d((1, 6), (1, 6))
        self.pool2 = nn.MaxPool2d((1, 3), (1, 3))
        self.pool3 = nn.MaxPool2d((1, 2), (1, 2))
        self.fc1 = nn.Sequential(
            fc_layer(cv_sz * 192, fc_sz, dropout),
            fc_layer(fc_sz, fc_sz, dropout),
        )
        self.fc2 = nn.Linear(fc_sz, 2)
    def forward(self, x, batch_size):
        x = self.conv1(x)
        x = self.pool1(self.conv2(x) + x)
        x = self.pool2(self.conv3(x) + x)
        x = self.pool2(self.conv4(x) + x)
        x = self.pool3(self.conv5(x) + x)
        x = self.pool3(self.conv6(x) + x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return self.fc2(x)

def conv_layer2(in_sz, out_sz, k_sz, padding, dropout):
    return nn.Sequential(
        nn.Conv1d(in_sz, out_sz, k_sz, padding = padding),
        nn.BatchNorm1d(out_sz),
        nn.ReLU(),
        nn.Dropout(dropout)
    )

def fc_layer2(in_sz, out_sz, dropout):
    return nn.Sequential(
        nn.Linear(in_sz, out_sz),
        nn.ReLU(),
        nn.Dropout(dropout)
    )
    
class CNN2(nn.Module):
    def __init__(self, cv_sz, fc_sz, out_sz, dropout):
        super().__init__()
        self.conv1 = conv_layer2(1, cv_sz, 5, 2, dropout)
        self.conv2 = conv_layer2(cv_sz, cv_sz, 3, 1, dropout) 
        self.conv3 = conv_layer2(cv_sz, cv_sz, 3, 1, dropout)
        self.conv4 = conv_layer2(cv_sz, cv_sz, 3, 1, dropout)
        self.pool = nn.MaxPool1d(10)
        self.fc1 = nn.Sequential(
            fc_layer2(cv_sz * 10 + 9, fc_sz, dropout),
            fc_layer2(fc_sz, fc_sz, dropout),
        )
        self.fc2 = nn.Linear(fc_sz, out_sz)
        self.out_sz = out_sz
    def forward(self, x, y):
        batch_size = torch.numel(y) // 4
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = x.view(batch_size, -1)
        z = torch.zeros(batch_size, 9)
        for i in range(batch_size):
            z[i][int(y[i][1])], z[i][7], z[i][8] = 1, y[i][2], y[i][3]
        z = z.to(device)
        x = torch.cat([x, z], 1)
        x = self.fc1(x)
        return self.fc2(x)
    
class NN(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(56, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8)
        )
    def forward(self, x):
        x = x.view(-1, 56)
        return self.fc(x)

# model1
model1a = CNN1(32, 500, 0)
model1a.load_state_dict(torch.load('lung_sound_model_spectral_crackles_v0.pkl', map_location = torch.device(device)))
model1a = model1a.to(device)
model1a.eval()
model1b = CNN1(32, 500, 0)
model1b.load_state_dict(torch.load('lung_sound_model_spectral_wheezes_v0.pkl', map_location = torch.device(device)))
model1b = model1b.to(device)
model1b.eval() 

# run model1
with torch.no_grad():
    predict = model1a(spectrals, 7)
    #print(predict)
    pred_label_a = torch.max(predict.data, 1).indices
    pred_label_a = pred_label_a.to("cpu").numpy()
    predict = model1b(spectrals, 7)
    #print(predict)
    pred_label_b = torch.max(predict.data, 1).indices
    pred_label_b = pred_label_b.to("cpu").numpy()

# specs
labels = np.zeros((7, 4))
for i in range(7):
    labels[i][1] = i
    labels[i][2] = int(pred_label_a[i])
    labels[i][3] = int(pred_label_b[i])
labels = torch.LongTensor(labels)

# model2
model2 = CNN2(32, 500, 8, 0)
model2.load_state_dict(torch.load('classify_model_wave_v3.7_0313_success2.pkl', map_location = torch.device(device)))
model2 = model2.to(device)
model2.eval()

# run model2
predict = []
with torch.no_grad():
    for i in range(7):
        predict.append(model2(waves[i].unsqueeze(0), labels[i].unsqueeze(0))[0])
    #print(predict)
predict = torch.stack(predict)
#predict[0] = torch.ones(8) * 0.125
#print(nn.functional.softmax(predict, dim = 0))

# model3
model3 = NN(0)
model3.load_state_dict(torch.load('comprehensive_model_v0_0311.pkl', map_location = torch.device(device)))
model3 = model3.to(device)
model3.eval()

# run model3
#print(predict.shape)
results = torch.zeros(8)
with torch.no_grad():
    results = model3(predict)
    #print(results)
    results = results.to("cpu")
results = nn.functional.softmax(results[0], dim = 0)

# report
report = []
for i in range(8):
    report.append((results[i], diseases[i]))
report.sort(reverse=True)
print('INSPECTION REPORT')
print('-' * 20)
print('Respiratory Sound Inspection', end = '\n\n')
no_crackles = True
for i in range(7):
    if pred_label_a[i] or pred_label_b[i]:
        if no_crackles:
            no_crackles = False
            print('Caution: Crackles detected at ')
        else:
            print(', ', end = '')
        print(positions[i], end = '')
if no_crackles == False:
    print(' of your chest. We suggest to consult healthcare professionals for a more detailed diagnosis and further advice.')
no_wheezes = True
for i in range(7):
    if pred_label_a[i] or pred_label_b[i]:
        if no_wheezes:
            no_wheezes = False
            print('Caution: Wheezes detected at ')
        else:
            print(', ', end = '')
        print(positions[i], end = '')
if no_wheezes == False:
    print(' of your chest. We suggest to consult healthcare professionals for a more detailed diagnosis and further advice.')
if no_crackles and no_wheezes:
    print('We detect no odd respiratory sounds from your recordings.')
print('-' * 20)
print('Respiratory Disease Prediction', end = '\n\n')
for i in range(8):
    print('{:}'.format(report[i][1]) + ': {:>.3%}'.format(float(report[i][0])))
print()
if report[0][0] < 0.5:
    print('Caution: No disease or health state was predicted with a confidence over 50%, so we suggest to record your respiratory sounds again with less noise.')
elif report[0][1] != diseases[0]:
    print('Caution: We predicted that you might have an underlying disease, which is ' + '{}'.format(report[0][1]) + '. We suggest to consult healthcare professionals for a more detailed diagnosis and further advice.')
else:
    print('We predict that you are healthy.')
print('-' * 20)

if torch.cuda.is_available():
	torch.cuda.empty_cache()