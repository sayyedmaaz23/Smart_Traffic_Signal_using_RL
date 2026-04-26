from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)
CORS(app)

class DQN(nn.Module):
    def __init__(self, state_dim=9, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# Load saved model
model = DQN()
checkpoint = torch.load('dqn_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['policy_net'])
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    state = request.json['state']          # list of 9 floats
    x = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        qvals = model(x).squeeze().tolist()
    return jsonify({'qvals': qvals, 'action': int(np.argmax(qvals))})


@app.route('/q_table.json')
def get_qtable():
    return send_from_directory('.', 'q_table.json')


if __name__ == '__main__':
    app.run(port=5050)


