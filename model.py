import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(x + self.block(x))

class LandmarkClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [256, 256, 256, 128]):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(0.3)
        )
        layers = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ResidualBlock(hidden_dims[i]))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU(), nn.Dropout(0.2)
                ))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.layers(self.input_proj(x)))


class HybridRNN(nn.Module):
    def __init__(self, pretrained_model, hidden_rnn_size=64, num_classes=7, num_layers=1):
        super(HybridRNN, self).__init__()
        self.encoder_proj = pretrained_model.input_proj
        self.encoder_layers = pretrained_model.layers
        
        with torch.no_grad():
            dummy = torch.zeros(2, 125)
            out = self.encoder_layers(self.encoder_proj(dummy))
            encoder_out_dim = out.shape[1]
        
        # 2. Temporal Layer: LSTM
        self.lstm = nn.LSTM(
            input_size=encoder_out_dim,
            hidden_size=hidden_rnn_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_rnn_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        b, s, f = x.size()
        x = x.view(b * s, f) 
        features = self.encoder_layers(self.encoder_proj(x)) # [Batch*Seq, 128]
        features = features.view(b, s, -1) # [Batch, Seq, 128]
        lstm_out, (h_n, c_n) = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        logits = self.classifier(last_output)
        
        return logits