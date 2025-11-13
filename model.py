import os
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F
import joblib
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch_geometric.nn import GCNConv
from makeData import ImuHidDataset
from utils.argparser import parser
from SA_ConvLSTM_Pytorch.convlstm.model import ConvLSTMParams
from SA_ConvLSTM_Pytorch.convlstm.seq2seq import Seq2Seq
from SA_ConvLSTM_Pytorch.self_attention_memory_convlstm.seq2seq import SAMSeq2Seq
from SA_ConvLSTM_Pytorch.core.constants import DEVICE, WeightsInitializer


train_log_path = ''
test_log_path = ''
weights_path = ''
device_ids = DEVICE
device_list = [torch.device("cuda:" + str(id)) for id in device_ids]
min_loss = [99999999]


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        output = self.classifier(x)
        return output


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_index):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, node_dim=1)
        self.conv2 = GCNConv(hidden_dim, output_dim, node_dim=1)
        self.relu = nn.ReLU()
        self.edge_index = edge_index

    def forward(self, x):
        self.edge_index = self.edge_index.to(x.device)
        x = self.conv1(x, self.edge_index)
        x = self.relu(x)
        x = self.conv2(x, self.edge_index)
        return x
    
    
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        xshape = x.shape
        return x.view(xshape[0], xshape[1], xshape[2] * xshape[3], xshape[4])
    

class DownSeqEn(nn.Module):
    def __init__(self, seq_length, seq_layers, conv_lstm_params, device_list):
        super(DownSeqEn, self).__init__()
        if len(device_list) == 1:
            self.seq = nn.Sequential(
                Seq2Seq(input_seq_length=seq_length, num_layers=seq_layers, num_kernels=3, convlstm_params=conv_lstm_params, return_sequences=True).to(device_list[0]),
                View((2048, 16, 180, 6)),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1),
                nn.ReLU(),
            ).to(device_list[0])
        else:
            self.seq = nn.Sequential(
                Seq2Seq(input_seq_length=seq_length, num_layers=seq_layers, num_kernels=3, convlstm_params=conv_lstm_params, return_sequences=True),
                View((2048, 16, 180, 6)),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1),
                nn.ReLU(),
            )
    
    def forward(self, x):
        x = self.seq(x)
        return x
    

class DHD(nn.Module):
    def __init__(self, channels, seq_length, seq_layers, output_dim, down_params, hover_params, conv_lstm_params: ConvLSTMParams):
        super(DHD, self).__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.seq_layers = seq_layers
        self.output_dim = output_dim
        self.conv_lstm_params = conv_lstm_params.copy()
        conv_lstm_params['frame_size'] = (10, 1)
        conv_lstm_params['in_channels'] = 2
        self.conv_lstm_params2 = conv_lstm_params.copy()
        self.down_params = down_params
        self.hover_params = hover_params
        
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=12, kernel_size=4, padding=(1, 0))
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.tpconv = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=2, output_padding=1)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.downseqen = DownSeqEn(seq_length=seq_length, seq_layers=seq_layers, conv_lstm_params=self.conv_lstm_params, device_list=device_list)
        self.hoverseqen = DownSeqEn(seq_length=seq_length, seq_layers=seq_layers, conv_lstm_params=self.conv_lstm_params, device_list=device_list)
        self.line = nn.Linear(1652, 1024)
        self.line2 = nn.Linear(1652 * 4, 1024)
        self.downline = nn.Linear(1652, 400)
        self.downbiline = nn.Bilinear(400, 400, 400)
        self.penlstm = nn.LSTM(input_size=2, hidden_size=4, num_layers=2, batch_first=True, bidirectional=True)
        self.penline = nn.Linear(seq_length * 10 * 2 * 4, 1024)
        
        edge_index = torch.tensor([
                [0, 0, 1, 1, 1, 2, 2, 3, 4, 5],
                [1, 3, 0, 2, 5, 1, 4, 0, 2, 1]
            ])
        self.gcn = GCN(input_dim=1024, hidden_dim=512, output_dim=256, edge_index=edge_index)
        
        self.decoder = nn.Sequential(
            nn.Linear(256 * 6, hover_params['hidden_dim1']),
            nn.Dropout(hover_params['dropout1']),
            nn.ReLU(),
            nn.Linear(hover_params['hidden_dim1'], hover_params['hidden_dim2']),
            nn.Dropout(hover_params['dropout2']),
            nn.ReLU(),
            nn.Linear(hover_params['hidden_dim2'], output_dim)
        )
        
        self.linehx = nn.Linear(47, 1024)

        self.classifier = Classifier(256 * 6, 256, 3, 0.1)
        
    def forward(self, x, hx):
        batch_size = x.shape[0]
        x = torch.stack([torch.cat((x[:, :, :, :3], x[:, :, :, 6:]), dim=3), x[:, :, :, 3:]], dim=3)
        x = x.view(batch_size, 3, self.seq_length, 10, 2, 4)
        x = torch.transpose(x, 3, 4)
        x = x.view(batch_size * self.seq_length * 3, 2, 10, 4)
        x = self.conv2(x)
        x = x.squeeze(3).view(batch_size * self.seq_length * 3, 3, 4, 9)
        x = self.tpconv(x)
        x = x.view(batch_size, 3, self.seq_length, self.channels, 6, 16)
        
        x = torch.transpose(x, 2, 3)
        x = self.relu(x)
        x = torch.split(x, 1, dim=1)
        x1 = x[0].squeeze(1)
        x2 = x[1].squeeze(1)
        x3 = x[2].squeeze(1)
        
        x1 = self.downseqen(x1)
        x2 = self.hoverseqen(x2)
        x3 = self.downseqen(x3)
        
        x1 = self.maxPool(x1)
        x3 = self.maxPool(x3)
        
        x1 = x1.reshape(batch_size, -1)
        x2 = x2.reshape(batch_size, -1)
        x3 = x3.reshape(batch_size, -1)
        
        d1 = self.downline(x1)
        d2 = self.downline(x3)
        d1 = self.relu(d1)
        d2 = self.relu(d2)
        d1 = self.downbiline(d1, d1)
        d2 = self.downbiline(d2, d2)
        d1 = d1.view(batch_size, 200, 2)
        d2 = d2.view(batch_size, 200, 2)
        
        x1 = self.line(x1)
        x2 = self.line2(x2)
        x3 = self.line(x3)
        
        y1, _ = self.penlstm(d1)
        y2, _ = self.penlstm(d2)
        y1 = y1.reshape(batch_size, -1)
        y2 = y2.reshape(batch_size, -1)
        y1 = self.penline(y1)
        y2 = self.penline(y2)
        
        hx = self.linehx(hx)
        
        h = torch.stack([x1, x2, x3, y1, y2, hx], dim=1)
        
        h = self.gcn(h)
        
        h = h.view(batch_size, -1)
        h1 = self.decoder(h)
        
        hc = self.classifier(h)
        
        return d1, d2, h1, hc


# Loss function
def loss_function(od1, od2, oh, oc, td, th, tc, loss_k):
    k = 263
    od1, od2, oh, td, th = od1 * k, od2 * k, oh * k, td * k, th * k

    batch_size = od1.shape[0]
    td1 = td[:, 0, :, :]
    td2 = td[:, 1, :, :]

    def align_center(od, td):
        od_center = od.mean(dim=1, keepdim=True)
        td_center = td.mean(dim=1, keepdim=True)
        od = od - od_center + td_center
        return od
    
    def nle(od, td):
        sq_diff = (od - td)**2
        dx = sq_diff[..., 0]
        dy = sq_diff[..., 1]
        tx, ty = td[..., 0], td[..., 1]
        tx_min, tx_max = tx.min(dim=1, keepdim=True)[0], tx.max(dim=1, keepdim=True)[0]
        ty_min, ty_max = ty.min(dim=1, keepdim=True)[0], ty.max(dim=1, keepdim=True)[0]
        diag_squ = torch.clamp((tx_max - tx_min)**2 + (ty_max - ty_min)**2, min=1.0)
        sumloss = (dx + dy) / diag_squ
        return torch.sqrt(sumloss).mean()
    
    def aed(od, td):
        sq_dist = torch.sum((od - td)**2, dim=-1)
        return torch.sqrt(sq_dist).mean()

    def ed(oh, th):
        sq_dist = torch.sum((oh - th)**2, dim=-1)
        return torch.sqrt(sq_dist).mean()

    def acc(oc, tc, mask=None):
        _, predicted = torch.max(oc, 1)
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(1.0, device=oc.device)
            return (predicted[mask] == tc[mask]).sum() / mask.sum()
        return (predicted == tc).sum() / tc.size(0)

    od1, od2 = align_center(od1, td1), align_center(od2, td2)
    
    loss1 = F.mse_loss(od1, td1, reduction='mean')
    loss2 = F.mse_loss(od2, td2, reduction='mean')
    
    loss_ed = torch.sqrt(torch.sum((oh - th)**2, dim=-1))
    loss_h = torch.log1p(torch.sqrt(torch.sum((oh - th)**2, dim=-1)))
    loss_cos = F.cosine_embedding_loss(oh, th, torch.tensor([1.0]).to(th.device), reduction='none')
    mask = tc != 2
    if mask is not None:
        if mask.sum() == 0:
            loss3 = torch.tensor(0.0, device=oh.device)
            lossh = torch.tensor(0.0, device=oh.device)
            loss4 = torch.tensor(0.0, device=oh.device)
        else:
            loss_ed_masked = loss_ed[mask]
            loss_h_masked = loss_h[mask]
            loss_cos_masked = loss_cos[mask]
            loss3 = loss_ed_masked.mean()
            lossh = loss_h_masked.mean()
            loss4 = loss_cos_masked.mean()
    else:
        loss3 = loss_ed.mean()
        lossh = loss_h.mean()
        loss4 = loss_cos.mean()
    
    loss5 = nle(od2, td2)
    loss6 = aed(od2, td2)
    
    weight = torch.FloatTensor([1 / 0.79, 1 / 0.2, 1 / 0.01]).to(th.device)
    loss7 = F.cross_entropy(oc, tc, reduction='mean', weight=weight)
    loss8 = acc(oc, tc, mask)

    loss = loss1 * loss_k[0] + loss2 * loss_k[1] + lossh * loss_k[2] + loss4 * loss_k[3] + loss7 * loss_k[4]
    
    return loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8

# Training function
def train(model, train_loader, optimizer, scheduler, epoch, log_interval=50, loss_k=[1, 1, 1, 1], device_list=None):
    model.train()
    train_loss = 0
    train_loss1, train_loss2, train_loss3, train_loss4, train_loss5 = 0, 0, 0, 0, 0
    for batch_idx, data in enumerate(train_loader):
        if len(device_list) == 1:
            for i in range(len(data)):
                data[i] = data[i].to(device_list[0])
        else:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        optimizer.zero_grad()
        output = model(data[0], data[3])
        loss = loss_function(output[0], output[1], output[2], output[3], data[2], data[1], data[4], loss_k)
        loss[0].backward()
        train_loss += loss[0].item()
        train_loss1 += loss[1].item()
        train_loss2 += loss[2].item()
        train_loss3 += loss[3].item()
        train_loss4 += loss[4].item()
        train_loss5 += loss[7].item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data[0])}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss[0].item() / len(data[0]):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.6f}')

# Test function
def test(model, test_loader, epoch, loss_k=[1, 1, 1, 1], device_list=None, base_dir='', id=''):
    global l_k

    model.eval()
    test_losses = [0] * 9
    outputs = []
    with torch.no_grad():
        for data in test_loader:
            if len(device_list) == 1:
                for i in range(len(data)):
                    data[i] = data[i].to(device_list[0])
            else:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            output = model(data[0], data[3])
            outputs.append(output)
            loss = loss_function(output[0], output[1], output[2], output[3], data[2], data[1], data[4], loss_k)
            for i in range(len(loss)):
                test_losses[i] += loss[i].item()

    print(f'====> Test set loss: {test_losses[0]:.6f}')
    print({
        'Test set loss': test_losses[0] / len(test_loader),
        'mse1': test_losses[1] / len(test_loader),
        'mse2': test_losses[2] / len(test_loader),
        'ed': test_losses[3] / len(test_loader),
        'cos': test_losses[4] / len(test_loader),
        'nle': test_losses[5] / len(test_loader),
        'aed': test_losses[6] / len(test_loader),
        'ce': test_losses[7] / len(test_loader),
        'acc': test_losses[8] / len(test_loader)
    })
    

def load_data(pkl_file):
    data = joblib.load(pkl_file)
    return data


def check_nan_inf_in_loader(dataloader):
    for data, sample in dataloader:
        if torch.isnan(data).any():
            return True
        if torch.isnan(sample).any():
            return True
        if torch.isinf(data).any():
            return True
        if torch.isinf(sample).any():
            return True
    return False


def start_run(params):
    global device, train_log_path, test_log_path, weights_path

    base_dir = params['base_dir']
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    train_log_path = base_dir + 'log_train_' + str(params['id'])
    test_log_path = base_dir + 'log_test_' + str(params['id'])
    weights_path = base_dir + 'work_weights_' + str(params['id']) + '.pth'
    
    if not params['overwrite']:
        fs = ''
        if os.path.exists(train_log_path):
            fs += train_log_path + '  '
        if os.path.exists(test_log_path):
            fs += test_log_path + '  '
        if os.path.exists(weights_path):
            fs += weights_path + '  '
        if fs != '':
            assert False, fs + ' : file exists\n'

    # Parameters
    channels = params['channels']
    seq_length = params['seq_length']
    seq_layers = params['seq_layers']
    hidden_dim = params['hidden_dim']
    output_dim = params['output_dim']
    lr = params['lr']
    loss_type = params['loss_type']
    loss_k = params['loss_k']

    if params['weights_initializer'] == 'zeros':
        weights_initializer = WeightsInitializer.Zeros
    else:
        assert False, 'unknown weights_initializer'
    conv_lstm_params: ConvLSTMParams = {
        'in_channels': params['in_channels'],
        'out_channels': params['out_channels'],
        'kernel_size': params['kernel_size'],
        'padding': params['padding'],
        'activation': params['activation'],
        'frame_size': params['frame_size'],
        'weights_initializer': weights_initializer
    }
    
    down_params = {
        'hidden_dim1': 2048,
        'dropout1': 0.2,
        'hidden_dim2': 1024,
        'dropout2': 0.1
    }
    
    hover_params = {
        'hidden_dim1': 2048,
        'dropout1': 0.2,
        'hidden_dim2': 1024,
        'dropout2': 0.1
    }
    
    model = DHD(channels, seq_length, seq_layers, output_dim, down_params, hover_params, conv_lstm_params)

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    params['device_ids'] = device_ids
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # weights_path = './weights/work_weights.pth'
    weights_path = './weights/work_weights.pth'
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    epochs = params['epochs']
    data_pkl = params['dataset']
    test_pkl = params['testset']

    print('loading')
    # train_data = load_data(data_pkl)
    test_data = load_data(test_pkl)
    print('loaded')

    batch_size = params['batch_size']
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print('begin')
    for epoch in range(1, epochs + 1):
        # train(model, train_loader, optimizer, scheduler, epoch, loss_k=loss_k, device_list=device_list)
        test(model, test_loader, epoch, loss_k=loss_k, device_list=device_list, base_dir=base_dir, id=str(params['id']))


def run_one(name):
    params = {
        'project': 'docpen',
        'name': name,
        'id': 1,
        # 'device_ids': device_ids,
        'base_dir': './outputs/' + name + '/',
        'overwrite': True,
        'dataset': '',
        'testset': './data/test_set.pkl',
        'device': DEVICE,
        'channels': 3,
        'seq_length': 20,
        'seq_layers': 1,
        'hidden_dim': 256,
        'output_dim': 2,
        'in_channels': 3,
        'out_channels': 16,
        'kernel_size': 3,
        'padding': 1,
        'activation': 'relu',
        'frame_size': (6, 16),
        'weights_initializer': 'zeros',
        'lr': 1e-4,
        'batch_size': 16384 * 1,
        'epochs': 1,
        'loss_type': 'MSE',
        'loss_k': [1, 1, 1.5, 50, 10]
    }
    
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value != None:
            params[key] = value
                        
    start_run(params)


if __name__ == '__main__':
    run_one('test')
    