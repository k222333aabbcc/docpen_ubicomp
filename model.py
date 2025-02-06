import os
import json
import sys
import torch
from torch import nn
from torch.nn import functional as F
import pickle
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch_geometric.nn import GCNConv
from makeData import CustomDataset
from utils.argparser import parser
from SA_ConvLSTM_Pytorch.convlstm.model import ConvLSTMParams
from SA_ConvLSTM_Pytorch.convlstm.seq2seq import Seq2Seq
from SA_ConvLSTM_Pytorch.self_attention_memory_convlstm.seq2seq import SAMSeq2Seq
from SA_ConvLSTM_Pytorch.core.constants import DEVICE, WeightsInitializer


train_log_path = ''
test_log_path = ''
weights_path = ''
min_loss = [999]


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_index):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, node_dim=1)
        self.conv2 = GCNConv(hidden_dim, output_dim, node_dim=1)
        self.leakyReLU = nn.LeakyReLU()
        self.edge_index = edge_index

    def forward(self, x):
        self.edge_index = self.edge_index.to(x.device)
        x = self.conv1(x, self.edge_index)
        x = self.leakyReLU(x)
        x = self.conv2(x, self.edge_index)
        return x
    
    
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        xshape = x.shape
        return x.view(xshape[0], xshape[1], xshape[2] * xshape[3], xshape[4])
    

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        xshape = x.shape
        return x.reshape(xshape[0] * xshape[1], xshape[2], xshape[3], xshape[4])
    
    
class DownSeqEn(nn.Module):
    def __init__(self, seq_length, seq_layers, conv_lstm_params):
        super(DownSeqEn, self).__init__()
        self.seq = nn.Sequential(
                Seq2Seq(input_seq_length=seq_length, num_layers=seq_layers, num_kernels=3, convlstm_params=conv_lstm_params, return_sequences=True).cuda(),
                View((2048, 16, 180, 6)),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1),
                nn.LeakyReLU(),
            ).cuda()
    
    def forward(self, x):
        x = self.seq(x)
        return x
    
    
class DownSeqDe(nn.Module):
    def __init__(self, output_dim, down_params):
        super(DownSeqDe, self).__init__()
        self.seq = nn.Sequential(
                nn.Linear(1652, down_params['hidden_dim1']),
                nn.Dropout(down_params['dropout1']),
                nn.LeakyReLU(),
                nn.Linear(down_params['hidden_dim1'], down_params['hidden_dim2']),
                nn.Dropout(down_params['dropout2']),
                nn.LeakyReLU(),
                nn.Linear(down_params['hidden_dim2'], output_dim)
            ).cuda()
    
    def forward(self, x):
        x = self.seq(x)
        return x
    
    
class PenSeq(nn.Module):
    def __init__(self, seq_length, seq_layers, conv_lstm_params):
        super(PenSeq, self).__init__()
        self.seq = nn.Sequential(
                Seq2Seq(input_seq_length=seq_length, num_layers=seq_layers, num_kernels=3, convlstm_params=conv_lstm_params, return_sequences=True).cuda(),
                nn.Flatten(),
                nn.Linear(3200, 1024)
            ).cuda()
        
    def forward(self, x):
        return self.seq(x)


class DHD(nn.Module):
    def __init__(self, channels, seq_length, seq_layers, output_dim, down_params, hover_params, conv_lstm_params: ConvLSTMParams, use_sam):
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
        
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=12, kernel_size=4, padding=(1, 0))
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, padding=1)
        self.tpconv = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=2, output_padding=1)
        self.leakyReLU = nn.LeakyReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.downseqen = DownSeqEn(seq_length=seq_length, seq_layers=seq_layers, conv_lstm_params=self.conv_lstm_params)
        self.hoverseqen = DownSeqEn(seq_length=seq_length, seq_layers=seq_layers, conv_lstm_params=self.conv_lstm_params)
        self.line = nn.Linear(1652, 1024)
        self.line2 = nn.Linear(1652 * 4, 1024)
        self.downseqde = DownSeqDe(output_dim=400, down_params=down_params)
        self.penseq = PenSeq(seq_length=seq_length, seq_layers=seq_layers, conv_lstm_params=self.conv_lstm_params2)
        
        edge_index = torch.tensor([
                [0, 0, 1, 1, 2, 2, 3, 4],
                [1, 3, 0, 2, 1, 4, 0, 2]
            ])
        self.gcn = GCN(input_dim=1024, hidden_dim=512, output_dim=256, edge_index=edge_index)
        
        self.decoder = nn.Sequential(
            nn.Linear(256 * 5, hover_params['hidden_dim1']),
            nn.Dropout(hover_params['dropout1']),
            nn.LeakyReLU(),
            nn.Linear(hover_params['hidden_dim1'], hover_params['hidden_dim2']),
            nn.Dropout(hover_params['dropout2']),
            nn.LeakyReLU(),
            nn.Linear(hover_params['hidden_dim2'], output_dim)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.stack([torch.cat((x[:, :, :, :3], x[:, :, :, 10:]), dim=3), torch.cat((x[:, :, :, 3:6], x[:, :, :, 10:]), dim=3), x[:, :, :, 6:10]], dim=3)
        x = x.view(batch_size, 3, self.seq_length, 10, self.channels, 4)
        x = torch.transpose(x, 3, 4)
        x = x.view(batch_size * self.seq_length * 3, self.channels, 10, 4)
        x = self.conv2(x)
        x = x.squeeze(3).view(batch_size * self.seq_length * 3, 3, 4, 9)
        x = self.tpconv(x)
        x = x.view(batch_size, 3, self.seq_length, self.channels, 6, 16)
        
        x = torch.transpose(x, 2, 3)
        x = self.leakyReLU(x)
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
        
        d1 = self.downseqde(x1)
        d2 = self.downseqde(x3)
        d1 = d1.view(batch_size, 200, 2)
        d2 = d2.view(batch_size, 200, 2)
        
        x1 = self.line(x1)
        x2 = self.line2(x2)
        x3 = self.line(x3)
        
        y1 = torch.transpose(d1, 1, 2)
        y2 = torch.transpose(d2, 1, 2)
        y1 = y1.view(batch_size, 2, 20, 10, 1)
        y2 = y2.view(batch_size, 2, 20, 10, 1)
        y1 = self.penseq(y1)
        y2 = self.penseq(y2)
        
        h = torch.stack([x1, x2, x3, y1, y2], dim=1)
        h = self.conv1(h)
        h = self.gcn(h)
        
        h = h.view(batch_size, -1)
        h = self.decoder(h)
        
        return d1, d2, h
    
    def _get_dim(self):
        # Dummy pass through the encoder to calculate the output size
        dummy_input = torch.zeros(1, self.channels, 10, self.conv_lstm_params['frame_size'][0], self.conv_lstm_params['frame_size'][1]).to(device)
        dummy_input2 = torch.zeros(1, 2, self.seq_length, self.conv_lstm_params2['frame_size'][0], self.conv_lstm_params2['frame_size'][1]).to(device)
        dummy_output = self.seq11(dummy_input)
        dummy_output2 = self.seq21(dummy_input2)
        return dummy_output.numel() + dummy_output2.numel()

# Loss function
def loss_function(od1, od2, oh, td, th):
    k = 263
    od1, od2, oh, td, th = od1 * k, od2 * k, oh * k, td * k, th * k

    batch_size = od1.shape[0]
    tds = torch.split(td, 1, dim=1)
    td1 = tds[0].squeeze(1)
    td2 = tds[1].squeeze(1)
    
    def nle(od, td):
        ox, oy = od[:, :, 0], od[:, :, 1]
        tx, ty = td[:, :, 0], td[:, :, 1]
        btx, bty = torch.max(tx, dim=1).values - torch.min(tx, dim=1).values, torch.max(ty, dim=1).values - torch.min(ty, dim=1).values
        diag_squ = btx ** 2 + bty ** 2
        diag_squ = torch.max(diag_squ, torch.ones_like(diag_squ))
        sumloss = ((ox - tx) ** 2 + (oy - ty) ** 2) / diag_squ.unsqueeze(1)
        nle_loss = torch.sqrt(sumloss).mean()
        return nle_loss
    
    def med(od, td):
        ox, oy = od[:, :, 0], od[:, :, 1]
        tx, ty = td[:, :, 0], td[:, :, 1]
        sumloss = ((ox - tx) ** 2 + (oy - ty) ** 2)
        med_loss = torch.sqrt(sumloss).mean()
        return med_loss

    def ed(oh, th):
        ox, oy = oh[:, 0], oh[:, 1]
        tx, ty = th[:, 0], th[:, 1]
        sumloss = ((ox - tx) ** 2 + (oy - ty) ** 2)
        ed_loss = torch.sqrt(sumloss).mean()
        return ed_loss

    loss1 = F.mse_loss(od1, td1, reduction='mean')
    loss2 = F.mse_loss(od2, td2, reduction='mean')
    loss3 = F.mse_loss(oh, th, reduction='mean')
    loss4 = F.cosine_embedding_loss(oh, th, torch.tensor([1.0]).to(th.device), reduction='mean')
    
    loss5 = nle(od2, td2)
    loss6 = med(od2, td2)
    loss7 = ed(oh, th)
    
    return loss1, loss2, loss3, loss4, loss5, loss6, loss7

# Training function
def train(model, train_loader, optimizer, scheduler, epoch, log_interval=50, loss_type='MSE', loss_k=[1, 1, 1, 1]):
    model.train()
    train_loss = 0
    train_loss1, train_loss2, train_loss3, train_loss4, train_loss5, train_loss6, train_loss7 = 0, 0, 0, 0, 0, 0, 0
    for batch_idx, (data, target, data2) in enumerate(train_loader):
        data, target, data2 = data.cuda(), target.cuda(), data2.cuda()
        optimizer.zero_grad()
        output1, output2, output3 = model(data)
        loss1, loss2, loss3, loss4, loss5, loss6, loss7 = loss_function(output1, output2, output3, data2, target)
        loss = loss1 * loss_k[0] + loss2 * loss_k[1] + loss3 * loss_k[2] + loss4 * loss_k[3]
        loss.backward()
        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss4 += loss4.item()
        train_loss5 += loss5.item()
        train_loss6 += loss6.item()
        train_loss7 += loss7.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    for param_group in optimizer.param_groups:
        if param_group['lr'] > 1e-4:
            param_group['lr'] = max(param_group['lr'] * 0.95, 1e-4)

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.6f}')

# Test function
def test(model, test_loader, epoch, loss_type='MSE', loss_k=[1, 1, 1, 1], base_dir='', id=''):
    model.eval()
    test_loss = 0
    test_loss1, test_loss2, test_loss3, test_loss4, test_loss5, test_loss6, test_loss7 = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for data, target, data2 in test_loader:
            data, target, data2 = data.cuda(), target.cuda(), data2.cuda()
            output1, output2, output3 = model(data)
            loss1, loss2, loss3, loss4, loss5, loss6, loss7 = loss_function(output1, output2, output3, data2, target)
            loss = loss1 * loss_k[0] + loss2 * loss_k[1] + loss3 * loss_k[2] + loss4 * loss_k[3]
            test_loss += loss.item()
            test_loss1 += loss1.item()
            test_loss2 += loss2.item()
            test_loss3 += loss3.item()
            test_loss4 += loss4.item()
            test_loss5 += loss5.item()
            test_loss6 += loss6.item()
            test_loss7 += loss7.item()

    print(f'====> Test set loss: {test_loss:.6f}')
    print({
        'Test set loss': test_loss / len(test_loader),
        'mse_d1': test_loss1 / len(test_loader),
        'mse_d2': test_loss2 / len(test_loader),
        'mse_h': test_loss3 / len(test_loader),
        'cos': test_loss4 / len(test_loader),
        'nle': test_loss5 / len(test_loader),
        'med': test_loss6 / len(test_loader),
        'ed': test_loss7 / len(test_loader),
    })
    
    min_loss.sort(reverse=True)
    if min_loss[0] > test_loss:
        weights_path = base_dir + 'work_weights_' + id + '_loss_' + str(min_loss[0]) + '.pth'
        if os.path.exists(weights_path):
            os.remove(weights_path)
        min_loss[0] = test_loss
        weights_path = base_dir + 'work_weights_' + id + '_loss_' + str(test_loss) + '.pth'
        torch.save(model.state_dict(), weights_path)

def load_data(pkl_file):
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
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
    use_sam = params['sam']
    
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
    
    model = DHD(channels, seq_length, seq_layers, output_dim, down_params, hover_params, conv_lstm_params, use_sam)

    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = params['epochs']
    data_pkl = params['dataset']
    test_pkl = params['testset']

    if not params['testset'] or len(params['testset']) == 0:
        cut = params['cut']
        all_data = load_data(data_pkl)
        
        indices = torch.randperm(len(all_data)).tolist()
        with open(base_dir + 'indices_' + str(params['id']) + '.json', 'w') as file:
            json.dump(indices, file)
        train_indices = indices[int(len(indices) * cut):]
        test_indices = indices[:int(len(indices) * cut)]
        train_data = Subset(all_data, train_indices)
        test_data = Subset(all_data, test_indices)
    else:
        # train_data = load_data(data_pkl)
        test_data = load_data(test_pkl)

    batch_size = params['batch_size']
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    weights_path = 'work_weights.pth'
    state_dict = torch.load(weights_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    for epoch in range(1, epochs + 1):
        # train(model, train_loader, optimizer, scheduler, epoch, loss_type=loss_type, loss_k=loss_k)
        test(model, test_loader, epoch, loss_type=loss_type, loss_k=loss_k, base_dir=base_dir, id=str(params['id']))
        return

    torch.save(model.state_dict(), weights_path)


def run_one(name):
    params = {
        'project': 'docpen',
        'name': name,
        'id': 1,
        'base_dir': './outputs/' + name + '/',
        'overwrite': True,
        'dataset': '',
        'testset': 'test_set.pkl',
        'cut': 0.2,
        'channels': 3,
        'seq_length': 20,
        'seq_layers': 1,
        'hidden_dim': 256,
        'output_dim': 2,
        'in_channels': 3,
        'out_channels': 16,
        'kernel_size': 3,
        'padding': 1,
        'activation': 'leakyRelu',
        'frame_size': (6, 16),
        'weights_initializer': 'zeros',
        'lr': 1e-4,
        'batch_size': 16384,
        'epochs': 8000,
        'loss_type': 'MSE',
        'loss_k': [1, 1, 1.5, 50],
        'sam': False
    }
    
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value != None:
            params[key] = value
            
    start_run(params)


if __name__ == '__main__':
    run_one('test')
