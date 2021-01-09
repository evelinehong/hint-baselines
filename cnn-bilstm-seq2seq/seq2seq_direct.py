from dataset import HINT, HINT_collate, SYMBOLS, SYMBOLS_AND_OPS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
       
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional = True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [batch size, src len]

        embedded = self.embedding(src)
        #embedded = [batch size, src len, emb dim]

        embedded = embedded.transpose(1,0)

        outputs, (hidden, cell) = self.rnn(embedded)    
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = len(SYMBOLS)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=True)
        
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        

        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        src = src.to(device)
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        hidden = hidden.to(device)
        cell = cell.to(device)
        
        for t in range(1, trg_len):
            input = input.to(device)
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs

OUTPUT_DIM = len(SYMBOLS)
INPUT_DIM = len(SYMBOLS_AND_OPS)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

BATCH_SIZE = 16

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

DRY_RUN = False
LOAD_BEST = True

if LOAD_BEST:
    model.load_state_dict(torch.load("./model.pt"))

train_set = HINT('train')
test_set = HINT('test')
val_set = HINT('val')

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=4, collate_fn=HINT_collate)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=4, collate_fn=HINT_collate)
eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                     shuffle=False, num_workers=4, collate_fn=HINT_collate)

def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    
    for sample in tqdm(dataloader):
        src = sample['expr_seq']
        trg = sample['label_seq']
        trg = trg.transpose(0,1)

        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        trg = trg.to(device)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()

        if DRY_RUN:
            break
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader):
    
    model.eval()
    
    epoch_acc = 0
    
    len_samples = 0
    with torch.no_grad():
    
        for sample in tqdm(dataloader):
            src = sample['expr_seq']
            trg = sample['label_seq']
            len_samples += trg.shape[0]
            
            trg = trg.transpose(0,1)
            output = model(src, trg, 0) #turn off teacher forcing

            output_dim = output.shape[-1]
            
            preds = torch.argmax(output, -1)
            
            preds = preds[1:].transpose(0,1).detach().cpu()
            trg = trg[1:].transpose(0,1)
            acc = torch.all(torch.eq(preds, trg), dim = 1)
            epoch_acc += np.sum(acc.numpy())

            if DRY_RUN:
                break
        
    return epoch_acc / len_samples

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 25
CLIP = 1

best_valid_loss = float('inf')
prev_test_acc = 0

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = 12
criterion = nn.CrossEntropyLoss()

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    print ("Training")
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    # valid_loss = evaluate(model, val_dataloader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if epoch % 5 == 4:
        print ("Testing")
        test_acc = evaluate(model, eval_dataloader)
        print(f'| Test Acc: {test_acc:.3f} |')
        if test_acc > prev_test_acc:
            torch.save(model.state_dict(), "model.pt")
            prev_test_acc = test_acc

