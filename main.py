import json
import numpy as np
import random
import argparse
import os 
import warnings
from utils import *
from load_data import load_data
import torch
from model import MyModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable as V
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  

criterion = torch.nn.CrossEntropyLoss()
windowSize = 13 
max_epoch=100
best_score=0
random_state=345
save=True
load=True
random.seed(69)
output_model = '/tmp/output/model.pth'
data_path = 'FL_T'   #'SF'  'ober'
    
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-3, type=float,help="learning_rate")
parser.add_argument("--train_bs", default=32, type=int,help="train_bs")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def create_dataloader(x,y,bs):
    zr=x.real
    zi=x.imag

    z=np.concatenate((zr,zi),axis=3)
    z=np.transpose(z, axes=(0, 3, 1, 2))  
    
    z=V(torch.FloatTensor(z))
    y=V(torch.LongTensor(y))

    dataset=TensorDataset(z,y)
    data_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=8,) 
    return data_loader

def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

def eval(model, optimizer,criterion,data_loader,save=save):
    print('\033[1;35m----Evaluating----\033[0m')
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0

    for i,(batch_x,batch_y) in enumerate(tqdm(data_loader)):
        batch_x=batch_x.to(device)
        batch_y=batch_y.to(device)
        with torch.no_grad():
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            logits = logits.detach().cpu().numpy()
            eval_loss += loss.item()
            label_ids = batch_y.cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    avg_loss = eval_loss / nb_eval_steps 
    avg_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")  
    #print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    global best_score
    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        if save:
            save(model, optimizer)
            print('\033[1;31m Better model saved.\033[0m')

    model.train()
    return avg_accuracy
    
def train(model,optimizer,criterion,train_loader,test_loader):
    print('-----------------training------------')
    for epoch in range(max_epoch):
        print('【Epoch:{}】'.format(epoch+1))
        for i,(batch_x,batch_y) in enumerate(tqdm(train_loader)):
            model.train()
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            logits=model(batch_x)

            loss=criterion(logits,batch_y)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        eval(model, optimizer,criterion,test_loader)

import shutil  
import torch.nn.init as init  
def pretrain(model, optimizer, criterion, train_loader, test_loader, max_cycles=30):  
    best_score_overall = 0  
    best_cycle = 0  
      
    for cycle in range(max_cycles):  

        model=MyModel(batch_size=args.train_bs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.01)   
        
        model.to(device)  
        print(f'Starting cycle {cycle+1}...')  
  
        for epoch in range(5):  # 5 epochs  
            print(f'【Cycle {cycle+1}, Epoch: {epoch+1}】')  
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):  
                model.train()  
                batch_x = batch_x.to(device)  
                batch_y = batch_y.to(device)  
                logits = model(batch_x)  
                loss = criterion(logits, batch_y)  
  
                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step()  
  
            # evaluating
            eval_score = eval(model, optimizer, criterion, test_loader, save=False)  
        
            if eval_score<0.8:
                break
            
            # check for saving
            if eval_score > 0.9 and eval_score > best_score_overall:  
                best_score_overall = eval_score  
                best_cycle = cycle  
                # save 
                save_path = f'pretrain/model{cycle+1}.pth'  
                torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)  
                print(f'Model saved at {save_path} with score {eval_score:.4f}')  
  
    # restore best model 
    if best_score_overall > 0.8:  
        best_model_path = f'pretrain/model{best_cycle+1}.pth'  
        shutil.copy(best_model_path, '/tmp/output/model.pth')  
        print(f'Best model copied to /tmp/output/model.pth with score {best_score_overall:.4f}')  
    else:  
        print('No model saved as no cycle achieved >90% accuracy.')  

if __name__ == '__main__':
    print('start')
    train_per = 1  
      
    data, gt = load_data(data_path)  
 
    data = Standardize_data(data)

    X_coh, y = createComplexImageCubes(data, gt, windowSize)
    del data, gt
    X_train, X_test, y_train, y_test = train_test_split(X_coh, y, test_size=0.9, random_state=random_state,stratify=y)
    del X_coh, y
    train_loader=create_dataloader(X_train,y_train,args.train_bs)
    del X_train,y_train
    test_loader=create_dataloader(X_test,y_test,args.train_bs)
    del X_test,y_test
    model=MyModel(batch_size=args.train_bs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.01)
    
    warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue, install nvrtc.so")

    if not os.path.exists('/tmp/output'):  
        os.makedirs('/tmp/output')  
    if not os.path.exists('pretrain'):  
        os.makedirs('pretrain')  
    pretrain(model,optimizer,criterion, train_loader, test_loader, max_cycles=20)
    model.to(device)
    if load:
        checkpoint = torch.load(output_model)  
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        print('Pretrained model loaded.')
    
    eval(model, optimizer,criterion,test_loader,save=False)
    train(model,optimizer,criterion,train_loader,test_loader)
    print(f'Best score:{best_score:.4f}')