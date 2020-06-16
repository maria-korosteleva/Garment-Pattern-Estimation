# Training loop func
import numpy as np
import torch
import wandb as wb

batch_size = 64
epochs_num = 100
learning_rate = 0.001
logdir = './logdir'

def fit(model, regression_loss, optimizer, scheduler, train_loader, valid_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model.to(device)
    for epoch in range (epochs_num):
        model.train()
        for i, batch in enumerate(train_loader):
            features, params = batch['features'].to(device), batch['pattern_params'].to(device)
            
            #with torch.autograd.detect_anomaly():
            preds = model(features)
            loss = regression_loss(preds, params)
            #print ('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss))
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # logging
            if i % 5 == 4:
                wb.log({'epoch': epoch, 'loss': loss})
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[(regression_loss(model(features), params), len(batch)) for batch in valid_loader]
            )
            
        valid_loss = np.sum(losses) / np.sum(nums)
        scheduler.step(valid_loss)
        
        print ('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))
        wb.log({'epoch': epoch, 'valid_loss': valid_loss, 'learning_rate': optimizer.param_groups[0]['lr']})