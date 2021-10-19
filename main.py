import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from custom_loader import CustomHeteroLinkDataLoader
from metrics import roc_auc
from model import HeteroGATConvGNN
from utils import load_prepare_data, get_opt

def train(train_data, model, optimizer, criterion, epoch, writer):
    model.train()
    optimizer.zero_grad()
    z = model(train_data.x_dict, train_data.edge_index_dict)
    out = model.decode(z, train_data[('user','rated','movie')]['edge_label_index'])
    loss = criterion(out, train_data[('user','rated','movie')]['edge_label'])
    loss.backward()
    optimizer.step()

    writer.add_scalar('training_loss', loss.item(), epoch)
    writer.add_scalar('training_roc_auc', roc_auc(train_data[('user','rated','movie')]['edge_label'],out.detach().numpy()), epoch)

    return loss.detach().numpy()

@torch.no_grad()
def val(val_data, model, criterion, epoch, writer):
    model.eval()
    z = model(val_data.x_dict, val_data.edge_index_dict)
    out = model.decode(z, val_data[('user','rated','movie')]['edge_label_index'])
    loss = criterion(out, val_data[('user','rated','movie')]['edge_label'])
    roc_auc_curr = roc_auc(val_data[('user','rated','movie')]['edge_label'],out.detach().numpy())

    writer.add_scalar('val_loss', loss.item(), epoch)
    writer.add_scalar('val_roc_auc',roc_auc_curr, epoch)

    return loss, roc_auc_curr

@torch.no_grad()
def test(model,test_data):
    model.eval()
    out = model(test_data.x_dict, test_data.edge_index_dict)
    out = model.decode(out,test_data[('user','rated','movie')]['edge_label_index'])
    out = out.sigmoid()

    return roc_auc(test_data[('user','rated','movie')]['edge_label'],out)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.5, help='GatConv dropout')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden_channels', type=int, default=64, help='')
    parser.add_argument('--negative_sample_ratio', type=int, default=0.15, help='negative sampling ratio')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')    
    parser.add_argument('--batch_size', type=int, default=0, help='batch size') 
    parser.add_argument('--opt', type=str, default='adam', help='choose optimizer Adam/SparseAdam/SGD')
    parser.add_argument('--lr', type=float, default=0.001, help='model learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization on model weights')
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=50)

    args = parser.parse_args()

    return args

def main(args):
    user_id_embeddings, movie_id_embeddings, train_data, val_data, test_data = load_prepare_data(movies_path = 'dataset/final_movies_dataset.pkl', connection_path = 'dataset/user_movie_connection.pkl', negative_sample_ratio = args.negative_sample_ratio)
    
    if args.batch_size != 0:
        train_loader = CustomHeteroLinkDataLoader(train_data,batch_size = args.batch_size)

    model = HeteroGATConvGNN(train_data.metadata(),args)
    writer = SummaryWriter()

    with torch.no_grad(): 
        _ = model(train_data.x_dict, train_data.edge_index_dict)

    optimizer = get_opt(args.opt)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT = [0, 1, 0]

    for epoch in range(args.epochs + 1):
        train_loss = train(train_data, model ,optimizer, criterion,epoch,writer)
        loss, roc_auc_curr = val(val_data, model, criterion, epoch, writer)
        if roc_auc_curr > BEST_VAL_PERF or loss.item() < BEST_VAL_LOSS:
            BEST_VAL_PERF = max(roc_auc_curr, BEST_VAL_PERF)
            BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)  
            PATIENCE_CNT = 0  
        else:
            PATIENCE_CNT += 1  
            if PATIENCE_CNT >= args.patience_period:
                print('Stopping the training')
                break
        
        if epoch % 10 == 0:
            print('Current epoch: {} ,Current training loss: {:.4f} ,Best validation loss: {:.4f} ,Best validation performance: {:.4f}'.format(epoch,train_loss,BEST_VAL_LOSS,BEST_VAL_PERF))
            
    print('Final Test(ROC_AUC_SCORE): ', test(model,test_data))

if __name__ == '__main__':
    main(get_args())
