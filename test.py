import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import os

def str2bool(x):
    return x.lower() in ("true","1","yes")

def get_args():
    # If user explicitly passed --config, go into “YAML mode”
    if "--config" in sys.argv:
        # ─── YAML MODE ───────────────────────────────────────────────────
        # Stage 1: grab only --config (ignore all else)
        tmp = argparse.ArgumentParser(add_help=False)
        tmp.add_argument('--config', type=str, default='config_x24y6.yaml',
                         help='Path to YAML config file')
        cfg_args, _ = tmp.parse_known_args()

        # Stage 2: load that YAML
        with open(cfg_args.config, 'r') as f:
            cfg = yaml.safe_load(f)

        # Stage 3: build a fresh parser seeded from YAML
        parser = argparse.ArgumentParser(description="Graph-WaveNet Training")
        parser.add_argument('--config', type=str,
                            default=cfg_args.config,
                            help='Path to YAML config file')
        for key, val in cfg.items():
            if isinstance(val, bool):
                parser.add_argument(
                    f'--{key}',
                    type=str2bool,
                    default=val,
                    help=f"{key} (true/false)"
                )
            else:
                parser.add_argument(f'--{key}', type=type(val), default=val)

        # Stage 4: parse everything (unknowns will error)
        return parser.parse_args()

    else:
        # ─── SWEEP/CLI MODE ────────────────────────────────────────────────
        parser = argparse.ArgumentParser(description="Graph-WaveNet Training")

        # Define every flag you expect the sweep to inject
        parser.add_argument('--device',type=str,default='cuda:0',help='')
        parser.add_argument('--data',type=str,default="/pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/train_data/x6y1",help='data path')
        parser.add_argument('--adjdata',type=str,default="/pfs/work9/workspace/scratch/ma_tofuchs-GraphWave-Seminar/Datasets/Mannheim/train_data/sensor_graph/adj_mx.csv",help='adj data path')
        parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
        parser.add_argument('--gcn_bool',    type=str2bool, default=True)
        parser.add_argument('--aptonly',     type=str2bool, default=False)
        parser.add_argument('--addaptadj',   type=str2bool, default=True)
        parser.add_argument('--randomadj',   type=str2bool, default=True)
        parser.add_argument('--evaluate_only',   type=str2bool, default=True)
        # parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
        # parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
        # parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
        # parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
        parser.add_argument('--seq_length',type=int,default=6,help='')
        parser.add_argument('--nhid',type=int,default=32,help='')
        parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
        parser.add_argument('--num_nodes',type=int,default=25,help='number of nodes')
        parser.add_argument('--batch_size',type=int,default=64,help='batch size')
        parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
        parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
        parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
        parser.add_argument('--epochs',type=int,default=100,help='')
        parser.add_argument('--print_every',type=int,default=50,help='')
        #parser.add_argument('--seed',type=int,default=99,help='random seed')
        parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
        parser.add_argument('--expid',type=int,default=1,help='experiment id')
        parser.add_argument('--diffusion_steps',type=int,default=2,help='')
        parser.add_argument('--checkpoint',type=str,help='')
        parser.add_argument('--plotheatmap',type=str,default='True',help='')
        return parser.parse_args()




def main():
    args = get_args()
    print(f"yaml tested was that from {args.config}")
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    #added out_dim=args.seq_length
    model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, out_dim = args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid*8, end_channels=args.nhid*16, diffusion_steps=args.diffusion_steps)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
            print(preds.shape)
        outputs.append(preds) #.squeeze

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]
    print(f"yhat shape is: {yhat.shape} and realy is shape {realy.shape}")
    yhat = yhat[:, -1, :, :]  

    amae = []
    amape = []
    armse = []
    for i in range(1): #hard code the pred length of 1 here
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    #alter this to seq_len maybe
    log = 'On average over all horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))


    if args.plotheatmap == True:
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="Greens")
        #plt.savefig("./emb"+ '.pdf')
        plt.savefig("/home/ma/ma_ma/ma_tofuchs/Graph-WaveNet-Mannheim/bayes_hyper_adaptadj"+ '.pdf') #bayes bayes_hyper

    # y12 = realy[:,99,11].cpu().detach().numpy()
    # yhat12 = scaler.inverse_transform(yhat[:,99,11]).cpu().detach().numpy()

    # y3 = realy[:,99,2].cpu().detach().numpy()
    # yhat3 = scaler.inverse_transform(yhat[:,99,2]).cpu().detach().numpy()

    # df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    # df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    main()
