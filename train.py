import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import wandb
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
        return parser.parse_args()

#parser = argparse.ArgumentParser()
# parser.add_argument('--device',type=str,default='cuda:0',help='')
# parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
# parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
# parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
# parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
# parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
# parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
# parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
# parser.add_argument('--seq_length',type=int,default=12,help='')
# parser.add_argument('--nhid',type=int,default=32,help='')
# parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
# parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
# parser.add_argument('--batch_size',type=int,default=64,help='batch size')
# parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
# parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
# parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
# parser.add_argument('--epochs',type=int,default=100,help='')
# parser.add_argument('--print_every',type=int,default=50,help='')
# #parser.add_argument('--seed',type=int,default=99,help='random seed')
# parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
# parser.add_argument('--expid',type=int,default=1,help='experiment id')

def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    args = get_args()
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # wandb setup
    wandb.init(
        project="graph-wavenet",
        name=f"run-exp-{args.expid}",
        config=vars(args)  # logs all argparse params
    )
   

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, args.diffusion_steps)
    wandb.watch(engine.model, log="gradients", log_freq=50)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    best_loss = float('inf')
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            loss, mape, rmse = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(loss)
            train_mape.append(mape)
            train_rmse.append(rmse)
            # wandb.log({
            #     "train_loss": loss,
            #     "train_mape": mape,
            #     "train_rmse": rmse
            # })
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        # minimal, combined logging
        wandb.log({
            "train/loss": mtrain_loss,
            "train/mape": mtrain_mape,
            "train/rmse": mtrain_rmse,
            "val/loss":   mvalid_loss,
            "val/mape":   mvalid_mape,
            "val/rmse":   mvalid_rmse
        }, step=i)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        if args.evaluate_only == False:
            #torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            for i in range(1, args.epochs+1):
                # … compute mvalid_loss …
                if mvalid_loss < best_loss:
                    best_loss = mvalid_loss
                    torch.save(engine.model.state_dict(), 
                            os.path.join(args.save, f"best_vloss_{mvalid_loss:.2f}.pth"))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing - commented out for now because want to decide on best model via validation loss
    # bestid = np.argmin(his_loss)
    # engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


    # outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1,3)[:,0,:,:]

    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1,3)
    #     with torch.no_grad():
    #         preds = engine.model(testx).transpose(1,3)
    #     outputs.append(preds.squeeze())

    # yhat = torch.cat(outputs,dim=0)
    # yhat = yhat[:realy.size(0),...]


    print("Training finished")
    save_path = os.path.join(args.save, f"best_vloss_{mvalid_loss:.2f}.pth")
    torch.save(engine.model.state_dict(), 
                            save_path)
    print(f"Model saved successfully as {save_path}")
    #print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    # amae = []
    # amape = []
    # armse = []
    # #was hard coded seq_len previously
    # for i in range(args.seq_length):
    #     pred = scaler.inverse_transform(yhat[:,:,i])
    #     real = realy[:,:,i]
    #     metrics = util.metric(pred,real)
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])

    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
   # torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
