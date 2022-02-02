import torch
import torch.nn as nn
from config import cfg
from train_conditional import train
from data import getData
from models import getModels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint = None
    if cfg.START_FROM_EPOCH > 0:
        checkpoint = torch.load(cfg.CHKPT_PATH + "/epoch_" + str(cfg.START_FROM_EPOCH-1) + ".chkpt", map_location='cpu')
    text_model, eft_all_with_caption, dataset_train, dataset_val = getData(cfg)    
    net_g, net_d = getModels(cfg, device, checkpoint) 

    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    dataLoader_val = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=True)
    
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=cfg.LEARNING_RATE_G, betas=(cfg.BETA_1, cfg.BETA_2), eps=1e-09)
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=cfg.LEARNING_RATE_D, betas=(cfg.BETA_1, cfg.BETA_2), eps=1e-09)
    print(device)
    train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, dataLoader_val, text_model)
    