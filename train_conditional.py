import torch
import torch.nn.functional as F
import math
import time
from torch.autograd import grad
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
from utils import get_noise_tensor, print_performances, save_models
from torch.autograd import Variable
from PIL import Image
algorithm = 'wgan-gp'
# weight clipping (WGAN)
c = 1


def prepare_data(data, device):
    imgs, captions, captions_lens, rot_vec = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    """real_imgs = []
    print(imgs.size())
    print(sorted_cap_indices.size())
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        real_imgs.append(Variable(imgs[i]).to(device))"""
    imgs = imgs[sorted_cap_indices]
    captions = captions[sorted_cap_indices].squeeze()
    rot_vec = rot_vec[sorted_cap_indices]
    
    captions = Variable(captions).to(device)
    sorted_cap_lens = Variable(sorted_cap_lens).to(device)

    return [imgs, captions, sorted_cap_lens,rot_vec]


def get_grad_penalty(batch_size, device, net_d, image, image_fake, sentence_match, rot_vec):  
    epsilon = torch.rand(batch_size, dtype=torch.float32).to(device)  
    ##########################
    # get so3_interpolated
    ##########################
    fake_interpolated = torch.empty_like(image, dtype=torch.float32).to(device) 
    for j in range(batch_size):
        fake_interpolated[j] = epsilon[j] * image[j] + (1 - epsilon[j]) * image_fake[j]
    
    fake_interpolated = Variable(fake_interpolated, requires_grad=True)    
    # calculate gradient penalty
    score_interpolated_fake = net_d(fake_interpolated, sentence_match, rot_vec).mean()
    gradient_fake = grad(outputs=score_interpolated_fake, 
                    inputs=fake_interpolated, 
                    grad_outputs=torch.ones_like(score_interpolated_fake).to(device),
                    create_graph=True, 
                    retain_graph=True)[0]
    grad_fake_norm = torch.sqrt(torch.sum(gradient_fake.reshape(batch_size, -1) ** 2, dim=1) + 1e-5)
    grad_penalty_fake = ((grad_fake_norm - 1) ** 2).mean()  
    return grad_penalty_fake

def get_d_score(so3_d):
    so3_d = so3_d.masked_fill(so3_d == 0, -1e9)
    pred = F.normalize(so3_d*so3_d, p=1, dim=-1)[:,:,:1]
    return pred.mean()

def get_g_so3(output):
    return output

def get_d_loss(cfg, device, net_g, net_d, batch, batch_index, optimizer_d=None, update_d=True, text_model=None):
    if update_d:
        net_d.zero_grad()
    #sentence_match = batch.get('sentence').to(device) # torch.Size([128, 128])
    caption = batch.get('caption').to(device)
    caption_len = batch.get('caption_len').to(device)
    image = batch.get('image').to(device)
    image_mismatch = batch.get('image_wrong').to(device)
    rot_vec = batch.get('rot_vec').to(device)
    image, caption, caption_len,  rot_vec = prepare_data((image, caption, caption_len,  rot_vec), device)
    if text_model == None:
        print(dfsdf)
    hidden = text_model.init_hidden(cfg.BATCH_SIZE)
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    words_embs, sent_emb = text_model(caption, caption_len, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)
    grad_penalty_fake = 0
    grad_penalty_wrong = 0
    score_fake = 0  
    if update_d:  
        # real
        score_right = torch.nn.ReLU()(1.0 - net_d(image, sent_emb, rot_vec)).mean()
        #score_right = net_d(image, sentence_match, rot_vec).mean()
        # wrong
        score_wrong = torch.nn.ReLU()(1.0 + net_d(image_mismatch, sent_emb, rot_vec)).mean()
        #score_wrong = net_d(image_mismatch, sentence_match, rot_vec).mean()
        # fake
        image_fake = net_g(noise, sent_emb, rot_vec).detach() 
        score_fake = torch.nn.ReLU()(1.0 + net_d(image_fake, sent_emb, rot_vec)).mean()       
        #score_fake = net_d(image_fake, sentence_match, rot_vec).mean()

        grad_penalty_fake = get_grad_penalty(cfg.BATCH_SIZE, device, net_d, image, image_fake, sent_emb, rot_vec)
        loss_d = score_fake + score_wrong  + 2*score_right + 2 * grad_penalty_fake         
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        if batch_index % 10 == 2:
            #############################################################
            im = Image.fromarray(image_fake[0].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/sample_0.jpg')
            im = Image.fromarray(image_fake[10].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/sample_10.jpg')
            im = Image.fromarray(image_fake[20].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/sample_20.jpg')
            im = Image.fromarray(image_fake[22].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/sample_22.jpg')
           
            im = Image.fromarray(image[0].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/real_0.jpg')
            im = Image.fromarray(image[10].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/real_10.jpg')
            im = Image.fromarray(image[20].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/real_20.jpg')
            im = Image.fromarray(image[22].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('/media/remote_home/chang/thesis_rgbimg/real_22.jpg')
            
            #############################################################            
                
    if update_d:
        net_d.zero_grad()     
    return score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong

def get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g=None, update_g=True, text_model=None):
    if update_g:
        net_g.zero_grad()

    # get text vectors and noises
    #sentence_match = batch.get('sentence').to(device) # torch.Size([128, 128])
    image = batch.get('image').to(device)
    rot_vec = batch.get('rot_vec').to(device)
    caption = batch.get('caption').to(device)
    caption_len = batch.get('caption_len').to(device)
    hidden = text_model.init_hidden(cfg.BATCH_SIZE)
    image, caption, caption_len,  rot_vec = prepare_data((image, caption, caption_len,  rot_vec), device)
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    words_embs, sent_emb = text_model(caption, caption_len, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    noise = get_noise_tensor(cfg.BATCH_SIZE, cfg.NOISE_SIZE).to(device)

    if update_g:
        # so3 fake
        image_fake = net_g(noise, sent_emb, rot_vec)
        score_fake = net_d(image_fake, sent_emb, rot_vec).mean()
        
        # 'wgan', 'wgan-gp'
        loss_g = - score_fake
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()    
    
    if update_g:
        net_g.zero_grad()
    return score_fake

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer=None, e=None, text_model=None):
    #print('learning rate: g ' + str(optimizer_g._optimizer.param_groups[0].get('lr')) + ' d ' + str(optimizer_d._optimizer.param_groups[0].get('lr')))
    net_d.train()
    net_g.train()  
    total_loss_g = 0
    total_loss_d = 0
    if text_model == None:
        print(fdgfdg)

    
    for i, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ', leave=False)):        
        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###############################################################
        score_fake, score_wrong, score_right, grad_penalty_fake, grad_penalty_wrong = get_d_loss(cfg, device, net_g, net_d, batch, i, optimizer_d, text_model=text_model)         
        loss_d = score_fake + score_wrong - 2 * score_right + 10 * grad_penalty_fake 
        if i % cfg.N_TRAIN_ENC ==0:
            total_loss_d += loss_d.item()
            if tb_writer != None:
                tb_writer.add_scalars('loss_d_', {'score_fake': score_fake, 'score_wrong': score_wrong, 'score_right': score_right, 'grad_penalty_fake': grad_penalty_fake, 'grad_penalty_wrong':grad_penalty_wrong}, e*len(dataLoader_train)+i)
                tb_writer.add_scalars('loss_d_wf', {'R_W': score_right-score_wrong, 'W_F': score_wrong-score_fake, 'w_loss': score_right-score_fake}, e*len(dataLoader_train)+i)
        
        ###############################################################
        # (2) Update G network: maximize log(D(G(z)))
        ###############################################################
        # after training discriminator for N times, train gernerator for 1 time
        if i % cfg.N_TRAIN_D_1_TRAIN_G == 0:            
            #get losses
            score_fake = get_g_loss(cfg, device, net_g, net_d, batch, optimizer_g, text_model=text_model)
            loss_g =  - score_fake
            total_loss_g += loss_g.item()
            if tb_writer != None:
                tb_writer.add_scalars('loss_g_', {'score_fake': score_fake}, e*len(dataLoader_train)+i)
    return total_loss_g/ (cfg.BATCH_SIZE/cfg.N_TRAIN_D_1_TRAIN_G), total_loss_d/(cfg.BATCH_SIZE)

def train(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, dataLoader_val, text_model):   
    if text_model == None:
        print(fdgfdg)
    # tensorboard
    if cfg.USE_TENSORBOARD:
        print("[Info] Use Tensorboard")  
        tb_writer = SummaryWriter(log_dir=cfg.TB_DIR) 
       
    start_of_all_training = time.time()
    for e in range(cfg.START_FROM_EPOCH, cfg.END_IN_EPOCH):   
        print("=====================Epoch " + str(e) + " start!=====================")     
        # Train!!
        start = time.time()
        train_loss_g, train_loss_d = train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer,e, text_model)
        
        """lr_g=optimizer_g._optimizer.param_groups[0].get('lr')
        lr_d=optimizer_d._optimizer.param_groups[0].get('lr')
        print_performances('Training', start, train_loss_g, train_loss_d, lr_g, lr_d, e)"""
        
        # save model for each 5 epochs
        if e % cfg.SAVE_MODEL_ITR == 0 and e != 0:
            save_models(cfg, e, net_g, net_d, cfg.CHKPT_PATH,  save_mode='all')
        elapse_mid=(time.time()-start_of_all_training)/60
        print('\n till episode ' + str(e) + ": " + str(elapse_mid) + " minutes (" + str(elapse_mid/60) + " hours)")

        if cfg.USE_TENSORBOARD:
            tb_writer.add_scalars('loss_g', {'train': train_loss_g/2, 'val': 0}, e)
            tb_writer.add_scalars('loss_d', {'train': train_loss_d, 'val': 0}, e)
            """tb_writer.add_scalar('loss_g', train_loss_g, e)
            tb_writer.add_scalar('loss_d', train_loss_d, e)"""
            """tb_writer.add_scalar('learning_rate_g', lr_g, e)
            tb_writer.add_scalar('learning_rate_d', lr_d, e)"""        

    elapse_final=(time.time()-start_of_all_training)/60
    print('\nfinished! ' + str(elapse_final) + " minutes")

if __name__ == "__main__":
    imgFullPath =os.path.join('/media/remote_home/chang/datasets/coco/train2014',"COCO_train2014_000000000036.jpg")
    img  = Image.open(imgFullPath).convert('RGB').resize((256, 256))
    img = np.array(img, dtype=float)
    img = np.fliplr(img).transpose(2, 0, 1)
    img = torch.tensor(img.copy(), dtype=torch.float32).sub_(127.5).div_(127.5)
    im = Image.fromarray(img.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
    im.save('/media/remote_home/chang/thesis_rgbimg/JOJOJO_r.jpg')
