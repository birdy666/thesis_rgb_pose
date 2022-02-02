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
    imgs, imgs_wrong, captions, captions_lens, rot_vec = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        real_imgs.append(Variable(imgs[i]).to(device))
    
    if imgs_wrong != None:
        wrong_imgs = []
        for i in range(len(imgs_wrong)):
            imgs_wrong[i] = imgs_wrong[i][sorted_cap_indices]
            wrong_imgs.append(Variable(imgs_wrong[i]).to(device))
    else:
        wrong_imgs = None
    
    captions = captions[sorted_cap_indices].squeeze()
    rot_vec = rot_vec[sorted_cap_indices]
    
    captions = Variable(captions).to(device)
    sorted_cap_lens = Variable(sorted_cap_lens).to(device)
    rot_vec = Variable(rot_vec).to(device)

    return [real_imgs, wrong_imgs, captions, sorted_cap_lens,rot_vec]

def get_d_score(score):
    return score.mean()

def get_grad_penalty(image, sent_emb, rot_vec, net_d, device):  
    interpolated = (image.data).requires_grad_()
    sent_inter = (sent_emb.data).requires_grad_()
    rot_inter = (rot_vec.data).requires_grad_()
    features = net_d(interpolated)
    out = net_d.COND_DNET(features,sent_inter, rot_inter)
    grads = torch.autograd.grad(outputs=out,
                                inputs=(interpolated,sent_inter,rot_inter),
                                grad_outputs=torch.ones(out.size()).to(device),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad2 = grads[2].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1,grad2),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 6, dim=1)) 
    return torch.mean((grad_l2norm))

def get_g_so3(output):
    return output

def safe_sample(real, fake):
    for i in range(len(fake)):
        if i % 3 == 0:
            im = fake[i].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            im.save('/media/remote_home/chang/thesis_rgbimg/sample_' + str(i) + '.jpg')
            im = real[i].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            im.save('/media/remote_home/chang/thesis_rgbimg/real_' + str(i) + '.jpg')
    

def train_epoch(cfg, device, net_g, net_d, optimizer_g, optimizer_d, dataLoader_train, tb_writer=None, e=None, text_model=None):
    #print('learning rate: g ' + str(optimizer_g._optimizer.param_groups[0].get('lr')) + ' d ' + str(optimizer_d._optimizer.param_groups[0].get('lr')))
    net_d.train()
    net_g.train()  
    total_loss_g = 0
    total_loss_d = 0
    if text_model == None:
        print(fdgfdg)
    
    for batch_index, batch in enumerate(tqdm(dataLoader_train, desc='  - (Training)   ', leave=False)):        
        #sentence_match = batch.get('sentence').to(device) # torch.Size([128, 128])
        captions = batch.get('caption')
        caption_lens = batch.get('caption_len')
        images = batch.get('image')
        #images_wrong = batch.get('image_wrong')
        images_wrong = None
        rot_vecs = batch.get('rot_vec')
        images, images_wrong, captions, caption_lens,  rot_vecs = prepare_data((images, images_wrong, captions, caption_lens,  rot_vecs), device)
        if text_model == None:
            print(dfsdf)
        hidden = text_model.init_hidden(cfg.BATCH_SIZE)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        _, sent_emb = text_model(captions, caption_lens, hidden)
        sent_emb = sent_emb.detach()
        
        # real
        images = images[0]
        real_features = net_d(images)
        output = net_d.COND_DNET(real_features,sent_emb,rot_vecs)
        score_right = torch.nn.ReLU()(1.0 - output).mean()
        # wrong
        """images_wrong = images_wrong[0]
        wrong_features = net_d(images_wrong)
        output = net_d.COND_DNET(wrong_features, sent_emb, rot_vecs)"""
        output = net_d.COND_DNET(real_features[:(cfg.BATCH_SIZE - 1)], sent_emb[1:cfg.BATCH_SIZE], rot_vecs[1:cfg.BATCH_SIZE])
        score_wrong = torch.nn.ReLU()(1.0 + output).mean()
        
        # fake
        noises = torch.randn(cfg.BATCH_SIZE, 100).to(device)
        fake = net_g(noises,sent_emb, rot_vecs)  
        fake_features = net_d(fake.detach()) 
        output = net_d.COND_DNET(fake_features,sent_emb,rot_vecs) 
        score_fake = torch.nn.ReLU()(1.0 + output).mean()     

        loss_d = (score_fake + score_wrong)/2.0 + score_right   
        optimizer_g.zero_grad()   
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        
        d_loss_gp = get_grad_penalty(images, sent_emb, rot_vecs, net_d, device)
        d_loss_gp = 2.0 * d_loss_gp
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        d_loss_gp.backward()
        optimizer_d.step()

        # update G
        for i in range(4):
            noises = torch.randn(cfg.BATCH_SIZE, 100).to(device)
            fake = net_g(noises,sent_emb, rot_vecs)
            features = net_d(fake)
            output = net_d.COND_DNET(features,sent_emb,rot_vecs)
            loss_g = - output.mean()
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        if batch_index % 20 == 2:
            safe_sample(images, fake)
            
        
        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        if tb_writer != None:
            tb_writer.add_scalars('loss_d_', {'score_fake': score_fake-1, 'score_wrong': score_wrong-1, 'score_right': 1-score_right, 'grad_penalty': d_loss_gp}, e*len(dataLoader_train)+batch_index)

    return total_loss_g/ (cfg.BATCH_SIZE), total_loss_d/(cfg.BATCH_SIZE)

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
