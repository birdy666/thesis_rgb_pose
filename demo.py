import json
import torch
from models import Generator as G1
from model_origin import NetG as G2
from config import cfg
from pycocotools.coco import COCO
from geometry import rotation2so3
import numpy as np
from DAMSM import RNN_ENCODER
import pickle
import os
from tqdm import tqdm
from PIL import Image
from utils import get_noise_tensor
import shutil

#keywords = ['frisbee','skateboard', 'tennis', 'ski']
keywords = ['baseball',  'surf', 'ski', 'tennis', 'frisbee', 'snow']
not_keywords = ["stand", "sit", "walk", "observ", "parked", "picture", "photo", "post"]
device = torch.device('cpu')

def saveImgOrNot(caption):
    save_this = False
    category = 0
    for nnk in range(len(not_keywords)):
        if not_keywords[nnk] in caption:
            return save_this, category

    for nk in range(len(keywords)):
        if keywords[nk] in caption:                    
            save_this = True
            category = nk
    return save_this, category
    
def pad_text(text, d_word_vec):
    batch_s = text.size(0)
    new_text = torch.zeros((batch_s,24,d_word_vec), dtype=torch.float32)
    for i in range(batch_s):
        if len(text[i]) < 24:
            new_text[i][0:len(text[i])] = text[i]
            for j in range(len(text[i]), 24):
                new_text[i][j] = torch.zeros(d_word_vec, dtype=torch.float32).unsqueeze(0)
        
    return new_text


if __name__ == "__main__":
    if os.path.exists('/media/remote_home/chang/thesis_rgbimg/output_pics/'):
        shutil.rmtree('/media/remote_home/chang/thesis_rgbimg/output_pics/')
        os.mkdir('/media/remote_home/chang/thesis_rgbimg/output_pics/')
    else:
        os.mkdir('/media/remote_home/chang/thesis_rgbimg/output_pics/')


    checkpoint_1 = torch.load('/media/remote_home/chang/thesis_rgbimg/models/checkpoints/epoch_120' + ".chkpt", map_location=torch.device('cpu')) #in docker
    net_g_1 = G1().to(device)
    net_g_1.load_state_dict(checkpoint_1['model_g'])

    checkpoint_2 = torch.load('/media/remote_home/chang/thesis_rgbimg/models/netG_120.pth', map_location=torch.device('cpu')) #in docker
    net_g_2 = G2().to(device)
    net_g_2.load_state_dict(checkpoint_2)

    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    text_encoder = RNN_ENCODER()
    state_dict = torch.load('/media/remote_home/chang/thesis_rgbimg/coco/text_encoder100.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()   
    print("Text model loaded")
    text_model = text_encoder
    
    #with open('../eft/eft_fit/COCO2014-All-ver01.json','r') as f: # in docker
    with open('/media/remote_home/chang/eft/eft_fit/COCO2014-All-ver01.json','r') as f:
        eft_data = json.load(f)
  
    eft_data = eft_data['data'] [1600:3200]
    #eft_all_with_caption = eft_all_with_caption
    print(len(eft_data))
    net_g_1.eval()
    
    output = []
    min_list = []
    max_list = []
    previous_ids = []
    filenamepath = os.path.join('/media/remote_home/chang/thesis_rgbimg/coco', 'filenames.pickle')    
    with open(filenamepath, 'rb') as f:
        filenames = np.array(pickle.load(f))
    captionpath = os.path.join('/media/remote_home/chang/thesis_rgbimg/coco', 'captions.pickle')
    
    with open(captionpath, 'rb') as f:
        x = pickle.load(f)
        captions = np.array(x[0])
    captions_list = []
    for i in tqdm(range(len(eft_data)), desc='  - (Dataset)   ', leave=False):
        data = {}      
        #text_match = batch.get('vector').to(device) # torch.Size([128, 24, 300])
        #text_match_mask = batch.get('vec_mask').to(device)
        img_id = coco_keypoint.loadAnns(eft_data[i]['annotId'])[0]['image_id']
        keypoint_ids = coco_keypoint.getAnnIds(imgIds=img_id)
        if img_id in previous_ids:
            continue
        if len(keypoint_ids) != 1:
            continue
        else:
            previous_ids.append(img_id)
        #previous_ids.append(img_id)
        # 但對於同一個圖片會有很多語意相同的captions
        caption_ids = coco_caption.getAnnIds(imgIds=img_id)
        captions_anns = coco_caption.loadAnns(ids=caption_ids)

        save_this, category = saveImgOrNot(captions_anns[0]['caption'])
        if not save_this:
            continue
        kk = np.where(filenames == eft_data[i]['imageName'][:-4]) # remove ".jpg"
        if True:
            new_sent_ix = kk[0][0] * 5 + 1
            caption = captions[new_sent_ix]
            caption = torch.tensor(caption).unsqueeze(0)
            caption_len = torch.tensor(len(caption[0])).unsqueeze(0)
            mask = []
            for _ in range(caption_len):
                mask.append(1)
            for _ in range(caption_len, 24):
                mask.append(0)
            mask = torch.tensor(mask).unsqueeze(0)
            hidden = text_model.init_hidden(1)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            _, sent_emb = text_model(caption, caption_len, hidden)
            sent_emb = sent_emb.detach().to(device)
            noise = get_noise_tensor(1, cfg.NOISE_SIZE).to(device)
            rot_vec = torch.tensor(np.array([rotation2so3(R) for R in eft_data[i]['parm_pose']]), dtype=torch.float32)
            image_fake_1 = net_g_1(noise, sent_emb, rot_vec)
            image_fake_2 = net_g_2(noise, sent_emb)
            if 18-caption_len >= 0:
                im = Image.fromarray(image_fake_1[0].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('/media/remote_home/chang/thesis_rgbimg/output_pics/'+'sample_'+str(i)+'.jpg')   
                im = Image.fromarray(image_fake_2[0].data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('/media/remote_home/chang/thesis_rgbimg/output_pics/'+'sample_'+str(i)+'_origin.jpg')   
                print('=======================================================')   
                print(str(i) + "  " + str(caption_len[0]))
                print(caption_len)
                print(captions_anns[0]['caption'])
                print('=======================================================')  
                cap_with_index = str(i) + ": " +  captions_anns[0]['caption']
                captions_list.append(cap_with_index)
            
    with open("/media/remote_home/chang/thesis_rgbimg/sample_caps.txt", "w") as fhandle:
        for line in captions_list:
            fhandle.write(f'{line}\n')

        