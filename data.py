import fasttext
import fasttext.util
import numpy as np
from pycocotools.coco import COCO
import torch
import json
import os
import random
from tqdm import tqdm
#import cv2
import string
import math
from PIL import Image
from geometry import rotation2so3
from skimage.util import random_noise
from skimage.filters import gaussian
from config import cfg
from DAMSM import RNN_ENCODER
from nltk.tokenize import RegexpTokenizer
import pickle

#keywords = ['ski', 'baseball', 'motor','tennis','skateboard','kite']
keywords = ['baseball', 'skateboard', 'surf','ski']
#keywords = ['frisbee','skateboard', 'tennis', 'ski']
not_keywords = ["stand", "sit", "walk", "observ", "parked", "picture", "photo", "post"]


def getEFTCaption(cfg):
    with open(cfg.EFT_FIT_PATH,'r') as f:
        eft_data = json.load(f)
        print("EFT data: ver {}".format(eft_data['ver']))
        eft_data_all = eft_data['data']    
    return eft_data_all

def get_textModel(cfg):
    # load text model
    print("Loading text model")
    """
    text_model = fasttext.load_model(cfg.TEXT_MODEL_PATH) 
    if  cfg.D_SENTENCE_VEC < 300:
        fasttext.util.reduce_model(text_model, cfg.D_SENTENCE_VEC)
    """
    text_encoder = RNN_ENCODER()
    state_dict = torch.load('/media/remote_home/chang/thesis_rgbimg/coco/text_encoder100.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()   
    print("Text model loaded")
    return text_encoder

def getData(cfg):
    # load coco  
    coco_caption = COCO(cfg.COCO_CAPTION_TRAIN)
    coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    # load text model
    text_model = get_textModel(cfg)
    # load eft data
    eft_data_all = getEFTCaption(cfg)        
    # get the dataset (single person, with captions)
    train_size = int(len(eft_data_all))
    print("dataset size: ", train_size)
    print("Creating dataset_train")
    dataset_train = TheDataset(cfg, eft_data_all[:int(train_size*0.9)], coco_caption, coco_keypoint, text_model=text_model)
    print("Creating dataset_val")
    dataset_val = TheDataset(cfg, eft_data_all[int(train_size*0.9):train_size], coco_caption, coco_keypoint, text_model=text_model)
    print("Datasets created")
    #return text_model, dataset, dataset_val, data_loader#, text_match_val, label_val
    return text_model, eft_data_all, dataset_train, dataset_val

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
    

class TheDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, eft_data_all, coco_caption, coco_keypoint, text_model=None, val=False):
        self.dataset = []
        self.cfg = cfg
        previous_img_ids = []
        self.img_size = cfg.IMG_SIZE
        ###########################################
        filenamepath = os.path.join('/media/remote_home/chang/thesis_rgbimg/coco', 'filenames.pickle')    
        with open(filenamepath, 'rb') as f:
            self.filenames = np.array(pickle.load(f))
        ###########################################
        captionpath = os.path.join('/media/remote_home/chang/thesis_rgbimg/coco', 'captions.pickle')
    
        with open(captionpath, 'rb') as f:
            x = pickle.load(f)
            self.captions = np.array(x[0])
        ##########################################    
        for i in tqdm(range(len(eft_data_all)), desc='  - (Dataset)   ', leave=False):
            # one eft data correspond to one keypoint in one img
            keypoint_ann = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]
            img_id = keypoint_ann['image_id']
            keypoint_ids = coco_keypoint.getAnnIds(imgIds=img_id)
            if img_id in previous_img_ids:
                continue
            if len(keypoint_ids) != 1:
                continue
            else:
                previous_img_ids.append(img_id)
            # many captions for one img
            caption_ids = coco_caption.getAnnIds(imgIds=img_id)
            captions_anns = coco_caption.loadAnns(ids=caption_ids)
            
            save_this, category = saveImgOrNot(captions_anns[0]['caption'])
            if not save_this:
                continue
            # n caption with the same pose
            for j, caption_ann in enumerate(captions_anns):
                if j > 0:
                    break
                data = {'caption': caption_ann['caption'],
                        'annotId': eft_data_all[i]['annotId'],
                        'imageName': eft_data_all[i]['imageName'],
                        'parm_pose': eft_data_all[i]['parm_pose'],
                        'bbox': keypoint_ann['bbox'],
                        'category': category}
                data['rot_vec'] = np.array([rotation2so3(R) for R in data['parm_pose']])
                kk = np.where(self.filenames == eft_data_all[i]['imageName'][:-4]) # remove ".jpg"
                puOrNot = True
                for gg in range(5):
                    # 82783個filename 每個file有5個caption
                    new_sent_ix = kk[0][0] * 5 + gg
                    caption_index = self.captions[new_sent_ix]
                    if 18-len(caption_index) < 0:    
                        puOrNot = False                
                if puOrNot:
                    self.dataset.append(data)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()
        #item['sentence'] = torch.tensor(data['sentence'], dtype=torch.float32)
        #item['sentence_wrong'] = self.get_sentence_wrong(index)
        item['image'] = self.get_image(data['imageName'], data['bbox'])
        item['image_wrong'] = self.get_image_wrong(index)
        item['rot_vec'] = torch.tensor(data['rot_vec'], dtype=torch.float32)
        #item['pose'] = self.get_pose(data['imageName'])
        item['caption'], item['caption_len'] = self.get_caption_index(data['imageName'])
        return item
    
    def get_caption_index(self, imageName):
        kk = np.where(self.filenames == imageName[:-4]) # remove ".jpg"
        sent_ix = random.randint(0, 5-1)
        # 82783個filename 每個file有5個caption
        new_sent_ix = kk[0][0] * 5 + sent_ix
        caption_index = self.captions[new_sent_ix]
        if 18-len(caption_index) < 0:
            print(gfhfghfg)
        caption_len = len(caption_index)
        caption_index = np.pad(caption_index, (0, 18-len(caption_index)))
        caption_index = np.expand_dims(caption_index, axis=0)
        return torch.tensor(caption_index.transpose(1,0)), torch.tensor(caption_len)

    def get_image_wrong(self, index):
        while True:
            data_random = self.dataset[random.choice(list(range(0, len(self.dataset))))]
            if data_random['category'] != self.dataset[index]['category']:
                return self.get_image(data_random['imageName'], data_random['bbox'])

    def get_sentence_wrong(self, index):
        while True:
            data_random = self.dataset[random.choice(list(range(0, len(self.dataset))))]
            if data_random['category'] != self.dataset[index]['category']:
                return torch.tensor(data_random['sentence'], dtype=torch.float32)

    def get_pose(self, imageName):
        poseFullPath =os.path.join('/media/remote_home/chang/z_master-thesis/render_eft', imageName)
        if os.path.exists(poseFullPath) ==False:
            print(f"Pose path is not valid: {poseFullPath}")
            print(fsdfsd)
        img  = Image.open(poseFullPath).convert('L').resize((256, 256))
        img = np.expand_dims(np.array(img, dtype=float), axis=-1).transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32).sub_(127.5).div_(127.5)

    def get_image(self, imageName, bbox):
        imgFullPath =os.path.join('/media/remote_home/chang/datasets/coco/train2014', imageName)
        if os.path.exists(imgFullPath) ==False:
            print(f"Img path is not valid: {imgFullPath}")
            print(fsdfsd)
        img  = Image.open(imgFullPath).convert('RGB')
        img = self.get_crop_img(img, bbox).resize((self.img_size, self.img_size))
        img = np.array(img, dtype=float)
        s = random.randint(0, 9)
        if s > 4:
            img = gaussian(img,sigma=1,multichannel=True).transpose(2, 0, 1)
            #img = np.fliplr(img).transpose(2, 0, 1)
            return torch.tensor(img.copy(), dtype=torch.float32).sub_(127.5).div_(127.5)
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32).sub_(127.5).div_(127.5)

    def get_crop_img(self, img, bbox):
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        return img.crop([x1, y1, x2, y2])

    def get_caption_vector(self,text_model, caption):
        return text_model.get_sentence_vector(caption.replace('\n', '').lower())
    

        return img.transpose(2, 0, 1)


if __name__ == "__main__":
    """coco_keypoint = COCO(cfg.COCO_keypoints_TRAIN)
    eft_data_all = getEFTCaption(cfg)
    previous_img_ids = []
    for i in tqdm(range(len(eft_data_all)), desc='  - (Dataset)   ', leave=False):
        if i > 50:
            break
        # one eft data correspond to one keypoint in one img
        keypoint_ann = coco_keypoint.loadAnns(eft_data_all[i]['annotId'])[0]
        img_id = keypoint_ann['image_id']
        keypoint_ids = coco_keypoint.getAnnIds(imgIds=img_id)
        if img_id in previous_img_ids:
            continue
        if len(keypoint_ids) != 1:
            continue
        else:
            previous_img_ids.append(img_id)

        imageName = eft_data_all[i]['imageName']

        imgFullPath =os.path.join('/media/remote_home/chang/datasets/coco/train2014', imageName)
        if os.path.exists(imgFullPath) ==False:
            print(f"Img path is not valid: {imgFullPath}")
            print(fsdfsd)
        bbox=keypoint_ann['bbox']
        print(bbox)
        img  = Image.open(imgFullPath).convert('RGB')
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2]).resize((256, 256))
        img = np.array(img, dtype=float)
        #img = gaussian(img,sigma=1,multichannel=True)
        img = random_noise(img,var=0.155**2)
        img = Image.fromarray((img).astype(np.uint8))

        img.save('/media/remote_home/chang/thesis_rgbimg/gg.jpg')"""
    
    """filepath = os.path.join('/media/remote_home/chang/thesis_rgbimg/coco', 'captions.pickle')
    
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        train_captions, test_captions = x[0], x[1]
        ixtoword, wordtoix = x[2], x[3]
        del x
        n_words = len(ixtoword)
        print('n_words: ', n_words)
        print(ixtoword[0])
        print(ixtoword[1])"""
    
    """filepath = os.path.join('/media/remote_home/chang/thesis_rgbimg/coco', 'filenames.pickle')
    
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        print(len(x))
        print(x[0])"""
        
    a = np.zeros((5), dtype='int64')
    a = np.expand_dims(a, axis=0)
    print(a.transpose(1,0))