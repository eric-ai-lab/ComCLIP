import torch
import argparse
from barbar import Bar
from utils.ImageTextDataset import ImageTextDataset
from torchvision import transforms, utils
from utils.ImageTextDataset import ImageTextDataset
from torchvision.transforms import Resize, CenterCrop, Normalize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from clip.model import CLIP
import multiprocessing as mp
import pandas as pd
import copy

parser = argparse.ArgumentParser(description='Process input params')
parser.add_argument('--image_folder', type=str, default="../../../data1/anonymous/Visual_Genome/VG_100K/", help='folder storing images')
parser.add_argument('--relation_folder', type=str, default = 'parsed_relation')
parser.add_argument('--text_path', type=str, default = 'fintune_clip_data.pkl')
parser.add_argument('--with_black', type=bool, default = True)
parser.add_argument('--batch_size', type=int, default = 10)
parser.add_argument('--epoch', type=int, default = 10)
parser.add_argument('--checkpoint_path', type=str, default = 'checkpoints/ablation-study-resnet-finetune.pt')
args = parser.parse_args()
device = 'cuda:6'
torch.manual_seed(42)

transforms = transforms.Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                                CenterCrop(size=(224, 224)),
                                transforms.ToTensor(),
                                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
                                )

def main():
    ## make dataloader
    dataset = ImageTextDataset(args.relation_folder, args.text_path, args.image_folder, with_black=args.with_black, transform = transforms)
    train, val = torch.utils.data.random_split(dataset, [56490, len(dataset)-56490])
    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=12)

    ## make model and load trained CLIP weights
    model = CLIP(1024, 224, (3,4, 6, 3), 64, 1, 77, 49408, 512, 8, 12, "original")
    model.load_state_dict(torch.load('resnet50_clip.pth'), strict=False)
    model.to(device)
    

    loss_image = torch.nn.CrossEntropyLoss() 
    loss_text = torch.nn.CrossEntropyLoss() 

    for param in model.ln_final.parameters():
        param.requires_grad = True
    for param in model.visual.attnpool.parameters():
        param.requires_grad = True

    param_groups = [
            {"params": model.ln_final.parameters()},
            {"params": model.visual.attnpool.parameters()}
        ]

    optimizer = torch.optim.AdamW(param_groups, lr=0.003, weight_decay=0.05)

    ## train
    for epoch in range(args.epoch):
        for i_batch, batch in enumerate(Bar(train_dataloader)):
            optimizer.zero_grad()
            text = torch.squeeze(batch['text']).to(device)
            image = batch['image'].to(device)
            words = (torch.squeeze(batch['subject_text']).to(device), torch.squeeze(batch['object_text']).to(device), torch.squeeze(batch['predicate']).to(device))
            image_tuple = (batch["subject_image"].to(device), batch['object_image'].to(device), batch['subject_object_image'].to(device))

            logits_per_image, logits_per_text, _, _ = model(image, text, words, image_tuple, None, None, None)

            ground_truth = torch.arange(len(image),dtype=torch.long,device=device)
            loss = (loss_image(logits_per_image,ground_truth) + loss_text(logits_per_text, ground_truth)) / 2

            loss.backward()
            optimizer.step()
            if i_batch == 1:
                print(loss)
        print("Finish epoch {}".format(epoch))
        ## save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, args.checkpoint_path)

if __name__=="__main__":
    main()