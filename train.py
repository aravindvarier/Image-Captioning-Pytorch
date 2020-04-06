import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
# import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import eval
import bleu
import utils
import string

from models import *

img_dir = './dataset/Flickr8k_Dataset/'
ann_dir = './dataset/Flickr8k_text/Flickr8k.token.txt'
train_dir = './dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
val_dir = './dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test_dir = './dataset/Flickr8k_text/Flickr_8k.testImages.txt'

vocab_file = './vocab.txt'

SEED = 123
torch.manual_seed(SEED)

mode = 'train'

class Flickr8kDataset(Dataset):
    """Flickr8k dataset."""
    
    def __init__(self, img_dir, split_dir, ann_dir, vocab_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            ann_dir (string): Directory with all the tokens
            split_dir (string): Directory with all the file names which belong to a certain split(train/dev/test)
            vocab_file (string): File which has the entire vocabulary of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split_dir = split_dir
        self.SOS = self.EOS = None
        self.word_2_token = None
        self.vocab_size = None
        self.image_file_names, self.captions, self.tokenized_captions= self.tokenizer(self.split_dir, self.ann_dir)
        
        if(transform == None):
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def tokenizer(self, split_dir, ann_dir):
        image_file_names = []
        captions = []
        tokenized_captions = []
        
        with open(split_dir, "r") as split_f:
            sub_lines = split_f.readlines()
        
        with open(ann_dir, "r") as ann_f:
            for line in ann_f:
                if line.split("#")[0] + "\n" in sub_lines:
                    caption = utils.clean_description(line.split()[1:])
                    image_file_names.append(line.split()[0])
                    captions.append(caption)


        vocab = []
        # for caption in captions:
        #     for word in caption:
        #         if word not in vocab:
        #             vocab.append(word)
        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                vocab.append(line.strip())
        
        self.vocab_size = len(vocab) + 2 #The +2 is to accomodate for the SOS and EOS
        self.SOS = 0
        self.EOS = self.vocab_size - 1
        
        
        self.word_2_token = dict(zip(vocab, list(range(1, self.vocab_size - 1))))

        for caption in captions:
            temp = []
            for word in caption:
                temp.append(self.word_2_token[word])
            temp.insert(0, self.SOS)
            temp.append(self.EOS)
            tokenized_captions.append(temp)
            
        assert(len(image_file_names) == len(captions))
            
        return image_file_names, captions, tokenized_captions
        

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name, cap_tok, caption = self.image_file_names[idx], self.tokenized_captions[idx], self.captions[idx]
        img_name, instance = img_name.split('#')
        img_name = os.path.join(self.img_dir,
                                img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        cap_tok = torch.tensor(cap_tok)
        sample = {'image': image, 'cap_tok': cap_tok, 'caption': caption}

        

        return sample




def collater(batch):
    '''This functions pads the cpations and makes them equal length
    '''
    
    cap_lens = torch.tensor([len(item['cap_tok']) for item in batch]) #Includes SOS and EOS as part of the length
    caption_list = [item['cap_tok'] for item in batch]
#     padded_captions = pad_sequence(caption_list, padding_value=9631) 
    images = torch.stack([item['image'] for item in batch])

    return images, caption_list, cap_lens


def display_sample(sample):
    image = sample['image']
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    image = inv_normalize(image)
    caption = ' '.join(sample['caption'])
    cap_tok = sample['cap_tok']
    plt.figure()
    plt.imshow(image.permute(1,2,0))
    print("Caption: ", caption)
    print("Tokenized Caption: ", cap_tok)
    plt.show()

def predict(model, device, image_name):
    vocab = []
    with open(vocab_file, "r") as vocab_f:
        for line in vocab_f:
            vocab.append(line.strip())
    image_path = os.path.join(img_dir, image_name)
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
#                 transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    hypotheses = eval.get_output_sentence(model, device, image, vocab)

    for i in range(len(hypotheses)):
        hypotheses[i] = [vocab[token - 1] for token in hypotheses[i]]
        hypotheses[i] = " ".join(hypotheses[i])

    print(hypotheses) 


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def adjust_optim(optimizer, n_iter, warmup_steps):
    optimizer.param_groups[0]['lr'] = (word_embedding_size**(-0.5)) * min(n_iter**(-0.5), n_iter*(warmup_steps**(-1.5)))




def train_for_epoch(model, dataloader, optimizer, device, n_iter):
    '''Train an EncoderDecoder for an epoch

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    
    criterion1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    criterion2 = nn.MSELoss(reduction='sum')
    total_loss = 0 
    total_num = 0
    for data in tqdm(dataloader):
        images, captions, cap_lens = data
        captions = pad_sequence(captions, padding_value=model.target_eos) #(seq_len, batch_size)
        images, captions, cap_lens = images.to(device), captions.to(device), cap_lens.to(device)
        optimizer.zero_grad()
        if model.decoder_type == 'rnn': 
            logits, total_attention_weights = model(images, captions) #total_attention_weights -> (L, N, 1)
            total_attention_weights = total_attention_weights.sum(axis=0).squeeze(2).T
        else:
            logits = model(images, captions)

        captions = captions[1:]
        mask = model.get_target_padding_mask(captions)
        captions = captions.masked_fill(mask,-1)
        loss1 = criterion1(torch.flatten(logits, 0, 1), torch.flatten(captions))
        if model.decoder_type == 'rnn':
            loss2 = criterion2(total_attention_weights, torch.ones_like(total_attention_weights))
            loss = loss1 + lamda * loss2
        else:
            loss = loss1
        total_loss += loss.item()
        total_num += len(cap_lens)
        # print(total_loss/total_num)
        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()
        if model.decoder_type == 'transformer':
            adjust_optim(optimizer, n_iter, warmup_steps)
        n_iter += 1
    return total_loss/total_num, n_iter


decoder_type = 'rnn' #transformer, rnn
warmup_steps = 4000

CNN_channels = 1024 #DO SOMETHING ABOUT THIS, 2048 for resnet101
max_epochs = 100
beam_width = 4

word_embedding_size = 512
attention_dim = 512
model_save_path = './model_saves/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lamda = 1.
if decoder_type == 'rnn':
    learning_rate = 0.005
    decoder_hidden_size = 1800
    dropout = 0.5
else:    
    learning_rate = (word_embedding_size**(-0.5)) * min(n_iter**(-0.5), n_iter*(warmup_steps**(-1.5)))
    decoder_hidden_size = CNN_channels
    dropout = 0.1

batch_size = 64
grad_clip = 5.
transformer_layers = 6
heads = 3

use_checkpoint = False
checkpoint_path = 'epoch50.pt'

if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

train_data = Flickr8kDataset(img_dir, train_dir, ann_dir, vocab_file)
val_data = eval.TestDataset(img_dir, val_dir, ann_dir, vocab_file)
test_data = eval.TestDataset(img_dir, test_dir, ann_dir, vocab_file)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collater)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=eval.collater)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=eval.collater)

encoder_class = Encoder
if decoder_type == 'rnn':
    decoder_class = Decoder
else:
    decoder_class = TransformerDecoder

model = EncoderDecoder(encoder_class, decoder_class, train_data.vocab_size, target_sos=train_data.SOS, 
                      target_eos=train_data.EOS, encoder_hidden_size=CNN_channels, 
                       decoder_hidden_size=decoder_hidden_size, 
                       word_embedding_size=word_embedding_size, attention_dim=attention_dim, decoder_type=decoder_type, cell_type='lstm', beam_width=beam_width, dropout=dropout,
                       transformer_layers=transformer_layers, num_heads=heads)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

fixed_image = "2090545563_a4e66ec76b.jpg"

if mode == "train":
    n_iter = 1
    best_bleu = 0.
    epoch = 1

    if use_checkpoint:
        checkpoint = torch.load(model_save_path + checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Loss of checkpoint model: ", loss)
    print("Ground Truth captions: ", [" ".join(caption) for caption in val_data.all_captions[fixed_image]])
    while epoch <= max_epochs:
        model.to(device)
        model.train()
        loss, n_iter = train_for_epoch(model, train_dataloader, optimizer, device, n_iter)
        model.eval()
        # bleu_score = bleu.compute_average_bleu_over_dataset(
        #     model, val_dataloader,
        #     val_data.SOS,
        #     val_data.EOS,
        #     device,
        # )
        print(f'Epoch {epoch}: loss={loss}')
        eval.print_metrics(model, device, val_data, val_dataloader)
        print("Predicted caption: ",predict(model, device, fixed_image))
    #     print(f'Epoch {epoch}: loss={loss}')
    #         if bleu_score < best_bleu:
    #             num_poor += 1
    #         else:
    #             num_poor = 0
    #             best_bleu = bleu_score
        if epoch % 10 == 0:
            model.cpu()
            print('Saving Model on Epoch', epoch)
            torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": loss
                        }, model_save_path + f'epoch{epoch}.pt')
            
        epoch += 1
        if epoch > max_epochs:
            print(f'Finished {max_epochs} epochs')
        torch.cuda.empty_cache()
elif mode == "test":
    model.load_state_dict(torch.load(model_save_path + 'epoch50.pt'))
    model.to(device)
    model.eval()

    predict(model, device, fixed_image)
    # eval.print_metrics(model, device, test_data, test_dataloader)
    


