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
import copy
import argparse

from models import *

img_dir = './dataset/Flickr8k_Dataset/'
ann_dir = './dataset/Flickr8k_text/Flickr8k.token.txt'
train_dir = './dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
val_dir = './dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test_dir = './dataset/Flickr8k_text/Flickr_8k.testImages.txt'

vocab_file = './vocab.txt'

SEED = 123
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
                    caption = utils.clean_description(line.replace("-", " ").split()[1:])
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

    return hypotheses


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




def train_for_epoch(model, dataloader, optimizer, device, n_iter, args):
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
            logits = model(images, captions).permute(1, 0, 2)

        captions = captions[1:]
        mask = model.get_target_padding_mask(captions)
        captions = captions.masked_fill(mask,-1)
        loss1 = criterion1(torch.flatten(logits, 0, 1), torch.flatten(captions))
        if model.decoder_type == 'rnn':
            loss2 = criterion2(total_attention_weights, torch.ones_like(total_attention_weights))
            loss = loss1 + lamda * loss2
        else:
            if args.smoothing:
                eps = args.Lepsilon
                captions = captions.masked_fill(mask,0) #just to make the scatter work so no indexing issue occurs
                gold = captions.contiguous().view(-1)

                logits = torch.flatten(logits, 0, 1)
                n_class = logits.shape[-1]
                one_hot = torch.zeros_like(logits, device=logits.device).scatter(1, gold.view(-1, 1), 1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
                log_prb = torch.log_softmax(logits, dim=1)

                captions = captions.masked_fill(mask,-1) #puttng the right mask back on
                gold = captions.contiguous().view(-1)
                non_pad_mask = gold.ne(-1)

                loss = -(one_hot * log_prb).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).sum()  # average later

                del gold, log_prb, non_pad_mask #hoping that this saves a bit of memory
            else:
                loss = loss1
        total_loss += loss.item()
        total_num += len(cap_lens)
        # print(total_loss/total_num)
        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()
        # if model.decoder_type == 'transformer':
        #     adjust_optim(optimizer, n_iter, warmup_steps)
        n_iter += 1
        torch.cuda.empty_cache()
    return total_loss/total_num, n_iter


parser = argparse.ArgumentParser(description='Training Script for Encoder+LSTM decoder')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--batch-size', type=int, help='batch size', default=64)
parser.add_argument('--batch-size-val', type=int, help='batch size validation', default=64)
parser.add_argument('--encoder-type', choices=['resnet18', 'resnet50', 'resnet101'], default='resnet18',
                    help='Network to use in the encoder (default: resnet18)')
parser.add_argument('--fine-tune', type=int, choices=[0,1], default=0)
parser.add_argument('--decoder-type', choices=['rnn', 'transformer'], default='rnn')
parser.add_argument('--beam-width', type=int, default=4)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--decoder-hidden-size', help="Hidden size for lstm", type=int, default=512)
parser.add_argument('--experiment-name', type=str, default="autobestmodel")
parser.add_argument('--num-tf-layers', help="Number of transformer layers", type=int, default=3)
parser.add_argument('--num-heads', help="Number of heads", type=int, default=2)
parser.add_argument('--beta1', help="Beta1 for Adam", type=float, default=0.9)
parser.add_argument('--beta2', help="Beta2 for Adam", type=float, default=0.999)
parser.add_argument('--dropout-lstm', help="Dropout_LSTM", type=float, default=0.5)
parser.add_argument('--dropout-trans', help="Dropout_Trans", type=float, default=0.1)
parser.add_argument('--smoothing', help="Label smoothing", type=int, default=1)
parser.add_argument('--Lepsilon', help="Label smoothing epsilon", type=float, default=0.1)
parser.add_argument('--use-checkpoint', help="Use checkpoint or start from beginning", type=int, default=0)
parser.add_argument('--checkpoint-name', help="Checkpoint model file name", type=str, default=None)

args = parser.parse_args()

encoder_type = args.encoder_type
decoder_type = args.decoder_type #transformer, rnn
warmup_steps = 4000
n_iter = 1

if encoder_type == 'resnet18':
    CNN_channels = 512 #DO SOMETHING ABOUT THIS, 2048 for resnet101
else:
    CNN_channels = 2048

max_epochs = args.num_epochs
beam_width = args.beam_width

print("Epochs are read correctly: ", max_epochs)
print("Encoder type is read correctly: ", encoder_type)
print("Number of CNN channels being used: ", CNN_channels)
print("Fine tune setting is set to: ", bool(args.fine_tune))


word_embedding_size = 512
attention_dim = 512
model_save_path = './model_saves/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lamda = 1.
if decoder_type == 'rnn':
    learning_rate = args.lr
    decoder_hidden_size = args.decoder_hidden_size
    dropout = args.dropout_lstm
else: 
    print("Label smoothing set to: ", bool(args.smoothing))   
    learning_rate = 0.00004
    # learning_rate = (CNN_channels**(-0.5)) * min(n_iter**(-0.5), n_iter*(warmup_steps**(-1.5)))
    decoder_hidden_size = CNN_channels
    dropout = args.dropout_trans

batch_size = args.batch_size
batch_size_val = args.batch_size_val
grad_clip = 5.
transformer_layers = args.num_tf_layers
heads = args.num_heads
beta1 = args.beta1
beta2 = args.beta2

use_checkpoint = args.use_checkpoint
checkpoint_path = args.checkpoint_name
mode = 'train'

if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

train_data = Flickr8kDataset(img_dir, train_dir, ann_dir, vocab_file)
val_data = eval.TestDataset(img_dir, val_dir, ann_dir, vocab_file)
test_data = eval.TestDataset(img_dir, test_dir, ann_dir, vocab_file)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collater)
val_dataloader = DataLoader(val_data, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)
test_dataloader = DataLoader(test_data, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)

encoder_class = Encoder
if decoder_type == 'rnn':
    decoder_class = Decoder
else:
    decoder_class = TransformerDecoder

model = EncoderDecoder(encoder_class, decoder_class, train_data.vocab_size, target_sos=train_data.SOS, 
                      target_eos=train_data.EOS, fine_tune=bool(args.fine_tune), encoder_type=args.encoder_type, encoder_hidden_size=CNN_channels, 
                       decoder_hidden_size=decoder_hidden_size, 
                       word_embedding_size=word_embedding_size, attention_dim=attention_dim, decoder_type=decoder_type, cell_type='lstm', beam_width=beam_width, dropout=dropout,
                       transformer_layers=transformer_layers, num_heads=heads)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2)) # used to experiment with (0.9, 0.98) for transformer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

fixed_image = "2090545563_a4e66ec76b.jpg"

if mode == "train":
    
    best_bleu4 = 0.
    poor_iters = 0
    epoch = 1
    num_iters_change_lr = 4
    max_poor_iters = 10
    best_model = None
    best_optimizer = None
    best_loss = None
    best_epoch = None
    best_metrics = None

    if use_checkpoint:
        checkpoint = torch.load(model_save_path + checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Loss of checkpoint model: ", loss)
    model.to(device)
    print("Ground Truth captions: ", [" ".join(caption) for caption in val_data.all_captions[fixed_image]])
    while epoch <= max_epochs:
        model.train()
        loss, n_iter = train_for_epoch(model, train_dataloader, optimizer, device, n_iter, args)
        

        # EVALUATE AND ADJUST LR ACCORDINGLY
        model.eval()
        print(f'Epoch {epoch}: loss={loss}')
        metrics = eval.get_pycoco_metrics(model, device, val_data, val_dataloader)
        print(metrics)
        is_epoch_better = metrics['Bleu_4'] > best_bleu4
        if is_epoch_better:
            poor_iters = 0
            best_bleu4 = metrics['Bleu_4']
            best_model = copy.deepcopy(model)
            best_epoch = copy.deepcopy(epoch)
            # best_optimizer = copy.deepcopy(optimizer)
            best_loss = copy.deepcopy(loss)
            best_metrics = copy.deepcopy(metrics)
        else:
            poor_iters += 1
        # if poor_iters > 0 and poor_iters % num_iters_change_lr == 0:
        #     print("Adjusting learning rate on epoch ", epoch)
        #     utils.adjust_learning_rate(optimizer, 0.6)
        if poor_iters > max_poor_iters:
            print("Hasn't improved for ", max_poor_iters, " epochs...I give up :(")
            test_metrics = eval.get_pycoco_metrics(best_model, device, test_data, test_dataloader)
            utils.save_model_and_result(model_save_path, args.experiment_name, best_model, decoder_type, best_optimizer, best_epoch, best_bleu4, best_loss, best_metrics, test_metrics)
            break
        print("Predicted caption: ",predict(model, device, fixed_image))
        
        # # SAVE MODEL EVERY 10 EPOCHS
        # if epoch % 10 == 0:
        #     model.cpu()
        #     utils.save_model_and_result(model_save_path, args.experiment_name, best_model, best_optimizer, best_epoch, best_bleu4, best_loss)
            
        epoch += 1
        if epoch > max_epochs:
            test_metrics = eval.get_pycoco_metrics(best_model, device, test_data, test_dataloader)
            utils.save_model_and_result(model_save_path, args.experiment_name, best_model, decoder_type, best_optimizer, best_epoch, best_bleu4, best_loss, best_metrics, test_metrics)
            print(f'Finished {max_epochs} epochs')
        torch.cuda.empty_cache()
elif mode == "test":
    checkpoint = torch.load(model_save_path + checkpoint_path)
    # print("This model has bleu4 of: ", checkpoint['best_bleu4'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # predict(model, device, fixed_image)
    print(eval.get_pycoco_metrics(model, device, test_data, test_dataloader))
    


