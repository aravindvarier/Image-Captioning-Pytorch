import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import warnings
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
import utils


class TestDataset(Dataset):
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
        self.vocab = None
        self.vocab_size = None
        self.images = self.captions = []
        self.all_captions = {}
        self.preprocess_files(self.split_dir, self.ann_dir, vocab_file)
        
        if(transform == None):
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
#                 transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    
    def preprocess_files(self, split_dir, ann_dir, vocab_file):
        # all_captions = {}
        
        with open(split_dir, "r") as split_f:
            sub_lines = split_f.readlines()
        
        with open(ann_dir, "r") as ann_f:
            for line in ann_f:
                if line.split("#")[0] + "\n" in sub_lines:
                    image_file = line.split('#')[0]
                    caption = utils.clean_description(line.split()[1:])
                    if image_file in self.all_captions:
                        self.all_captions[image_file].append(caption)
                    else:
                        self.all_captions[image_file] = [caption]

        self.images = list(self.all_captions.keys())
        self.captions = list(self.all_captions.values())
        assert(len(self.images) == len(self.captions))
        assert(len(self.captions[-1]) == 5)
        vocab = []
        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                vocab.append(line.strip())
        
        self.vocab_size = len(vocab) + 2 #The +2 is to accomodate for the SOS and EOS
        self.SOS = 0
        self.EOS = self.vocab_size - 1
        self.vocab = vocab        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name, caps = self.images[idx], self.captions[idx]
        img_name = os.path.join(self.img_dir,
                                img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'captions': caps}

def collater(batch):
    images = torch.stack([item['image'] for item in batch])
    all_caps = [item['captions'] for item in batch]

    return images, all_caps

def get_output_sentence(model, device, images, vocab):


    # hypotheses = []
    
    with torch.no_grad():
        torch.cuda.empty_cache()

        images = images.to(device)
        target_eos = len(vocab) + 1
        target_sos = 0

        b_1 = model(images, on_max='halt')
        captions_cand = b_1[..., 0]

        cands = captions_cand.T
        cands_list = cands.tolist()
        for i in range(len(cands_list)): #Removes sos tags
            cands_list[i] = list(filter((target_sos).__ne__, cands_list[i]))
            cands_list[i] = list(filter((target_eos).__ne__, cands_list[i]))

    #     hypotheses += cands_list

    
    return cands_list

def print_metrics(model, device, dataset, dataloader):
    references = []
    hypotheses = []
    assert(len(dataset.captions) == len(dataset.images))
    with torch.no_grad():
        for data in tqdm(dataloader):
            torch.cuda.empty_cache()
            images, captions = data

            references += captions
            hypotheses += get_output_sentence(model, device, images, dataset.vocab)

        for i in range(len(references)):
            hypotheses[i] = [dataset.vocab[j - 1] for j in hypotheses[i]]

        assert(len(references) == len(hypotheses))
        
        # bleu scores
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        print('BLEU-1 ({})\t'
                  'BLEU-2 ({})\t'
                  'BLEU-3 ({})\t'
                  'BLEU-4 ({})\t'.format(bleu_1, bleu_2, bleu_3, bleu_4))

        # meteor score
        total_m_score = 0.0
        
        for i in range(len(references)):
            actual = [" ".join(ref) for ref in references[i]]
            total_m_score += meteor_score(actual, " ".join(hypotheses[i]))
        
        print('Meteor Score: {}'.format(total_m_score/len(references)))

