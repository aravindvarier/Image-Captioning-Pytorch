import utils
ann_file = './dataset/Flickr8k_text/Flickr8k.token.txt'
train_file = './dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
val_file = './dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test_file = './dataset/Flickr8k_text/Flickr_8k.testImages.txt'
output_file = './vocab.txt'


captions = []
train_lines = open(train_file, "r").readlines()
val_lines = open(val_file, "r").readlines()
test_lines = open(test_file, "r").readlines()

with open(ann_file, "r") as ann_f:
    for line in ann_f:
        img = line.split('#')[0] + "\n"
        if (img in train_lines) or (img in val_lines) or (img in test_lines):
            caption = utils.clean_description(line.split()[1:])
            captions.append(caption)

vocab = []
for caption in captions:
    for word in caption:
        if word not in vocab:
            vocab.append(word)

print("Vocabulary length: ",len(vocab))
with open(output_file, "w") as out_f:
    for word in vocab:
        out_f.write(word + "\n")
