ann_file = './dataset/Flickr8k_text/Flickr8k.token.txt'
output_file = './vocab.txt'


captions = []

with open(ann_file, "r") as ann_f:
    for line in ann_f:
        captions.append(line.split()[1:])

vocab = []
for caption in captions:
    for word in caption:
        if word not in vocab:
            vocab.append(word)

with open(output_file, "w") as out_f:
    for word in vocab:
        out_f.write(word + "\n")
