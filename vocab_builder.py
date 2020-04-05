import string
ann_file = './dataset/Flickr8k_text/Flickr8k.token.txt'
output_file = './vocab.txt'


captions = []

def clean_description(desc):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # # tokenize
    # desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word)>1]

    return desc

with open(ann_file, "r") as ann_f:
    for line in ann_f:
    	caption = clean_description(line.split()[1:])
    	captions.append(caption)

vocab = []
for caption in captions:
    for word in caption:
        if word not in vocab:
            vocab.append(word)

print(len(vocab))
with open(output_file, "w") as out_f:
    for word in vocab:
        out_f.write(word + "\n")
