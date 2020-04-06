import string

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
