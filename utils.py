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
    #remove numbers
    table = str.maketrans('', '', string.digits)
    desc = [w.translate(table) for w in desc]
    # remove one letter words except 'a'
    desc = [word for word in desc if len(word)>1 or word == 'a']


    return desc

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))