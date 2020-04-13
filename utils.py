import string
import torch
import eval
from csv import writer
import datetime

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

def append_new_metric(file_name, metric):
    with open('./experiments/'+file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(metric)

def save_model_and_result(save_path, experiment, model, decoder_type, optimizer, best_epoch, bleu4, loss, val_metrics, test_metrics):
    print(f"Saving Best Model with Bleu_4 score of {bleu4}")
    torch.save({
                "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                "epoch": best_epoch,
                "loss": loss,
                "best_bleu4": bleu4
                }, save_path + f'{experiment}.pt')
    now = datetime.datetime.now()
    val_row = [experiment, best_epoch, loss, now] + list(val_metrics.values()) 
    test_row = [experiment, best_epoch, loss, now] + list(test_metrics.values()) 
    append_new_metric(f"{decoder_type}Validation.csv", val_row)
    append_new_metric(f"{decoder_type}Test.csv", test_row)
