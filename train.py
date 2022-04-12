import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--max_length', type=int, default=128, help='max_length')
parser.add_argument('--num_labels', type=int, default=3, help='num_labels')
parser.add_argument('--fold_k', type=int, default=5, help='k_fold: 1, 2, 3, 4, 5')

parser.add_argument('--train_files', default='./data/train_data.csv', type=str, help='train_files_dir')

parser.add_argument('--pretrained_model', default='klue/roberta-large', type=str, help='pretrained_model_name')
parser.add_argument('--seed', type=int, default=1234, help='seed')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#seed
if args.seed is not None:
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#train_valid_fold
def process_fold(paths, files, fold_k):
    #len(raw_data) = 24998 -> 24998 * 0.2 = 5000

    #load_data
    file_path = os.path.join(paths, files)
    raw_data = pd.read_csv(file_path, sep=',', header=0)
    train_df = pd.concat([raw_data[:5000*(fold_k-1)], raw_data[5000*fold_k:]])
    valid_df = raw_data[5000*(fold_k-1):5000*fold_k]

    return train_df, valid_df

def read_data(tokenizer, fold):
    #max_length of train_dataset: 90 -> 128

    raw_data = fold

    premise = raw_data['premise'].values.tolist()
    hypothesis = raw_data['hypothesis'].values.tolist()
    label = raw_data['label'].values.tolist()

    #label_list
    label_list = ['entailment', 'contradiction', 'neutral']
    label_map = {label: i for i, label in enumerate(label_list)}

    #tokenize
    inputs = []
    segs = []
    targets = []
    for i in range(len(raw_data)):
        input_dict = tokenizer(premise[i], hypothesis[i], padding = 'max_length', max_length = args.max_length, return_tensors = 'pt', return_attention_mask = False)
        inputs.append(input_dict['input_ids'])
        segs.append(input_dict['token_type_ids'])
        targets.append(label_map[label[i]])

    input_tensor = torch.stack(inputs, dim=0)
    seg_tensor = torch.stack(segs, dim=0)
    mask_tensor = ~ (input_tensor == 0)

    output_tensor = torch.cat([input_tensor, seg_tensor, mask_tensor], dim=1)
    output_labels = torch.tensor(targets)

    return output_tensor, output_labels

#read_data
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
paths = os.getcwd()
train_fold, valid_fold = process_fold(paths, args.train_files, args.fold_k)
train_inputs, train_labels = read_data(tokenizer, train_fold)
valid_inputs, valid_labels = read_data(tokenizer, valid_fold)

#data_loader
train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

validation_data = TensorDataset(valid_inputs, valid_labels)
validation_dataloader = DataLoader(validation_data, sampler=None, batch_size=args.batch_size)

model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels)
model.cuda()

optimizer = AdamW(model.parameters(), lr = args.lr,  eps = 1e-8)

#model.load_state_dict(torch.load(f'./save_models/batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}.pth'))

stop_loss = np.inf
count = 0
for epoch_i in range(args.epochs):

    #train
    model.train()
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')

    total_loss = 0
    train_acc_sum = 0
    train_loss = []
    for step, (data, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        src = data[:, 0, :]
        segs = data[:, 1, :]
        mask = data[:, 2, :]

        src = src.to(device)
        segs = segs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        outputs = model(src, 
                        token_type_ids=segs, 
                        attention_mask=mask, 
                        labels=labels)

        loss = outputs.loss
        total_loss += loss.item()
        train_loss.append(total_loss/(step+1))

        targets = labels.detach().cpu().numpy()
        preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis = 1)
        train_acc = np.equal(targets, preds).sum()
        train_acc_sum += train_acc

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'  Average training loss: {avg_train_loss:.2f}')
    print(f'  Train Accuracy: {100 * train_acc_sum / len(train_dataloader.dataset):.4f}')

    #validation
    with torch.no_grad():
        model.eval()
        print('Running Real Validation...')

        val_acc_sum = 0
        targets_list = []
        preds_list = []
        for data, labels in validation_dataloader:

            src = data[:, 0, :]
            segs = data[:, 1, :]
            mask = data[:, 2, :]

            src = src.to(device)
            segs = segs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(src,
                            token_type_ids=segs,
                            attention_mask=mask)

            targets = labels.detach().cpu().numpy()
            preds = np.argmax(outputs.logits.detach().cpu().numpy(), axis = 1)
            val_acc = np.equal(targets, preds).sum()
            val_acc_sum += val_acc
            targets_list.append(targets)
            preds_list.append(preds)

        targets_list = np.concatenate(targets_list, axis = 0)
        preds_list = np.concatenate(preds_list, axis = 0)
        f1_scores = f1_score(targets_list, preds_list, average="macro") * 100.0
        total_acc = (preds_list == targets_list).mean() * 100.0

    print(f'  Real Validation Accuracy: {100 * val_acc_sum / len(validation_dataloader.dataset):.4f}')
    print(f'  f1_score: {f1_scores:.4f}')
    print(f'  total_acc: {total_acc:.4f}')

    #early_stopping
    if not os.path.exists('./save_models'):
        os.makedirs('./save_models')
    if np.mean(valid_loss) < stop_loss:
        stop_loss = np.mean(valid_loss)
        print('best_loss:: {:.4f}'.format(stop_loss))
        torch.save(model.state_dict(), f'./save_models/batch_{args.batch_size}_lr_{args.lr}_epochs_{epoch_i}_acc_{total_acc:.4f}_fold_{args.fold_k}.pth')
        count = 0
    else:
        count += 1
        print(f'EarlyStopping counter: {count} out of {args.patience}')
        print(f'best_loss:: {stop_loss:.4f}\t valid_loss:: {np.mean(valid_loss):.4f}' )
        if count >= args.patience:
            print('Ealry stopping')
            break



    






