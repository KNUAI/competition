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

parser.add_argument('--test_files', default='./data/test_data.csv', type=str, help='test_files_dir')
parser.add_argument('--sample_submission', default='./data/sample_submission.csv', type=str, help='sample_files_dir')

parser.add_argument('--pretrained_model', default='klue/roberta-large', type=str, help='pretrained_model_name')
parser.add_argument('--seed', type=int, default=1234, help='seed')

parser.add_argument('--des', type=str, default=None, help='description')

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

def test_read_data(tokenizer, paths, files):
    #max_length of train_dataset: 90 -> 128

    #load_data
    file_path = os.path.join(paths, files)
    raw_data = pd.read_csv(file_path, sep=',', header=0)

    premise = raw_data['premise'].values.tolist()
    hypothesis = raw_data['hypothesis'].values.tolist()

    #tokenize
    inputs = []
    segs = []
    for i in range(len(raw_data)):
        input_dict = tokenizer(premise[i], hypothesis[i], padding = 'max_length', max_length = args.max_length, return_tensors = 'pt', return_attention_mask = False)
        inputs.append(input_dict['input_ids'])
        segs.append(input_dict['token_type_ids'])

    input_tensor = torch.stack(inputs, dim=0)
    seg_tensor = torch.stack(segs, dim=0)
    mask_tensor = ~ (input_tensor == 0)

    output_tensor = torch.cat([input_tensor, seg_tensor, mask_tensor], dim=1)

    return output_tensor

#read_data
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
paths = os.getcwd()
test_inputs = test_read_data(tokenizer, paths, args.test_files)

#data_loader
test_data = TensorDataset(test_inputs, test_inputs)
test_dataloader = DataLoader(test_data, sampler=None, batch_size=args.batch_size)

model1 = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels).cuda()
model2 = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels).cuda()
model3 = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels).cuda()
model4 = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels).cuda()
model5 = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels).cuda()

#lr = 8e-6 && batch = 32 && additional data && epoch 50 && scheduler && dropout 0.2
model1.load_state_dict(torch.load(f'./save_models/fold_1_batch_32_lr_8e-06_acc_91.6071_epochs_17.pth'))
model2.load_state_dict(torch.load(f'./save_models/fold_2_batch_32_lr_8e-06_acc_92.3393_epochs_36.pth'))
model3.load_state_dict(torch.load(f'./save_models/fold_3_batch_32_lr_8e-06_acc_92.3036_epochs_15.pth'))
model4.load_state_dict(torch.load(f'./save_models/fold_4_batch_32_lr_8e-06_acc_91.6071_epochs_48.pth'))
model5.load_state_dict(torch.load(f'./save_models/fold_5_batch_32_lr_8e-06_acc_90.9253_epochs_22.pth'))

#test
with torch.no_grad():
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    print('Running Real Validation...')

    preds_list = []
    for data, labels in test_dataloader:

        src = data[:, 0, :]
        segs = data[:, 1, :]
        mask = data[:, 2, :]

        src = src.to(device)
        segs = segs.to(device)
        mask = mask.to(device)

        output1 = model1(src,
                        token_type_ids=segs,
                        attention_mask=mask)

        output2 = model2(src,
                        token_type_ids=segs,
                        attention_mask=mask)

        output3 = model3(src,
                        token_type_ids=segs,
                        attention_mask=mask)

        output4 = model4(src,
                        token_type_ids=segs,
                        attention_mask=mask)

        output5 = model5(src,
                        token_type_ids=segs,
                        attention_mask=mask)
        outputs = (output1.logits + output2.logits + output3.logits + output4.logits + output5.logits) / 5

        preds = np.argmax(outputs.detach().cpu().numpy(), axis = 1)
        preds_list.append(preds)

    preds_list = np.concatenate(preds_list, axis = 0)

submission = pd.read_csv(args.sample_submission)
label_dict = {'entailment' : 0, 'contradiction' : 1, 'neutral' : 2}
out = [list(label_dict.keys())[r] for r in preds_list]
submission['label'] = out
submission.to_csv(f'sample_submission_{args.des}.csv', index = False)







