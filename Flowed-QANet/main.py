from config import config, device
from preproc import preproc
from absl import app
import math
import os
import numpy as np
import json
import re
from collections import Counter
import string
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import pickle
import csv
writer = SummaryWriter(log_dir='./log1')
'''
Some functions are from the official evaluation script.
'''

class SQuADDataset(Dataset):
    def __init__(self, npz_file, batch_size):
        dataset = np.load(npz_file)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()

# SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
        batch_size, c_len, w_len = self.context_char_idxs.size()
        ones = torch.ones((batch_size, 1), dtype=torch.int64)
        self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
        self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

        ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
        self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
        self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

        self.y1s += 1
        self.y2s += 1
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if self.y1s[idx].item() >= 0]
        self.num = len(self.valid_idxs)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx=self.valid_idxs[idx]

        return self.context_idxs[idx],self.context_char_idxs[idx], self.question_idxs[idx],self.question_char_idxs[idx],self.y1s[idx],self.y2s[idx],self.ids[idx]
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1+num_updates)/(10+num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

def collate(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, ids = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    ids = torch.from_numpy(np.array(ids)).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, ids

def collate_fn(examples):
    def merge_0d(scalars, dtype=torch.int64):
            return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded
    context_idxs, context_char_idxs, \
            question_idxs, question_char_idxs, \
            y1s, y2s, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)
    # print(context_idxs.shape, context_char_idxs.shape,
    #         question_idxs.shape, question_char_idxs.shape)
    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)




def get_loader(npz_file, batch_size):
    dataset = SQuADDataset(npz_file, batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=5,
                                              collate_fn=collate_fn)
    return data_loader


def discretize(p_start, p_end, max_len=15, no_answer=False):
    """Discretize soft predictions to get start and end indices.

    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.

    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.

    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
    if p_start.min() < 0 or p_start.max() > 1 \
            or p_end.min() < 0 or p_end.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probabilities
    p_start = p_start.unsqueeze(dim=2)
    p_end = p_end.unsqueeze(dim=1)
    p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

    # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
    c_len, device = p_start.size(1), p_start.device
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                                diagonal=max_len)
    if no_answer:
        # Index 0 is no-answer
        p_no_answer = p_joint[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_joint *= is_legal_pair

    # Take pair (i, j) that maximizes p_joint
    max_in_row, _ = torch.max(p_joint, dim=2)
    max_in_col, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(max_in_row, dim=-1)
    end_idxs = torch.argmax(max_in_col, dim=-1)

    if no_answer:
        # Predict no-answer whenever p_no_answer > max_prob
        max_prob, _ = torch.max(max_in_col, dim=-1)
        start_idxs[p_no_answer > max_prob] = 0
        end_idxs[p_no_answer > max_prob] = 0

    return start_idxs, end_idxs


def convert_tokens(eval_file, qa_id, pp1, pp2,no_answer=True):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        if no_answer and(p1==0 or p2==0):
            answer_dict[str(qid)]=''
            remapped_dict[uuid]=''
        else:
            if no_answer:
                p1,p2=p1-1,p2-1
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
            remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict

def compute_avna(prediction,ground_truths):
    return float(bool(prediction) == bool(ground_truths))

def evaluate(eval_file, answer_dict,no_answer=True):
    avna=f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        if no_answer:
            avna+=compute_avna(prediction,ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    if no_answer:
        avna=100.0*avna/total
        return {'exact_match': exact_match, 'f1': f1,'AvNA':avna}

    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        #exclude.update('，', '。', '、', '；', '「', '」')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens)==0 or len(ground_truth_tokens)==0:
        return int(prediction_tokens==ground_truth_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction,'')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def train(model, optimizer, scheduler, dataset, dev_dataset, dev_eval_file, start, ema):
    model.train()
    losses = []
    print(f'Training epoch {start}')
    for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
        optimizer.zero_grad()
        Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        # print(Cwid.shape,Qwid.shape,Ccid.shape,Qcid.shape,'9999999')
        print(i)
        # print()
        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        y1, y2 = y1.to(device), y2.to(device)
        p1 = F.log_softmax(p1, dim=1)
        p2 = F.log_softmax(p2, dim=1)
        loss1 = F.nll_loss(p1, y1)
        loss2 = F.nll_loss(p2, y2)
        loss = (loss1 + loss2)
        writer.add_scalar('data/loss', loss.item(), i+start*len(dataset))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        ema(model, i+start*len(dataset))

        scheduler.step()
        if (i+1) % config.checkpoint == 0 and (i+1) < config.checkpoint*(len(dataset)//config.checkpoint):
            ema.assign(model)
            metrics = test(model, dev_dataset, dev_eval_file, i+start*len(dataset))
            ema.resume(model)
            model.train()
        for param_group in optimizer.param_groups:
            #print("Learning:", param_group['lr'])
            writer.add_scalar('data/lr', param_group['lr'], i+start*len(dataset))
        print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
    loss_avg = np.mean(losses)
    print("STEP {:8d} Avg_loss {:8f}\n".format(start, loss_avg))

def test(model, dataset, eval_file, test_i):
    print("\nTest")
    model.eval()
    answer_dict = {}
    sub_dict={}
    losses = []
    # for k,v in model.named_parameters():
    #     print(k,v.view(-1)[0])
    num_batches = config.val_num_batches
    with torch.no_grad():
        for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
            # import pdb
            # pdb.set_trace()
            Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)

            
            P1, P2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(device), y2.to(device)
            p1 = F.log_softmax(P1, dim=1)
            p2 = F.log_softmax(P2, dim=1)
            loss1 = F.nll_loss(p1, y1)
            loss2 = F.nll_loss(p2, y2)
            loss = torch.mean(loss1 + loss2)
            losses.append(loss.item())

            # p1 = F.softmax(P1, dim=1)
            # p2 = F.softmax(P2, dim=1)
            p1,p2=p1.exp(),p2.exp()
            starts, ends = discretize(p1, p2, config.max_ans_limit)

            #ymin = []
            # ymax = []Cwid, Ccid, Qwid, Qcid = Cwid.cuda(), Ccid.cuda(),Qwid.cuda(), Qcid.cuda()
            outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
            for j in range(outer.size()[0]):
                outer[j] = torch.triu(outer[j])
                #outer[j] = torch.tril(outer[j], config.ans_limit)
            # a1, _ = torch.max(outer, dim=2)
            # a2, _ = torch.max(outer, dim=1)
            # ymin = torch.argmax(a1, dim=1)
            # ymax = torch.argmax(a2, dim=1)

            answer_dict_, sub_dict_ = convert_tokens(eval_file, ids.tolist(), starts.tolist(), ends.tolist())
            answer_dict.update(answer_dict_)
            sub_dict.update(sub_dict_)
            print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
            if((i+1) == num_batches):
                break
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log_large/answers.json", "w")
    json.dump(answer_dict, f)
    f.close()
    metrics["loss"] = loss
    print("EVAL loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))
    # sub_path = join(config.save_dir, config.split + '_' + config.sub_file)
    # log.info(f'Writing submission file to {sub_path}...')
    if config.mode =="test":
        sub_path = os.path.join(config.save_dir, config.split + '_' + config.sub_file)
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(sub_dict):
                csv_writer.writerow([uuid, sub_dict[uuid]])

    if config.mode == "train":

        print('------')
        print("finish:  {}".format(test_i))
        writer.add_scalar('data/test_loss', loss, test_i)
        writer.add_scalar('data/F1', metrics["f1"], test_i)
        writer.add_scalar('data/EM', metrics["exact_match"], test_i)
    return metrics

def train_entry(config):
    from models import QANet

    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = get_loader(config.train_record_file, config.batch_size)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)

    lr = config.learning_rate
    base_lr = 1
    lr_warm_up_num = config.lr_warm_up_num

    model = QANet(word_mat, char_mat).to(device)
    if torch.cuda.device_count() > 1:
      print('i can use gpu')
      model = torch.nn.DataParallel(model,device_ids=[0,1])
    model.load_state_dict(torch.load('/home/cn/AI/QANet-pytorch-/model_state_dict.pt'))
    ema = EMA(config.decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(lr=base_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=5e-8, params=parameters)
    cr = lr / math.log2(lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < lr_warm_up_num else lr)
    best_f1 = 0
    best_em = 0
    patience = 0
    unused = False
    for iter in range(config.num_epoch):
        train(model, optimizer, scheduler, train_dataset, dev_dataset, dev_eval_file, iter, ema)
        print(iter)
        ema.assign(model)
        metrics = test(model, dev_dataset, dev_eval_file, (iter+1)*len(train_dataset))
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)

        fn = os.path.join(config.save_dir, "model.pt")
        torch.save(model, fn)
        torch.save(model.state_dict(),'model_state_dict.pt')
        ema.resume(model)


def test_entry(config):
    # with open(config.dev_eval_file, "r") as fh:
    #     dev_eval_file = json.load(fh)
    # dev_dataset = get_loader(config.dev_record_file, config.batch_size)
    # fn = os.path.join(config.save_dir, "model.pt")
    from models import QANet
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)
    fn = os.path.join(config.save_dir, "model.pt")
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # import pdb
    model = QANet(word_mat, char_mat).cuda()
    # # pdb.set_trace()
    # # torch.save(model.state_dict(),'model_ph.pkl')
    # model.load_state_dict(torch.load('/disc1/nuo.chen/QANet-pytorch-/model_ph.pkl'))
    # for k,v in model.named_parameters():
    #     print(k,v.view(-1)[0])
    # pdb.set_trace()
    # # model = torch.load(fn).to(device)
    # if torch.cuda.device_count() > 1:
    #     print('i can use gpu')
    #     model = torch.nn.DataParallel(model,device_ids=[0,1,2,3])
    # if isinstance(model,torch.nn.DataParallel):
    #     model=model.module
    # torch.save(model.state_dict(),'model_ph.pkl')
    # pdb.set_trace()
    model=torch.load(fn)
    test(model, dev_dataset, dev_eval_file, 0)
   


def main(_):
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.batch_size = 2
        config.num_steps = 32
        config.val_num_batches = 2
        config.checkpoint = 2
        config.period = 1
        train_entry(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    app.run(main)
