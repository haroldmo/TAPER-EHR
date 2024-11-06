import torch

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, roc_curve, precision_recall_curve
def roc_auc(output,target):
    # temporary place holders..
    # these will be run at the end of the epoch once all probabilities are obtained..
    # refer to pr_auc_1

    #output = output.detach().cpu().numpy()
    #target = target.cpu().numpy()
    #if (torch.sum(target) == 0 or torch.sum(target) == len(target)):
    #    return 1.0

    #return roc_auc_score(target, output)
    return 1.0

def pr_auc(output, target):
    # temporary place holders.
    # these will be run at the end of the epoch once all probabilities are obtained..
    # refer to pr_auc_1


    #output = output.detach().cpu().numpy()
    #target = target.cpu().numpy()
    #if (torch.sum(target) == 0 or torch.sum(target) == len(target)):
    #    return 1.0
    #return average_precision_score(target, output)
    return 1.0

def roc_auc_1(output,target):
    # evaluate after eac
    fpr, tpr, thresholds = roc_curve(target, output)
    area = auc(fpr, tpr)
    return area #roc_auc_score(target, output)

def pr_auc_1(output, target):
    precision, recall, _ = precision_recall_curve(target, output)
    area = auc(recall, precision)
    return area #average_precision_score(target, output)

def accuracy(output, target):
    with torch.no_grad():
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy2(output, target, t=0):
    with torch.no_grad():
        if (len(target.shape) == 1):
            target = torch.unsqueeze(target, 1)
        if (len(output.shape) == 1):
            output = torch.unsqueeze(output, 1)

        pred = output >= 0.5#torch.argmax(output, dim=1)
        pred = pred.long()

        assert pred.shape[0] == len(target)

        correct = 0
        correct += torch.sum(pred == target).item()

    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def recall_k(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_10_diag(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, :232]
    target = target[:, :232]

    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_20_diag(output, target, mask, k=20, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, :232]
    target = target[:, :232]

    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_30_diag(output, target, mask, k=30, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, :232]
    target = target[:, :232]

    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_40_diag(output, target, mask, k=40, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, :232]
    target = target[:, :232]
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r


def recall_10_proc(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, 232:]
    target = target[:, 232:]

    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)

        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))

        if r != r:
            r = 0
    return r

def recall_20_proc(output, target, mask, k=20, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, 232:]
    target = target[:, 232:]

    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_30_proc(output, target, mask, k=30, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, 232:]
    target = target[:, 232:]

    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_40_proc(output, target, mask, k=40, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    output = output[:, 232:]
    target = target[:, 232:]
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_10(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_20(output, target, mask, k=20, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_30(output, target, mask, k=30, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_40(output, target, mask, k=40, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_50(output, target, mask, k=50, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r






def specificity(output, target, t=0.5):
    with torch.no_grad():
        preds = output > t#torch.argmax(output, dim=1)
        preds = preds.long()
        num_true_0s = torch.sum((target == 0) & (preds == target), dtype=torch.float).item()
        num_false_1s = torch.sum((target == 0) & (preds != target), dtype=torch.float).item()

    if (num_false_1s == 0):
        return 1
    s = num_true_0s / (num_true_0s + num_false_1s)

    if (s != s):
        s = 1

    return s

def sensitivity(output, target, t=0.5):
    with torch.no_grad():
        preds = output > t#torch.argmax(output, dim=1)
        preds = preds.long()
        num_true_1s = torch.sum((preds == target) & (preds == 1), dtype=torch.float)
        num_false_1s = torch.sum((preds != target) & (preds == 1), dtype=torch.float)

    s = num_true_1s / (num_true_1s + num_false_1s)

    if (s != s):
        s = 1
    return s

def precision(output, target, t=0.5):
    with torch.no_grad():
        preds = output > t
        preds = preds.long()
        num_true_1s = torch.sum((preds == target) & (preds == 1), dtype=torch.float)
        num_false_0s = torch.sum((preds != target) & (preds == 0), dtype=torch.float)

    s = num_true_1s / (num_true_1s + num_false_0s)
    if (s != s):
        s = 1
    return s


def accuracy_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()


        male_accuracy = (male_tp + male_tn) / (male_tp + male_tn + male_fp + male_fn) if male_tp + male_tn + male_fp + male_fn > 0 else None
        female_accuracy = (female_tp + female_tn) / (female_tp + female_tn + female_fp + female_fn) if female_tp + female_tn + female_fp + female_fn > 0 else None
    return male_accuracy, female_accuracy



def FPR_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()

        male_FPR = male_fp / (male_fp + male_tn) if male_fp + male_tn > 0 else None
        female_FPR = female_fp / (female_fp + female_tn) if female_fp + female_tn > 0 else None

    return male_FPR, female_FPR


def FNR_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()

        male_FNR = male_fn / (male_fn + male_tp) if male_fn + male_tp > 0 else None
        female_FNR = female_fn / (female_fn + female_tp) if female_fn + female_tp > 0 else None

    return male_FNR, female_FNR

def TPR_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()

        male_TPR = male_tp / (male_tp + male_fn) if male_tp + male_fn > 0 else None
        female_TPR = female_tp / (female_tp + female_fn) if female_tp + female_fn > 0 else None

    return male_TPR, female_TPR

def TNR_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()

        male_TNR = male_tn / (male_tn + male_fp) if male_tn + male_fp > 0 else None
        female_TNR = female_tn / (female_tn + female_fp) if female_tn + female_fp > 0 else None

    return male_TNR, female_TNR

def PPV_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()

        male_PPV = male_tp / (male_tp + male_fp) if male_tp + male_fp > 0 else None
        female_PPV = female_tp / (female_tp + female_fp) if female_tp + female_fp > 0 else None

    return male_PPV, female_PPV

def NPV_gender(output, target, gender_labels):
    with torch.no_grad():
        if len(target.shape) == 1:
            target = torch.unsqueeze(target, 1)
        if len(output.shape) == 1:
            output = torch.unsqueeze(output, 1)

        pred = output > 0.5
        pred = pred.long()

        male_indices = gender_labels == 0
        female_indices = gender_labels == 1

        male_fp = ((pred[male_indices] == 1) & (target[male_indices] == 0)).sum().item()
        female_fp = ((pred[female_indices] == 1) & (target[female_indices] == 0)).sum().item()

        male_tn = ((pred[male_indices] == 0) & (target[male_indices] == 0)).sum().item()
        female_tn = ((pred[female_indices] == 0) & (target[female_indices] == 0)).sum().item()

        male_fn = ((pred[male_indices] == 0) & (target[male_indices] == 1)).sum().item()
        female_fn = ((pred[female_indices] == 0) & (target[female_indices] == 1)).sum().item()

        male_tp = ((pred[male_indices] == 1) & (target[male_indices] == 1)).sum().item()
        female_tp = ((pred[female_indices] == 1) & (target[female_indices] == 1)).sum().item()

        male_NPV = male_tn / (male_tn + male_fn) if male_tn + male_fn > 0 else None
        female_NPV = female_tn / (female_tn + female_fn) if female_tn + female_fn > 0 else None

    return male_NPV, female_NPV