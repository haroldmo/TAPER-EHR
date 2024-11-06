import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.metric import *
from train import get_instance
from train import import_module

import pickle

def main(config, resume):
    # setup data_loader instances
    print("HERE")
    data_loader = get_instance(module_data, 'data_loader', config)
    total_data_points = 0
    for batch in data_loader:
        # print('batch: ', batch)s
        batch_size = len(batch[0])  # Assuming batch[0] contains the data
        total_data_points += batch_size

    print("Total data points before:", total_data_points)

    #data_loader = getattr(module_data, config['data_loader']['type'])(
    #    config['data_loader']['args']
    #    batch_size=512,
    #    shuffle=False,
    #    validation_split=0.0,
    #    training=False,
    #    num_workers=2
    #)

    # data_loader = data_loader.split_validation()

    # build model architecture

    model = import_module('model', config)(**config['model']['args'])
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    predictions = {"output": [], "target": []}

    male_acc_total = 0
    male_acc_total_runs = 0
    female_acc_total = 0
    female_acc_total_runs = 0
    male_FPR_total = 0
    male_FPR_total_runs = 0
    female_FPR_total = 0
    female_FPR_total_runs = 0
    male_FNR_total = 0
    male_FNR_total_runs = 0
    female_FNR_total = 0
    female_FNR_total_runs = 0
    male_TPR_total = 0
    male_TPR_total_runs = 0
    female_TPR_total = 0
    female_TPR_total_runs = 0
    male_TNR_total = 0
    male_TNR_total_runs = 0
    female_TNR_total = 0
    female_TNR_total_runs = 0
    male_PPV_total = 0
    male_PPV_total_runs = 0
    female_PPV_total = 0
    female_PPV_total_runs = 0
    male_NPV_total = 0
    male_NPV_total_runs = 0
    female_NPV_total = 0
    female_NPV_total_runs = 0
    total_male = 0
    total_female = 0
    male_PPV_total = 0
    total_ones_percentage = 0.0
    total_samples = 0
    male_ones_percentage = 0.0
    female_ones_percentage = 0.0
    male_samples = 0
    female_samples = 0
    all_t, all_o = [], []

    data_loader = get_instance(module_data, 'data_loader', config)
    total_data_points = 0
    for batch in data_loader:
        # print('batch: ', batch)s
        batch_size = len(batch[0])  # Assuming batch[0] contains the data
        total_data_points += batch_size

    print("Total data points after:", total_data_points)

    total_data_points = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            total_data_points += len(data[5])
            #data, target = data.to(device), target.to(device)
            target = target.to(device)
            output = model(data, device)
            #
            # save sample images, or do something with output here
            #

            all_t.append(target.cpu().numpy())


            output, logits = output
            predictions['output'].append(output.cpu().numpy())
            predictions['target'].append(target.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size

            # print('demo: ', data[5])
            # print('gender ', data[5][:,1])
            # print('target ', target)
            gender_labels = data[5][:, 1]

            male_mask = data[5][:, 1] == 0
            female_mask = data[5][:, 1] == 1

            # print('male mask', male_mask)
            # print('female mask', female_mask)

            male_ones_percentage += torch.sum(target[male_mask] == 1).item() / max(torch.sum(male_mask).item(), 1)
            female_ones_percentage += torch.sum(target[female_mask] == 1).item() / max(torch.sum(female_mask).item(), 1)
            male_samples += torch.sum(male_mask).item()
            female_samples += torch.sum(female_mask).item()
            total_samples += 1
            total_male += torch.sum(gender_labels == 0).item()
            total_female += torch.sum(gender_labels == 1).item()
            # print('output', output)
            

            male_accuracy, female_accuracy = accuracy_gender(output, target, gender_labels)
            male_FPR, female_FPR = FPR_gender(output, target, gender_labels)
            male_FNR, female_FNR = FNR_gender(output, target, gender_labels)
            male_TPR, female_TPR = TPR_gender(output, target, gender_labels)
            male_TNR, female_TNR = TNR_gender(output, target, gender_labels)
            male_PPV, female_PPV = PPV_gender(output, target, gender_labels)
            male_NPV, female_NPV = NPV_gender(output, target, gender_labels)

            if male_accuracy is not None:
                male_acc_total += male_accuracy
                male_acc_total_runs += 1
            if female_accuracy is not None:
                female_acc_total += female_accuracy
                female_acc_total_runs += 1
            if male_FPR is not None:
                male_FPR_total += male_FPR
                male_FPR_total_runs += 1
            if female_FPR is not None:
                female_FPR_total += female_FPR
                female_FPR_total_runs += 1
            if male_FNR is not None:
                male_FNR_total += male_FNR
                male_FNR_total_runs += 1
            if female_FNR is not None:
                female_FNR_total += female_FNR
                female_FNR_total_runs += 1
            if male_PPV is not None:
                male_PPV_total += male_PPV
                male_PPV_total_runs += 1
            if female_PPV is not None:
                female_PPV_total += female_PPV
                female_PPV_total_runs += 1
            if male_NPV is not None:
                male_NPV_total += male_NPV
                male_NPV_total_runs += 1
            if female_NPV is not None:
                female_NPV_total += female_NPV
                female_NPV_total_runs += 1

            if male_TPR is not None:
                male_TPR_total += male_TPR
                male_TPR_total_runs += 1
            if female_TPR is not None:
                female_TPR_total += female_TPR
                female_TPR_total_runs += 1
            if male_TNR is not None:
                male_TNR_total += male_TNR
                male_TNR_total_runs += 1
            if female_TNR is not None:
                female_TNR_total += female_TNR
                female_TNR_total_runs += 1

            all_o.append(output.detach().cpu().numpy())


            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    print('len total data: ', total_data_points)
    if total_samples > 0:
        total_ones_percentage /= total_samples
    male_accuracy_avg = male_acc_total / male_acc_total_runs
    female_accuracy_avg = female_acc_total / female_acc_total_runs
    male_FPR_avg = male_FPR_total / male_FPR_total_runs
    female_FPR_avg = female_FPR_total / female_FPR_total_runs
    male_FNR_avg = male_FNR_total / male_FNR_total_runs
    female_FNR_avg = female_FNR_total / female_FNR_total_runs
    male_PPV_avg = male_PPV_total / male_PPV_total_runs
    female_PPV_avg = female_PPV_total / female_PPV_total_runs
    male_NPV_avg = male_NPV_total / male_NPV_total_runs
    female_NPV_avg = female_NPV_total / female_NPV_total_runs

    # Average the percentages over all batches
    male_ones_percentage /= total_samples
    female_ones_percentage /= total_samples

    male_TPR_avg = male_TPR_total / male_TPR_total_runs
    female_TPR_avg = female_TPR_total / female_TPR_total_runs
    male_TNR_avg = male_TNR_total / male_TNR_total_runs
    female_TNR_avg = female_TNR_total / female_TNR_total_runs

    all_o = np.hstack(all_o)
    all_t = np.hstack(all_t)

    pr_auc = pr_auc_1(all_o, all_t)
    roc_auc = roc_auc_1(all_o, all_t)


    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    log.update({
        'male_accuracy_avg': male_accuracy_avg,
        'female_accuracy_avg': female_accuracy_avg,
        'male_FPR_avg': male_FPR_avg,
        'female_FPR_avg': female_FPR_avg,
        'male_FNR_avg': male_FNR_avg,
        'female_FNR_avg': female_FNR_avg,
        'male_TPR_avg': male_TPR_avg,
        'female_TPR_avg': female_TPR_avg,
        'male_TNR_avg': male_TNR_avg,
        'female_TNR_avg': female_TNR_avg,
        'male_total': total_male,
        'female_total': total_female,
        'male_PPV_avg': male_PPV_avg,
        'female_PPV_avg': female_PPV_avg,
        'male_NPV_avg': male_NPV_avg,
        'female_NPV_avg': female_NPV_avg,
        'male_ones_percentage': male_ones_percentage,
        'female_ones_percentage': female_ones_percentage,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    })
    print(log)
    save_dir = os.path.join(os.path.abspath(os.path.join(resume, '..', '..')))
    predictions['output'] = np.hstack(predictions['output'])
    predictions['target'] = np.hstack(predictions['target'])
    print(save_dir + '/predictions.pkl')
    with open(os.path.join(save_dir, 'predictions.pkl'), 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
        # print('HERE')
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    # print("HERE 2")

    num_runs = 10
    seeds = [97, 123, 456, 789, 321, 654, 987, 246, 135, 802]
    all_metrics = []
    for seed in seeds:
        config['data_loader']['args']['seed'] = seed
        print('balanced data or not: ', config['data_loader']['args']['balanced_data'])
        # config['data_loader']['args']['balanced_data'] = False
        print('balanced data or not: ', config['data_loader']['args']['balanced_data'])

        
        # print("NEW SEED:", config['data_loader']['args']['seed'])
        log = main(config, args.resume)
        # print("MAIN LOG: ", log)

        # Extract metrics from the log
        metrics = {key: value for key, value in log.items() if isinstance(value, (int, float))}
        # print("MAIN METRICS: ", metrics)
        all_metrics.append(metrics)

    # Calculate average and standard error for each metric
    avg_metrics = {}
    std_error_metrics = {}

    print("ALL METRICS: ", all_metrics)

    for key in all_metrics[0].keys():
        values = [metric[key] for metric in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_error_metrics[key] = np.std(values) / np.sqrt(len(values))

    # Print average and standard error for each metric on the same line
    for key in avg_metrics.keys():
        print(f"{key}: Avg = {avg_metrics[key]}, Std Err = {std_error_metrics[key]}")