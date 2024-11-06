import numpy as np
import nni
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.metric import *

class ClassificationTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(ClassificationTrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step', int(np.sqrt(data_loader.batch_size)))

        if (self.config['model']['args']['num_classes'] == 1):
            weight_0 = self.config['trainer'].get('class_weight_0', 1.0)
            weight_1 = self.config['trainer'].get('class_weight_1', 1.0)
            self.weight = [weight_0, weight_1]
            self.loss = lambda output, target: loss(output, target, self.weight)
        self.prauc_flag = pr_auc in self.metrics and roc_auc in self.metrics

                # Print model summary
        print("Here")
        print(self.model)

        # Explore model attributes
        # Example: Print 'features' attribute if available
        if hasattr(self.model, 'features'):
            print(self.model.features)



    def _eval_metrics(self, output, target, **kwargs):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, **kwargs)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        all_t = []
        all_o = []
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
        for batch_idx, (data, target) in enumerate(self.data_loader):

            all_t.append(target.numpy())
            target = target.to(self.device)
            self.optimizer.zero_grad()
            # # Iterate over each tensor in the data tuple
            # for tensor_idx, tensor in enumerate(data):
            #     # Print the dimensions of the tensor
            #     print(f"Dimensions of tensor {tensor_idx + 1}: {tensor.size()}")

            # Calculate accuracy, FPR, and FNR for male and female predictions
            
            gender_labels = data[5][:, 1]
            # old_demographic_tensor = data[5]
            # Remove the second element from the 91-dimensional vectors in the demographic tensor using slicing
            # new_demographic_tensor = old_demographic_tensor[:, [i for i in range(91) if i != 1]]
            # print("new demo tensor:", new_demographic_tensor.shape)
            # data = (*data[:5], new_demographic_tensor)
            # data = (*data[:4], demographic_tensor)
            # print("Shape of demographic tensor:", data[5].shape)
            # for tensor_idx, tensor in enumerate(data):
            #     # Print the dimensions of the tensor
            #     print(f"After change Dimensions of tensor {tensor_idx + 1}: {tensor.size()}")
            
            output, logits = self.model(data, device=self.device)
            predictions = output
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            print(gender_labels)
            total_male += torch.sum(gender_labels == 0).item()
            total_female += torch.sum(gender_labels == 1).item()
            # Calculate percentage of 1s in target labels for male and female separately
            male_mask = data[5][:, 1] == 0  # Assuming gender label index is 1
            female_mask = data[5][:, 1] == 1
            male_ones_percentage += torch.sum(target[male_mask] == 1).item() / max(torch.sum(male_mask).item(), 1)
            female_ones_percentage += torch.sum(target[female_mask] == 1).item() / max(torch.sum(female_mask).item(), 1)
            male_samples += torch.sum(male_mask).item()
            female_samples += torch.sum(female_mask).item()
            total_samples += 1
            
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


            total_loss += loss
            total_metrics += self._eval_metrics(output, target)
            all_o.append(output.detach().cpu().numpy())
        

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    'loss', loss))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if total_samples > 0:
            total_ones_percentage /= total_samples
        # Check if divisor is zero, if not, calculate the average, otherwise set the average to zero
        male_accuracy_avg = male_acc_total / male_acc_total_runs if male_acc_total_runs != 0 else 0
        female_accuracy_avg = female_acc_total / female_acc_total_runs if female_acc_total_runs != 0 else 0
        male_FPR_avg = male_FPR_total / male_FPR_total_runs if male_FPR_total_runs != 0 else 0
        female_FPR_avg = female_FPR_total / female_FPR_total_runs if female_FPR_total_runs != 0 else 0
        male_FNR_avg = male_FNR_total / male_FNR_total_runs if male_FNR_total_runs != 0 else 0
        female_FNR_avg = female_FNR_total / female_FNR_total_runs if female_FNR_total_runs != 0 else 0
        male_TPR_avg = male_TPR_total / male_TPR_total_runs if male_TPR_total_runs != 0 else 0
        female_TPR_avg = female_TPR_total / female_TPR_total_runs if female_TPR_total_runs != 0 else 0
        male_TNR_avg = male_TNR_total / male_TNR_total_runs if male_TNR_total_runs != 0 else 0
        female_TNR_avg = female_TNR_total / female_TNR_total_runs if female_TNR_total_runs != 0 else 0

        male_PPV_avg = male_PPV_total / male_PPV_total_runs if male_PPV_total_runs != 0 else 0
        female_PPV_avg = female_PPV_total / female_PPV_total_runs if female_PPV_total_runs != 0 else 0
        male_NPV_avg = male_NPV_total / male_NPV_total_runs if male_NPV_total_runs != 0 else 0
        female_NPV_avg = female_NPV_total / female_NPV_total_runs if female_NPV_total_runs != 0 else 0

            # Average the percentages over all batches
        male_ones_percentage /= len(self.data_loader)
        female_ones_percentage /= len(self.data_loader)

        # male_TPR_avg = male_TPR_total / male_TPR_total_runs
        # female_TPR_avg = female_TPR_total / female_TPR_total_runs
        # male_TNR_avg = male_TNR_total / male_TNR_total_runs
        # female_TNR_avg = female_TNR_total / female_TNR_total_runs

        total_metrics = total_metrics / len(self.data_loader)
        if (self.prauc_flag):
            all_o = np.hstack(all_o)
            all_t = np.hstack(all_t)
            total_metrics[-2] = pr_auc_1(all_o, all_t)
            total_metrics[-1] = roc_auc_1(all_o, all_t)

        log = {
            'loss': total_loss / len(self.data_loader),
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
            'metrics': total_metrics,
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log



    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        all_t = []
        all_o = []
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
        total_ones_percentage = 0.0
        total_samples = 0
        male_ones_percentage = 0.0
        female_ones_percentage = 0.0
        male_samples = 0
        female_samples = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                all_t.append(target.numpy())
                target = target.to(self.device)
                # for tensor_idx, tensor in enumerate(data):
                #     print(f"Dimensions of tensor (VALID) {tensor_idx + 1}: {tensor.size()}")


                output, logits = self.model(data, self.device)
                loss = self.loss(output, target.reshape(-1,))
                all_o.append(output.detach().cpu().numpy())

                # Calculate accuracy, FPR, and FNR for male and female predictions
                predictions = output
                gender_labels = data[5][:, 2]
                total_male += torch.sum(gender_labels == 0).item()
                total_female += torch.sum(gender_labels == 1).item()
                # Calculate percentage of 1s in target labels for male and female separately
                male_mask = data[5][:, 1] == 0  # Assuming gender label index is 1
                female_mask = data[5][:, 1] == 1
                male_ones_percentage += torch.sum(target[male_mask] == 1).item() / max(torch.sum(male_mask).item(), 1)
                female_ones_percentage += torch.sum(target[female_mask] == 1).item() / max(torch.sum(female_mask).item(), 1)
                male_samples += torch.sum(male_mask).item()
                female_samples += torch.sum(female_mask).item()
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
        

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))


        # Check if divisor is zero, if not, calculate the average, otherwise set the average to zero
        male_accuracy_avg = male_acc_total / male_acc_total_runs if male_acc_total_runs != 0 else 0
        female_accuracy_avg = female_acc_total / female_acc_total_runs if female_acc_total_runs != 0 else 0
        male_FPR_avg = male_FPR_total / male_FPR_total_runs if male_FPR_total_runs != 0 else 0
        female_FPR_avg = female_FPR_total / female_FPR_total_runs if female_FPR_total_runs != 0 else 0
        male_FNR_avg = male_FNR_total / male_FNR_total_runs if male_FNR_total_runs != 0 else 0
        female_FNR_avg = female_FNR_total / female_FNR_total_runs if female_FNR_total_runs != 0 else 0
        male_TPR_avg = male_TPR_total / male_TPR_total_runs if male_TPR_total_runs != 0 else 0
        female_TPR_avg = female_TPR_total / female_TPR_total_runs if female_TPR_total_runs != 0 else 0
        male_TNR_avg = male_TNR_total / male_TNR_total_runs if male_TNR_total_runs != 0 else 0
        female_TNR_avg = female_TNR_total / female_TNR_total_runs if female_TNR_total_runs != 0 else 0

        male_PPV_avg = male_PPV_total / male_PPV_total_runs if male_PPV_total_runs != 0 else 0
        female_PPV_avg = female_PPV_total / female_PPV_total_runs if female_PPV_total_runs != 0 else 0
        male_NPV_avg = male_NPV_total / male_NPV_total_runs if male_NPV_total_runs != 0 else 0
        female_NPV_avg = female_NPV_total / female_NPV_total_runs if female_NPV_total_runs != 0 else 0

        if total_samples > 0:
            total_ones_percentage /= total_samples  

        total_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        if self.prauc_flag:
            all_o = np.hstack(all_o)
            all_t = np.hstack(all_t)
            total_val_metrics[-2] = pr_auc_1(all_o, all_t)
            total_val_metrics[-1] = roc_auc_1(all_o, all_t)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_male_accuracy_avg': male_accuracy_avg,
            'val_female_accuracy_avg': female_accuracy_avg,
            'val_male_FPR_avg': male_FPR_avg,
            'val_female_FPR_avg': female_FPR_avg,
            'val_male_FNR_avg': male_FNR_avg,
            'val_female_FNR_avg': female_FNR_avg,
            'val_male_TPR_avg': male_TPR_avg,
            'val_female_TPR_avg': female_TPR_avg,
            'val_male_TNR_avg': male_TNR_avg,
            'val_female_TNR_avg': female_TNR_avg,
            'val_male_total': total_male,
            'val_female_total': total_female,
            'val_male_PPV_avg': male_PPV_avg,
            'val_female_PPV_avg': female_PPV_avg,
            'val_male_NPV_avg': male_NPV_avg,
            'val_female_NPV_avg': female_NPV_avg,
            'val_male_ones_percentage': male_ones_percentage,
            'val_female_ones_percentage': female_ones_percentage,
            'val_metrics': total_val_metrics

        }