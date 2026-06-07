import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import *
from utils import _precision, _recall, _f1_score

from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, average_precision_score,accuracy_score

from pprint import pprint

from data_process import *
from metric import get_evaluation_metrics

import csv
import wandb
num_ent = {'drugbank': 1710, 'twosides': 645, 'HetioNet': 34124}
num_rel = {'drugbank': 86, 'twosides': 209} # 209, 309, 188

# import warnings
# warnings.filterwarnings('always')

class Trainer():
    def __init__(self, args):
        super(Trainer, self).__init__()

        self.args = args

        ### things need to be recorded in the record name: dataset, model, setting, time
        self.file_name = self.args.dataset + '_' + self.args.model + '_'  + str(self.args.gpu) + '_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.txt'
        checkpoint_dir = args.paths['checkpoint_dir']
        self.save_path = os.path.join(checkpoint_dir, self.args.dataset + '_' + self.args.model + '_' + '_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        pprint(vars(self.args))

        record_dir = args.paths['record_dir']
        with open(os.path.join(record_dir, self.file_name), 'w') as f:
            f.write(str(vars(self.args)) + '\n')
            # f.close()
        
        self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
        args.device = self.device

        self.data_record = Data_record(args)

        self.model = add_model(args, self.data_record, self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.patience = args.patience
        self.label_mappings = args.label_mappings

    def run(self):
        self.model.train()
        self.valid_split = [j for j in self.data_record.split_not_train if 'valid' in j]
        self.test_split = [j for j in self.data_record.split_not_train if 'test' in j]
        self.best_val_f1 = {j:0. for j in self.valid_split}
        self.no_update_epoch = {j:0 for j in self.valid_split}

        for epoch in range(self.args.epoch):
            train_loss  = self.run_epoch(epoch)

            print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + ' [Epoch {}]: Training Loss: {:.5}'.format(epoch, train_loss))
            with open(os.path.join('record', self.file_name), 'a+') as f:
                f.write(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + ' [Epoch {}]: Training Loss: {:.5}\n'.format(epoch, train_loss))

            if epoch % self.args.eval_skip == 0:
                val_results = self.evaluate('valid', epoch)

                break_flag = self.update_result(val_results)
                if break_flag:
                    print("Early Stopping!!")
                    break

        print('Loading best model, Evaluating on Test data')
        test_results = self.evaluate('test', epoch)
        # return test_results['accuracy']
        return 

    def run_epoch(self, epoch):
        self.model.train()
        losses = []

        train_iter = iter(self.data_record.data_iter['train'])
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            split = 'train'
            data, label = read_batch(batch, split, self.device, self.args, self.data_record) 

            pred = self.model.forward(data)
            loss = self.model.loss(pred, label)

            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            if step % 100 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' [E:{}| {}]: Train Loss:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.args.name))

        loss = np.mean(losses)
        # Log training loss to wandb
        wandb.log({"train/loss": loss, "epoch": epoch})
        return loss

    def evaluate(self, split, epoch):
        results = {}
        result_record = []
        split_this = self.valid_split if split == 'valid' else self.test_split
        for j in split_this:
            if 'test' in j:
                self.load_model(self.save_path)
            valid_results, valid_record = self.predict(j, epoch)
            result_record.append(valid_record)
            results[j] = valid_results
        for j in result_record:
            print(j)
            with open(os.path.join('record', self.file_name), 'a+') as f:
                f.write(j)
        return results

    def predict(self, split, epoch):
        self.model.eval()
        with torch.no_grad():
            train_iter = iter(self.data_record.data_iter[split])

            label_list = []
            pred_list = []
            output_list = []

            for step, batch in enumerate(train_iter):
                data, label	= read_batch(batch, split, self.device, self.args, self.data_record) 
                pred = self.model.forward(data)
                
                if self.args.eval_skip:
                    pred = pred[:,:num_rel[self.args.dataset]]
                
                output_list.append(pred.cpu().numpy())
                pred_list.append(pred.argmax(1).cpu().numpy())
                label_list.append(label.argmax(1).cpu().numpy())
            
            if self.args.dataset == 'drugbank':
                pred_final = np.concatenate(pred_list)
                label_final = np.concatenate(label_list)
                accuracy = np.sum(pred_final == label_final) / len(pred_final)
                f1 = f1_score(label_final, pred_final, average='macro')
                kappa = cohen_kappa_score(label_final, pred_final)

                results['accuracy'] = accuracy
                results['f1'] = f1
                results['kappa'] = kappa
                str_record = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' {} [Epoch {} {}]: F1-score : {:.5}, Accuracy : {:.5}, Kappa : {:.5}\n'.format(split ,epoch, split, results['f1'], results['accuracy'], results['kappa'])
                wandb.log({
                    f"{split}/f1": results['f1'],
                    f"{split}/accuracy": results['accuracy'],
                    f"{split}/kappa": results['kappa'],
                    "epoch": epoch
                })
            elif self.args.dataset == 'twosides':
                pred_final = np.concatenate(pred_list)
                label_final = np.concatenate(label_list)
                pred_cun = []
                label_cun = []
                for j in range(pred_final.shape[1]):
                    where_is = np.where(label_final[:,j]==1)[0]
                    pred_cun.append(pred_final[where_is,j])
                    label_cun.append(label_final[where_is,j]*label_final[where_is,-1])
                roc_auc = [ roc_auc_score(label_cun[l], pred_cun[l]) if label_cun[l].shape[0] > 0 else 0 for l in range(pred_final.shape[1])]
                prc_auc = [ average_precision_score(label_cun[l], pred_cun[l]) if label_cun[l].shape[0] > 0 else 0 for l in range(pred_final.shape[1])]
                ap = [accuracy_score(label_cun[l], (pred_cun[l] > 0.5).astype('float')) if label_cun[l].shape[0] > 0 else 0 for l in range(pred_final.shape[1])]
                
                # Calculate AP@K
                apk_list = []
                for j in range(pred_final.shape[1]):
                    where_is = np.where(label_final[:,j]==1)[0]
                    if len(where_is) > 0:
                        score = pred_cun[j]
                        label = label_cun[j]
                        sort_label = np.array(sorted(zip(score, label), reverse=True))
                        k = int(len(label)//2)
                        apk = np.sum(sort_label[:k,1])
                        apk_list.append(apk/k if k > 0 else 0)
                    else:
                        apk_list.append(0)

                results['PR-AUC'] = np.array(prc_auc).mean()
                results['AUC-ROC'] = np.array(roc_auc).mean()
                results['accuracy'] = np.array(ap).mean()
                results['AP@K'] = np.array(apk_list).mean()
                str_record = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' {} [Epoch {} {}]: PR-AUC : {:.5},  AUC-ROC: {:.5}, Accuracy : {:.5}, AP@K : {:.5}\n'.format(split ,epoch, split, results['PR-AUC'], results['AUC-ROC'], results['accuracy'], results['AP@K'])
                wandb.log({
                    f"{split}/pr_auc": results['PR-AUC'],
                    f"{split}/auc_roc": results['AUC-ROC'],
                    f"{split}/accuracy": results['accuracy'],
                    f"{split}/apk": results['AP@K'],
                    "epoch": epoch
                })
        return results, str_record

    def update_result(self, results):
        for j in results:
            if results[j]['f1'] > self.best_val_f1[j]:
                self.best_val_f1[j] = results[j]['f1']
                self.no_update_epoch[j] = 0
                self.save_model(self.save_path)
            else:
                self.no_update_epoch[j] += 1
        for j in self.no_update_epoch:
            if self.no_update_epoch[j] <= self.args.patience:
                return 0
        return 1

    def save_model(self, save_path):
        state = {
			'state_dict'	: self.model.state_dict(),
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.args)
		}
        torch.save(state, save_path)

    def load_model(self, load_path):
        # print(torch.cuda.device_count())
        state = torch.load(load_path, map_location='cpu', weights_only=False)
        state_dict		= state['state_dict']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

