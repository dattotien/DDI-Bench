import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import *

from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, average_precision_score,accuracy_score

from pprint import pprint

from data_process import *

import csv
import wandb

from dataset_registry import is_multiclass, is_multilabel
from metric import get_evaluation_metrics

# import warnings
# warnings.filterwarnings('always')

class Trainer():
    def __init__(self, args):
        super(Trainer, self).__init__()

        self.args = args

        ### things need to be recorded in the record name: dataset, model, setting, time
        self.file_name = self.args.dataset + '_' + self.args.model + '_'  + str(self.args.gpu) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime()) + '.txt'
        ### output dirs are dataset-independent, so they stay in config.yaml's paths section
        checkpoint_dir = args.paths.get('checkpoint_dir', './checkpoints')
        self.record_dir = args.paths.get('record_dir', './record')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)
        self.save_path = os.path.join(checkpoint_dir, self.args.dataset + '_' + self.args.model + '_' + '_' + time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime()))

        pprint(vars(self.args))

        with open(os.path.join(self.record_dir, self.file_name), 'w') as f:
            f.write(str(vars(self.args)) + '\n')
            # f.close()
        
        self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
        args.device = self.device

        self.data_record = Data_record(args)

        ### MUDI-style datasets store each pair in both directions and are scored by
        ### metric.py; everything else keeps the accuracy / macro-F1 / kappa report
        self.directed_eval = bool(args.dataset_cfg.get('directed_eval'))
        self.eval_options = args.dataset_cfg.get('eval_options') or [1]
        if self.directed_eval and not args.dataset_cfg.get('label_mapping'):
            raise ValueError(
                "dataset '{}' sets directed_eval but has no label_mapping in "
                "dataset_registry.py".format(args.dataset))

        if is_multilabel(self.args):
            occur = (np.array([j[2] for j in self.data_record.triplets['train']]).sum(0))[:-1]
            args.loss_weight = occur.min()/occur

        self.model = add_model(args, self.data_record, self.device) ###
        if self.args.adversarial:
            if is_multiclass(self.args):
                self.random_layer = RandomLayer([self.model.cdan_dim, self.data_record.num_rel], 500).to(self.device)
            else:
                self.random_layer = RandomLayer([self.model.cdan_dim, 2], 500).to(self.device)
            self.random_layer.device(self.device)
            self.ad_net = AdversarialNetwork(500, 500).to(self.device)
            self.optimizer_ad = optim.AdamW(self.ad_net.parameters(), lr=args.lr, weight_decay=args.weight_decay) ###
            pass

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay) ###

        self.patience = args.patience

    def run(self):
        self.model.train()
        self.valid_split = [j for j in self.data_record.split_not_train if 'valid' in j]
        self.test_split = [j for j in self.data_record.split_not_train if 'test' in j]
        self.best_val_acc = {j:0. for j in self.valid_split}
        self.no_update_epoch = {j:0 for j in self.valid_split}

        for epoch in range(self.args.epoch):
            train_loss  = self.run_epoch(epoch)

            print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + ' [Epoch {}]: Training Loss: {:.5}'.format(epoch, train_loss))
            with open(os.path.join(self.record_dir, self.file_name), 'a+') as f:
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
        if self.args.adversarial:
            train_adv_iter = iter(self.data_record.data_iter['train_adv'])
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            if self.args.adversarial:
                self.optimizer_ad.zero_grad()

            split = 'train'
            data, label = read_batch(batch, split, self.device, self.args, self.data_record) 

            if self.args.adversarial and is_multiclass(self.args):
                data_adv, label_adv = read_batch(next(train_adv_iter), split, self.device, self.args, self.data_record)
                pred_adv, final_layer_adv = self.model.forward(data_adv)
                pred, final_layer = self.model.forward(data)
                loss_label = self.model.loss(pred, label) ### label loss for source domain prediction
                softmax_pred = nn.Softmax(dim=1)(pred)
                softmax_pred_adv = nn.Softmax(dim=1)(pred_adv)
                pred_comb = torch.cat([softmax_pred, softmax_pred_adv], 0) ### whether need softmax
                final_layer_comb = torch.cat([final_layer, final_layer_adv], 0)
                loss = CDAN([final_layer_comb, pred_comb], self.ad_net, self.device, None, None, self.random_layer) * 0.01 + loss_label # 0.01 = adversarial weight
            elif self.args.adversarial and is_multilabel(self.args):
                data_adv, label_adv = read_batch(next(train_adv_iter), split, self.device, self.args, self.data_record)
                pred_adv, final_layer_adv = self.model.forward(data_adv)
                pred, final_layer = self.model.forward(data)
                loss_label = self.model.loss(pred, label) ### label loss for source domain prediction
                pred_out = torch.flatten(torch.cat([1 - torch.sigmoid(pred).unsqueeze(2),torch.sigmoid(pred).unsqueeze(2)], dim=2).permute(1,0,2),start_dim=0,end_dim = 1) ### consider to repeat
                pred_out_adv = torch.flatten(torch.cat([1 - torch.sigmoid(pred_adv).unsqueeze(2),torch.sigmoid(pred_adv).unsqueeze(2)], dim=2).permute(1,0,2),start_dim=0,end_dim = 1)
                pred_comb = torch.cat([pred_out, pred_out_adv], 0) ### whether need softmax
                final_layer_comb = torch.cat([final_layer.repeat(209,1), final_layer_adv.repeat(209,1)], 0)
                loss = CDAN([final_layer_comb, pred_comb], self.ad_net, self.device, None, None, self.random_layer) * 0.01 + loss_label # 0.01 = adversarial weight
            else:
                pred = self.model.forward(data)
                loss = self.model.loss(pred, label)

            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
            if self.args.adversarial:
                self.optimizer_ad.step()

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
        if split == 'valid' and self.args.model in ['CSMDDI']:
            self.model.pre_process()
        for j in split_this:
            if 'test' in j:
                self.load_model(self.save_path + j[-3:])
            valid_results, valid_record = self.predict(j, epoch)
            result_record.append(valid_record)
            results[j] = valid_results
        for j in result_record:
            print(j)
            with open(os.path.join(self.record_dir, self.file_name), 'a+') as f:
                f.write(j)
        return results

    def predict(self, split, epoch):
        self.model.eval()
        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_record.data_iter[split])

            label_list = []
            pred_list = []
            output_list = []

            for step, batch in enumerate(train_iter):
                data, label	= read_batch(batch, split, self.device, self.args, self.data_record) 
                if self.args.adversarial:
                    pred, _ = self.model.forward(data)
                else:
                    pred = self.model.forward(data)
                if self.args.eval_skip:
                    pred = pred[:,:self.data_record.num_rel]
                if is_multiclass(self.args):
                    logits = pred.cpu().numpy()
                    pred_list.append(logits.argmax(1))
                    label_list.append(label.argmax(1).cpu().numpy())
                    if self.directed_eval:
                        output_list.append(logits)
                elif is_multilabel(self.args):
                    pred = torch.sigmoid(pred).cpu().numpy()
                    label = label.cpu().numpy()
                    pred_list.append(pred)
                    label_list.append(label)
            
            if is_multiclass(self.args) and self.directed_eval:
                results, str_record = self.directed_metrics(
                    split, epoch, np.concatenate(label_list), np.concatenate(pred_list),
                    np.concatenate(output_list))
            elif is_multiclass(self.args):
                pred_final = np.concatenate(pred_list)
                label_final = np.concatenate(label_list)
                accuracy = np.sum(pred_final == label_final) / len(pred_final)
                f1 = f1_score(label_final, pred_final, average='macro')
                kappa = cohen_kappa_score(label_final, pred_final)

                results['accuracy'] = accuracy
                results['f1'] = f1
                results['kappa'] = kappa
                results['score'] = accuracy
                str_record = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' {} [Epoch {} {}]: F1-score : {:.5}, Accuracy : {:.5}, Kappa : {:.5}\n'.format(split ,epoch, split, results['f1'], results['accuracy'], results['kappa'])
                wandb.log({
                    f"{split}/f1": results['f1'],
                    f"{split}/accuracy": results['accuracy'],
                    f"{split}/kappa": results['kappa'],
                    "epoch": epoch
                })
            elif is_multilabel(self.args):
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
                results['score'] = results['accuracy']
                str_record = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' {} [Epoch {} {}]: PR-AUC : {:.5},  AUC-ROC: {:.5}, Accuracy : {:.5}, AP@K : {:.5}\n'.format(split ,epoch, split, results['PR-AUC'], results['AUC-ROC'], results['accuracy'], results['AP@K'])
                wandb.log({
                    f"{split}/pr_auc": results['PR-AUC'],
                    f"{split}/auc_roc": results['AUC-ROC'],
                    f"{split}/accuracy": results['accuracy'],
                    f"{split}/apk": results['AP@K'],
                    "epoch": epoch
                })
        return results, str_record

    def directed_metrics(self, split, epoch, label_final, pred_final, output_final):
        """Metrics for datasets whose eval files are laid out as [forward | inverse].

        `metric.py` compares row i against row i + N/2, so the loader order has to
        survive untouched: no shuffling and no dropped partial batch.
        """
        if len(label_final) % 2:
            raise ValueError(
                "split '{}' has {} rows; a directed_eval dataset needs an even number so "
                "that row i and row i+N/2 are the two directions of one pair.".format(
                    split, len(label_final)))
        detail = get_evaluation_metrics(
            label_final.tolist(), pred_final.tolist(), output_final,
            label_mapping=self.args.dataset_cfg['label_mapping'],
            options=self.eval_options, is_test=('test' in split),
            epoch=epoch, prefix=split)

        results = {}
        for option in self.eval_options:
            for key in ('macro', 'micro', 'micro_precision', 'micro_recall',
                        'f1_no_interaction'):
                results['opt{}_{}'.format(option, key)] = detail[option][key]
        main = self.eval_options[0]
        results['score'] = detail[main]['macro']
        results['accuracy'] = detail[main]['micro'] ### keep a familiar key around
        str_record = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
            ' {} [Epoch {}]: '.format(split, epoch) + ', '.join(
                'opt{} macro-F1 {:.5} micro-F1 {:.5}'.format(
                    o, detail[o]['macro'], detail[o]['micro']) for o in self.eval_options) + '\n'
        return results, str_record

    def update_result(self, results):
        for j in results:
            if results[j]['score'] > self.best_val_acc[j]:
                self.best_val_acc[j] = results[j]['accuracy']
                self.no_update_epoch[j] = 0
                self.save_model(self.save_path + j[-3:])
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

