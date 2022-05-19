"""
Name: dev_process
Date: 2022/4/11 上午10:26
Version: 1.0
"""
import math

from model import ModelParam
import torch
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from util.compare_to_save import compare_to_save
import test_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf


def dev_process(opt, critertion, cl_model, dev_loader, test_loader=None, last_F1=None, last_Accuracy=None, train_log=None, log_summary_writer:SummaryWriter=None):
    y_true = []
    y_pre = []
    total_labels = 0
    dev_loss = 0

    orgin_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc='Dev Iteration')
        epoch_step_num = train_log['epoch'] * dev_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(dev_loader_tqdm):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels, \
            texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, _ = data
            # continue

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask)
            origin_res = cl_model(orgin_param)

            loss = critertion(origin_res, labels) / opt.acc_batch_size
            dev_loss += loss.item()
            _, predicted = torch.max(origin_res, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())

            dev_loader_tqdm.set_description("Dev Iteration, loss: %.6f" % loss)
            if log_summary_writer:
                log_summary_writer.add_scalar('dev_info/loss', loss.item(), global_step=step_num + epoch_step_num)
            step_num += 1

        dev_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        dev_accuracy = accuracy_score(y_true, y_pre)
        dev_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        dev_R_weighted = recall_score(y_true, y_pre, average='weighted')
        dev_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        dev_F1 = f1_score(y_true, y_pre, average='macro')
        dev_R = recall_score(y_true, y_pre, average='macro')
        dev_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Dev  : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (dev_accuracy, dev_F1_weighted, dev_precision_weighted, dev_R_weighted, dev_F1, dev_precision, dev_R, dev_loss)

        print(save_content)

        if log_summary_writer:
            log_summary_writer.add_scalar('dev_info/loss_epoch', dev_loss, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/acc', dev_accuracy, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/f1_w', dev_F1_weighted, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/r_w', dev_R_weighted, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/p_w', dev_precision_weighted, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/f1_ma', dev_F1, global_step=train_log['epoch'])
            log_summary_writer.flush()

        if last_F1 is not None:
            WriteFile(
                opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
            # 运行测试集
            test_process.test_process(opt, critertion, cl_model, test_loader, last_F1, log_summary_writer, train_log['epoch'])

            dev_log = {
                "dev_accuracy": dev_accuracy,
                "dev_F1": dev_F1,
                "dev_R": dev_R,
                "dev_precision": dev_precision,
                "dev_F1_weighted": dev_F1_weighted,
                "dev_precision_weighted": dev_precision_weighted,
                "dev_R_weighted": dev_R_weighted,
                "dev_loss": dev_loss
            }

            last_Accuracy, is_save_model, model_name = compare_to_save(last_Accuracy, dev_accuracy, opt, cl_model, train_log, dev_log, 'Acc', opt.save_acc, add_enter=False)
            if is_save_model is True:
                if opt.data_type == 'HFM':
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1, opt, cl_model, train_log, dev_log, 'F1-marco', opt.save_F1, 'F1-marco', model_name)
                else:
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1_weighted, opt, cl_model, train_log, dev_log, 'F1', opt.save_F1, 'F1', model_name)
            else:
                if opt.data_type == 'HFM':
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1, opt, cl_model, train_log, dev_log, 'F1-marco', opt.save_F1)
                else:
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1_weighted, opt, cl_model, train_log, dev_log, 'F1', opt.save_F1)

            return last_F1, last_Accuracy