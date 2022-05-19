"""
Name: test_process
Date: 2022/4/11 上午10:26
Version: 1.0
"""

from model import ModelParam
import torch
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np
from  torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
import math


def test_process(opt, critertion, cl_model, test_loader, last_F1=None, log_summary_writer: SummaryWriter=None, epoch=None):
    y_true = []
    y_pre = []
    total_labels = 0
    test_loss = 0

    orgin_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        test_loader_tqdm = tqdm(test_loader, desc='Test Iteration')
        epoch_step_num = epoch * test_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(test_loader_tqdm):
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
            test_loss += loss.item()
            _, predicted = torch.max(origin_res, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())

            test_loader_tqdm.set_description("Test Iteration, loss: %.6f" % loss)
            if log_summary_writer:
                log_summary_writer.add_scalar('test_info/loss', loss.item(), global_step=step_num + epoch_step_num)
            step_num += 1

        test_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        test_accuracy = accuracy_score(y_true, y_pre)
        test_F1 = f1_score(y_true, y_pre, average='macro')
        test_R = recall_score(y_true, y_pre, average='macro')
        test_precision = precision_score(y_true, y_pre, average='macro')
        test_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        test_R_weighted = recall_score(y_true, y_pre, average='weighted')
        test_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        save_content = 'Test : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
            (test_accuracy, test_F1_weighted, test_precision_weighted, test_R_weighted, test_F1, test_precision, test_R, test_loss)

        print(save_content)

        if log_summary_writer:
            log_summary_writer.add_scalar('test_info/loss_epoch', test_loss, global_step=epoch)
            log_summary_writer.add_scalar('test_info/acc', test_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_w', test_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/r_w', test_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/p_w', test_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_ma', test_F1, global_step=epoch)
            log_summary_writer.flush()

        if last_F1 is not None:
            WriteFile(
                opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
