"""
Name: model
Date: 2022/4/11 上午10:25
Version: 1.0
"""

import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import os
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
import matplotlib.pyplot as plt
from pre_model import RobertaEncoder
import copy


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token


def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class BertClassify(nn.Module):
    def __init__(self, opt, in_feature, dropout_rate=0.1):
        super(BertClassify, self).__init__()
        self.classify_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_feature, 3),
            ActivateFun(opt)
        )

    def forward(self, inputs):
        return self.classify_linear(inputs)


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = ''

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output


class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        if opt.image_model == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif opt.image_model == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif opt.image_model == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif opt.image_model == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif opt.image_model == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)
        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        self.output_dim = self.resnet_encoder[7][2].conv3.out_channels

        for param in self.resnet.parameters():
            if opt.fixed_image_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        image_encoder = self.resnet_encoder(images)
        # image_encoder = self.conv_output(image_encoder)
        image_cls = self.resnet_avgpool(image_encoder)
        image_cls = torch.flatten(image_cls, 1)
        return image_encoder, image_cls


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)

        self.text_config = copy.deepcopy(self.text_model.get_config())
        self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.image_encoder = RobertaEncoder(self.image_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_cls_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)

        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)
        image_encoder, image_cls = self.image_model(image_inputs)
        if self.image_output_type == 'all':
            image_encoder = image_encoder.contiguous().view(image_encoder.size(0), -1, image_encoder.size(1))
            image_encoder_init = self.image_change(image_encoder)
            image_cls_init = self.image_cls_change(image_cls)
            image_init = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)
        else:
            image_cls_init = self.image_cls_change(image_cls)
            image_init = image_cls_init.unsqueeze(1)

        image_mask = text_image_mask[:, -image_init.size(1):]
        extended_attention_mask = get_extended_attention_mask(image_mask, image_init.size())

        image_init = self.image_encoder(image_init,
                                             attention_mask=None,
                                             head_mask=None,
                                             encoder_hidden_states=None,
                                             encoder_attention_mask=extended_attention_mask,
                                             past_key_values=None,
                                             use_cache=self.use_cache,
                                             output_attentions=self.text_config.output_attentions,
                                             output_hidden_states=(self.text_config.output_hidden_states),
                                             return_dict=self.text_config.use_return_dict
                                             )
        image_init = image_init.last_hidden_state

        text_image_cat = torch.cat((text_init, image_init), dim=1)

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_inputs.size())
        text_image_transformer = self.text_image_encoder(text_image_cat,
                                                 attention_mask=extended_attention_mask,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=extended_attention_mask,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_image_transformer = text_image_transformer.last_hidden_state
        text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()

        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()

            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:text_image_output.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention(text_image_output)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

            text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(2)
            text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
        else:
            raise Exception('fuse_type设定错误')
        return text_image_output, None, None


class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        self.temperature = opt.temperature
        self.set_cuda = opt.cuda

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, data_orgin: ModelParam, data_augment: ModelParam = None, labels=None, target_labels=None):
        orgin_res, orgin_text_cls, orgin_image_cls = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                     data_orgin.images, data_orgin.text_image_mask)
        output = self.output_classify(orgin_res)

        if data_augment:
            augment_res, augment_text_cls, augment_image_cls = self.fuse_model(data_augment.texts, data_augment.bert_attention_mask,
                                                                               data_augment.images, data_augment.text_image_mask)
            orgin_res_change = self.orgin_linear_change(orgin_res)
            augment_res_change = self.augment_linear_change(augment_res)

            l_pos_neg = torch.einsum('nc,ck->nk', [orgin_res_change, augment_res_change.T])
            cl_lables = torch.arange(l_pos_neg.size(0))
            if self.set_cuda:
                cl_lables = cl_lables.cuda()
            l_pos_neg /= self.temperature

            l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_res_change, orgin_res_change.T])
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
            l_pos_neg_self = l_pos_neg_self.view(-1)

            cl_self_labels = target_labels[labels[0]]
            for index in range(1, orgin_res.size(0)):
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index*labels.size(0)), 0)

            l_pos_neg_self = l_pos_neg_self / self.temperature
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

            return output, l_pos_neg, cl_lables, cl_self_loss
        else:
            return output


class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = CLModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
