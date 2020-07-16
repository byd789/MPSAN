from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf

from src.model.model_template import ModelTemplate
from src.nn_utils.nn import linear
from src.nn_utils.integration_func import generate_embedding_mat
from src.nn_utils.baselines.interface import sentence_encoding_models


class ModelContextFusion(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, output_cls_num, scope):
        super(ModelContextFusion, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, output_cls_num, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        _logger.add()
        _logger.add('building %s neural network structure...' % cfg.network_type)
        tds, cds = self.tds, self.cds
        tl = self.tl
        tel, cel, cos, ocd, fh = self.tel, self.cel, self.cos, self.ocd, self.fh
        hn = self.hn
        bs = self.bs

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
                                                   scope='gene_token_emb_mat')
            emb = tf.nn.embedding_lookup(token_emb_mat, self.token_seq)  # bs,sl1,tel

        with tf.variable_scope('sent_encoding'):
            rep = sentence_encoding_models(
                emb, self.token_mask, cfg.context_fusion_method, 'relu',
                'ct_based_sent2vec', cfg.wd, self.is_train, cfg.dropout,
                block_len=cfg.block_len)

        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(linear([rep], hn, True, scope='pre_logits_linear',
                                           wd=cfg.wd, input_keep_prob=cfg.dropout,
                                           is_train=self.is_train))  # bs, hn
            logits = linear([pre_logits], self.output_class, False, scope='get_output',
                            wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train) # bs, 5
        _logger.done()
        return logits