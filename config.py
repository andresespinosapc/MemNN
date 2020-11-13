from __future__ import division
import os

class Config():
    def __init__(self):
        self.n_embed = 186841
        self.d_embed = 300
        self.sent_len = 20
        self.win_len = 7 + 3

        self.ir_size = 1000
        self.filter_size = 100
        self.n_neg_samples = 1000

        self.num_workers = 4
        self.batch_size = 128
        self.epoch = 100
        self.lr = 0.005
        self.use_lr_decay = False
        self.l2 = 0.00001
        self.save_every = 50

        self.margin = 0.2

        self.rnn_fea_size = self.d_embed

        #model file
        self.pre_embed_file = "./model/{}/embedding.pre".format(self.d_embed)
        self.reader_model_dir = "./model"
        self.reader_model = os.path.join(self.reader_model_dir, "reader_{}.torch".format(self.d_embed))

        self.title_dict = "./pkl/dict/title.dict"
        self.entity_dict = "./pkl/dict/entity.dict"

        #data dir
        self.data_dir = "./data"

        #memory network
        self.hop = 2

        # RL
        self.K = 1


config = Config()
