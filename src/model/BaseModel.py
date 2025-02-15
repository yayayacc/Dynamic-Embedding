from src.utils import *
from torch_scatter import scatter_add, scatter_mean, scatter_max


class BaseModel(nn.Module):
    def __init__(self, args, kg):
        super(BaseModel, self).__init__()
        self.args = args
        self.kg = kg  # information of snapshot sequence, self.kg.snapshots[i] is the i-th snapshot

        '''initialize the entity and relation embeddings for the first snapshot'''
        self.ent_embeddings = nn.Embedding(self.kg.snapshots[0].num_ent, self.args.emb_dim).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.snapshots[0].num_rel, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)

        '''loss function'''
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")#.to(self.args.device)  #

        # self.trainable_ent_embeddings = self.ent_embeddings
        # self.trainable_rel_embeddings = self.rel_embeddings

        # self.frozen_ent_embeddings = None
        # self.frozen_rel_embeddings = None

        self.embedding_dim = self.args.emb_dim

    def merge_parameters(self):
        dim_list = self.get_dim_list()
        # 结合trainable和frozen的embedding
        this_ss_ent_embedding = self.mapping(self.this_snapshot_ent_embeddings.data)
        this_ss_rel_embedding = self.mapping(self.this_snapshot_rel_embeddings.data)
        # this_ss_rel_embedding = torch.cat([self.frozen_rel_embeddings.weight, self.trainable_rel_embeddings.weight], dim=1).clone()
        # 构建新的embedding来放置这两个embedding
        self.ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot].num_ent, dim_list[self.args.snapshot + 1]).to(self.args.device).double()
        self.rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot].num_rel, dim_list[self.args.snapshot + 1]).to(self.args.device).double()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)

        self.ent_embeddings.weight.data = this_ss_ent_embedding
        self.rel_embeddings.weight.data = this_ss_rel_embedding

        # self.trainable_ent_embeddings = self.ent_embeddings
        # self.trainable_rel_embeddings = self.rel_embeddings
        # self.frozen_ent_embeddings = None
        # self.frozen_rel_embeddings = None

    def reinit_param(self):
        '''
        Re-initialize all model parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                xavier_normal_(p)

    def get_dim_list(self):
        self.args.emb_dim1 = self.args.emb_dim
        dim_list = []
        dim_list.append(self.args.emb_dim1)
        dim_list.append(self.args.emb_dim2)
        dim_list.append(self.args.emb_dim3)
        dim_list.append(self.args.emb_dim4)
        dim_list.append(self.args.emb_dim5)
        return dim_list
    
    def expand_embedding_size(self):
        '''
        Initialize entity and relation embeddings for next snapshot
        '''
        # 修改了原始维度
        dim_list = self.get_dim_list()
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, dim_list[self.args.snapshot + 1]).to(
            self.args.device).double()
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, dim_list[self.args.snapshot + 1]).to(
            self.args.device).double()
        # rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(
        #     self.args.device).double()
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return deepcopy(ent_embeddings), deepcopy(rel_embeddings)

    def switch_snapshot(self):
        '''
        After the training process of a snapshot, prepare for next snapshot
        '''
        pass

    def pre_snapshot(self):
        '''
        Preprocess before training on a snapshot
        '''
        pass

    def epoch_post_processing(self, size=None):
        '''
        Post process after a training iteration
        '''
        pass

    def snapshot_post_processing(self):
        '''
        Post process after training on a snapshot
        '''
        pass

    def store_old_parameters(self):
        '''
        Store the learned model after training on a snapshot
        '''
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer('old_data_{}'.format(name), value.clone().detach())

    def initialize_old_data(self):
        '''
        Initialize the storage of old parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                self.register_buffer('old_data_{}'.format(n), p.data.clone())

    def embedding(self, stage=None):
        '''
        :param stage: Train / Valid / Test
        :return: entity and relation embeddings
        '''
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def new_loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: loss of new facts
        '''
        return self.margin_loss(head, rel, tail, label)/head.size(0)



    def margin_loss(self, head, rel, tail, label=None):
        '''
        Pair Wise Margin loss: L1-norm(s + r - o)
        :param head:
        :param rel:
        :param tail:
        :param label:
        :return:
        '''
        ent_embeddings, rel_embeddings = self.embedding('Train')

        s = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, rel)
        o = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(s, r, o)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        loss = self.margin_loss_func(p_score, n_score, y)
        return loss

    def split_pn_score(self, score, label):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return:
        '''
        p_score = score[torch.where(label>0)]
        n_score = (score[torch.where(label<0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def score_fun(self, s, r, o):
        '''
        score function f(s, r, o) = L1-norm(s + r - o)
        :param h:
        :param r:
        :param t:
        :return:
        '''
        s = self.norm_ent(s)
        r = self.norm_rel(r)
        o = self.norm_ent(o)
        return torch.norm(s + r - o, 1, -1)

    def predict(self, sub, rel, stage='Valid'):
        '''
        Scores all candidate facts for evaluation
        :param head: subject entity id
        :param rel: relation id
        :param stage: object entity id
        :return: scores of all candidate facts
        '''

        '''get entity and relation embeddings'''
        if self.args.expand_dim == False: # 正常情况
            if stage != 'Test':
                num_ent = self.kg.snapshots[self.args.snapshot].num_ent
            else:
                num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
            ent_embeddings, rel_embeddings = self.embedding(stage)
            s = torch.index_select(ent_embeddings, 0, sub)
            r = torch.index_select(rel_embeddings, 0, rel)
            o_all = ent_embeddings[:num_ent]
            s = self.norm_ent(s)
            r = self.norm_rel(r)
            o_all = self.norm_ent(o_all)

            '''s + r - o'''
            pred_o = s + r
            score = 9.0 - torch.norm(pred_o.unsqueeze(1) - o_all, p=1, dim=2)
            score = torch.sigmoid(score)

            return score
        
        else: # 在扩展维度上进行预测
            if stage != 'Test':
                num_ent = self.kg.snapshots[self.args.snapshot].num_ent
            else:
                num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
            
            head = sub
            relation = rel
            tail = None
            # all_ent_emb = torch.cat([self.frozen_ent_embeddings.weight, self.trainable_ent_embeddings.weight], dim=1)
            # h1 = torch.index_select(self.frozen_ent_embeddings.weight, 0, head)
            # r1 = torch.index_select(self.frozen_rel_embeddings.weight, 0, relation)
            # h2 = torch.index_select(self.trainable_ent_embeddings.weight, 0, head)
            # r2 = torch.index_select(self.trainable_rel_embeddings.weight, 0, relation)
            # h = torch.cat([h1, h2], dim=1)
            # r = torch.cat([r1, r2], dim=1)

            t_all = self.ent_embeddings.weight[:num_ent]
            # h = torch.index_select(self.trainable_ent_embeddings.weight, 0, head)
            # r = torch.index_select(self.trainable_rel_embeddings.weight, 0, relation)
            h = torch.index_select(self.ent_embeddings.weight, 0, head)
            r = torch.index_select(self.rel_embeddings.weight, 0, relation)

            h = self.norm_ent(h)
            r = self.norm_rel(r)
            t_all = self.norm_ent(t_all)

            """ h + r - t """
            # 这个score 是对当前阶段见过的所有实体进行打分
            pred_t = h + r
            score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)
            score = torch.sigmoid(score)
            return score

    def norm_rel(self, r):
        return nn.functional.normalize(r, 2, -1)

    def norm_ent(self, e):
        return nn.functional.normalize(e, 2, -1)
    
