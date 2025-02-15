from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable

class TrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        '''prepare data'''
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            '''get loss'''
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()

            '''update'''
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            '''post processing'''
            model.epoch_post_processing(bh.size(0))
        return total_loss

    def evaluate_on_training_data(self, model):
        model.eval()
        results = {}
        num = 0
        retrain_data = []
        # results['retrain_data'] = [] # 这里面只会选label为1的数据
        # results['right_data'] = []
        all_training_data = []
        all_entity = []
        all_relation = []
        all_data_head = torch.tensor([], dtype=torch.int64)
        all_data_relation = torch.tensor([], dtype=torch.int64)
        all_data_tail = torch.tensor([], dtype=torch.int64)
        all_data_label = torch.tensor([], dtype=torch.int64)
        # 下面这三个用来存贮所有label为1的数据
        new_data_head = []
        new_data_relation = []
        new_data_tail = []
        new_data_label = []
        for b_id, batch in enumerate(self.data_loader):
            """ Get loss   y中的值是0或者1，1表示这个样本是正确的，0表示这个样本是错误的 """
            head, relation, tail, label = batch

            # 记录一下有哪些实体和关系
            all_entity = all_entity + head.tolist() # 用来记录本轮训练数据里面一共有哪些实体/关系
            all_entity = all_entity + tail.tolist()
            all_relation = all_relation + relation.tolist()
            all_data_head = torch.cat([all_data_head, head], dim=0) # 把数据存下来，便于后面提取
            all_data_relation = torch.cat([all_data_relation, relation], dim=0)
            all_data_tail = torch.cat([all_data_tail, tail], dim=0)
            all_data_label = torch.cat([all_data_label, label], dim=0)
        # TODO:我只想要label为1的数据，然后根据他们的预测概率进行采样，预测logit高的被采样概率就低
        for idx in range(len(all_data_head)):
            if all_data_label[idx] == 1:
                new_data_head.append(all_data_head[idx].item())
                new_data_relation.append(all_data_relation[idx].item())
                new_data_tail.append(all_data_tail[idx].item())
                new_data_label.append(all_data_label[idx].item())
        new_data_head = torch.tensor(new_data_head, dtype=torch.int64)
        new_data_relation = torch.tensor(new_data_relation, dtype=torch.int64)
        new_data_tail = torch.tensor(new_data_tail, dtype=torch.int64)
        new_data_label = torch.tensor(new_data_label, dtype=torch.int64)

        all_training_data = [new_data_head, new_data_relation, new_data_tail, new_data_label]
        all_entity = list(set(all_entity))
        all_relation = list(set(all_relation))

        def chunks(lst, n):
            for i in range(0, len(lst[0]), n):
                yield (lst[0][i:i + n], lst[1][i:i + n], lst[2][i:i + n], lst[3][i:i + n])
        
        all_prob = [] # 用来进行采样
        for chunk in chunks(all_training_data, int(self.args.batch_size/32)):
            head, relation, tail, label = chunk
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            # label = label.to(self.args.device) # (batch_size, ent_num) 1表示正确的，0表示错误的
            num += len(head)
            pred = model.predict(head, relation, stage='Valid') # (batch_size, num_ent)
            pred = pred.to(self.args.device)

            # 下面这几行的意思是：有可能同一个头实体和关系对应不同的尾实体，但根据我们这个数据的要求，我们只认为某个尾实体是对的，因此要把别的尾实体对应的地方设置成负无穷
            # new_label 模仿测试集中的label，和上面那个label不一样
            new_label = torch.zeros_like(pred)
            for idx in range(tail.shape[0]):
                new_label[idx, tail[idx]] = 1 
            """ filter: If there is more than one tail in the label, we only think that the tail in this triple is right """
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail] # take off the score of tail in this triple
            pred = torch.where(new_label.bool(), -torch.ones_like(pred) * 10000000, pred) # Set all other tail scores to negative infinity
            pred[batch_size_range, tail] = target_pred # restore the score of the tail in this triple
            
            """ rank all candidate entities """
            """ Two sorts can be optimized into one """
            # ranks表示的是当前阶段见过的所有实体中，我们正在预测的这个实体的排名
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float() # all right tail ranks, (batch_size, 1)
            # results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            # results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            # results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            target_logit = pred[batch_size_range, tail]
            # 这里采用的不是困难样本，而是用的softmax的概率，也就是简单样本被选中的概率更高
            if self.args.use_difficult_samples:
                all_prob += torch.exp(1-target_logit).tolist()
            else:
                all_prob += torch.nn.Softmax(target_logit).dim.tolist()
            # all_prob += torch.nn.Softmax(target_logit).dim.tolist()
   
        # 计算每个样本被采样的概率
        all_prob = np.array(all_prob)
        sum_prob = 0.
        for prob in all_prob:
            sum_prob += prob
        for idx, prob in enumerate(all_prob):
            all_prob[idx] = prob / sum_prob
        # count = float(results['count'])

        # 随机采样若干个样本
        num_samples = min(self.args.num_samples_for_retrain, len(all_prob)/3)
        chosen_index = np.random.choice(len(all_prob), p=all_prob, size=num_samples, replace=False)

        # 找到所采集的样本没有覆盖的关系和实体
        for idx in chosen_index:
            head = new_data_head[idx].item()
            relation = new_data_relation[idx].item()
            tail = new_data_tail[idx].item()
            if head in all_entity:
                all_entity.remove(head)
            if tail in all_entity:
                all_entity.remove(tail)
            if relation in all_relation:
                all_relation.remove(relation)
            retrain_data.append((head, relation, tail, new_data_label[idx].item()))
        # 此时的 all_entity 和 all_relation 就是没有覆盖到的实体和关系，从all_entity开始遍历，加入数据
        for idx in range(len(new_data_head)):
            if idx%1000 == 0:
                print(idx)
            if new_data_head[idx] in all_entity:
                retrain_data.append((new_data_head[idx].item(), new_data_relation[idx].item(), new_data_tail[idx].item(), new_data_label[idx]))
                all_entity.remove(new_data_head[idx].item())
                if new_data_relation[idx] in all_relation:
                    all_relation.remove(new_data_relation[idx].item())
                continue
            if new_data_tail[idx] in all_entity:
                retrain_data.append((new_data_head[idx].item(), new_data_relation[idx].item(), new_data_tail[idx].item(), new_data_label[idx]))
                all_entity.remove(new_data_tail[idx].item())
                if new_data_relation[idx] in all_relation:
                    all_relation.remove(new_data_relation[idx].item())
                continue
            if new_data_relation[idx] in all_relation:
                retrain_data.append((new_data_head[idx].item(), new_data_relation[idx].item(), new_data_tail[idx].item(), new_data_label[idx]))
                all_relation.remove(new_data_relation[idx].item())
            if all_entity == [] and all_relation == []:
                break

            # TODO:这行代码是测试用的
            # if idx > 100:
            break

        return retrain_data



class DevBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = 100
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        sr2o = self.kg.snapshots[self.args.snapshot].sr2o_all
        '''start evaluation'''
        num = 0
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                if k not in [0, 2, 4, 9]:
                    continue
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results
    
    def process_epoch_again(self, model):
        model.eval()
        num = 0
        results = dict()
        '''start evaluation'''
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            model.args.expand_dim = True
            pred = model.predict(sub, rel, stage=stage)
            model.args.expand_dim = False

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                if k not in [0, 2, 4, 9]:
                    continue
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results


class DevBatchProcessor_MEANandLAN():
    '''
    To save memory, we collect the queries with the same relation and then perform evaluation.
    '''
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = 1
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        '''start evaluation'''
        sub, rel, obj, label = None, None, None, None

        for step, batch in enumerate(self.data_loader):
            sub_, rel_, obj_, label_ = batch
            if sub == None:
                sub, rel, obj, label = sub_, rel_, obj_, label_
                continue
            elif rel[0] == rel_ and rel.size(0) <= 50:
                sub = torch.cat((sub, sub_), dim=0)
                rel = torch.cat((rel, rel_), dim=0)
                obj = torch.cat((obj, obj_), dim=0)
                label = torch.cat((label, label_), dim=0)
                continue

            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
            sub, rel, obj, label = None, None, None, None

        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results