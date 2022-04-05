import torch
import torch.nn as nn
from .Model import Model
import sys

class ComplEx_NNE_AER(Model):
    def __init__(self, ent_tot, rel_tot, rule_list, mu, dim = 100):
        super(ComplEx_NNE_AER, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rule_list = rule_list
        self.mu = mu

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

    def _calc_rule(self):
        # r_re =  self.rel_re_embeddings
        # r_im =  self.rel_im_embeddings
        # for each rule, modify gradient(weight) for corresponding relations
        # print(r_re.requires_grad)
        # print(r_im.requires_grad)
        for i, rule in enumerate(self.rule_list):
            r_p, r_q, conf, r_dir = rule
            r_p = torch.LongTensor([r_p]).cuda()
            r_q = torch.LongTensor([r_q]).cuda()

            r_p_re = self.rel_re_embeddings(r_p)
            r_p_im = self.rel_im_embeddings(r_p)
            r_p_re *= conf
            r_p_im *= r_dir
            
            r_q_re = self.rel_re_embeddings(r_q)
            r_q_im = self.rel_im_embeddings(r_q)
            r_q_re *= conf
            # print("rule grad exists?: " + str(r_q_im.requires_grad))
            # real penalty
            if not i:
                rule_score = torch.sum(torch.max(torch.zeros(self.dim).cuda(), (r_p_re - r_q_re))) 
            else:
                rule_score += torch.sum(torch.max(torch.zeros(self.dim).cuda(), (r_p_re - r_q_re))) 
            # imaginary penalty
            rule_score += torch.sum(torch.square(r_p_im - r_q_im) * conf).cuda() 

        rule_score /= len(self.rule_list)
        rule_score *= self.mu
        # print(rule_score.requires_grad)
        return rule_score

    def _calc_rule_grad(self):
        r_re = self.rel_re_embeddings
        r_im = self.rel_im_embeddings
        # for each rule, modify gradient(weight) for corresponding relations
        for rule in self.rule_list:
            r_p, r_q, conf, r_dir = rule
            r_p_re = r_re.weight.data[r_p]
            r_p_im = r_im.weight.data[r_p] * r_dir
            r_q_re = r_re.weight.data[r_q]
            r_q_im = r_im.weight.data[r_q]
            for i in range(self.dim):
                if r_q_re[i] > r_p_re[i]: # real penalty
                    self.rel_re_embeddings.weight.grad[r_p][i] -= self.mu * conf
                    self.rel_re_embeddings.weight.grad[r_q][i] += self.mu * conf
            # imaginary penalty
            dif_im = r_q_im - r_p_im
            self.rel_im_embeddings.weight.grad[r_p] -= 2 * dif_im * self.mu
            self.rel_im_embeddings.weight.grad[r_q] += 2 * dif_im * self.mu * r_dir * conf
        # normalize the gradient
        for i, grad in enumerate(self.rel_re_embeddings.weight.grad):
            dnorm = torch.sqrt(torch.sum(torch.square(grad)))
            if dnorm > 1:
                self.rel_re_embeddings.weight.grad[i] /= dnorm
        for i, grad in enumerate(self.rel_im_embeddings.weight.grad):
            dnorm = torch.sqrt(torch.sum(torch.square(grad)))
            if dnorm > 1:
                self.rel_im_embeddings.weight.grad[i] /= dnorm


    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        # print("original forward grad?: " + str(h_im.requires_grad))
        # print('score dimension: ' + str(score.size()))
        # print('embedding dimension: ' + str(h_re.size()) + str(r_re.size()))
        # print('original embedding: ' + str(self.rel_re_embeddings.num_embeddings) + ' ' + str(self.rel_re_embeddings.embedding_dim))
        # sys.exit()
        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        regul = (torch.mean(h_re ** 2) + 
                 torch.mean(h_im ** 2) + 
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()