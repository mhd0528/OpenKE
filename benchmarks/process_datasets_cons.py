# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import pandas as pd

import numpy as np

from collections import defaultdict

# DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')
# DATA_PATH = './data/'
DATA_PATH = Path('/blue/daisyw/ma.haodi/OpenKE/benchmarks')

def translate_cons(dataset, path, train_data, relation_dict, rule_type = 0):
    rel2id = relation_dict
    if rule_type == 0:
        print(path)
        with open(path+'/_cons.txt') as f,open(path+'/cons.txt','w') as out:
            for line in f:
                rule_str, conf = line.strip().split()
                body,head = rule_str.split(',')
                prefix = ''
                if '-' in body:
                    prefix = '-'
                    body = body[1:]
                try:
                    rule = prefix + str(rel2id[body])+','+str(rel2id[head])
                    out.write('%s\t%s\n' % (rule,conf))
                    # out2.write(line)
                except KeyError:
                    print("rule not found: " + line)
    elif rule_type == 3:
        # read in each rule, translate, extract triples from all training triples
        # format: p, q, r, conf, triple_ids
        # read in rule set
        rule_df = pd.read_excel(path+'/Freebase_Rules.xlsx', sheet_name=None)['Type 3']
        # print(rule_df.head())
        with open(path+'/all_cons_3.txt','w') as out:
            for id, row in rule_df.iterrows():
                rel_p = '/' + row['p(x,y) <-'].replace('.', '/')
                rel_q = '/' + row['q(z,x)'].replace('.', '/')
                re_r = '/' + row['r(z,y)'].replace('.', '/')
                conf = row['Confidence']
                if conf >= 0.5:
                    try:
                        rule = str(rel2id[rel_p])+','+str(rel2id[rel_q])+','+str(rel2id[re_r])
                        # extract triples from training set
                        triple_ids = []
                        for i, triple in enumerate(train_data):
                            if triple[2] == rel2id[rel_q]:
                                triple_ids.append(str(i))
                        triple_ids_str = ' '.join(triple_ids)
                        out.write('%s\t%s\t%s\n' % (rule, conf, triple_ids_str))
                        print("rule found: " + str(rel_q))
                        # out2.write(line)
                    except KeyError:
                        continue
    elif rule_type == 4:
        # read in each rule, translate, extract triples from all training triples
        # format: p, q, r, conf: triple_ids
        # read in rule set
        rule_df = pd.read_excel(path+'/original/Freebase_Rules.xlsx', sheet_name=None)['Type 4']
        # print(rule_df.head())
        with open(path+'/all_cons_4.txt','w') as out:
            for id, row in rule_df.iterrows():
                rel_p = '/' + row['p(x,y) <-'].replace('.', '/')
                rel_q = '/' + row['q(x,z)'].replace('.', '/')
                re_r = '/' + row['r(z,y)'].replace('.', '/')
                conf = row['Confidence']
                if conf >= 0.8:
                    try:
                        rule = str(rel2id[rel_p])+','+str(rel2id[rel_q])+','+str(rel2id[re_r])
                        # extract triples from training set
                        triple_ids = []
                        for i, triple in enumerate(train_data):
                            if triple[2] == rel2id[rel_q]:
                                triple_ids.append(str(i))
                        triple_ids_str = ' '.join(triple_ids)
                        out.write('%s\t%s\t%s\n' % (rule, conf, triple_ids_str))
                        print("rule found: " + str(rule))
                        # out2.write(line)
                    except KeyError:
                        continue

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    n_relations = len(relations)
    n_entities = len(entities)
    #### check if file exists
    os.makedirs(os.path.join(DATA_PATH, name, 'exist'))

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['entity2id.txt', 'relation2id.txt']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w')
        ff.write(str(len(dic)) + '\n')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    # OpenKE's format: (e1, e2, r)
    train_triples = []
    for f, outf in zip(files, ['train2id.txt', 'valid2id.txt', 'test2id.txt']):
        # file_path = os.path.join(path+'/original/', f)
        out = open(os.path.join(DATA_PATH, name, outf), 'w')
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        sample_cnt = 0
        sample_content = ''
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                sample_cnt += 1
                sample_content += (str(entities_to_id[lhs]) + ' ' + str(entities_to_id[rhs]) + ' ' + str(relations_to_id[rel]) + '\n')
                if outf == 'train2id.txt':
                    train_triples.append([entities_to_id[lhs], entities_to_id[rhs], relations_to_id[rel]])
            except ValueError:
                continue
        out.write(str(sample_cnt) + '\n')
        out.write(sample_content)
        out.close()
    
    # translate rules
    translate_cons(name, path, train_triples, relations_to_id, 0)


#### original format: (e1, r, e2)
#### OpenKE format: (e1, e2, r)

if __name__ == "__main__":
    # datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    # datasets = ['FB15K', 'FB15K237', 'NELL-One']
    datasets = ['NELL-One', 'FB237']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise