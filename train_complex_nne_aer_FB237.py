import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx_NNE_AER
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import collections


device = 'cuda'
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB237/1207-0_shot/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB237/1207-0_shot/", "link")

# dataloader for rule set
original_rule = './benchmarks/FB237/1207-0_shot/cons.txt'
rule_list = []
with open(original_rule, 'r') as f:
	while True:
		line = f.readline()
		if line:
			flag = 1
			# two relations split by ',', confidence split by tab
			line.replace('\n', '')
			if line[0] == '-': # negative rules
				flag = -1
				line = line[1:]
			tokens = line.split('\t')
			r_p = tokens[0].split(',')[0]
			r_q = tokens[0].split(',')[1]
			r_p = int(r_p)
			r_q = int(r_q)
			conf = float(tokens[1])
			# print(r_p, r_q, conf)
			rule_list.append((r_p, r_q, conf, flag))
		else:
			break
print ("\n======> Number of rules: " + str(len(rule_list)))
print ("=======> Sample rule: " + str(rule_list[0]))

# define the model
complEx_nne_aer = ComplEx_NNE_AER(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	rule_list = rule_list, 
	mu = 0.1, 
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = complEx_nne_aer, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)
model.to(device)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
# trainer.to(device)
trainer.run()
# complEx_nne_aer.save_checkpoint('./checkpoint/complEx.ckpt')

# test the model
# complEx_nne_aer.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx_nne_aer, data_loader = test_dataloader, use_gpu = True)
# tester.to(device)
tester.run_link_prediction(type_constrain = False)