import torch
import sys
import datetime
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.dataset import MD17
from dig.threedgraph.method import SphereNet,SchNet, DimeNetPP,EAA
from dig.threedgraph.method import run
from dig.threedgraph.evaluation import ThreeDEvaluator

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
dataset_md17 = MD17(root='dataset/', name='uracil')

split_idx_md17 = dataset_md17.get_idx_split(len(dataset_md17.data.y), train_size=1000, valid_size=1000, seed=42)

train_dataset_md17, valid_dataset_md17, test_dataset_md17 = dataset_md17[split_idx_md17['train']], dataset_md17[split_idx_md17['valid']], dataset_md17[split_idx_md17['test']]
print('train, validaion, test:', len(train_dataset_md17), len(valid_dataset_md17), len(test_dataset_md17))
length=len(train_dataset_md17)
all = []
for item in train_dataset_md17:
    all.append(item['y'].numpy()/len(item['z']))
all = torch.tensor(all)
mean = torch.mean(all)
std = torch.std(all)
print(mean)
print(std)
#:obj:`aspirin`, :obj:`benzene_old`, :obj:`ethanol`, :obj:`malonaldehyde`,
#:obj:`naphthalene`, :obj:`salicylic`, :obj:`toluene`, :obj:`uracil`. (default: :obj:`benzene_old`)
model_md17 = EAA(energy_and_force=True, cutoff=5.0, num_layers=1,num_gaussians=25,num_heads=7,
        hidden_channels=128,use_stand=True,mean=mean,std=std)
# model_md17 = EAA(energy_and_force=True, cutoff=5.0, num_layers=3,num_gaussians=25,
#         hidden_channels=128)
loss_func_md17 = torch.nn.L1Loss()
evaluation_md17 = ThreeDEvaluator()
starttime = datetime.datetime.now()
run3d_md17 = run()
run3d_md17.run(device, train_dataset_md17, valid_dataset_md17, test_dataset_md17, model_md17,
               loss_func_md17, evaluation_md17,p=100,
               epochs=2000, batch_size=36, vt_batch_size=36,
               lr=0.001, lr_decay_factor=0.8, lr_decay_step_size=150, energy_and_force=True,
               save_dir='malonaldehydemodel', log_dir='malonaldehydelog')
#Params: 1890118

endtime = datetime.datetime.now()
print("training use:"+str(endtime - starttime))
#Params: 1890118