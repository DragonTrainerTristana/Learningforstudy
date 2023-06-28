import organ
from organ import ORGAN

model = ORGAN('test', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1})

#model.ad_training_set('data/toy.csv')

model.load_training_set('data/toy.csv')

model.set_training_program(['novelty'], [1])
model.load_metrics()
#model.train(ckpt_dir='ckpt_dir')

model.train(ckpt_dir='C:/Users/sengh/OneDrive/바탕 화면/ORGAN-master/darksoul')
