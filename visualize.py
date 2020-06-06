import torch
from sklearn.manifold import TSNE
import numpy as np
import os
from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager

from options import get_best_file, get_assigned_file

# t-sne feature
def tsne(model, data_loader):
  all_feats = None
  all_labels = None
  for (x, y) in data_loader:
    x = x.cuda()
    feats = model(x)
    feats = feats.reshape(feats.shape[0], -1).data.cpu().numpy()
    labels = y.numpy()

    all_feats = feats if all_feats is None else np.concatenate((all_feats, feats), axis=0)
    all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
    break

  all_embeddings = TSNE(n_components=2).fit_transform(all_feats)
  return all_embeddings, all_labels


def get_visualize_data(
        dataset='multi', save_epoch=399, name='tmp', method='baseline', model='ResNet10', split='novel',
        data_dir='./filelists', save_dir='./output'):
  print('Visualizing! {} dataset with {} epochs of {}({})'.format(dataset, save_epoch, name, method))

  print('\nStage 1: saving features')
  # dataset
  print('  build dataset')
  if 'Conv' in model:
    image_size = 84
  else:
    image_size = 224
  split = split
  loadfile = os.path.join(data_dir, dataset, split + '.json')
  datamgr         = SimpleDataManager(image_size, batch_size = 64)
  data_loader      = datamgr.get_data_loader(loadfile, aug = False)

  print('  build feature encoder')
  # feature encoder
  checkpoint_dir = '%s/checkpoints/%s'%(save_dir, name)
  if save_epoch != -1:
    modelfile   = get_assigned_file(checkpoint_dir,save_epoch)
  else:
    modelfile   = get_best_file(checkpoint_dir)
  if method in ['relationnet', 'relationnet_softmax']:
    if model == 'Conv4':
      model = backbone.Conv4NP()
    elif model == 'Conv6':
      model = backbone.Conv6NP()
    else:
      model = model_dict[model]( flatten = False )
  else:
    model = model_dict[model]()
  model = model.cuda()
  tmp = torch.load(modelfile)
  try:
    state = tmp['state']
  except KeyError:
    state = tmp['model_state']
  except:
    raise
  state_keys = list(state.keys())
  for i, key in enumerate(state_keys):
    if "feature." in key and not 'gamma' in key and not 'beta' in key:
      newkey = key.replace("feature.","")
      state[newkey] = state.pop(key)
    else:
      state.pop(key)

  model.load_state_dict(state)
  model.eval()

  return tsne(model, data_loader)
