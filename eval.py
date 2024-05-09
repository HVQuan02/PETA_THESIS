import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models import create_model
from src.utils.evaluation import AP_partial, spearman_correlation
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datasets import CUFED

parser = argparse.ArgumentParser(description='PETA: Photo Album Event Recognition')
parser.add_argument('--model_path', type=str, default='./weights/PETA-cufed.pt')
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec', 'holidays'])
parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
parser.add_argument('--split_path', type=str, default='/kaggle/input/cufed-full-split')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # change
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
parser.add_argument('--ema', action='store_true', help='use ema model or not')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--transformers_pos', type=int, default=1)
args = parser.parse_args()

def evaluate(model, test_loader, test_dataset, device):
  model.eval()
  scores = torch.zeros((len(test_dataset), len(test_dataset.event_labels)), dtype=torch.float32)
  attentions = []
  importance_labels = []
  gidx = 0
    
  with torch.no_grad():
    for batch in test_loader:
      feats, _, importance_scores = batch
      feats = feats.to(device)
      logits, attention = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape
      attentions.append(attention)
      importance_labels.append(importance_scores)
        
  attention_tensor = torch.cat(attentions).to(device)
  importance_labels = torch.cat(importance_labels).to(device)
    
  map = AP_partial(test_dataset.labels, scores.numpy())[1]
  spearman = spearman_correlation(attention_tensor[:, 0, 1:], importance_labels)

  return map, spearman

def main():
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if args.dataset == 'cufed':
    dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length)
  else:
    exit("Unknown dataset!")

  test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

  if args.verbose:
    print("running on {}".format(device))
    print("num of test sample={}".format(len(dataset)))

  # Setup model
  print('creating and loading the model...')
  state = torch.load(args.model_path, map_location='cpu')
  model = create_model(args).to(device)
  if args.ema:
    model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
  model.load_state_dict(state['model_state_dict'], strict=True)

  t0 = time.perf_counter()
  map, spearman = evaluate(model, test_loader, dataset, device)
  t1 = time.perf_counter()
  
  print('map={:.2f} spearman={:.2f} dt={:.2f}sec'.format(map, spearman, t1 - t0))

if __name__ == '__main__':
  main()
