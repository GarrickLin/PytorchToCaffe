import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from model.trans_models import TransformerNet_self_bn
import pytorch_to_caffe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
args = parser.parse_args()
print(args)

def transfer_to_caffe():
    name = "TransformerNet"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    transformer = TransformerNet_self_bn().to(device)
    transformer.load_state_dict(torch.load(args.checkpoint_model))
    transformer.eval()
    input=Variable(torch.ones([1,3,480,640]))
    pytorch_to_caffe.trans_net(transformer, input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))

if __name__ == "__main__":
    transfer_to_caffe()
