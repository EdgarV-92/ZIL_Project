import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--batch_size", type=int, default=16,
                    help="batch size")
parser.add_argument("--froze_layer_index", type=int, default=20,
                    help="Froze layer index")
# parser.add_argument("-v", "--verbosity", action="count",
#                     help="increase output verbosity")
args = parser.parse_args()
print(args.batch_size)