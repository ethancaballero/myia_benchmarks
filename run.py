import argparse

"""
from mlp.myia_mlp import run_model as run_model_myia_mlp
from mlp.pytorch_mlp import run_model as run_model_pytorch_mlp

from lstm.myia_lstm import run_model as run_model_myia_lstm
from lstm.pytorch_lstm import run_model as run_model_pytorch_lstm
#"""

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='benchmarks')

parser.add_argument('--mlp_fast', type=str2bool, default=False,
                    help='do fast mlp benchmarks')

parser.add_argument('--mlp', type=str2bool, default=False,
                    help='do mlp benchmarks')
parser.add_argument('--lstm', type=str2bool, default=False,
                    help='do lstm benchmarks')
parser.add_argument('--rnn', type=str2bool, default=False,
                    help='do rnn benchmarks')

parser.add_argument('--my', type=str2bool, default=False,
                    help='do myia benchmarks')
parser.add_argument('--pt', type=str2bool, default=False,
                    help='do pytorch benchmarks')

parser.add_argument('--iters', type=int, default=50, metavar='i',
                    help='rnumber of iters')
parser.add_argument('--timesteps', type=int, default=10, metavar='i',
                    help='recurrent timesteps')
parser.add_argument('--warmup', type=int, default=5, metavar='w',
                    help='rnumber of iters')
parser.add_argument('--lr', type=float, default=1.00, metavar='l',
                    help='learning rate')
parser.add_argument('--dev', type=str, default='cuda', metavar='d',
                    help='hardware')

parser.add_argument('--print_all_iters', type=str2bool, default=False,
                    help='print_all_iters')

parser.add_argument('--lstm_input_size', type=int, default=256, help="LSTM Input size")
parser.add_argument('--lstm_hidden_size', type=int, default=512, help="LSTM Hidden size")

parser.add_argument('--cuda_sync', type=str2bool, default=True, help="torch.cuda.synchronize")

parser.add_argument('--break_cr1', type=str2bool, default=False,
                    help='break and compiling once and running once')
parser.add_argument('--break_cr2', type=str2bool, default=False,
                    help='break and compiling once and running twice')

parser.add_argument('--break_bm', type=str2bool, default=False,
                    help='break before model put on device')
parser.add_argument('--break_mod_on_d', type=str2bool, default=False,
                    help='break right after model put on device')
parser.add_argument('--break_mod_on_d_and_gen_data', type=str2bool, default=False,
                    help='break right after model put on device and data generator created')

parser.add_argument('--break_after_pt_optim', type=str2bool, default=False,
                    help='break right after pytorch optimizer is created')

parser.add_argument('--save_txt', type=str2bool, default=False,
                    help='save benchmark stats as text file')


args = parser.parse_args()

if args.mlp:
	if args.my:
		from mlp.myia_mlp import run_model as run_model_myia_mlp
		run_model_myia_mlp(args)

	if args.pt:
		from mlp.pytorch_mlp import run_model as run_model_pytorch_mlp
		run_model_pytorch_mlp(args)

if args.mlp_fast:
	if args.my:
		from mlp.myia_mlp_fast import run_model as run_model_myia_mlp_fast
		run_model_myia_mlp_fast(args)

if args.lstm:
	if args.my:
		from lstm.myia_lstm import run_model as run_model_myia_lstm
		run_model_myia_lstm(args)

	if args.pt:
		from lstm.pytorch_lstm import run_model as run_model_pytorch_lstm
		run_model_pytorch_lstm(args)

if args.rnn:
	if args.my:
		from rnn.myia_rnn import run_model as run_model_myia_rnn
		run_model_myia_rnn(args)

	if args.pt:
		from rnn.pytorch_rnn import run_model as run_model_pytorch_rnn
		run_model_pytorch_rnn(args)