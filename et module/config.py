import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--warmup", type=float, default=0.1)
parser.add_argument("--early_stop", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--t_total", action='store_true')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay_epochs", type=int, default=50)
parser.add_argument("--lr_decay_rate", type=float, default=0.1)
parser.add_argument("--weight_decay_rate", type=float, default=0.01)
parser.add_argument("--bert_lr", type=float, default=2e-5)
parser.add_argument("--num_epochs", type=int, default=100)

parser.add_argument("--model_dir", type=str, default="checkpoints")
parser.add_argument("--model_path", type=str, default='bert_1.pt')

parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--train_dataset", type=str, default="wiki/train_new.json")
parser.add_argument("--eval_dataset", type=str, default="wiki/dev_new.json")
parser.add_argument("--test_dataset", type=str, default="wiki/dev_new.json")

parser.add_argument("--bert_low_case", type=bool, default=True)
parser.add_argument("--bert_type", type=str, default='bert')
parser.add_argument("--bert_hidden_size", type=int, default=768)
parser.add_argument("--bert_dropout", type=float, default=0.1)
parser.add_argument("--bert_mention_max_len", type=int, default=50)
parser.add_argument("--bert_max_len", type=int, default=256)
parser.add_argument("--bert_char_max_len", type=int, default=100)
parser.add_argument("--bert_entity_threshold", type=float, default=0.3)
parser.add_argument("--bert_freeze", action='store_true')
parser.add_argument("--bert_threshold", type=float, default=0.5)
parser.add_argument("--bert_adam", action='store_true')

parser.add_argument("--enhance_mention", action='store_true')
parser.add_argument("--interaction", action='store_true')
parser.add_argument("--context_dropout", type=float, default=0.2)
parser.add_argument("--mention_dropout", type=float, default=0.5)
parser.add_argument("--rnn_hidden_size", type=int, default=768)
parser.add_argument("--rnn_num_layers", type=int, default=1)
parser.add_argument("--rnn_dropout", type=float, default=0.)
parser.add_argument("--rnn_num_dirs", type=int, default=2)

parser.add_argument("--cnn_embedding_dim", type=int, default=100)
parser.add_argument("--cnn_output_dim", type=int, default=150)
parser.add_argument("--cnn_filters", type=str, default="[[2,50],[3,50],[4,50]]")

parser.add_argument("--hierarchy_alpha", type=float, default=0.3)

# args = parser.parse_args(sys.argv)
args, unknown = parser.parse_known_args()
