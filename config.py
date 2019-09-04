import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--bert_lr", type=float, default=2e-5)

parser.add_argument("--warm_up", type=float, default=0.1)
parser.add_argument("--t_total", action='store_true')
parser.add_argument("--lr_decay_epochs", type=int, default=10)
parser.add_argument("--lr_decay_rate", type=float, default=0.1)
parser.add_argument("--weight_decay_rate", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=20)

parser.add_argument("--model_dir", type=str, default="checkpoints")
parser.add_argument("--model_path", type=str)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=0)

# parser.add_argument("--train_dataset", type=str, default="data/locationcrowd/train_version2.txt")
# parser.add_argument("--eval_dataset", type=str, default="data/locationcrowd/dev.txt")
# parser.add_argument("--test_dataset", type=str, default="data/locationcrowd/test.txt")
parser.add_argument("--train_dataset", type=str, default="data/train.txt")
parser.add_argument("--eval_dataset", type=str, default="data/dev.txt")
parser.add_argument("--test_dataset", type=str, default="data/test.txt")


parser.add_argument("--rnn_num_layers", type=int, default=2)
parser.add_argument("--rnn_num_dirs", type=int, default=2)
parser.add_argument("--rnn_hidden_size", type=int, default=768)

parser.add_argument("--bert_freeze", action='store_true')
parser.add_argument("--bert_low_case", type=bool, default=False)
parser.add_argument("--bert_hidden_size", type=int, default=768)

parser.add_argument('--char_max_len', type=int, default=32)
parser.add_argument("--bert_max_len", type=int, default=256)

parser.add_argument("--cnn_embedding_dim", type=int, default=100)
parser.add_argument("--cnn_output_size", type=int, default=150)
parser.add_argument("--cnn_filters", type=str, default='[[2,50],[3,50],[4,50]]')

parser.add_argument("--attn_key_dim", type=int, default=64)
parser.add_argument("--attn_val_dim", type=int, default=64)
# a bug, must can be divided
parser.add_argument("--attn_num_heads", type=int, default=3)

parser.add_argument("--attn_dropout", type=float, default=0.)
parser.add_argument("--feat_dropout", type=float, default=.5)
parser.add_argument("--rnn_dropout", type=float, default=.5)

args = parser.parse_args()
