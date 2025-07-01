import sys
import json

from . import pretrain_chessformer


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python -m src <config path>\n')
        sys.exit(0)
    config_path = sys.argv[1]
    with open(config_path) as config_file:
        config = json.load(config_file)
    
    match config['type']:
        case 'chessformer-pretraining':
            pretrain_chessformer(config, config_path)
        case _:
            sys.stderr.write('Unknown training type\n')