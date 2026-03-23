import argparse
import os
import sys
import time
from logging import getLogger
from pathlib import Path

import torch
import pickle
import numpy as np

from numpy_compat import patch_numpy_compat

patch_numpy_compat()

from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer

SEQ_DIR = Path(__file__).resolve().parent

# Ensure local imports work regardless of current working directory.
if str(SEQ_DIR) not in sys.path:
    sys.path.insert(0, str(SEQ_DIR))

from utils import get_model, create_dataset


@torch.no_grad()
def evaluate_cand_hit_rate(model, eval_data, topk=1000):
    model.eval()
    hits = []
    device = next(model.parameters()).device

    for batch in eval_data:
        interaction = batch[0] if isinstance(batch, tuple) else batch
        interaction = interaction.to(device)
        scores = model.full_sort_predict(interaction)
        k = min(int(topk), scores.size(1))
        cand_idx = torch.topk(scores, k=k, dim=1).indices
        pos_items = interaction[model.POS_ITEM_ID]
        hit = (cand_idx == pos_items.unsqueeze(1)).any(dim=1).float()
        hits.append(hit)

    if not hits:
        return 0.0
    return torch.cat(hits).mean().item()


def _is_better(score, best_score, bigger=True):
    if best_score is None:
        return True
    return score > best_score if bigger else score < best_score


def fit_with_epoch_cand(trainer, model, train_data, valid_data, test_data, config, logger):
    epochs = int(config['epochs'])
    stopping_step = int(config['stopping_step'])
    bigger = bool(config['valid_metric_bigger'])

    best_valid_score = None
    best_valid_result = None
    best_state = None
    best_epoch = -1
    cur_step = 0

    for epoch in range(epochs):
        train_start = time.time()
        train_loss = trainer._train_epoch(train_data, epoch, show_progress=config['show_progress'])
        logger.info(
            f'epoch {epoch + 1} training [time: {time.time() - train_start:.2f}s, train loss: {train_loss}]'
        )

        valid_start = time.time()
        valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
        logger.info(
            f'epoch {epoch + 1} evaluating [time: {time.time() - valid_start:.2f}s, valid_score: {valid_score:.6f}]'
        )
        logger.info('valid result:')
        logger.info(valid_result)

        cand_hit_rate_1000 = evaluate_cand_hit_rate(model, test_data, topk=1000)
        logger.info(f'epoch {epoch + 1} test cand_hit_rate@1000: {cand_hit_rate_1000:.6f}')

        if _is_better(valid_score, best_valid_score, bigger=bigger):
            best_valid_score = float(valid_score)
            best_valid_result = valid_result
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            cur_step = 0
        else:
            cur_step += 1
            if cur_step >= stopping_step:
                logger.info(f'Early stopping triggered at epoch {epoch + 1}. Best epoch: {best_epoch}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_valid_score, best_valid_result


def run_single(model_name, dataset, pretrained_file='', **kwargs):
    # RecBole config/data paths in yaml are relative; run inside seq_rec_results/.
    orig_cwd = os.getcwd()
    os.chdir(SEQ_DIR)

    # configurations initialization
    props = [
        str(SEQ_DIR / 'config' / 'overall.yaml'),
        str(SEQ_DIR / 'config' / f'{model_name}.yaml'),
    ]
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        # If a relative path is provided, resolve it from the original cwd.
        ckpt_path = Path(pretrained_file)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(orig_cwd) / ckpt_path
        try:
            checkpoint = torch.load(str(ckpt_path))
        except pickle.UnpicklingError:
            # PyTorch 2.6+ defaults weights_only=True; RecBole checkpoints may require full unpickling.
            checkpoint = torch.load(str(ckpt_path), weights_only=False)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training（trainer.fit就是RecBole自动调用模型来训练）
    best_valid_score, best_valid_result = fit_with_epoch_cand(
        trainer, model, train_data, valid_data, test_data, config, logger
    )

    # model evaluation（RecBole自动预测并输出指标，在overall.yaml定义的）
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    cand_hit_rate_1000 = evaluate_cand_hit_rate(model, test_data, topk=1000)
    test_result['cand_hit_rate@1000'] = cand_hit_rate_1000

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='UniSRec', help='model name')
    parser.add_argument('-d', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    args, unparsed = parser.parse_known_args()
    print(args)

    run_single(args.m, args.d, pretrained_file=args.p)
