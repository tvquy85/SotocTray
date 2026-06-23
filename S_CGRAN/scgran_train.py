import argparse
import random
import numpy as np
import os
import torch
import pickle
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scgran_model import get_scgran_loss, SCGRAN

# Evaluator port from parent directory if needed
base_src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_src, "original"))
try:
    from quant_evaluator import evaluate
except ImportError:
    print("Failed to import quant_evaluator")
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NASDAQ')
    parser.add_argument('--seed', type=int, default=123456789)
    parser.add_argument('--alpha', type=float, default=0.1, help="Base Weight for pairwise rank loss")
    parser.add_argument('--lambda_alpha', type=float, default=0.05, help="Base Weight for RA-Loss")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=25)
    return parser.parse_args()

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.random.manual_seed(12345678) # Hardcoded to match original
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Dataset: {args.dataset}, Seed: {args.seed}, Device: {device}")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset_path = os.path.join(BASE_DIR, "dataset", args.dataset)
    
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)
        
    stock_num = eod_data.shape[0]
    lookback_length = 16
    steps = 1
    valid_index = 756
    test_index = 1008
    trade_dates = mask_data.shape[1]
    fea_num = eod_data.shape[2]
    latent_dim = 32

    model = SCGRAN(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        latent_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    max_train_offset = valid_index - lookback_length - steps + 1
    batch_offsets = np.arange(start=1, stop=max_train_offset, dtype=int)
    
    def get_batch(offset=None):
        if offset is None:
            offset = random.randrange(1, max_train_offset)
        seq_len = lookback_length
        m_batch = mask_data[:, offset: offset + seq_len + steps]
        m_batch = np.min(m_batch, axis=1)
        
        return (
            eod_data[:, offset: offset + seq_len, :],
            np.expand_dims(m_batch, axis=1),
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
        )

    def validate(start_index, end_index):
        model.eval()
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            
            loss_sum = 0.0
            for cur_offset in range(
                    start_index - lookback_length - steps + 1,
                    end_index - lookback_length - steps + 1,
            ):
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.Tensor(x).to(device), get_batch(cur_offset)
                )
                pred, alpha_pred, lambda_alpha, lambda_beta = model(data_batch)
                
                cur_loss, cur_reg_loss, cur_rank_loss, cur_alpha_loss, cur_rr = get_scgran_loss(
                    pred, alpha_pred, gt_batch, price_batch, mask_batch, stock_num, 
                    lambda_alpha, lambda_beta, args
                )
                loss_sum += cur_loss.item()
                
                idx = cur_offset - (start_index - lookback_length - steps + 1)
                cur_valid_pred[:, idx] = cur_rr[:, 0].cpu()
                cur_valid_gt[:, idx] = gt_batch[:, 0].cpu()
                cur_valid_mask[:, idx] = mask_batch[:, 0].cpu()
            
            loss_avg = loss_sum / (end_index - start_index)
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        return loss_avg, cur_valid_perf

    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None
    best_epoch = -1
    patience_counter = 0

    print("Starting Training S-CGRAN...")
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        
        for j in range(len(batch_offsets)):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            
            pred, alpha_pred, lambda_alpha, lambda_beta = model(data_batch)
            
            cur_loss, cur_reg_loss, cur_rank_loss, cur_alpha_loss, _ = get_scgran_loss(
                pred, alpha_pred, gt_batch, price_batch, mask_batch, stock_num, 
                lambda_alpha, lambda_beta, args
            )
            
            cur_loss.backward()
            optimizer.step()
            tra_loss += cur_loss.item()
            
        tra_loss = tra_loss / len(batch_offsets)
        
        val_loss, val_perf = validate(valid_index, test_index)
        test_loss, test_perf = validate(test_index, trade_dates)

        print(f"epoch{epoch+1}##########################################################")
        print(f"Train : loss:{tra_loss:.2e}")
        print(f"Valid : loss:{val_loss:.2e}")
        print(f"Test  : loss:{test_loss:.2e}")
        print('Valid performance:\n mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            val_perf['mse'], val_perf['IC'], val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
        print('Test performance:\n mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            test_perf['mse'], test_perf['IC'], test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']))

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save(model.state_dict(), f"scgran_best_{args.dataset}_{args.seed}.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    print("\n================ FINAL BEST PERFORMANCE (Min Valid Loss) ================")
    print(f"Best Epoch: {best_epoch}")
    print('Valid performance:\n mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        best_valid_perf['mse'], best_valid_perf['IC'], best_valid_perf['RIC'], best_valid_perf['prec_10'], best_valid_perf['sharpe5']))
    print('Test performance:\n mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))
    print("=========================================================================\n")

if __name__ == '__main__':
    main()
