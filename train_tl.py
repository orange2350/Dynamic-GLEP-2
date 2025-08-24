import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from deepchem.data import DiskDataset

from model.DNN import DNNNet_mean


# ======================
# Utility Functions
# ======================
def parse_to_integers(data_string):
    return [float(num) for num in data_string.split(',')]


def scale_data(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = [scaler.transform(bag) for bag in x_train]
    x_test_scaled = [scaler.transform(bag) for bag in x_test]
    return np.array(x_train_scaled), np.array(x_test_scaled)


# ======================
# Main
# ======================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create save dir
    save_dir = os.path.join(
        "./model_save",
        f"EquiScore_{args.target}_DNN_mean"
    )
    os.makedirs(save_dir, exist_ok=True)

    # ======================
    # Load data
    # ======================
    # input features
    df_1 = pd.read_csv(
        f"../data/5-features/"
        f"{args.target}_EquiScore_leadopt_LAST_SECOND_label_1_fillwith0.csv"
    )

    # proteins list
    proteins_df = pd.read_csv(
        f"../data/3-deduplicated/Proteins_rrcs_{args.target}.csv"
    )

    valid_columns = [protein for protein in proteins_df['Proteins'] if protein in df_1.columns]
    df = df_1[valid_columns]
    features_df = pd.concat([df_1[['Smiles']], df], axis=1)
    smilesList = features_df['Smiles']
    LabelList = df_1['Label']
    features_df.drop(['Smiles'], axis=1, inplace=True)
    features_df_train = np.array(features_df.values)

    # training configs
    n_splits = args.splits
    n_repeats = args.repeats
    batch_size = args.batch_size
    num_epochs = args.epochs

    # ======================
    # Cross-validation
    # ======================
    fold_test_results = [[] for _ in range(n_splits)]

    for repeat in range(n_repeats):
        for i in range(n_splits):
            train = DiskDataset(
                f"../data/2-splits/20241127_split_{args.target}_ecfp_2048/czy_fpsplit_train_{i}"
            )
            test = DiskDataset(
                f"../data/2-splits/20241127_split_{args.target}_ecfp_2048/czy_fpsplit_test_{i}"
            )

            training_index = []
            for s, v in zip(smilesList, LabelList):
                if s in train.ids:
                    training_index.append("Training")
                elif s in test.ids:
                    training_index.append("Testing")
            training_index = np.array(training_index)

            x_train = features_df_train[training_index == "Training"]
            y_train = LabelList[training_index == "Training"]
            x_test = features_df_train[training_index == "Testing"]
            y_test = LabelList[training_index == "Testing"]
            x_test_smile = smilesList[training_index == "Testing"]
            x_test_emax = LabelList[training_index == "Testing"]

            # convert features
            x_train_data = np.array([ [parse_to_integers(item) for item in group] for group in x_train ])
            x_test_data = np.array([ [parse_to_integers(item) for item in group] for group in x_test ])

            # scale
            x_train_data, x_test_data = scale_data(x_train_data, x_test_data)
            x_train_new = x_train_data.mean(axis=1)
            x_test_new = x_test_data.mean(axis=1)

            # pytorch dataset
            train_dataset = TensorDataset(
                torch.tensor(x_train_new, dtype=torch.float32).to(device),
                torch.tensor(y_train.values, dtype=torch.float32).to(device),
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            test_dataset = TensorDataset(
                torch.tensor(x_test_new, dtype=torch.float32).to(device),
                torch.tensor(y_test.values, dtype=torch.float32).to(device),
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # model
            model = DNNNet_mean().to(device)
            criterion = nn.BCELoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            losses, ex_losses, accuracy_train, accuracy_val = [], [], [], []

            # ======================
            # Train
            # ======================
            for epoch in range(num_epochs):
                model.train()
                epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

                for x_batch, y_batch in train_loader:
                    outputs = model(x_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    predicted = (outputs > 0.5).float()
                    epoch_total += y_batch.size(0)
                    epoch_correct += (predicted == y_batch).sum().item()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(train_loader)
                train_accuracy = 100 * epoch_correct / epoch_total
                losses.append(avg_epoch_loss)

                # validation
                model.eval()
                test_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        outputs = model(x_batch)
                        loss = criterion(outputs.squeeze(), y_batch)
                        test_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()

                avg_test_loss = test_loss / len(test_loader)
                val_accuracy = 100 * val_correct / val_total
                ex_losses.append(avg_test_loss)
                accuracy_train.append(train_accuracy)
                accuracy_val.append(val_accuracy)

                print(
                    f'Target {args.target} | Repeat {repeat+1}/{n_repeats}, Fold {i+1}/{n_splits}, '
                    f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_epoch_loss:.4f}, '
                    f'Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_test_loss:.4f}, '
                    f'Val Acc: {val_accuracy:.2f}%'
                )

            # save model
            best_model_state = model.state_dict()
            best_model_path = os.path.join(
                save_dir,
                f"best_model_{args.target}_repeat{repeat}_fold{i}.pth"
            )
            torch.save(best_model_state, best_model_path)

            # ======================
            # Final evaluation
            # ======================
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                outputs_all, labels_all = [], []
                for x_batch, y_batch in test_loader:
                    outputs = model(x_batch)
                    outputs_all.append(outputs.cpu().numpy())
                    labels_all.append(y_batch.cpu().numpy())
                outputs_all = np.concatenate(outputs_all, axis=0)
                labels_all = np.concatenate(labels_all, axis=0)
                auc = roc_auc_score(labels_all, outputs_all)
                print(f"Target {args.target} | Repeat {repeat+1}, Fold {i+1}, AUC: {auc:.4f}")

                test_result = pd.DataFrame({
                    'Smiles': x_test_smile,
                    'Emax_Label': labels_all,
                    'Label': x_test_emax,
                    'Predicted_Probability': outputs_all.squeeze(),
                    'Repeat': repeat + 1
                })
                fold_test_results[i].append(test_result)

    # ======================
    # Save results
    # ======================
    for fold in range(n_splits):
        all_test_results_fold = pd.concat(fold_test_results[fold], ignore_index=True)
        pivoted_results = all_test_results_fold.pivot_table(
            index=['Smiles', 'Emax_Label', 'Label'],
            columns='Repeat',
            values='Predicted_Probability'
        ).reset_index()
        pivoted_results.columns = ['Smiles', 'Emax_Label', 'Label'] + [
            f'{i}' for i in range(1, n_repeats + 1)
        ]
        out_dir = f'../data/6-test_results/Equi2_dnn_fp_{args.target}/'
        os.makedirs(out_dir, exist_ok=True)
        pivoted_results.to_csv(
            os.path.join(out_dir, f'{args.target}_results_fold{fold}.csv'),
            index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Target name, e.g., 5ht1a or a2a")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    main(args)
