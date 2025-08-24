import os
import pandas as pd
import torch
import numpy as np
import argparse
from model.DNN import DNNNet_mean
from sklearn.preprocessing import MinMaxScaler
from deepchem.data import DiskDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_to_integers(data_string):
    """Convert comma-separated string to list of floats"""
    return [float(num) for num in data_string.split(',')]


def inference_model(model, test_features, smiles_list, label_list, device):
    """Run inference using trained model"""
    model.eval()
    with torch.no_grad():
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
        outputs = model(test_features_tensor)
        probabilities = outputs.squeeze().cpu().numpy()
        inference_results = pd.DataFrame({
            'Smiles': smiles_list,
            'Label': label_list,
            '1': probabilities
        })
    return inference_results


def scale_data(train_data, external_data=None, external=False):
    """Scale training and external data using MinMaxScaler"""
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_data))  
    train_data_scaled = train_data.copy()
    if external:
        if external_data is not None:
            external_data_scaled = external_data.copy()
            for i, molecule in enumerate(external_data):
                external_data_scaled[i] = scaler.transform(molecule.reshape(-1, molecule.shape[-1]))
            return np.array(external_data_scaled)
    else:
        for i, molecule in enumerate(train_data):
            train_data_scaled[i] = scaler.transform(molecule.reshape(-1, molecule.shape[-1]))
        return np.array(train_data_scaled)


def main(args):
    # =====================
    # Load training data
    # =====================
    df_train = pd.read_csv(
        f'../data/5-features/{args.target}_EquiScore_leadopt_LAST_SECOND_label_1_fillwith0.csv'
    )

    # Load selected proteins for the target
    proteins_df = pd.read_csv(
        f'../data/3-deduplicated/Proteins_rrcs_{args.target}.csv'
    )
    print(proteins_df)

    # Keep only valid protein columns
    valid_columns = [protein for protein in proteins_df['Proteins'] if protein in df_train.columns]
    df_train_rrcs = df_train[valid_columns]
    features_df_train = pd.concat([df_train[['Smiles']], df_train_rrcs], axis=1)
    smilesList_train = features_df_train['Smiles']
    LabelList_train = df_train['Label']
    features_df_train.drop(['Smiles'], axis=1, inplace=True)

    # Load train/test split
    train = DiskDataset(f"../data/2-splits/20241127_split_{args.target}_ecfp_2048/czy_fpsplit_train_1")
    test = DiskDataset(f"../data/2-splits/20241127_split_{args.target}_ecfp_2048/czy_fpsplit_test_1")

    training_index = []
    for s, v in zip(smilesList_train, LabelList_train):
        if s in train.ids:
            training_index.append("Training")
        elif s in test.ids:
            training_index.append("Test")

    training_index = np.array(training_index)
    features_df_train = np.array(features_df_train)
    LabelList = np.array(LabelList_train)

    train_features = features_df_train[training_index == "Training"]
    results_train = []
    for group in train_features:
        list_of_lists = [parse_to_integers(item) for item in group]
        results_train.append(list_of_lists)
    train_features_split = np.array(results_train)

    # =====================
    # Load external dataset
    # =====================
    df_1 = pd.read_csv(
        f'../data/5-features/{args.target}_external_EquiScore_leadopt_LAST_SECOND_label_1_fillwith0.csv'
    )
    df = df_1[valid_columns]
    LabelList_ex = df_1['Label']
    features_df = pd.concat([df_1[['SMILES']], df], axis=1)
    smilesList_ex = features_df['SMILES']
    features_df.drop(['SMILES'], axis=1, inplace=True)
    features_df = np.array(features_df)

    results_ex = []
    for group in features_df:
        list_of_lists = [parse_to_integers(item) for item in group]
        results_ex.append(list_of_lists)
    test_features_ex = np.array(results_ex)

    # Scale external features using training data scaler
    test_features_ex = scale_data(train_features_split, external_data=test_features_ex, external=True)
    test_features_ex = test_features_ex.mean(axis=1)

    # =====================
    # Model paths (ensemble models)
    # =====================
    model_paths = [
        f"../data/7.model_save/EquiScore_{args.target}_DNN_mean/best_model_EquiScore_DNN_last_second_fp_leapopt_full_{i}_k_1_best_AUCepoch_25.pth"
        for i in range(5)
    ]

    # =====================
    # Inference with ensemble
    # =====================
    all_inference_results = []
    all_probabilities = []

    for model_path in model_paths:
        model = DNNNet_mean()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        inference_results = inference_model(model, test_features_ex, smilesList_ex, LabelList_ex, device)
        all_inference_results.append(inference_results)
        all_probabilities.append(inference_results['1'].values)

    # Average predictions
    mean_probabilities = np.mean(all_probabilities, axis=0)

    final_inference_results = pd.DataFrame({
        'Smiles': smilesList_ex,
        'Label': LabelList_ex,
        '1': mean_probabilities
    })

    # Save results
    output_path = f"../data/8.external_results/inference_results_mean_{args.target}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_inference_results.to_csv(output_path, index=False)
    print(f"Inference results with mean predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Target name, e.g., 5ht1a_holo")
    args = parser.parse_args()
    main(args)
