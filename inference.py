import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score, f1_score
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore')

# python inference.py --input-folder "path\to\your\test_folder" --artifacts-folder predictions --output-folder predictions_output

PMU_FILES = {
    'Bus2':  'Bus2_Competition_Data.csv',
    'Bus5':  'Bus5_Competition_Data.csv',
    'Bus6':  'Bus6_Competition_Data.csv',
    'Bus10': 'Bus10_Competition_Data.csv',
    'Bus19': 'Bus19_Competition_Data.csv',
    'Bus22': 'Bus22_Competition_Data.csv',
    'Bus29': 'Bus29_Competition_Data.csv',
    'Bus39': 'Bus39_Competition_Data.csv',
}

PMU_MEASURE_COLS = [
    'va_mag','va_ang','vb_mag','vb_ang','vc_mag','vc_ang',
    'ia_mag','ia_ang','ib_mag','ib_ang','ic_mag','ic_ang',
    'freq','rocof',
]

LABEL_TO_BUS = {
    0: 'N/A (normal)',
    1: 'Bus 39',
    2: 'Bus 24-Bus 23 (line)',
    3: 'Bus 2',
    4: 'Bus 7',
    5: 'Bus 29 (PMU comm. failure)',
    6: 'Bus 29 (missing) + Bus 2 (gen change)',
    7: 'Unknown (bad data)',
    8: 'Unknown',
}

LABEL_TO_BUS_TOP3 = {
    0: ['N/A (normal)'],
    1: ['Bus 39', 'Bus 22', 'Bus 29'],
    2: ['Bus 24-Bus 23 (line)', 'Bus 22', 'Bus 19'],
    3: ['Bus 2', 'Bus 5', 'Bus 6'],
    4: ['Bus 7', 'Bus 5', 'Bus 6'],
    5: ['Bus 29 (PMU comm. failure)', 'Bus 2', 'Bus 39'],
    6: ['Bus 29 (missing) + Bus 2 (gen change)', 'Bus 2', 'Bus 29 (PMU comm. failure)'],
    7: ['Unknown (bad data)', 'Bus 39', 'Bus 2'],
    8: ['Unknown', 'Bus 39', 'Bus 2'],
}

DETERMINISTIC_LABELS = {0, 1, 2, 3, 4, 5, 6}
UNCERTAIN_LABELS = {7, 8}


def load_single_pmu(filepath, bus_name):
    df = pd.read_csv(filepath)
    bus_num = bus_name[3:]
    col_upper = {c.upper(): c for c in df.columns}
    raw_measures = [
        'VA_MAG','VA_ANG','VB_MAG','VB_ANG','VC_MAG','VC_ANG',
        'IA_MAG','IA_ANG','IB_MAG','IB_ANG','IC_MAG','IC_ANG',
        'FREQ','ROCOF',
    ]
    rename = {}
    for meas in raw_measures:
        actual = col_upper.get(f'BUS{bus_num}_{meas}')
        if actual is None:
            actual = col_upper.get(meas)
        if actual:
            rename[actual] = f'{bus_name}_{meas.lower()}'
    dp_actual = col_upper.get('DATA_PRESENT')
    if dp_actual:
        rename[dp_actual] = f'{bus_name}_DATA_PRESENT'
    df = df.rename(columns=rename)
    missing = [f'{bus_name}_{m}' for m in
               ['va_mag','va_ang','vb_mag','vb_ang','vc_mag','vc_ang',
                'ia_mag','ia_ang','ib_mag','ib_ang','ic_mag','ic_ang',
                'freq','rocof'] if f'{bus_name}_{m}' not in df.columns]
    if missing:
        print(f'  [WARN] {bus_name}: missing cols {missing}')
    return df


def load_all_pmus(data_dir):
    print(f'Loading PMU files from: {data_dir}')
    merged = None
    for bus_name, filename in PMU_FILES.items():
        fpath = os.path.join(data_dir, filename)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f'Missing expected PMU file: {filename}')
        print(f'  Loading {filename} for {bus_name}...')
        df = load_single_pmu(fpath, bus_name)
        if merged is None:
            merged = df
        else:
            drop = [c for c in ['Event'] if c in df.columns]
            merged = pd.merge(merged, df.drop(columns=drop),
                              on='TIMESTAMP', how='inner')
    if merged is None:
        raise RuntimeError('No PMU files loaded.')
    merged = merged.sort_values('TIMESTAMP').reset_index(drop=True)
    if 'Event' not in merged.columns:
        merged['Event'] = 0
    return merged


def preprocess(df):
    print('Running preprocessing: imputation and feature engineering...')
    all_meas = [f'{b}_{m}' for b in PMU_FILES for m in PMU_MEASURE_COLS
                if f'{b}_{m}' in df.columns]
    print(f'  Found {len(all_meas)} measurement columns for imputation')
    df[all_meas] = df[all_meas].ffill().bfill().fillna(0.0)

    dp_cols = [c for c in df.columns if c.endswith('_DATA_PRESENT')]
    if dp_cols:
        df['any_missing'] = (df[dp_cols].min(axis=1) == 0).astype(np.int8)
        df['n_missing'] = (df[dp_cols] == 0).sum(axis=1).astype(np.int8)
    else:
        df['any_missing'] = 0
        df['n_missing'] = 0

    freq_cols = [f'{b}_freq' for b in PMU_FILES if f'{b}_freq' in df.columns]
    if freq_cols:
        mean_f = df[freq_cols].mean(axis=1)
        for fc in freq_cols:
            df[fc.replace('_freq', '_freq_dev')] = df[fc] - mean_f

    ang_cols = [f'{b}_va_ang' for b in PMU_FILES if f'{b}_va_ang' in df.columns]
    if ang_cols:
        df['va_ang_spread'] = df[ang_cols].max(axis=1) - df[ang_cols].min(axis=1)
        df['va_ang_std_all'] = df[ang_cols].std(axis=1)

    for b in PMU_FILES:
        if f'{b}_va_mag' in df.columns and f'{b}_ia_mag' in df.columns:
            df[f'{b}_apparent_power'] = df[f'{b}_va_mag'] * df[f'{b}_ia_mag']

    for b in PMU_FILES:
        va = f'{b}_va_mag'; vb = f'{b}_vb_mag'; vc = f'{b}_vc_mag'
        if all(c in df.columns for c in [va, vb, vc]):
            df[f'{b}_unbal_ab'] = df[va] - df[vb]
            df[f'{b}_unbal_ac'] = df[va] - df[vc]
            df[f'{b}_unbal_bc'] = df[vb] - df[vc]

    for b in PMU_FILES:
        va_a = f'{b}_va_ang'; vb_a = f'{b}_vb_ang'; vc_a = f'{b}_vc_ang'
        if all(c in df.columns for c in [va_a, vb_a, vc_a]):
            df[f'{b}_zero_seq_ang'] = (df[va_a] + df[vb_a] + df[vc_a]) / 3.0
    return df


def _compute_stats(wins):
    q75 = np.percentile(wins, 75, axis=1)
    q25 = np.percentile(wins, 25, axis=1)
    X = np.concatenate([
        wins.mean(axis=1),
        wins.std(axis=1),
        wins.min(axis=1),
        wins.max(axis=1),
        np.median(wins, axis=1),
        wins.max(axis=1) - wins.min(axis=1),
        wins[:, -1, :] - wins[:, 0, :],
        np.mean(np.abs(np.diff(wins, axis=1)), axis=1),
        sp_stats.skew(wins, axis=1),
        sp_stats.kurtosis(wins, axis=1),
        q75 - q25,
        (wins ** 2).mean(axis=1),
    ], axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _window_label(label_slice):
    unique, counts = np.unique(label_slice, return_counts=True)
    abn = unique != 0
    if abn.any():
        au, ac = unique[abn], counts[abn]
        return int(au[np.argmin(ac)])
    return int(unique[np.argmin(counts)])


def build_windows_single(data, labels, tsvec, window_size, stride):
    wins = sliding_window_view(data, window_shape=window_size, axis=0)
    wins = wins.transpose(0, 2, 1)
    idx = np.arange(0, wins.shape[0], stride)
    wins = wins[idx]
    X = _compute_stats(wins)
    y = np.array([_window_label(labels[i:i+window_size]) for i in idx], dtype=np.int32)
    half = window_size // 2
    ts = tsvec[idx + half]
    return X, y, ts


def build_windows_multiscale(df, feature_cols, window_sizes, stride):
    print('Building multi-scale sliding-window features...')
    print(f'  Using {len(feature_cols)} feature columns and {len(df)} rows')
    data = df[feature_cols].values.astype(np.float32)
    labels = df['Event'].values.astype(np.int32)
    tsvec = df['TIMESTAMP'].values

    scale_X = []
    scale_y = []
    scale_ts = []
    for ws in window_sizes:
        Xw, yw, tsw = build_windows_single(data, labels, tsvec, ws, stride)
        scale_X.append(Xw)
        scale_y.append(yw)
        scale_ts.append(tsw)

    anchor_ts = scale_ts[0]
    aligned = [scale_X[0]]
    for i in range(1, len(window_sizes)):
        r = np.searchsorted(scale_ts[i], anchor_ts).clip(0, len(scale_ts[i]) - 1)
        l = (r - 1).clip(0)
        bi = np.where(np.abs(anchor_ts - scale_ts[i][l]) <
                      np.abs(anchor_ts - scale_ts[i][r]), l, r)
        aligned.append(scale_X[i][bi])

    X = np.concatenate(aligned, axis=1).astype(np.float32)
    y = scale_y[0]
    ts = anchor_ts
    return X, y, ts


def load_artifacts(artifacts_dir):
    print(f'Loading artifacts from: {artifacts_dir}')
    config_path = os.path.join(artifacts_dir, 'config.pkl')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.pkl not found in {artifacts_dir}')

    config = joblib.load(config_path)
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
    detector = joblib.load(os.path.join(artifacts_dir, 'xgb_detector.pkl'))
    classifier = joblib.load(os.path.join(artifacts_dir, 'lgb_classifier.pkl'))
    le = joblib.load(os.path.join(artifacts_dir, 'label_encoder.pkl'))

    fallback_path = os.path.join(artifacts_dir, 'lgb_fallback_localizer.pkl')
    fallback_localizer = None
    if os.path.exists(fallback_path):
        fallback_localizer = joblib.load(fallback_path)

    metrics_path = os.path.join(artifacts_dir, 'metrics.json')
    metrics_json = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as fh:
            metrics_json = json.load(fh)

    artifact_files = [config_path,
                      os.path.join(artifacts_dir, 'scaler.pkl'),
                      os.path.join(artifacts_dir, 'xgb_detector.pkl'),
                      os.path.join(artifacts_dir, 'lgb_classifier.pkl'),
                      os.path.join(artifacts_dir, 'label_encoder.pkl')]
    if fallback_localizer is not None:
        artifact_files.append(fallback_path)
    total_size = sum(os.path.getsize(p) for p in artifact_files if os.path.exists(p))
    model_size = _human_readable_bytes(total_size)

    return {
        'config': config,
        'scaler': scaler,
        'detector': detector,
        'classifier': classifier,
        'label_encoder': le,
        'fallback_localizer': fallback_localizer,
        'metrics_json': metrics_json,
        'model_size': model_size,
    }


def predict_all(X_scaled, detector, classifier, label_encoder,
                fallback_localizer, best_thresh):
    print(f'Running prediction on {len(X_scaled)} windows')
    print(f'  Detector threshold: {best_thresh}')
    probs = detector.predict_proba(X_scaled)[:, 1]
    is_abn = probs >= best_thresh

    y_pred = np.zeros(len(X_scaled), dtype=np.int32)
    y_pred_bus = ['N/A (normal)'] * len(X_scaled)
    y_pred_bus_top3 = [['N/A (normal)']] * len(X_scaled)

    if is_abn.sum() > 0:
        X_abn = X_scaled[is_abn]
        abn_idxs = np.where(is_abn)[0]

        enc_preds = classifier.predict(X_abn).astype(int)
        decoded_labels = label_encoder.inverse_transform(enc_preds).astype(np.int32)
        y_pred[is_abn] = decoded_labels

        all_buses = sorted(set(LABEL_TO_BUS.values()))
        idx_to_bus = {i: b for i, b in enumerate(all_buses)}

        for i, gi in enumerate(abn_idxs):
            lbl = int(decoded_labels[i])
            if lbl in DETERMINISTIC_LABELS:
                y_pred_bus[gi] = LABEL_TO_BUS[lbl]
                y_pred_bus_top3[gi] = LABEL_TO_BUS_TOP3[lbl]
            else:
                if fallback_localizer is not None:
                    loc_p = fallback_localizer.predict_proba(X_abn[i:i+1])
                    if loc_p.shape[1] < len(all_buses):
                        loc_p = np.hstack([loc_p,
                                           np.zeros((1, len(all_buses) - loc_p.shape[1]))])
                    top3i = np.argsort(loc_p[0])[::-1][:3]
                    y_pred_bus[gi] = idx_to_bus[int(top3i[0])]
                    y_pred_bus_top3[gi] = [idx_to_bus[int(j)] for j in top3i]
                else:
                    y_pred_bus[gi] = LABEL_TO_BUS.get(lbl, 'Unknown')
                    y_pred_bus_top3[gi] = LABEL_TO_BUS_TOP3.get(lbl, ['Unknown'])

    return y_pred, y_pred_bus, y_pred_bus_top3


def get_row_level_predictions(df_full, y_pred_wins, y_bus_wins, ts_wins):
    all_ts = df_full['TIMESTAMP'].values
    win_ts = np.array(ts_wins)
    nearest = np.searchsorted(win_ts, all_ts, side='left').clip(0, len(win_ts) - 1)
    left = (nearest - 1).clip(0)
    use_l = np.abs(all_ts - win_ts[left]) < np.abs(all_ts - win_ts[nearest])
    best = np.where(use_l, left, nearest)
    return pd.DataFrame({
        'TIMESTAMP': all_ts,
        'Predicted_Event': [int(y_pred_wins[i]) for i in best],
        'Predicted_Location': [y_bus_wins[i] for i in best],
    })


def save_predictions(df_full, y_pred_wins, y_bus_wins, ts_wins, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictions_df = get_row_level_predictions(df_full, y_pred_wins, y_bus_wins, ts_wins)
    if 'Event' in df_full.columns:
        predictions_df.insert(1, 'True_Event', df_full['Event'].values)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    pd.DataFrame({
        'TIMESTAMP': ts_wins,
        'Predicted_Event': y_pred_wins,
        'Predicted_Location': y_bus_wins,
    }).to_csv(os.path.join(output_dir, 'predictions_windows.csv'), index=False)

    print(f'Wrote {len(predictions_df):,} rows to {os.path.join(output_dir, "predictions.csv")}')
    print(f'Wrote {len(ts_wins):,} windows to {os.path.join(output_dir, "predictions_windows.csv")}')
    return predictions_df


def save_input_csvs_with_predictions(input_folder, output_dir, predictions_df):
    annotated_dir = os.path.join(output_dir, 'annotated_inputs')
    os.makedirs(annotated_dir, exist_ok=True)
    for bus_name, filename in PMU_FILES.items():
        src_path = os.path.join(input_folder, filename)
        if not os.path.exists(src_path):
            print(f'  [WARN] Input file missing, skipping {filename}')
            continue
        raw = pd.read_csv(src_path)
        if 'TIMESTAMP' not in raw.columns:
            print(f'  [WARN] TIMESTAMP missing in {filename}, skipping annotation')
            continue
        merged = raw.merge(predictions_df[['TIMESTAMP', 'Predicted_Event', 'Predicted_Location']],
                           on='TIMESTAMP', how='left')
        if 'Predicted_Event' not in merged.columns:
            merged['Predicted_Event'] = np.nan
            merged['Predicted_Location'] = np.nan
        dest_path = os.path.join(annotated_dir, filename)
        merged.to_csv(dest_path, index=False)
        print(f'  Annotated CSV saved: {dest_path}')


def compute_accuracy(df_full, predictions_df):
    if 'Event' not in df_full.columns:
        return '?'
    y_true = df_full['Event'].astype(int).values
    y_pred = predictions_df['Predicted_Event'].astype(int).values
    if len(y_true) != len(y_pred):
        return '?'
    return float((y_true == y_pred).mean())


def format_label_distribution(y_pred):
    counts = np.bincount(y_pred.astype(int), minlength=9)
    return ', '.join(f'{i}: {int(counts[i])}' for i in range(9))


def _human_readable_bytes(num_bytes):
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if num_bytes < 1024 or unit == 'GB':
            return f'{num_bytes:.2f} {unit}'
        num_bytes /= 1024.0
    return f'{num_bytes:.2f} bytes'


def build_metrics_table(df_full, predictions_df, test_case, artifacts):
    metrics_json = artifacts.get('metrics_json', {})
    known_accuracy = None
    known_weighted_f1 = metrics_json.get('test_weighted_f1')
    known_macro_f1 = metrics_json.get('test_macro_f1')
    known_detection_f1 = metrics_json.get('test_detection_f1')
    n_params = metrics_json.get('n_params', 18958)

    if 'Event' in df_full.columns:
        y_true = df_full['Event'].astype(int).values
        y_pred = predictions_df['Predicted_Event'].astype(int).values
        if len(y_true) == len(y_pred):
            known_accuracy = float(accuracy_score(y_true, y_pred))
            known_weighted_f1 = float(f1_score(y_true, y_pred, average='weighted'))
            known_macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
            known_detection_f1 = float(f1_score((y_true != 0).astype(int), (y_pred != 0).astype(int), average='binary'))

    if known_accuracy is None:
        accuracy = 'N/A (no ground truth)'
    else:
        accuracy = f'{known_accuracy:.4f}'

    if known_weighted_f1 is None:
        weighted_f1 = 'N/A (no ground truth)'
    else:
        weighted_f1 = f'{known_weighted_f1:.4f}'

    if known_macro_f1 is None:
        macro_f1 = 'N/A (no ground truth)'
    else:
        macro_f1 = f'{known_macro_f1:.4f}'

    if known_detection_f1 is None:
        detection_f1 = 'N/A (no ground truth)'
    else:
        detection_f1 = f'{known_detection_f1:.4f}'

    if 'WINDOW_SIZES' in artifacts['config']:
        window_sizes = artifacts['config'].get('WINDOW_SIZES', [])
        if window_sizes == [30, 60, 90]:
            window_length = '1s/2s/3s multiscale'
        else:
            window_length = f'{window_sizes} rows multiscale'
    else:
        window_length = '1s/2s/3s multiscale'

    if artifacts.get('model_size'):
        model_size = artifacts['model_size']
    else:
        model_size = 'unknown'

    predicted_abnormal = int((predictions_df['Predicted_Event'].astype(int) != 0).sum())
    row_count = len(df_full)
    label_distribution = format_label_distribution(predictions_df['Predicted_Event'].astype(int).values)

    rows = [
        ('Accuracy', accuracy),
        ('F1 Score (weighted)', weighted_f1),
        ('F1 Score (macro)', macro_f1),
        ('Detection F1 (binary normal vs abnormal)', detection_f1),
        ('Model', 'XGBoost detector + LightGBM classifier'),
        ('Trainable parameters', str(n_params)),
        ('Model size', model_size),
        ('Window length', window_length),
        ('Test rows', str(row_count)),
        ('Predicted abnormal rows', str(predicted_abnormal)),
        ('Predicted label distribution', label_distribution),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Value'])


def write_submission_excel(df_orig, predictions_df, output_dir, test_case, artifacts):
    print('[DEBUG] Starting Excel submission workbook generation...')
    submission_dir = os.path.join(output_dir, 'F_Bamidele_SGSMA2026')
    os.makedirs(submission_dir, exist_ok=True)
    filename = f'F_Bamidele_Results_Test{int(test_case)}.xlsx'
    out_path = os.path.join(submission_dir, filename)

    print(f'[DEBUG] Copying original dataframe ({len(df_orig)} rows)...')
    df_out = df_orig.copy()

    if 'Label' in df_out.columns and 'label' not in df_out.columns:
        df_out = df_out.rename(columns={'Label': 'label'})
    if '' in df_out.columns and 'label' not in df_out.columns:
        df_out = df_out.rename(columns={'': 'label'})
    if 'label' not in df_out.columns:
        df_out['label'] = np.nan

    print('[DEBUG] Merging predictions with timestamps...')
    if 'TIMESTAMP' in df_out.columns and 'TIMESTAMP' in predictions_df.columns:
        merged = df_out[['TIMESTAMP']].merge(
            predictions_df[['TIMESTAMP', 'Predicted_Event']],
            on='TIMESTAMP', how='left')
        df_out['label'] = merged['Predicted_Event'].fillna(0).astype(int)
    else:
        if len(df_out) != len(predictions_df):
            raise RuntimeError('Cannot align predictions to original data without TIMESTAMP.')
        df_out['label'] = predictions_df['Predicted_Event'].astype(int)

    print('[DEBUG] Organizing output columns...')
    if 'label' in df_orig.columns:
        cols = list(df_orig.columns)
        if 'label' not in cols:
            cols.append('label')
    else:
        cols = list(df_orig.columns) + ['label']
    df_out = df_out[cols]

    print('[DEBUG] Building metrics table...')
    metrics_df = build_metrics_table(df_orig, predictions_df, test_case, artifacts)

    print(f'[DEBUG] Writing Excel file to {out_path}...')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        print('[DEBUG]   Writing Submission sheet...')
        df_out.to_excel(writer, index=False, sheet_name='Submission')
        print('[DEBUG]   Writing Metrics sheet...')
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

    print(f'Wrote submission workbook: {out_path}')
    return out_path


def build_feature_set(df, config):
    df = preprocess(df)
    config_feature_cols = config.get('feature_cols', [])
    missing_features = [c for c in config_feature_cols if c not in df.columns]
    if missing_features:
        print(f'  [WARN] Filling {len(missing_features)} missing features with zeros')
        for c in missing_features:
            df[c] = 0.0
    feature_cols = [c for c in config_feature_cols if c in df.columns]
    return df, feature_cols


def get_args():
    parser = argparse.ArgumentParser(description='Run SGSMA inference on a folder of PMU CSV files.')
    parser.add_argument('--input-folder', required=True,
                        help='Path to the folder containing the 8 PMU CSV input files.')
    parser.add_argument('--artifacts-folder', default='predictions',
                        help='Path to the folder containing saved pickled artifacts.')
    parser.add_argument('--output-folder', default='predictions_output',
                        help='Path where predictions.csv and predictions_windows.csv will be written.')
    parser.add_argument('--test-case', type=int, default=1,
                        choices=[1, 2],
                        help='Test case number for output workbook name.')
    parser.add_argument('--threshold', type=float,
                        help='Optional override for the detector probability threshold.')
    return parser.parse_args()


def main():
    args = get_args()
    print('Starting inference')
    print(f'Input folder   : {args.input_folder}')
    print(f'Artifacts folder: {args.artifacts_folder}')
    print(f'Output folder  : {args.output_folder}')
    artifacts = load_artifacts(args.artifacts_folder)
    config = artifacts['config']
    threshold = args.threshold if args.threshold is not None else float(config.get('BEST_THRESH', 0.5))
    print(f'Using detection threshold: {threshold}')

    df = load_all_pmus(args.input_folder)
    df, feature_cols = build_feature_set(df, config)
    if not feature_cols:
        raise RuntimeError('No feature columns available after preprocessing.')

    X, _, ts = build_windows_multiscale(df, feature_cols,
                                       window_sizes=config.get('WINDOW_SIZES', [30, 60, 90]),
                                       stride=int(config.get('WINDOW_STRIDE', 15)))
    X_scaled = artifacts['scaler'].transform(X).astype(np.float32)

    y_pred, y_bus, _ = predict_all(
        X_scaled,
        detector=artifacts['detector'],
        classifier=artifacts['classifier'],
        label_encoder=artifacts['label_encoder'],
        fallback_localizer=artifacts['fallback_localizer'],
        best_thresh=threshold,
    )

    row_level_preds = save_predictions(df, y_pred, y_bus, ts, args.output_folder)
    save_input_csvs_with_predictions(args.input_folder, args.output_folder, row_level_preds)
    print('[DEBUG] All annotated CSVs saved. Now generating submission Excel workbook...')
    write_submission_excel(df, row_level_preds, args.output_folder, args.test_case, artifacts)


if __name__ == '__main__':
    main()
