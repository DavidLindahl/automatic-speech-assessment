from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_quali_speech(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset('tsinghua-ee/QualiSpeech', cache_dir=str(cache_dir))


def summarize_split(name: str, df: pd.DataFrame) -> None:
    print(f'=== {name} ===')
    print(f'rows: {len(df)}')
    print('columns:', df.columns.tolist())
    print('\nRating summary:')
    rating_cols = [
        'Speed',
        'Naturalness',
        'Background noise',
        'Distortion',
        'Listening effort',
        'Continuity',
        'Overall quality',
        'Feeling of voice',
    ]
    print(df[rating_cols].describe().transpose())
    print('\nTop 3 by Overall quality:')
    print(df.sort_values('Overall quality', ascending=False).head(3)[['id', 'Overall quality']])
    print('\nBottom 3 by Overall quality:')
    print(df.sort_values('Overall quality', ascending=True).head(3)[['id', 'Overall quality']])


def main() -> None:
    parser = ArgumentParser(description='Analyze the QualiSpeech dataset metadata.')
    parser.add_argument('--cache-dir', default='data-temp/hf_cache', help='Hugging Face cache directory')
    parser.add_argument('--save-csv', action='store_true', help='Export each split to CSV in data-temp')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    dataset = load_quali_speech(cache_dir)

    for split, ds in dataset.items():
        df = ds.to_pandas()
        summarize_split(split, df)
        if args.save_csv:
            out_path = Path('data-temp') / f'QualiSpeech_{split}.csv'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f'Saved: {out_path}\n')


if __name__ == '__main__':
    main()
