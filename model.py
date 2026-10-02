"""Leakage-conscious SMS classification with a held-out evaluation split."""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def key(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text))).strip().casefold()

def prepare(frame):
    if not {'label', 'text'} <= set(frame.columns):
        raise ValueError('Input must contain label and text columns.')
    frame = frame[['label', 'text']].dropna().copy()
    frame['label'] = frame.label.astype(str).str.strip().str.lower()
    if not frame.label.isin(['ham', 'spam']).all():
        raise ValueError('Labels must be ham or spam.')
    frame['key'] = frame.text.map(key)
    frame = frame[frame.key.ne('')]
    if (frame.groupby('key').label.nunique() > 1).any():
        raise ValueError('Conflicting labels found for identical normalized messages.')
    return frame.drop_duplicates('key').reset_index(drop=True)

def models():
    return {
        'majority': Pipeline([('tfidf', TfidfVectorizer()), ('classifier', DummyClassifier(strategy='most_frequent'))]),
        'naive_bayes': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('classifier', MultinomialNB())]),
        'logistic_regression': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
    }

def train(path, output):
    path, output = Path(path), Path(output)
    raw = pd.read_csv(path) if path.suffix == '.csv' else pd.read_csv(path, sep='\t', header=None, names=['label', 'text'])
    frame = prepare(raw)
    if frame.label.nunique() != 2 or frame.label.value_counts().min() < 5:
        raise ValueError('Need at least five distinct messages per class.')
    train_df, test_df = train_test_split(frame, test_size=.2, random_state=42, stratify=frame.label)
    assert set(train_df.key).isdisjoint(test_df.key)
    output.mkdir(parents=True, exist_ok=True)
    report = {'seed': 42, 'raw_rows': len(raw), 'unique_rows': len(frame), 'train_rows': len(train_df), 'test_rows': len(test_df), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'models': {}}
    for name, model in models().items():
        model.fit(train_df.text, train_df.label)
        predictions = model.predict(test_df.text)
        report['models'][name] = {'metrics': classification_report(test_df.label, predictions, output_dict=True, zero_division=0), 'confusion_matrix_ham_spam': confusion_matrix(test_df.label, predictions, labels=['ham', 'spam']).tolist()}
        if name == 'logistic_regression':
            joblib.dump(model, output / 'model.joblib')
            # Fixed model choice; held-out results are not used to select it.
            pd.DataFrame({'text': test_df.text, 'actual': test_df.label, 'predicted': predictions}).query('actual != predicted').to_csv(output / 'errors.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/demo.csv')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    train(args.data, args.output)
