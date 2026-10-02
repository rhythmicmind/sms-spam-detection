import pandas as pd
import pytest
from model import prepare, models

def test_normalized_duplicates_removed():
    df = pd.DataFrame({'label': ['ham', 'ham', 'spam'], 'text': [' Hello  there ', 'hello there', 'Claim prize']})
    assert len(prepare(df)) == 2

def test_conflicting_labels_rejected():
    with pytest.raises(ValueError, match='Conflicting'):
        prepare(pd.DataFrame({'label': ['ham', 'spam'], 'text': ['hello', 'HELLO']}))

def test_classifier_learns_and_predicts():
    model = models()['logistic_regression']
    model.fit(['meet at library', 'class tomorrow', 'win free prize', 'claim cash reward'], ['ham', 'ham', 'spam', 'spam'])
    assert model.predict(['free cash prize'])[0] == 'spam'
