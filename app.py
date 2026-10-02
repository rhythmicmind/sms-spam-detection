from pathlib import Path
import joblib
import streamlit as st
st.set_page_config(page_title='SMS Spam Detection', page_icon='✉️')
st.title('SMS Spam Detection')
st.caption('TF-IDF + logistic regression. Educational baseline; English SMS only.')
path = Path('results/model.joblib')
if not path.exists():
    st.info('Train the model first: python model.py --data data/demo.csv')
    st.stop()
@st.cache_resource
def load_model(stamp):
    return joblib.load(path)
model = load_model(path.stat().st_mtime_ns)
text = st.text_area('Message', 'Are we meeting at the library today?')
if st.button('Classify') and text.strip():
    label = model.predict([text])[0]
    classes = list(model.classes_)
    score = model.predict_proba([text])[0][classes.index('spam')]
    st.metric('Prediction', label.upper())
    st.write(f'Model spam score: {score:.3f} (not a calibrated guarantee).')
st.warning('Demo training data is synthetic. Use the UCI dataset for meaningful evaluation.')
