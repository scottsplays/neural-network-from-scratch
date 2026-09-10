import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import mathematics
from model import Model
from visualization_helpers import display_values, architecture_figure, layer_figure
from backprop_inspector import backward_snapshot

st.set_page_config(page_title='NeuralLab', page_icon='🧠', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.nl-hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
.nl-kicker {font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; opacity: .65;}
.nl-title {font-size: 2.35rem; font-weight: 750; margin: .15rem 0;}
.nl-sub {font-size: 1rem; opacity: .78;}
.mathbox {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; padding: .8rem 1rem; border-radius: 10px; background: rgba(128,128,128,.08);}
</style>
<div class="nl-hero">
  <div class="nl-kicker">FROM-SCRATCH NEURAL NETWORK LAB</div>
  <div class="nl-title">🧠 NeuralLab</div>
  <div class="nl-sub">Upload or choose a dataset, train your custom Vector → Matrix → Layer → Weight → Model stack, then inspect the actual math inside it.</div>
</div>
''', unsafe_allow_html=True)


def load_builtin_dataset(name):
    bunch = load_iris() if name == 'Iris' else load_wine()
    X = bunch.data.astype(float)
    y = bunch.target.astype(int)
    return X, y, list(bunch.feature_names), [str(name) for name in bunch.target_names]


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def prepare_uploaded_dataset(df, target_column, selected_features):
    if not selected_features:
        raise ValueError('Select at least one numeric feature column.')

    subset = df[selected_features + [target_column]].copy()
    if subset.empty:
        raise ValueError('The uploaded CSV has no rows to train on.')

    if subset[selected_features].isnull().any().any():
        raise ValueError('Selected feature columns contain missing values. Remove or fill them before training.')

    if subset[target_column].isnull().any():
        raise ValueError('The target column contains missing values. Remove or fill them before training.')

    try:
        X = subset[selected_features].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError('All selected feature columns must contain numeric values.') from exc

    if not np.isfinite(X).all():
        raise ValueError('Selected feature columns contain infinite or non-finite values.')

    # Factorize preserves a direct mapping from uploaded labels to output neurons.
    y, unique_labels = pd.factorize(subset[target_column], sort=True)
    if len(unique_labels) < 2:
        raise ValueError('Classification requires at least two distinct target classes.')

    class_names = [str(label) for label in unique_labels.tolist()]
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        raise ValueError('Each target class needs at least two rows so NeuralLab can create a stratified train/test split.')

    return X.astype(float), y.astype(int), list(selected_features), class_names


def predict_proba(model, X):
    return np.array([display_values(model.forward(row.tolist())) for row in X], dtype=float)


def predict_classes(model, X):
    return np.argmax(predict_proba(model, X), axis=1)


def forward_snapshot(model, sample):
    probs = display_values(model.forward(sample.tolist()))
    steps = []
    for i, weight in enumerate(model.weights):
        activation = 'ReLU' if weight.activation_function == mathematics.relu else 'Linear'
        steps.append({
            'layer': f'fc{i+1}',
            'activation': activation,
            'x': display_values(weight.x),
            'xb': display_values(weight.xb),
            'W': display_values(weight.weight_matrix),
            'z': display_values(weight.z),
            'y': display_values(weight.y),
        })
    return probs, steps


def heatmap_figure(matrix, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(matrix, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Output unit')
    ax.set_ylabel('Input unit + bias row')
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    return fig


def confusion_figure(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header('Experiment')
    dataset_name = st.selectbox('Dataset', ['Iris', 'Wine', 'Upload CSV'])

    uploaded_file = None
    uploaded_df = None
    target_column = None
    selected_features = None

    if dataset_name == 'Upload CSV':
        st.caption('Custom CSV mode supports classification with numeric input features and one target column.')
        uploaded_file = st.file_uploader('CSV file', type=['csv'])

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f'Could not read this CSV: {exc}')
                uploaded_df = None

        if uploaded_df is not None and not uploaded_df.empty:
            target_column = st.selectbox('Target column', uploaded_df.columns.tolist())
            available_features = [
                column for column in uploaded_df.select_dtypes(include=np.number).columns.tolist()
                if column != target_column
            ]
            selected_features = st.multiselect(
                'Numeric feature columns',
                available_features,
                default=available_features
            )

            st.caption(
                f'{len(uploaded_df):,} rows · {len(selected_features)} selected features'
            )

    hidden_size = st.slider('Hidden units', 3, 24, 8)
    epochs = st.slider('Epochs', 10, 300, 100, 10)
    learning_rate = st.select_slider('Learning rate', options=[0.001, 0.003, 0.01, 0.03, 0.05], value=0.01)
    test_size = st.slider('Test split', 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input('Split seed', 0, 9999, 42)
    train_clicked = st.button('Train from scratch', type='primary', use_container_width=True)

    if st.button('Clear trained model', use_container_width=True):
        for key in ['model','losses','X_train','X_test','y_train','y_test','features','classes','scaler','dataset_name','hidden_size']:
            st.session_state.pop(key, None)
        st.rerun()


if dataset_name == 'Upload CSV':
    if uploaded_df is not None:
        st.subheader('Uploaded dataset preview')
        st.dataframe(uploaded_df.head(20), use_container_width=True, hide_index=True)

        if target_column is not None:
            preview_left, preview_mid, preview_right = st.columns(3)
            preview_left.metric('Rows', f'{len(uploaded_df):,}')
            preview_mid.metric('Columns', len(uploaded_df.columns))
            preview_right.metric('Detected classes', uploaded_df[target_column].nunique(dropna=True))
            st.caption('Only numeric columns are offered as neural-network inputs in this version. The target may be text, integer, or another categorical label.')

    if uploaded_df is None and 'model' not in st.session_state:
        st.info('Upload a CSV in the sidebar. Then choose the target and numeric feature columns you want NeuralLab to train on.')
else:
    X, y, feature_names, class_names = load_builtin_dataset(dataset_name)


if train_clicked:
    try:
        if dataset_name == 'Upload CSV':
            if uploaded_df is None:
                raise ValueError('Upload a CSV file before training.')
            if target_column is None:
                raise ValueError('Choose a target column before training.')
            X, y, feature_names, class_names = prepare_uploaded_dataset(
                uploaded_df, target_column, selected_features or []
            )
        else:
            X, y, feature_names, class_names = load_builtin_dataset(dataset_name)

        n_classes = len(np.unique(y))
        n_samples = len(y)
        test_count = max(int(np.ceil(n_samples * float(test_size))), n_classes)
        train_count = n_samples - test_count
        if test_count < n_classes or train_count < n_classes:
            raise ValueError(
                f'This dataset is too small for a stratified {float(test_size):.0%} test split across {n_classes} classes. '
                'Add more rows or adjust the split.'
            )

        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y, test_size=float(test_size), random_state=int(seed), stratify=y
        )

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        y_train_oh = one_hot(y_train_int, n_classes)
        model = Model(X.shape[1], int(hidden_size), n_classes)

        progress = st.progress(0, text='Training your custom network…')

        def on_epoch(epoch, loss):
            progress.progress(min(epoch / epochs, 1.0), text=f'Epoch {epoch}/{epochs} · loss {loss:.4f}')

        losses = model.train(
            X_train.tolist(), y_train_oh.tolist(), num_epochs=int(epochs),
            lr=float(learning_rate), visualize=False, on_epoch=on_epoch
        )
        progress.empty()

        st.session_state.update(
            model=model, losses=losses, X_train=X_train, X_test=X_test,
            y_train=y_train_int, y_test=y_test_int, features=feature_names,
            classes=class_names, scaler=scaler, dataset_name=dataset_name,
            hidden_size=int(hidden_size)
        )

    except ValueError as exc:
        st.error(str(exc))


if 'model' not in st.session_state:
    st.info('Choose settings in the sidebar and click **Train from scratch**. The app uses your custom Model, Weight, Layer, Matrix, and Vector classes for the neural-network computation.')
    st.stop()

model = st.session_state.model
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test
features = st.session_state.features
classes = st.session_state.classes
losses = st.session_state.losses

train_pred = predict_classes(model, X_train)
test_probs = predict_proba(model, X_test)
test_pred = np.argmax(test_probs, axis=1)
train_acc = float(np.mean(train_pred == y_train))
test_acc = float(np.mean(test_pred == y_test))

m1, m2, m3, m4 = st.columns(4)
m1.metric('Training samples', len(X_train))
m2.metric('Test samples', len(X_test))
m3.metric('Train accuracy', f'{train_acc:.1%}')
m4.metric('Test accuracy', f'{test_acc:.1%}')

st.caption(f"Trained: {st.session_state.dataset_name} · {len(features)} inputs → {st.session_state.hidden_size} → {st.session_state.hidden_size} → {len(classes)} outputs")

tab_train, tab_arch, tab_weights, tab_math, tab_backprop, tab_predict = st.tabs([
    '📈 Training', '🕸️ Architecture', '🌡️ Weights', '🔬 Forward Math', '↩️ Backprop', '🎯 Predictions'
])

with tab_train:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader('Loss over training')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(losses)+1), losses)
        ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Online SGD training loss')
        ax.grid(alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader('What your code is doing')
        st.markdown('''
<div class="mathbox">
for each epoch:<br>
&nbsp;&nbsp;for each sample:<br>
&nbsp;&nbsp;&nbsp;&nbsp;forward(x)<br>
&nbsp;&nbsp;&nbsp;&nbsp;CrossEntropy(y, p)<br>
&nbsp;&nbsp;&nbsp;&nbsp;backward(dy)<br>
&nbsp;&nbsp;&nbsp;&nbsp;W ← W − η dW
</div>
''', unsafe_allow_html=True)
        st.write('The loss curve comes from `Model.train()`; the Streamlit layer only records and displays the returned epoch losses.')

with tab_arch:
    st.subheader('Network architecture')
    st.pyplot(architecture_figure([len(features), st.session_state.hidden_size, st.session_state.hidden_size, len(classes)]), clear_figure=True)
    st.latex(r'x_b=[x;1]\qquad z=x_bW\qquad y=f(z)')
    st.caption('Bias is represented as an extra input row in each weight matrix. The final linear layer is followed by softmax in Model.forward().')

with tab_weights:
    st.subheader('Learned weight matrices')
    for i, weight in enumerate(model.weights):
        W = display_values(weight.weight_matrix)
        with st.expander(f'fc{i+1} · shape {W.shape}', expanded=(i == 0)):
            st.pyplot(heatmap_figure(W, f'fc{i+1} learned weights'), clear_figure=True)
            st.dataframe(pd.DataFrame(W).round(4), use_container_width=True)

with tab_math:
    st.subheader('Forward-pass Math Inspector')
    sample_index = st.slider('Test sample', 0, len(X_test)-1, 0, key='forward_sample')
    sample = X_test[sample_index]
    probs, steps = forward_snapshot(model, sample)
    predicted = int(np.argmax(probs))
    true = int(y_test[sample_index])
    c1, c2 = st.columns(2)
    c1.metric('Prediction', classes[predicted])
    c2.metric('True class', classes[true])
    st.write('**Standardized input**')
    st.dataframe(pd.DataFrame([sample], columns=features).round(4), use_container_width=True)

    for step in steps:
        with st.expander(f"{step['layer']} · {step['activation']}", expanded=True):
            st.pyplot(layer_figure(step), clear_figure=True)
            st.latex(r'z=x_bW')
            a, b = st.columns(2)
            with a:
                st.write('**Biased input  xᵦ**')
                st.dataframe(pd.DataFrame([step['xb']]).round(4), use_container_width=True)
                st.write('**Pre-activation  z**')
                st.dataframe(pd.DataFrame([step['z']]).round(4), use_container_width=True)
            with b:
                st.write('**Weight matrix  W**')
                st.dataframe(pd.DataFrame(step['W']).round(4), use_container_width=True)
                st.write('**Activated output  y**')
                st.dataframe(pd.DataFrame([step['y']]).round(4), use_container_width=True)
    st.write('**Final softmax probabilities**')
    st.dataframe(pd.DataFrame([probs], columns=classes).round(5), use_container_width=True)

with tab_backprop:
    st.subheader('Backprop Inspector')
    st.caption('Inspection runs on a deep copy of the trained model, so viewing gradients does not change the trained weights.')
    bp_index = st.slider('Test sample', 0, len(X_test)-1, 0, key='bp_sample')
    target = one_hot(np.array([y_test[bp_index]]), len(classes))[0]
    bp_loss, bp_steps = backward_snapshot(model, X_test[bp_index].tolist(), target.tolist())
    st.metric('Sample cross-entropy loss', f'{bp_loss:.5f}')
    st.latex(r'\delta z=f\'(z)\odot\delta y\qquad dW=x_b^T\delta z\qquad \delta x=(\delta zW^T)_{drop\ bias}')
    for step in bp_steps:
        with st.expander(step['layer'], expanded=(step['layer'] == 'fc3')):
            a, b, c, d = st.columns(4)
            a.metric('||dy||', f"{np.linalg.norm(step['dy']):.4f}")
            b.metric('||dz||', f"{np.linalg.norm(step['dz']):.4f}")
            c.metric('||dW||', f"{np.linalg.norm(step['dW']):.4f}")
            d.metric('||dx||', f"{np.linalg.norm(step['dx']):.4f}")
            st.pyplot(heatmap_figure(step['dW'], f"{step['layer']} gradient dW"), clear_figure=True)
            st.write('**dy**', step['dy'])
            st.write('**f\'(z)**', step['derivative'])
            st.write('**dz**', step['dz'])
            st.write('**dx**', step['dx'])

with tab_predict:
    st.subheader('Test-set predictions')
    cm = confusion_matrix(y_test, test_pred, labels=np.arange(len(classes)))
    left, right = st.columns([1, 1.35])
    with left:
        st.pyplot(confusion_figure(cm, classes), clear_figure=True)
    with right:
        rows = []
        for i in range(len(X_test)):
            rows.append({
                'sample': i,
                'actual': classes[int(y_test[i])],
                'predicted': classes[int(test_pred[i])],
                'confidence': float(np.max(test_probs[i]))
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
