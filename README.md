# NeuralLab — Better Streamlit Edition

Interactive UI around the custom from-scratch neural-network framework.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What is visualized
- Training loss from `Model.train()`
- Architecture
- Learned weight matrices
- Cached forward-pass values (`x`, `xb`, `W`, `z`, `y`)
- Final softmax probabilities
- Inspection-only backprop (`dy`, activation derivative, `dz`, `dW`, `dx`)
- Test predictions and confusion matrix

The neural-network computation remains in the custom `Vector`, `Matrix`, `Layer`, `Weight`, and `Model` classes.
