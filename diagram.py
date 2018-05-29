from scipy.sparse import load_npz
import pandas as pd


X = load_npz('data/dummy/uiswf0/X.npz')
df = pd.DataFrame(X.toarray(), dtype=int)
df.to_latex('diagram.tex', index=False)
