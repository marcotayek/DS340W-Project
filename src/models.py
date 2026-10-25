from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def ridge(alpha=1.0):
    return Ridge(alpha=alpha)

def lasso(alpha=0.01):
    return Lasso(alpha=alpha)

def random_forest(random_state=42):
    return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
