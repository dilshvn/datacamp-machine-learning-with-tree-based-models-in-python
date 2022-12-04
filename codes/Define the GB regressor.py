# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR

# Instantiate gb
gb = GBR(max_depth=4, 
            n_estimators=200,
            random_state=2)