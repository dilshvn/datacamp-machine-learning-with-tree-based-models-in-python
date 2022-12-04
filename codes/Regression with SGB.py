    # Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR

# Instantiate sgbr
sgbr = GBR(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)