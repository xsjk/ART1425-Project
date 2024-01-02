from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from utils import fit
from preprocess import X_indoor, y_indoor, X_sec_back_t, y_sec_back_t

model_indoor = fit(LinearRegression(), X_indoor, y_indoor)
# model_indoor = fit(SVR(kernel='rbf', C=10), X_indoor, y_indoor)
model_sec_back_t = fit(LinearRegression(), X_sec_back_t, y_sec_back_t)
