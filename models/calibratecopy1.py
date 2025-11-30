from sklearn.calibration import CalibratedClassifierCV

def calibrate_platt_cv(X_train, y_train, estimator):
    calibrator = CalibratedClassifierCV(estimator=estimator, cv=5, method='sigmoid')
    calibrator.fit(X_train, y_train)
    return calibrator

def calibrate_isotonic_cv(X_train, y_train, estimator):
    calibrator = CalibratedClassifierCV(estimator=estimator, cv=5, method='isotonic')
    calibrator.fit(X_train, y_train)
    return calibrator