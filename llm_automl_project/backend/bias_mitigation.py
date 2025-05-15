from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

def mitigate_bias(X, y, sensitive_features):
    estimator = LogisticRegression(solver='liblinear')
    mitigation = ExponentiatedGradient(estimator, constraints=DemographicParity())
    mitigation.fit(X, y, sensitive_features=sensitive_features)
    return mitigation
