import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats

df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

#transforming variables to categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
one_hot_encoder = OneHotEncoder()
encoded_categorical_data = one_hot_encoder.fit_transform(df[categorical_columns]).toarray()
encoded_df = pd.DataFrame(encoded_categorical_data, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, encoded_df], axis=1)

X = df.drop('G3', axis=1).values
y = df['G3'].values

#outer and inner cross-validation
K1, K2 = 10, 10  # Number of folds
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

outer_fold_data_size = []
optimal_hidden_units = []
optimal_regularization_strength = []
ann_generalization_errors = []
ridge_generalization_errors = []
baseline_generalization_errors = []

for fold_num, (train_index, test_index) in enumerate(outer_cv.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    outer_fold_data_size.append(len(X_test))

    best_ridge_score = np.inf
    best_ridge_model = None
    best_ann_score = np.inf
    best_ann_model = None

    #Ridge regression
    for alpha in [0.1, 1, 10]:
        ridge_model = Ridge(alpha=alpha)
        ridge_inner_scores = []
        for inner_train_index, inner_test_index in inner_cv.split(X_train):
            X_inner_train, X_inner_test = X_train[inner_train_index], X_train[inner_test_index]
            y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]

            ridge_model.fit(X_inner_train, y_inner_train)
            y_inner_pred = ridge_model.predict(X_inner_test)
            inner_score = mean_squared_error(y_inner_test, y_inner_pred)
            ridge_inner_scores.append(inner_score)

        average_inner_score = np.mean(ridge_inner_scores)
        if average_inner_score < best_ridge_score:
            best_ridge_score = average_inner_score
            best_ridge_model = ridge_model

    #ANN
    for h in [1, 5, 10]:  #the hidden units
        ann_model = MLPRegressor(hidden_layer_sizes=(h,), random_state=42)
        ann_inner_scores = []
        for inner_train_index, inner_test_index in inner_cv.split(X_train):
            X_inner_train, X_inner_test = X_train[inner_train_index], X_train[inner_test_index]
            y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]

            ann_model.fit(X_inner_train, y_inner_train)
            y_inner_pred = ann_model.predict(X_inner_test)
            inner_score = mean_squared_error(y_inner_test, y_inner_pred)
            ann_inner_scores.append(inner_score)

        average_inner_score = np.mean(ann_inner_scores)
        if average_inner_score < best_ann_score:
            best_ann_score = average_inner_score
            best_ann_model = ann_model

    #baseline model
    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_score = mean_squared_error(y_test, y_pred_baseline)

    #evaluating best models on outer test set
    best_ridge_model.fit(X_train, y_train)
    y_pred_ridge = best_ridge_model.predict(X_test)
    ridge_score = mean_squared_error(y_test, y_pred_ridge)

    best_ann_model.fit(X_train, y_train)
    y_pred_ann = best_ann_model.predict(X_test)
    ann_score = mean_squared_error(y_test, y_pred_ann)

    ridge_generalization_errors.append(ridge_score)
    ann_generalization_errors.append(ann_score)
    baseline_generalization_errors.append(baseline_score)

    optimal_hidden_units.append(best_ann_model.hidden_layer_sizes[0])
    optimal_regularization_strength.append(best_ridge_model.alpha)

print("Table 2: Summary of 2-level 10-fold CV for regression")
print("Outer Fold i\tData Size\tE^test\tANN (h*, Etest)\tLinear regression (λ*, Etest)\tBaseline Etest")
for i in range(K1):
    print(f"{i+1}\t\t{outer_fold_data_size[i]}\t\t{ann_generalization_errors[i]}\t\t({optimal_hidden_units[i]}, {ann_generalization_errors[i]})\t\t\t({optimal_regularization_strength[i]}, {ridge_generalization_errors[i]})\t\t\t{baseline_generalization_errors[i]}")



#paired t-tests
t_stat_ann_ridge, p_val_ann_ridge = stats.ttest_rel(ann_generalization_errors, ridge_generalization_errors)
t_stat_ann_baseline, p_val_ann_baseline = stats.ttest_rel(ann_generalization_errors, baseline_generalization_errors)
t_stat_ridge_baseline, p_val_ridge_baseline = stats.ttest_rel(ridge_generalization_errors, baseline_generalization_errors)

print(f"ANN vs. Ridge: t-statistic = {t_stat_ann_ridge}, p-value = {p_val_ann_ridge}")
print(f"ANN vs. Baseline: t-statistic = {t_stat_ann_baseline}, p-value = {p_val_ann_baseline}")
print(f"Ridge vs. Baseline: t-statistic = {t_stat_ridge_baseline}, p-value = {p_val_ridge_baseline}")

#normality check
_, p_normal_ann = stats.shapiro(ann_generalization_errors)
_, p_normal_ridge = stats.shapiro(ridge_generalization_errors)
_, p_normal_baseline = stats.shapiro(baseline_generalization_errors)

print(f"Normality check - ANN errors: p-value = {p_normal_ann}")
print(f"Normality check - Ridge errors: p-value = {p_normal_ridge}")
print(f"Normality check - Baseline errors: p-value = {p_normal_baseline}")

#If not then  use wilcoxon signed rank test
if p_normal_ann < 0.05 or p_normal_ridge < 0.05 or p_normal_baseline < 0.05:
    print("Normality assumption violated, consider using Wilcoxon signed-rank test.")
    w_stat_ann_ridge, w_p_val_ann_ridge = stats.wilcoxon(ann_generalization_errors, ridge_generalization_errors)
    w_stat_ann_baseline, w_p_val_ann_baseline = stats.wilcoxon(ann_generalization_errors, baseline_generalization_errors)
    w_stat_ridge_baseline, w_p_val_ridge_baseline = stats.wilcoxon(ridge_generalization_errors, baseline_generalization_errors)

    print(f"Wilcoxon - ANN vs. Ridge: statistic = {w_stat_ann_ridge}, p-value = {w_p_val_ann_ridge}")
    print(f"Wilcoxon - ANN vs. Baseline: statistic = {w_stat_ann_baseline}, p-value = {w_p_val_ann_baseline}")
    print(f"Wilcoxon - Ridge vs. Baseline: statistic = {w_stat_ridge_baseline}, p-value = {w_p_val_ridge_baseline}")
