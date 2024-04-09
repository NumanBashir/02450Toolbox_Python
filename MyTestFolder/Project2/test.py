from scipy.stats import shapiro
import numpy as np

#the errors from each model
ridge_errors = np.array([
    22.7049668826052, 12.786867782924244, 20.190853964423585, 17.415655555396203,
    19.184680726565098, 15.167257039063006, 18.369584942071846, 25.418799876380863,
    19.002821159976037, 14.286420187454913
])
ann_errors = np.array([
    22.782557632824705, 13.652666452713698, 19.57823134509192, 15.851320363550192,
    15.969634764073543, 12.637179306376114, 15.391110410067444, 26.666151552543433,
    16.651912872915016, 13.785404075757075
])
baseline_errors = np.array([
    24.69449414798651, 16.245978972426105, 24.87393671890498, 21.679445546518547,
    21.525616941083122, 12.040113872892247, 22.734521832583948, 30.008184791158847,
    19.49441763051156, 16.683446955350757
])

#differences between the errors
differences_ann_ridge = ann_errors - ridge_errors
differences_ann_baseline = ann_errors - baseline_errors
differences_ridge_baseline = ridge_errors - baseline_errors

#Shapiro-Wilk test to check normality
normality_ann_ridge = shapiro(differences_ann_ridge)
normality_ann_baseline = shapiro(differences_ann_baseline)
normality_ridge_baseline = shapiro(differences_ridge_baseline)

print(normality_ann_ridge, normality_ann_baseline, normality_ridge_baseline)
