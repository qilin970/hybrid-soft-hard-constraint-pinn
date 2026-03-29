# Constraint and Monotonicity Summary

## Boundary Constraints
- Training Set: negative_ratio=0, min_component_min=0.00411666, sum_mean=1.000000000, sum_range=[0.999999881, 1.000000119], sum_abs_err_mean=3.33085e-08, sum_abs_err_max=1.19209e-07
- Testing Set: negative_ratio=0, min_component_min=0.00433609, sum_mean=1.000000000, sum_range=[0.999999881, 1.000000119], sum_abs_err_mean=3.57628e-08, sum_abs_err_max=1.19209e-07

## Monotonic Compliance (Senior-Style PCD)

- Evaluation uses finite-difference monotonic checking on the real training/testing sets.
- Forward path uses the full model.
- The reported table value is compliance_rate = 1 - violation_rate.

### Against ER
| Data set | N2 | H2 | CO | CO2 | CH4 |
| --- | --- | --- | --- | --- | --- |
| Testing Set | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Training Set | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### Against Temperature
| Data set | H2 | CO | CO2 |
| --- | --- | --- | --- |
| Testing Set | 1.000000 | 1.000000 | 1.000000 |
| Training Set | 1.000000 | 1.000000 | 1.000000 |

## 2D-PDA Trend Check
- H2: Temp compliance=1.0000, ER compliance=1.0000. Overall trend is strongly consistent with the expected monotonic pattern.
- CO: Temp compliance=1.0000, ER compliance=1.0000. Overall trend is strongly consistent with the expected monotonic pattern.
- CO2: Temp compliance=1.0000, ER compliance=1.0000. Overall trend is strongly consistent with the expected monotonic pattern.

## Pairwise Monotonic Training/Evaluation Indicator
- E_M_diff_train = 0.0000000030
- E_M_seniorfd_train = 0.0000000000
- E_M_seniorfd_test = 0.0000000000
- In the senior-injection branch, E_M_diff_train is the training backprop metric on synthetic_samples_physics_branch, while the finite-difference PCD table is the paper reporting metric on real_train_test_sets.
- lambda_p trajectory note: earlier two-point diagnostics confirmed that lambda_p now changes the residual branch trajectory.