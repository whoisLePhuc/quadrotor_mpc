# CC-MPC Benchmark Results

| Scenario | Solver | FOV | Goal? | Steps | Avg ms | P95 ms | Failures | CTE RMSE | HDG RMSE |
|----------|--------|-----|-------|-------|--------|--------|----------|----------|----------|
| simulation | CLARABEL | OFF | ✅ |   247 |     49 |     49 |        1 | 0.685 | 13.3 |
| simulation | OSQP    | OFF | ✅ |   251 |     79 |    167 |        2 | 0.682 | 12.4 |
| simulation_static | CLARABEL | OFF | ❌ |   500 |     45 |     47 |        2 | 0.401 | 10.9 |
| simulation_static | OSQP    | OFF | ❌ |   500 |     64 |    132 |        3 | 0.393 | 11.0 |
| simulation_corridor | CLARABEL | OFF | ✅ |   203 |     48 |     48 |        3 | 1.670 | 8.0 |
| simulation_corridor | OSQP    | OFF | ✅ |   206 |    109 |    225 |        4 | 1.683 | 7.7 |
