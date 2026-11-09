teacher: val entropy 1.151 nats (max 2.079), KL(teacher || marginal) 0.877, collect return 5.23, max-prob median 0.516, states with max-prob > 0.5 0.541

| variant | head | params | best lr | val frac | val KL | train KL | student entropy | max-prob median | max-prob > 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| finetune | mlp128 | 2696 | 0.01 | 0.999±0.000 | 0.001±0.000 | 0.001±0.000 | 1.151±0.004 | 0.517±0.001 | 0.546±0.005 |
| finetune | vqc-nav-id | 117 | 0.01 | 0.854±0.008 | 0.128±0.007 | 0.122±0.005 | 1.407±0.004 | 0.438±0.003 | 0.320±0.011 |
| finetune | vqc-optim-id | 93 | 0.01 | 0.795±0.007 | 0.180±0.006 | 0.167±0.006 | 1.474±0.013 | 0.410±0.005 | 0.245±0.019 |
| finetune | vqc-optim-id-gray | 93 | 0.01 | 0.782±0.034 | 0.191±0.030 | 0.172±0.025 | 1.471±0.021 | 0.411±0.009 | 0.244±0.035 |
| frozen | mlp128 | 2696 | 0.01 | 0.999±0.000 | 0.001±0.000 | 0.001±0.000 | 1.152±0.002 | 0.516±0.001 | 0.544±0.005 |
| frozen | vqc-nav-id | 117 | 0.01 | 0.752±0.015 | 0.217±0.013 | 0.212±0.008 | 1.553±0.015 | 0.374±0.006 | 0.152±0.029 |
| frozen | vqc-optim-id | 93 | 0.01 | 0.647±0.015 | 0.309±0.013 | 0.297±0.006 | 1.613±0.005 | 0.348±0.006 | 0.108±0.007 |
| frozen | vqc-optim-id-gray | 93 | 0.01 | 0.660±0.008 | 0.298±0.007 | 0.294±0.007 | 1.631±0.017 | 0.348±0.008 | 0.081±0.018 |

Verdict (capacity limit if mean+std of the head < 0.5 * (mean-std) of mlp128, same variant):
- finetune vqc-nav-id: head/mlp128 = 0.86 -> no capacity limit
- finetune vqc-optim-id: head/mlp128 = 0.80 -> no capacity limit
- finetune vqc-optim-id-gray: head/mlp128 = 0.78 -> no capacity limit
- frozen vqc-nav-id: head/mlp128 = 0.75 -> no capacity limit
- frozen vqc-optim-id: head/mlp128 = 0.65 -> no capacity limit
- frozen vqc-optim-id-gray: head/mlp128 = 0.66 -> no capacity limit
