# MLWC RIS-assisted ISAC reproduction

Course implementation of Wei et al., *Machine Learning-Driven Joint RIS Partitioning, Beamforming, and Phase Shifts Design for mmWave ISAC Systems*.

## Run

From this folder, install the packages in `requirements.txt` if needed and run:

```powershell
python -m pip install -r requirements.txt
python train_ddpg.py --steps 1000 --eval-trials 100 --seed 0
```

The runner logs mean reward every 100 steps, saves per-step rewards and actor weights in `run/`, and evaluates the actor and a random-action reference on the same held-out CSI. It reports sum rate, the fractions of draws meeting every user or target threshold, and the fraction meeting both. `--antennas` and `--ris-side` let you rerun with different array sizes.

The original `kmeans_clustering.py` and `RIS_Phase_rate.py` can still be run independently.

## Model and implementation choices

- The simulator uses the paper's default coordinates, path-loss parameters, noise power, power budget, phase resolution, and QoS thresholds. It reuses the starter code's three paths per link and independently sampled path angles. A draw uses a common full-aperture channel, restricted to each assigned rectangle.
- NCC grouping runs once per draw. Equations (3)–(5) define one beamforming column per CU and one per sensing stream. The environment evaluates all other streams as interference at each CU and computes target illumination as the sum of received power across the streams.
- The geometry decoder uses recursive axis-aligned cuts guided by centers, desired sizes, and priorities. This is a feasible approximation to the paper's priority projection and rectangular expansion. Each active group gets a rectangle. A zero-sized sensing proposal allows all sensing to use the direct links.
- The action decoder enforces the transmit-power cap and quantizes phases. Rates and sensing targets remain soft constraints in the reward from Eq. (8). The sensing deficit is evaluated in watts, as in that equation.
- The training harness uses two 128-unit hidden layers, a persistent stream of independent CSI draws, Gaussian exploration, and a replay buffer. Default learning rates, target update coefficient, discount, and penalty weights use the reported values.

## Validation and reproduction status

The environment passed 400 partition checks spanning four grid shapes, including optional sensing allocation. A hand-calculated two-stream example checked both interference and target illumination. Additional checks covered discrete phase levels and the transmit-power cap. Both starter scripts ran successfully.

A 1,000-step run with 100 held-out draws stayed near a mean reward of -30. Both the trained actor and random reference had 0% joint QoS satisfaction. In these draws the simulator's per-link upper bounds were below threshold for at least one CU and for at least one target every time. Each bound gives a link the full aperture and full power budget, ignores interference, and uses the triangle inequality to bound the magnitude of the combined channel. This identifies infeasible draws under the current channel model even with ideal resource allocation.

Figure 2 remains open. First reconcile the simulated link budget with the paper, especially the channel and array-gain conventions. Then run convergence and array-size sweeps with the paper's AO benchmarks and imperfect-CSI evaluation. The random-action reference is a smoke check.

## Link-budget and Figure 2(c) pilot

```powershell
python reproduction_pilot.py --trials 100 --steps 192
```

This command runs the link-budget test on the default 36-element RIS, then compares three policies at 9, 16, 25, 36, 49, 64, and 100 elements. Each size uses 100 paired held-out CSI draws and 16 BS antennas. The fixed-partition heuristic divides the RIS into four rectangles, aligns each communication rectangle to its NCC center, and aligns the sensing rectangle to the strongest direct target. The shared-RIS heuristic aligns the full surface to the strongest NCC center. Both use regularized zero forcing for communication, MRT for sensing, and equal power per stream. These are diagnostic heuristics. The paper's alternating-optimization benchmarks and imperfect-CSI evaluation remain open.

| Default-size link check | Median over 100 draws | Required |
| --- | ---: | ---: |
| One CU received power with full RIS and MRT | 2.12e-15 W | 4.14e-13 W |
| One CU rate with zero interference | 0.00306 bit/s/Hz | 0.5 bit/s/Hz |
| One target, direct-only MRT | 1.43e-8 W | 3.00e-7 W |
| One target, reflected-only MRT | 2.47e-15 W | 3.00e-7 W |
| One target, combined direct and reflected MRT | 1.43e-8 W | 3.00e-7 W |

Each diagnostic uses the first sampled user or target from each default-size draw. For reflected paths it uses eight alternating beam and 3-bit phase updates. Communication uses sigma squared = 1e-12 W from -90 dBm, Pmax = 1 W from 30 dBm, and a required interference-free received power of (2^0.5 - 1) times the noise power. The median communication signal is 22.9 dB short and the median combined illumination is 13.2 dB short. The code uses the paper's unit-magnitude half-wavelength steering vectors and 1/sqrt(L) path-sum scaling. The shortfalls apply to the current channel implementation, which retains the starter's angle and path-count assumptions.

At 36 elements the mean sum rates were 0.000273 bit/s/Hz for the fixed partition, 0.003161 for the shared surface, and 0.000026 for the 192-step DDPG pilot. All three rates increased across the full sweep, and none of the 100 draws at any size passed every user and sensing constraint for any method. See [the pilot plot](ris_size_pilot.png). The gap to Fig. 2(c) remains open. These results support checking channel calibration before longer training.

## Step 1. Equation-to-code audit

Run the audit from this folder:

```powershell
python channel_audit/check_link_budget.py
```

The script checks the following in order.

| Check | Why it matters | Result |
| --- | --- | --- |
| Half-wavelength ULA and UPA steering vectors, including nonzero angles and row-major element order | An incorrect phase progression or element order misaligns the RIS weights with their channels. | Passed. The steering vectors have unit-magnitude entries, as in the paper. |
| Section IV power units and distance law | A dB/dBm conversion error or treating a power gain as an amplitude can change the budget by orders of magnitude. | Passed. The budget is 1 W, the noise variance is 1e-12 W, and the zero-shadowing per-link power gain is 10^(-61.4/10)/d^2. |
| Eqs. (1) and (2), single path and coherent three paths | This checks the amplitude square root and the 1/sqrt(L) normalization separately from random fading. | Passed. Three identical coherent paths produce sqrt(3) times the single-path channel amplitude. |
| Eqs. (3) and (4), communication signal and interference | Each user should see its own sub-RIS. Other communication streams in either group and sensing streams should enter its SINR denominator. | Passed with explicit full-RIS, split-RIS, and multi-stream cases. |
| Eq. (5), direct and reflected target illumination | The direct and reflected channels combine as complex amplitudes before squaring. The printed equation lists the target channel as a row, so the dimensionally consistent calculation is the squared norm of that row times the beamformer matrix. | Passed for direct-only, reflected-only, in-phase, quadrature, and complex-conjugation cases. |

The deterministic case uses a 16-antenna BS, a 6 by 6 RIS, the paper's node coordinates, one user and one target at the centers of their deployment disks, broadside unit-magnitude steering vectors, one path per link, and zero shadowing. Each path's complex gain is set to its RMS magnitude with zero phase. The full-aperture communication and sensing measurements are separate single-link checks.

| Quantity | Calculated and verified in code |
| --- | ---: |
| BS to RIS path power gain | -92.86 dB |
| RIS to user or target path power gain | -90.65 dB |
| BS to target direct path power gain | -90.22 dB |
| Full-aperture user received power | 9.240e-15 W |
| User SNR and rate | -20.34 dB, 0.01327 bit/s/Hz |
| Direct target power | 1.523e-8 W |
| Reflected target power | 9.240e-15 W |
| Combined target power | 1.525e-8 W |

The two-hop path gains multiply. Their combined loss is about 183.51 dB before coherent array gain. Even with all 36 RIS elements aligned and the 16-antenna BS beam at full power, this example falls below 0.5 bit/s/Hz for the user and 3e-7 W for the target. The earlier 100-draw feasibility check showed the gap across sampled channels as well.

No equation or unit error was found in the cases checked. The remaining reproduction assumptions include the number and distribution of paths, how steering angles relate to physical positions, and whether the numerical results use additional link or array gain beyond the stated equations. The current simulator samples path angles independently of geometry and uses a common full-aperture channel restricted to each partition. Those assumptions should be tested separately in Step 2, with the original settings retained as the reference case.
