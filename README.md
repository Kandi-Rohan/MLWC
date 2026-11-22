# MLWC RIS-assisted ISAC reproduction

Course implementation of Wei et al., *Machine Learning-Driven Joint RIS Partitioning, Beamforming, and Phase Shifts Design for mmWave ISAC Systems*.

## How the paper works

The base station sends communication and sensing streams through the same wireless system. The RIS is a surface with many reflecting elements. Changing their phases can make reflected signals add coherently in a desired direction. Dividing the surface into rectangles lets different user groups and the sensing group steer different parts of the surface. A larger rectangle for one group leaves fewer elements for the others.

1. **Generate the channels.** The simulator models propagation from the BS to the RIS, from the RIS to each receiver, and directly from the BS to sensing targets. Direct BS-to-user links are blocked in the paper's model.
2. **Group similar users.** Normalized channel correlation measures how similar their RIS channels are. The paper's k-means++-style procedure chooses well-separated initial representatives and assigns users to the closest representative by correlation.
3. **Choose the allocation.** A DDPG actor observes channel state information and proposes RIS rectangles, BS beamforming weights, and phases. The decoder forms a valid partition, caps transmit power, and quantizes phases to three bits. Our decoder uses recursive cuts to approximate the paper's rectangle projection and completion.
4. **Evaluate and learn.** User SINR accounts for interference from other streams. The receiver rate is log2(1 + SINR). Sensing performance is target illumination in watts, with direct and reflected amplitudes combined before squaring. The training reward adds the user rates and subtracts penalties for shortfalls. The critic estimates that reward and guides the actor.
5. **Compare methods.** On the same simulated channels, report both mean sum rate and the fraction of draws where every user and target meets its threshold. The fixed and shared RIS baselines isolate the effect of allocation and give the learned policy a conventional comparison.

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

## Step 2. Paired feasibility and gain sensitivity

```powershell
python channel_audit/feasibility_sensitivity.py --draws 100 --seed 100000
```

We compared two angle assumptions on the same draws with 16 BS antennas, 36 RIS elements, six users, and three sensing targets. Both use the same positions, shadowing, path gains, phase resolution, thresholds, and power budget. The geometry variant uses the endpoint direction of the largest-magnitude path on each link and retains the random angles for the other paths. It assumes a BS ULA along x and RIS rows and columns along x and z. The BS-to-RIS arrival direction points from the RIS toward the BS. These array orientations and the strongest-path rule are choices for the sensitivity experiment.

First, we used per-link upper bounds with all transmit power and the full RIS, ignoring interference. This tests whether the channels clear a necessary condition for joint feasibility. Then we used the existing fixed-partition beamforming heuristic to measure how much gain a concrete policy would need. The gain test scales received signal and interference power together at each CU, leaving its noise floor fixed. Target illumination power scales by the same tested factor. If a user's signal-to-interference ratio is already below its threshold, no finite gain can make that fixed beam pass. The experiment applies these gain factors to the evaluated links as a sensitivity test. The simulator's path loss and thresholds retain their original values.

| Angle model | Evaluation | All users pass | All targets pass | Joint pass | Median gain needed by weakest CU | Median gain needed by weakest target | Any CU interference-limited |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | Upper bounds | 0% | 0% | 0% | 28.0 dB | 18.5 dB | 0% |
| Geometry for strongest path | Upper bounds | 0% | 0% | 0% | 27.7 dB | 18.2 dB | 0% |
| Random | Fixed partition | 0% | 0% | 0% | No finite gain | 26.4 dB | 90% |
| Geometry for strongest path | Fixed partition | 0% | 0% | 0% | No finite gain | 24.3 dB | 100% |

| Added effective power gain | Random upper bounds pass jointly | Geometry upper bounds pass jointly | Random fixed partition passes jointly | Geometry fixed partition passes jointly |
| ---: | ---: | ---: | ---: | ---: |
| 0 dB | 0% | 0% | 0% | 0% |
| 10 dB | 0% | 0% | 0% | 0% |
| 20 dB | 12% | 7% | 0% | 0% |
| 30 dB | 61% | 61% | 0% | 0% |
| 40 dB | 95% | 94% | 0% | 0% |
| 50 dB | 100% | 100% | 0% | 0% |
| 60 dB | 100% | 100% | 5% | 0% |

The endpoint angles leave the link-budget gap open. At +30 dB both variants pass the necessary upper-bound check on 61% of draws, and the fixed policy passes on 0%. The geometry variant also produces an interference-limited user on every fixed-policy draw. Those results explain why a common power gain or longer DDPG training alone cannot establish a reproduction of Fig. 2 under the reference channel settings.

## Step 3. Interference-aware alternating optimization

```powershell
python ao_baseline.py --ris-sides 3 4 5 6 7 8 10
```

For both the fixed partition and the shared surface, initialize with the existing phase-alignment and regularized zero-forcing heuristic. Hold the RIS phases fixed and take one projected gradient step on the BS beamforming matrix. The gradient includes interference at every user and the sensing term from Eq. (8). Backtracking accepts an improving step and the projection respects the transmit-power cap. Then hold the beamforming matrix fixed and test all eight allowed phase values at each element. Repeat for up to six rounds. This baseline uses projected gradient and discrete coordinate search. The greedy layout described below uses the same AO solver.

The following mean sum rates are in bit/s/Hz. Each size uses 100 paired draws at 16 BS antennas and the original channel settings.

| RIS elements | Fixed heuristic | Fixed AO | Shared heuristic | Shared AO | Greedy initialization | Greedy AO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 0.0000167 | 0.000287 | 0.000279 | 0.00500 | 0.0000961 | 0.00234 |
| 16 | 0.0000806 | 0.00139 | 0.000749 | 0.0151 | 0.000393 | 0.00889 |
| 25 | 0.000114 | 0.00217 | 0.00142 | 0.0357 | 0.00127 | 0.0285 |
| 36 | 0.000273 | 0.00561 | 0.00316 | 0.0639 | 0.00265 | 0.0568 |
| 49 | 0.000452 | 0.00894 | 0.00628 | 0.109 | 0.00482 | 0.100 |
| 64 | 0.000802 | 0.0163 | 0.0103 | 0.171 | 0.00822 | 0.157 |
| 100 | 0.00186 | 0.0370 | 0.0198 | 0.305 | 0.0193 | 0.324 |

Every method had 0% all-user, all-target, and joint QoS satisfaction at every size. AO improved its initialization but did not resolve the link-budget gap. The objective uses the sensing deficit in watts, so a numerically small sensing penalty can coexist with a large relative sensing shortfall. We therefore continue to report the sensing threshold separately.

The original channel audit passed after the evaluator refactor. On 20 regression draws, the refactored evaluator reproduced the prior rates, illumination, and states. Finite differences checked the complex beam gradient. Additional checks verified the power cap, discrete phases, and increasing reward on accepted AO iterations.

### Greedy element assignment

Wei et al. describe the reference [11] baseline as greedy assignment of RIS elements followed by AO. Our adaptation uses the following explicit assignment rule. The exact selection rule in the cited work could not be verified from the accessible sources. [Wei et al.](https://doi.org/10.1109/LWC.2026.3699388) · [Wang et al., reference 11](https://doi.org/10.1109/TVT.2023.3326488)

1. Use the fixed-partition heuristic's BS beam as the starting beam. Visit elements starting with the strongest incoming BS-to-RIS channel.
2. For the current element, score the eligible communication and sensing groups at all eight phase values. Keep the group and phase with the highest Eq. (8) score given the allocations already made. Reserve one element for each group. Each element has one owner and groups can have scattered elements.
3. Freeze the resulting layout, then run the same beam and phase optimizer used for the fixed and shared RIS. The same command above runs all three methods on paired channel draws with the same power limit and QoS thresholds.

At 36 elements, the greedy adaptation reached 0.0568 bit/s/Hz after AO, compared with 0.00561 for fixed AO and 0.0639 for shared AO. At 100 elements, greedy AO reached 0.324 and shared AO reached 0.305 bit/s/Hz. Across the tested sizes, the sensing group received one element per draw, and joint QoS stayed at 0%. Thus the sum-rate ranking alone does not establish a sensing benefit.

Independent enumeration through the full evaluator verified the greedy choices on a small surface. Additional checks covered assignment coverage, deterministic repetition, phase quantization, the transmit-power cap, and increasing reward on accepted AO updates. The original channel audit also passed.

## Step 4. DDPG sanity check on a known-feasible channel

```powershell
python ddpg_sanity.py
```

The paper-model link audit found that some draws fail even optimistic per-link bounds. This experiment isolates the learning pipeline with a synthetic channel where a feasible action can be constructed. It uses three BS antennas, two users, one sensing target, and a 2 by 3 RIS. Each user has its own orthogonal BS direction and two reflecting elements. We vary the user gains and relative phases across draws. The two communication beams also illuminate the target over a direct path. This controlled design has a simpler interference pattern than the reference channel model.

The constructed action gives each user its two cells, compensates for the relative phase, and splits the power equally between the two orthogonal beams. In the weakest case allowed by the synthetic gain ranges, the aligned user rate exceeds 1.45 bit/s/Hz and the target receives at least 9.11e-7 W. The original thresholds remain 0.5 bit/s/Hz for each user and 3e-7 W for sensing. The script runs the constructed action through the same geometry, beam, and phase decoder and checks that it passes all held-out draws. The actor trains from scratch without access to the constructed action.

The controlled run uses seeds 0, 1, and 2 for 2,400 steps each. It uses a learning rate of 3e-4 and target-update coefficient of 0.005 to test learning over a short run. It scales the stronger synthetic observations by an additional factor of 0.1 before passing them to the networks. Every 200 steps, the current actor is evaluated without exploration noise on the same 128 draws from a separate seed. The plot shows each seed as a thin line and their mean as the thick line. The dashed reference shows the constructed feasible action.

The original channel generator and default training settings retain their behavior. The channel audit passed, 20 original-model draws reproduced identical CSI, rates, sensing and reward, and a 70-step original-model training check reproduced the previous rewards and actor weights exactly.

| Policy at evaluation | Mean reward | Mean sum rate (bit/s/Hz) | All users pass | Sensing passes | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constructed feasible action | 2.3011 | 2.3011 | 100% | 100% | 100% |
| Random action | -8.4388 | 0.1435 | 0% | 93.0% | 0% |
| DDPG seed 0, final | 1.9840 | 1.9840 | 100% | 100% | 100% |
| DDPG seed 1, final | 0.4701 | 1.5247 | 25% | 100% | 25% |
| DDPG seed 2, final | 1.9933 | 1.9933 | 100% | 100% | 100% |
| DDPG final, mean across seeds | 1.4825 | 1.8340 | 75% | 100% | 75% |

Every untrained actor had 0% joint satisfaction. The final agents averaged 75%, and some checkpoints dropped sharply during training. These results show that the learning pipeline can discover feasible actions on the controlled channel, with substantial variation across seeds and checkpoints. [Held-out learning curves](ddpg_sanity.png)

For comparison, the same controlled test with the paper's learning rate and target update coefficient stayed at 0% joint satisfaction across all 2,400 steps and all three seeds. Its final mean reward was -9.9052. This comparison is for the stated step budget on the controlled channel. The paper-model performance gap and Figure 2 remain open.

```powershell
python ddpg_sanity.py --lr 1e-6 --tau 1e-8 --plot ddpg_sanity_paper_settings.png
```
