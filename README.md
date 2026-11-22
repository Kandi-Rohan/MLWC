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

## Step 5. Paired DDPG versus AO at 36 RIS elements

```powershell
python compare_policies.py
```

This experiment uses the original simulated channel model with 16 BS antennas, a 6 by 6 RIS, six communication users, and three sensing targets. Train DDPG from scratch for 2,400 steps per seed with seeds 0, 1, and 2. Run two training settings with the same step budget. One uses the paper's learning rate and target coefficient of 1e-6 and 1e-8. The other uses the faster sanity-check settings of 3e-4 and 0.005. Score the initial and final actors without exploration noise.

All actors and all three AO layouts are evaluated on exactly the same 100 held-out draws from seed 100000. The environment evaluates the current action without advancing its channel state between methods. Each AO baseline runs up to six rounds with access to the current channel, while DDPG decodes one actor output per draw. Use sum rate to compare communication throughput. Use the three threshold columns to see whether all users pass, all targets pass, or both. The DDPG rows average over each seed's held-out draws and then across the three training seeds.

The channel audit passed again after extracting the non-advancing action evaluator. On 20 regression draws it matched the previous step interface, including the next state and every reported metric. The three AO means matched the prior 100-draw comparison.

| Method | Mean sum rate (bit/s/Hz) | Mean reward | All users pass | All targets pass | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed partition AO | 0.00560964 | -29.938303 | 0% | 0% | 0% |
| Shared RIS AO | 0.0639383 | -29.308341 | 0% | 0% | 0% |
| Greedy assignment AO | 0.0567903 | -29.375316 | 0% | 0% | 0% |
| DDPG before training | 2.09997e-5 | -29.999778 | 0% | 0% | 0% |
| DDPG, paper settings | 1.94689e-5 | -29.999795 | 0% | 0% | 0% |
| DDPG, faster settings | 1.79680e-5 | -29.999811 | 0% | 0% | 0% |

| DDPG training seed | Paper settings mean sum rate | Faster settings mean sum rate |
| ---: | ---: | ---: |
| 0 | 2.47391e-5 | 2.71262e-5 |
| 1 | 1.66455e-5 | 1.38470e-5 |
| 2 | 1.70219e-5 | 1.29308e-5 |

Shared RIS AO had the highest mean sum rate in this experiment. Both DDPG settings averaged slightly below the untrained actors after the stated step budget. The per-link upper bounds passed all users on 0% of these draws and all targets on 0%, confirming the existing feasibility gap for this held-out set. This comparison measures the complete policies under the current model. Figure 2 and the question of how much the learned partition contributes remain open.

## Step 6. Imperfect CSI at 36 RIS elements

```powershell
python imperfect_csi.py
```

Use the same 100 true channel draws and the same training seeds, step budget, and AO iteration limit as in Step 5. The script retrains the actors with perfect CSI using the same seeds, then holds their weights fixed across the three evaluation conditions. The AO baselines, CU grouping, and DDPG input all use the supplied channel estimate. The evaluator keeps the user-to-group assignments from that estimate and calculates rates and sensing power on the true channel. Both noisy conditions share the same underlying complex Gaussian samples.

Wei et al. describe the estimate as the true CSI plus independent, zero-mean complex Gaussian errors of variance 0.01. They do not specify a normalization convention in the numerical-results description. We test the raw additive interpretation and a separate per-link normalization sensitivity case.

- **Perfect CSI.** Use each true channel as the estimate.
- **Raw additive error.** Add noise with complex variance 0.01 directly to each complex channel coefficient. Its real and imaginary parts each have variance 0.005.
- **Per-link relative error.** Set the error variance to 0.01 times the mean squared true-coefficient magnitude of its link. This gives approximately 1% error energy relative to the true channel energy. The normalization covers the full BS-to-RIS matrix, and each RIS-to-user, RIS-to-target, and direct BS-to-target link separately.

The script prints sum rate and QoS pass rates for all methods, how often estimated group labels differ from the perfect-CSI labels, and the error-energy ratio by link type. Zero-variance checks gave identical scores for all three conditions. An independent reference check confirmed that decisions selected using a noisy channel were scored with the true coefficients and the estimated user grouping.

| Method | Perfect CSI | Raw additive error | Per-link relative error |
| --- | ---: | ---: | ---: |
| Fixed partition AO | 0.00560964 | 1.74699e-5 | 0.00584886 |
| Shared RIS AO | 0.0639383 | 6.40427e-5 | 0.0660309 |
| Greedy assignment AO | 0.0567903 | 1.88828e-5 | 0.0590620 |
| DDPG, paper training settings | 1.94689e-5 | 2.48503e-5 | 2.00326e-5 |
| DDPG, faster training settings | 1.79680e-5 | 3.91404e-5 | 1.81496e-5 |

All entries are held-out mean sum rates in bit/s/Hz. Every method had 0% all-user, all-target, and joint pass in all three conditions. The perfect-CSI values reproduced Step 5 to the displayed precision. Estimated group labels differed from the perfect-CSI labels on 100 of 100 raw-error draws and 46 of 100 relative-error draws.

Raw error energy was 3.65 million to 11.2 million times the true channel energy across the four link types. It obscures the simulated CSI and gives much lower rates for all AO methods. Relative error energy was 0.96% to 1.03% of the true channel energy. In that case, the method ranking was preserved and the rates stayed close to the perfect-CSI runs. The small increases under relative noise apply to this draw set and the finite-iteration solvers. These comparisons document the effect of the error convention on our simulator and leave the paper's imperfect-CSI curves open.

## Step 7. BS antenna-count sweep for Fig. 2(b)

```powershell
python antenna_sweep.py
```

Sweep 4, 8, 12, 16, 20, 24, 28, and 32 BS antennas with the RIS fixed at 36 elements, six CUs, three CU groups, and three targets. Use 100 held-out true channel draws per count from seed 100000. The path gains, angles, and node positions are paired across antenna counts. For a smaller array the true BS channel is the corresponding prefix of the larger array. Estimation errors use the same complex Gaussian samples across antenna widths, generated for the largest array and sliced to each width before applying the per-link RMS scaling.

Compare the three six-round AO baselines with a DDPG pilot trained separately for each antenna count with 320 steps per seed, using seeds 0, 1, and 2 and the faster settings from Step 4. Report perfect CSI and 1% relative error energy. The raw additive stress test is documented in Step 6. Scores use the true channels in all cases. The figure uses a log scale to make both AO and the lower-rate DDPG pilot visible.

The 16-antenna perfect-CSI AO results match the prior 100-draw comparison. The noisy 16-antenna row uses the shared 32-antenna noise samples for this sweep, so it can differ from the standalone Step 6 noisy row. Checks verified that the smaller true and estimated-noise arrays use identical prefixes and that the standalone imperfect-CSI experiment preserves its original results.


Mean sum rate in bit/s/Hz

### Perfect CSI

| BS antennas | Fixed AO | Shared AO | Greedy AO | DDPG pilot |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.0014358 | 0.018729 | 0.015785 | 1.4291e-05 |
| 8 | 0.0028562 | 0.03472 | 0.029201 | 1.7684e-05 |
| 12 | 0.0042424 | 0.050246 | 0.044057 | 1.8376e-05 |
| 16 | 0.0056096 | 0.063938 | 0.05679 | 1.5702e-05 |
| 20 | 0.0069198 | 0.079262 | 0.071519 | 1.5435e-05 |
| 24 | 0.0083581 | 0.092607 | 0.079756 | 2.0272e-05 |
| 28 | 0.0097244 | 0.10623 | 0.089225 | 1.5828e-05 |
| 32 | 0.010852 | 0.11884 | 0.10754 | 2.1725e-05 |

### 1% per-link relative error energy

| BS antennas | Fixed AO | Shared AO | Greedy AO | DDPG pilot |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.0016281 | 0.017943 | 0.016132 | 1.4146e-05 |
| 8 | 0.0030071 | 0.034488 | 0.029422 | 1.6238e-05 |
| 12 | 0.0045024 | 0.050102 | 0.042382 | 1.7959e-05 |
| 16 | 0.0060096 | 0.062125 | 0.057557 | 1.4383e-05 |
| 20 | 0.0073847 | 0.076643 | 0.074299 | 1.5154e-05 |
| 24 | 0.0090216 | 0.093858 | 0.084233 | 1.8171e-05 |
| 28 | 0.010379 | 0.10387 | 0.098851 | 1.4414e-05 |
| 32 | 0.011921 | 0.1144 | 0.10711 | 1.5582e-05 |

All three AO baselines increased at every antenna count with perfect and relative-error CSI. Shared AO led across the sweep. The DDPG pilot remained near 1e-5 bit/s/Hz and did not show a consistent increase. All user, target, and joint threshold pass rates were 0% across methods and conditions. The AO trend agrees with the direction in Fig. 2(b), while the absolute rates and method ranking differ. [Antenna sweep plot](antenna_sweep.png)

## Step 8. RIS allocation maps for Fig. 2(d)

```powershell
python partition_visualization.py
```

Use a 64-element (8 by 8) RIS, 16 BS antennas, six CUs, and three sensing targets. The figure shows the first two of 100 held-out channels from seed 100000. On each draw, all policies use the same true CSI and NCC grouping. The two AO methods use up to six rounds. Train one DDPG actor from seed 0 for 2,400 steps with the faster settings from Step 4 and evaluate the same actor before and after training.

Each color identifies one group. The figure gives group cell counts in legend order and reports the weakest target's sensing power as a fraction of its requirement. This distinguishes a layout that allocates sensing elements from one that actually meets the sensing threshold. We count distinct labeled grids over 100 draws, with the untrained actor included for comparison. The fixed layout and DDPG use rectangles, and the greedy method can assign scattered cells. The layout and reported target powers were checked against direct evaluation on the example channels.

| Method | Distinct labeled layouts | Mean sensing cells | Sensing cell range | Mean sum rate (bit/s/Hz) | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed partition + AO | 1 | 16.00 | 16 to 16 | 0.016281 | 0% |
| Greedy assignment + AO | 100 | 1.00 | 1 to 1 | 0.157104 | 0% |
| DDPG before training | 15 | 16.00 | 16 to 16 | 4.74934e-5 | 0% |
| DDPG after training | 46 | 13.32 | 7 to 28 | 7.73324e-5 | 0% |

The two plotted DDPG layouts allocate 24 / 12 / 12 / 16 and 36 / 6 / 6 / 16 cells to CU groups 1–3 and sensing. The sensing count stayed the same on these two examples and varied across the full set. The fixed layout always uses 16 cells per group. The greedy rule gave 61 cells to a single communication group on both examples, with a different group chosen on each draw. All-user and all-target pass rates were 0% for all methods. This single DDPG run demonstrates variation in the decoded rectangles under the current channel model. The paper's successful sensing result remains open. [RIS allocation maps](partition_visualization.png)

## Step 9. Matched RIS-size sweep for Fig. 2(c)

```powershell
python ris_size_sweep.py
```

Hold the BS at 16 antennas with six CUs, three CU groups, and three sensing targets. Sweep square RIS surfaces with 9, 16, 25, 36, 49, 64, and 100 elements. Each point uses 100 held-out draws from seed 100000. Node positions, path gains, and path angles are paired across sizes. The smaller RIS uses the same coefficients as the overlapping rows and columns of the 10 by 10 surface. The relative-error test also pairs the standard complex Gaussian samples at overlapping RIS coordinates, then applies the RMS scaling for each link at its current size.

Use the same six-round AO baselines and DDPG pilot settings as the antenna sweep. Train one actor per antenna/RIS configuration and seed for 320 steps, using three seeds and the faster training settings from Step 4. Evaluate the fixed, shared, and greedy AO methods and the trained DDPG pilot under perfect CSI and 1% per-link relative error energy. Score both conditions on the true channels. This matched sweep replaces the earlier pilot as the main Fig. 2(c)-style comparison in this project. The earlier heuristic plot remains available for the initial implementation record.


Mean sum rate in bit/s/Hz

### Perfect CSI

| RIS elements | Fixed AO | Shared AO | Greedy AO | DDPG pilot |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00028736 | 0.0050028 | 0.0023426 | 2.7536e-06 |
| 16 | 0.0013876 | 0.015055 | 0.0088939 | 4.4166e-06 |
| 25 | 0.0021652 | 0.035666 | 0.028466 | 1.1384e-05 |
| 36 | 0.0056096 | 0.063938 | 0.05679 | 1.5702e-05 |
| 49 | 0.0089435 | 0.10858 | 0.10014 | 2.1505e-05 |
| 64 | 0.016281 | 0.17095 | 0.1571 | 2.7096e-05 |
| 100 | 0.037038 | 0.3048 | 0.32417 | 6.3907e-05 |

### 1% per-link relative error energy

| RIS elements | Fixed AO | Shared AO | Greedy AO | DDPG pilot |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00029233 | 0.0050025 | 0.0022301 | 2.8158e-06 |
| 16 | 0.0012897 | 0.014873 | 0.0098627 | 4.2051e-06 |
| 25 | 0.002099 | 0.031704 | 0.026397 | 1.0871e-05 |
| 36 | 0.0058109 | 0.059512 | 0.053118 | 1.4303e-05 |
| 49 | 0.0083429 | 0.10195 | 0.099803 | 2.2273e-05 |
| 64 | 0.01633 | 0.16703 | 0.16878 | 2.4248e-05 |
| 100 | 0.036444 | 0.30208 | 0.32245 | 6.5827e-05 |

Mean sum rate increased at every size for all four methods in both conditions. Greedy AO led at 100 elements with perfect CSI and at 64 and 100 elements with relative error. Shared AO led at the other sizes. The DDPG pilot remained well below the AO methods, and every method had 0% all-user, all-target, and joint threshold pass. The upward size trend appeared in our model, while the rates and method ranking in Fig. 2(c) remain open. [Matched RIS-size plot](ris_size_sweep.png)

Checks confirmed that the true channels and estimation samples match at overlapping RIS coordinates. The 36-element perfect-CSI row reproduced the shared point in the antenna sweep, and regression runs confirmed the existing antenna and CSI scripts preserve their results.

## Step 10. Final validation against Figure 2

```powershell
python reference_convergence.py
```

The reference convergence check uses the original 16-antenna, 36-element channel model, six CUs, and three sensing targets. It uses the reported learning rates and target-update coefficient, with 2,400 simulator steps for each of seeds 0, 1, and 2. The left plot shows non-overlapping 100-step means of exploratory training rewards. The right plot scores each actor without exploration on the same 100 held-out channels at step zero and every 400 steps. The horizontal axis uses simulator steps. The paper plots episodes, and their step counts are unspecified, so we compare the reward behavior over our stated budget.

All panels refer to Fig. 2 in Wei et al. [Paper](https://doi.org/10.1109/LWC.2026.3699388)

| Paper panel | What the paper reports | What this project verified |
| --- | --- | --- |
| 2(a), convergence | DDPG average reward rises and stabilizes on the simulated mmWave channel. | Reference-channel reward remains near -30 over our training budget. On the separate known-feasible channel the faster-setting DDPG reached 75% joint pass across three seeds, with variation across seeds. |
| 2(b), BS antenna count | Sum rate rises with antenna count and the proposed method leads the compared baselines. | All three AO rates rise at every tested count under both CSI conditions. Shared RIS AO leads in our experiment and the short DDPG pilot has no consistent upward trend. [Plot](antenna_sweep.png) |
| 2(c), RIS size | Sum rate rises with RIS size and the proposed method leads. | Mean sum rate rises at every size for all four tested methods. Shared AO leads most points and greedy AO leads at 100 elements with perfect CSI and at 64 and 100 elements with relative error. [Plot](ris_size_sweep.png) |
| 2(d), RIS partition | One example shows distinct, non-overlapping rectangular allocations. | One trained actor produced 46 distinct labeled rectangular layouts on 100 draws and used 7 to 28 sensing cells. No draw met every target requirement. [Maps](partition_visualization.png) |

Cross-checks confirmed identical perfect-CSI AO values in the paired policy and noisy-CSI experiments, identical perfect-CSI rates at the shared 16-antenna and 36-element point of the two sweeps, and the same perfect-CSI greedy rate at 64 elements in the RIS-size and partition-map experiments.

The implementation evidence covers the channel calculations, valid layout decoding, a runnable DDPG pipeline, and controlled comparisons. The reference model falls below the paper's rates and thresholds. Its optimistic all-user and all-target link-bound checks passed 0% of the reference draws. The exact channel assumptions and reference [11] implementation remain unverified, so the reproduction claims apply to the observed trends and layout behavior.

| Reference training checkpoint | Mean held-out reward | Mean held-out sum rate (bit/s/Hz) | Joint pass |
| ---: | ---: | ---: | ---: |
| 0 | -29.999777926 | 2.09997e-5 | 0% |
| 400 | -29.999779966 | 2.08143e-5 | 0% |
| 800 | -29.999781116 | 2.07098e-5 | 0% |
| 1,200 | -29.999778601 | 2.09385e-5 | 0% |
| 1,600 | -29.999787671 | 2.01142e-5 | 0% |
| 2,000 | -29.999790002 | 1.99024e-5 | 0% |
| 2,400 | -29.999794774 | 1.94689e-5 | 0% |

The initial and final values agree with the paper-setting runs in Step 5. All seeds had 0% all-user, all-target, and joint pass at the recorded checkpoints. Tiny rates keep the reward near the six-user shortfall penalty of -30. The zoomed axes show the variation around that value. [Reference convergence plot](reference_convergence.png) · [Known-feasible convergence plot](ddpg_sanity.png)

Nine reduced runs completed from a clean source snapshot using the installed project dependencies. They covered the deterministic channel audit, feasibility sensitivity, both DDPG experiments, the paired policy comparison, imperfect CSI, the antenna and RIS sweeps, and the allocation maps. Each plotting command produced a PNG.
