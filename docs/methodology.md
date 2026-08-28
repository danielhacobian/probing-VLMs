# Probe methodology

## What a probe sees

The frozen visual encoder produces one representation tensor for each sampled
image. At DINO block `l`, the notebooks evaluate three readouts: the CLS token,
the mean of patch tokens, and the learned projected aggregate. The dynamics
predictor is evaluated separately through its pooled visual-token channels.

If a cached readout has shape `[window, time, feature] = [N, T, D]`, a
single-frame probe receives `N*T` rows of `D` features. A first-difference probe
receives `N*(T-1)` rows, and a second-difference probe receives `N*(T-2)` rows.
Window membership is retained through the train/validation/test split so
frames from one trajectory cannot cross partitions.

## Targets

Position is the simulator's Cartesian state. Displacement velocity and
acceleration are computed from the same sampled frames as the representation
differences. Scalar speed is the norm of velocity; heading is its unit vector.
PushT additionally probes pusher and block translation, circular block
orientation, angular velocity, and angular acceleration.

## Linear readout

For training features `X` and targets `Y`, the notebooks fit

```text
argmin_W ||XW - Y||^2 + lambda ||W||^2,  lambda = 10.
```

Feature means and scales are estimated from the training set only. Each layer,
readout, target, feature construction, condition, and split receives its own
probe. The main regression metric is held-out R-squared; RMSE and MAE are also
recorded. Direction uses mean cosine similarity and angular error.

## Selection, testing, and controls

Complete trajectories are divided into 60% training, 20% validation, and 20%
locked test partitions. Exploratory layer curves and spatial/doorway checks use
development data only. For each target and model family, the layer, readout,
and temporal feature construction with the highest mean validation R-squared
across OFF and ON is selected. The same choice is used for both conditions,
then fixed before each ridge probe is refit on
training plus validation trajectories and evaluated once on locked test
trajectories. Controls include shuffled targets, a position-only model, and
motion targets residualized against position.

Every confirmatory R-squared, RMSE, MAE, and ON-minus-OFF value has a
2.5th--97.5th percentile interval from 1,000 test resamples. The sampling unit
is a complete trajectory window: all frames from a sampled window remain
together. OFF and ON differences use paired trajectory resamples. These
intervals measure test-trajectory sampling uncertainty, not variation across
checkpoint training seeds. The current assets contain model-training seed 0;
additional seeds require independently trained OFF and ON checkpoints.
When matching checkpoints for more seeds are available, the confirmatory block
should be run independently for each seed while preserving `model_seed` in the
exported rows. Report each seed, the across-seed mean and standard deviation,
and paired OFF/ON differences within seed. Three seeds would require two
additional OFF and two additional ON training runs for each environment (12
new model trainings across UMaze, Wall, and PushT).

## Interpretation boundary

Linear readability is evidence that a variable can be recovered by a simple
map. It does not prove that the model stores a unique variable, that the
variable is represented by a single direction, or that later model components
use it causally. Temporal-difference probes also construct temporal information
outside the framewise DINO encoder; they test the geometry traced by encoded
frames rather than claiming that one frame contains motion.
