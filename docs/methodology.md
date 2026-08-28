# Probe methodology

## What a probe sees

The frozen visual encoder produces one representation tensor for each sampled
image. At DINO block `l`, the notebooks evaluate three readouts: the CLS token,
the mean of patch tokens, and the learned projected aggregate. The dynamics
predictor is evaluated separately through its pooled visual-token channels.

If a cached readout has shape `[window, time, feature] = [N, T, D]`, a
single-frame probe receives `N*T` rows of `D` features. A first-difference probe
receives `N*(T-1)` rows, and a second-difference probe receives `N*(T-2)` rows.
Window membership is retained until after the train/test split so frames from a
held-out trajectory cannot leak into training.

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

## Splits and controls

The episode split holds out complete trajectories. The spatial split fits away
from a region and evaluates inside that region, with a buffer between them.
Wall includes a doorway-specific holdout. Controls include shuffled targets, a
position-only model, and motion targets residualized against position.

The notebooks show 2.5th--97.5th percentile intervals from 300 bootstrap
resamples of held-out prediction rows. These bands do not measure variation
across checkpoint training seeds, and rows from the same trajectory window are
not independent. They should therefore be read as descriptive probe-sampling
uncertainty.

## Interpretation boundary

Linear readability is evidence that a variable can be recovered by a simple
map. It does not prove that the model stores a unique variable, that the
variable is represented by a single direction, or that later model components
use it causally. Temporal-difference probes also construct temporal information
outside the framewise DINO encoder; they test the geometry traced by encoded
frames rather than claiming that one frame contains motion.
