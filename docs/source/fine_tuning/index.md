# Fine-tuning

A model fitted across a broad dataset is a good *starting point* for a specific
material family, and a poor substitute for one. Fine-tuning continues that fit
on the smaller set at a much lower learning rate: what the base model learned
about the general map is kept, and only the specialisation is learned.

```{note}
Off by default. Switch it on with `enable: true` in the `fine_tuning` block, or
with `--fine-tune` on the command line.
```

```bash
# 1. the base model, trained across everything you have
poraque-train --config configs/train.yaml

# 2. specialise it on one family
poraque-train --config configs/train.yaml \
    --fine-tune --pretrained models/poraque_models.pfno \
    --root data/oxides --checkpoint-dir models/oxides
```

## What comes from the checkpoint

Two things are taken from the base model rather than from this run's config,
and both matter.

**The architecture**, inferred from the stored tensors. A `width` or `modes`
left over in the config that disagreed with the weights could only load
mismatched tensors, so the tensors are the authority — the `model` section is
ignored for the *shape* of a fine-tuned network.

**The normalizations.** The datasets are re-pointed at the checkpoint's
transforms, discarding the ones just fitted to the new data.

```{important}
Refitting the normalizations would rescale the network's inputs out from under
weights trained against the old scale, and the model would spend the fine-tune
relearning the scale instead of the chemistry. It also means the new data has
to fall in a comparable range: fine-tuning on a family whose densities are an
order of magnitude away is transfer learning in name only.
```

## Freezing the lifting path

```yaml
fine_tuning:
  freeze_lifting_layers: true
```

The lifting map embeds the input field into the network's channel space before
any operator acts on it. It is the most general part of the model, so it is the
part least in need of specialising — and freezing it removes parameters a small
dataset could otherwise overfit. The cell encoder is frozen with it: that
embeds the lattice, a property of the input rather than of the mapping.

The **projection head stays trainable**. It decodes to physical units, which is
precisely what differs between material families.

The run reports what was actually frozen, so you never have to infer it:

```text
      fine-tuning from : models/poraque_models.pfno
      architecture     : inferred from the checkpoint (4,202,001 parameters)
      transforms       : taken from the checkpoint
      frozen           : lifting path, 1,392 parameters; 4,200,609 remain trainable
      learning rate    : 1e-05 (fine-tuning; base training uses 0.002)
```

```{tip}
Frozen parameters are kept out of the optimiser entirely, not merely denied a
gradient. AdamW's decoupled weight decay applies without one, so a frozen
weight left in the optimiser would shrink towards zero every step — quietly
undoing the pre-training the freeze was meant to preserve.
```

## Configuration

```yaml
fine_tuning:
  enable: false
  pretrained_checkpoint: models/poraque_models.pfno
  learning_rate: 1.0e-05
  freeze_lifting_layers: false
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | start from `pretrained_checkpoint` instead of a fresh init |
| `pretrained_checkpoint` | `models/poraque_models.pfno` | base bundle to adapt |
| `learning_rate` | `1e-05` | replaces `training.learning_rate` for this run |
| `freeze_lifting_layers` | `false` | hold the input lifting path fixed |

The fine-tuning learning rate is much smaller by design: the base rate would
walk the weights away from the solution being adapted before a small dataset
could constrain them.

## Output

A fine-tune is written as **`poraque_finetuned.pfno`**, never over the base
`poraque_models.pfno`. The two coexist in the same directory, and a run that
would genuinely overwrite its own base is refused — before training starts, not
after.

The PDF report leads its Configuration section with the fine-tuning facts and
carries a caveat at the top:

> This is a FINE-TUNED model, adapted from `models/poraque_models.pfno`. Its
> scores describe the material family it was specialised on, and say nothing
> about the broader set the base model was trained across.

The bundle's metadata records `fine_tuned_from`, the learning rate and whether
the lifting layers were frozen, so a checkpoint can always be traced back to
its parent.

## The `.pfno` format

Poraquê writes its models as **`.pfno`** — *Poraquê Fourier Neural Operator*:

| File | Contents |
| --- | --- |
| `models/poraque_models.pfno` | the universal model: both operators in one file |
| `models/poraque_finetuned.pfno` | a fine-tune, specialised to one family |

The container is a `torch.save` payload keyed by task (`ext2chg`, `chg2tau`)
with a format tag and metadata; the extension is a label, not a different
serialisation. Nothing inspects it when loading, so a bundle under any name
loads fine.

```{note}
Models written before this rename carry `.pth`. If the `.pfno` file is absent
and a `.pth` of the same stem is present, it is used and the substitution is
announced — an existing trained model should not become invisible because a
default filename changed. Rename it to silence the notice.
```
