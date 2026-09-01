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
  use_lora: false
  lora_rank: 8
  lora_alpha: 16.0
  lora_dropout: 0.0
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | start from `pretrained_checkpoint` instead of a fresh init |
| `pretrained_checkpoint` | `models/poraque_models.pfno` | base bundle to adapt |
| `learning_rate` | `1e-05` | replaces `training.learning_rate` for this run |
| `freeze_lifting_layers` | `false` | hold the input lifting path fixed |
| `use_lora` | `false` | freeze everything and learn a low-rank correction |
| `lora_rank` | `8` | $r$; cost and capacity are both linear in it |
| `lora_alpha` | `16.0` | the update is scaled by `lora_alpha / lora_rank` |
| `lora_dropout` | `0.0` | dropout on the adapter's input |

The fine-tuning learning rate is much smaller by design: the base rate would
walk the weights away from the solution being adapted before a small dataset
could constrain them.

## LoRA — adapting a model that does not fit

`use_lora: true` freezes every trained weight and learns a rank-$r$ correction
beside the model's **dense ($1\times1\times1$) lifting and projection**
convolutions:

$$ W' = W + \frac{\alpha}{r} B A, \qquad
   A \in \mathbb{R}^{r \times d_\mathrm{in}}, \quad
   B \in \mathbb{R}^{d_\mathrm{out} \times r} . $$

On the shipped model that is **1 320 trainable parameters against 3 152 353
frozen — 0.042 %**, so the optimiser state and the saved file shrink by three
orders of magnitude and the fine-tune fits where a full one does not:

```text
      LoRA             : 3 adapted layer(s), rank 8, alpha 16
      trainable        : 1,320 of 3,153,673 parameters (0.042%); the rest is frozen
      NOTE: the checkpoint will hold the ADAPTER only and name this base;
            it cannot be loaded without poraque_models.pfno.
```

$B$ starts at **zero**, so $BA = 0$ and the adapted model *is* the base model
until the first optimiser step — a fine-tune that began anywhere else would
already have discarded some of what it set out to adapt.

```{warning}
**The spectral weights are not adapted.** They hold ~99.8 % of the parameters
and look like the obvious target, but a rank-$r$ factorisation of a 5-index
complex kernel is a *choice of which axes to pair* rather than one
decomposition, and adapting them would also stop being cheap.

What LoRA says here is precise: *keep the learned operator, re-fit how fields
enter and leave it.* A family whose operator genuinely differs from the base
model's — not merely its input and output scales — is out of its reach, and the
honest answer there is a full fine-tune. It buys memory, not generality.
```

```{note}
**A LoRA checkpoint is not self-contained.** It holds the adapter and records
where its base lives, which is the whole economy of the method — the frozen
tensors are already on disk in the model being adapted. Move or delete that
base and the fine-tune cannot be loaded; the error says so and names the file.
`freeze_lifting_layers` is ignored under LoRA, which freezes everything anyway,
and the run says so rather than appearing to apply both.
```

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
