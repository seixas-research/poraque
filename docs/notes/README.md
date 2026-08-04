# Design and analysis notes

Working documents: rationale, measurements and open questions. The
user-facing documentation is the Sphinx site in `docs/source/` and the two
guides in `latex/`.

| Note | Question it answers |
| --- | --- |
| [`roadmap.md`](roadmap.md) | What is done, what blocks progress, what to do next |
| [`model2_architecture.md`](model2_architecture.md) | Why train `chg2tau` at all, and how it couples to `ext2chg` |
| [`pi_fno.md`](pi_fno.md) | How KS-DFT and OF-DFT enter as constraints; the staged plan |
| [`fno_physics.md`](fno_physics.md) | What each model corresponds to in DFT terms |
| [`vasp_analysis_report.md`](vasp_analysis_report.md) | How VASP writes `EXTCAR` and `TAUCAR`, from the Fortran source |
| [`committee.md`](committee.md) | Query by committee: `init_seed`, disagreement on 3D fields, and how to validate it |

Start with `roadmap.md`.
