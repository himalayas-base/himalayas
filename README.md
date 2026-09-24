# HiMaLAYAS

![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
[![PyPI](https://img.shields.io/pypi/v/himalayas.svg)](https://pypi.python.org/pypi/himalayas)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
![Tests](https://github.com/himalayas-base/himalayas/actions/workflows/ci.yml/badge.svg)

**Hierarchical Matrix Layout and Annotation Software** (**HiMaLAYAS**) is a
framework for post hoc enrichment-based annotation and visualization of
hierarchically clustered matrices. HiMaLAYAS treats dendrogram-defined clusters
as statistical units, tests categorical annotations for enrichment, controls
multiple testing, and renders significant annotations alongside clusters.
HiMaLAYAS supports both biological and non-biological domains.

For a full description of HiMaLAYAS and its applications, see:
<br>
Horecka, I., and Röst, H. (2026)
<br>
_HiMaLAYAS: enrichment-based annotation and visualization of hierarchically clustered matrices_
<br>
_bioRxiv_. [https://www.biorxiv.org/content/10.64898/2026.02.11.705303v2](https://www.biorxiv.org/content/10.64898/2026.02.11.705303v2)
<br>
Submitted to _Bioinformatics Advances_.

## Documentation and Tutorials

- **Full Documentation**: [himalayas-base.github.io/himalayas-docs](https://himalayas-base.github.io/himalayas-docs)
- **Figure Gallery**: [himalayas-base.github.io/himalayas-docs/11_figure_gallery](https://himalayas-base.github.io/himalayas-docs/11_figure_gallery/)
- **Try in Browser (Binder)**:
  - Quickstart: [![Launch Quickstart in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/himalayas-base/himalayas-docs/main?filepath=notebooks/quickstart.ipynb)
  - Advanced Quickstart: [![Launch Advanced Quickstart in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/himalayas-base/himalayas-docs/main?filepath=notebooks/quickstart_advanced.ipynb)
- **Documentation Repository**: [github.com/himalayas-base/himalayas-docs](https://github.com/himalayas-base/himalayas-docs)

## Installation

HiMaLAYAS is compatible with Python 3.8 or later and runs on major operating
systems.

```bash
pip install himalayas
# Optional: faster clustering + richer compressed-label text processing
pip install "himalayas[speed,text]"
```

Detailed installation options and fallback behavior are documented at
[himalayas-base.github.io/himalayas-docs/1_installation](https://himalayas-base.github.io/himalayas-docs/1_installation/).

## Key Features of HiMaLAYAS

- **Matrix-Based Input**: Works with real-valued matrices representing
  relationships among observations.
- **Configurable Clustering**: Supports linkage method, distance metric,
  dendrogram distance threshold, and minimum cluster size settings.
- **Cluster-Level Enrichment Testing**: Treats dendrogram-defined clusters as
  statistical units and tests categorical annotations for enrichment.
- **Multiple-Testing Control**: Controls false discovery rate across
  cluster-term tests.
- **Annotation-Aware Visualization**: Renders significant annotations alongside
  clustered matrices.
- **Zoomed and Condensed Views**: Supports zoomed cluster reanalysis,
  condensed hierarchy views, and post hoc row data tracks.
- **Publication-Ready Output**: Exports configurable figures in raster or
  vector formats.

## Example Usage

We applied HiMaLAYAS to a hierarchically clustered
_Saccharomyces cerevisiae_ genetic interaction profile similarity matrix
(Costanzo _et al_., 2016), focusing on 1,053 genes with high profile variance.
Dendrogram-defined clusters were tested for Gene Ontology Biological Process
(GO BP; Ashburner _et al_., 2000) enrichment, with top-ranked significant
annotations rendered alongside clusters.

![Figure 1](assets/figure_1.png)
**HiMaLAYAS workflow and application to a hierarchically clustered yeast
genetic interaction profile similarity matrix (Costanzo _et al_., 2016)**.
A real-valued matrix and categorical annotations serve as inputs. HiMaLAYAS
hierarchically clusters the matrix, cuts the dendrogram at a user-defined
distance threshold, tests categorical annotations for enrichment, controls
multiple testing, and renders significant annotations alongside clusters.

## Citation

### Primary citation

Horecka, I., and Röst, H. (2026)
<br>
_HiMaLAYAS: enrichment-based annotation and visualization of hierarchically clustered matrices_
<br>
_bioRxiv_. [https://www.biorxiv.org/content/10.64898/2026.02.11.705303v2](https://www.biorxiv.org/content/10.64898/2026.02.11.705303v2)
<br>
Submitted to _Bioinformatics Advances_.

### Software archive

HiMaLAYAS software archive.
<br>
Zenodo. [https://doi.org/10.5281/zenodo.18610373](https://doi.org/10.5281/zenodo.18610373)

## Contributing

We welcome contributions from the community:

- [Issues Tracker](https://github.com/himalayas-base/himalayas/issues)
- [Source Code](https://github.com/himalayas-base/himalayas/tree/main/src/himalayas)

## Support

If you encounter issues or have suggestions for new features, please use the
[Issues Tracker](https://github.com/himalayas-base/himalayas/issues) on GitHub.

## License

This project is distributed under the [BSD 3-Clause License](LICENSE).
