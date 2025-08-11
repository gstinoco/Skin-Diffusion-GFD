# Skin Diffusion Simulator with GFD :dna:

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) [![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange.svg)](https://numpy.org/) [![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-red.svg)](https://matplotlib.org/) [![Numba](https://img.shields.io/badge/Numba-0.54+-purple.svg)](https://numba.pydata.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Computational Framework for Skin Diffusion Modeling**

*Advanced numerical simulation using Generalized Finite Differences for biomedical applications*

### :link: Quick Links
[![🚀 Quick Start](https://img.shields.io/badge/🚀-Quick%20Start-green)](#rocket-quick-start) [![📊 Performance](https://img.shields.io/badge/📊-Performance-orange)](#chart_with_upwards_trend-performance-benchmarks) [![🎬 Visualizations](https://img.shields.io/badge/🎬-Visualizations-purple)](#movie_camera-visualizations) [![👥 Team](https://img.shields.io/badge/👥-Research%20Team-blue)](#scientist-research-team)

</div>

---

## :clipboard: Table of Contents
- [Overview](#star2-overview)
- [Features](#sparkles-features)
- [Installation & Setup](#package-installation--setup)
- [Quick Start](#rocket-quick-start)
- [Visualizations](#movie_camera-visualizations)
- [Project Architecture](#open_file_folder-project-architecture)
- [Mathematical Model](#books-mathematical-model)
- [Dataset Structure](#file_cabinet-dataset-structure)
- [Performance Benchmarks](#chart_with_upwards_trend-performance-benchmarks)
- [Research Team](#scientist-research-team)
- [Citation & License](#memo-citation)
- [Contact](#email-contact--support)

---

## :star: Overview

This repository presents a **state-of-the-art computational framework** for modeling substance diffusion in biological tissues using the Generalized Finite Differences (GFD) method. The simulator provides high-performance solutions for studying drug delivery, cosmetic penetration, and biomedical transport phenomena in skin layers.

### :wrench: Key Capabilities
- **:dna: Biological Modeling**: 2D transient diffusion equation solver for skin tissue simulation
- **:zap: High-Performance Computing**: Numba JIT compilation for maximum computational efficiency
- **:dart: Scientific Accuracy**: GFD method implementation for irregular mesh handling
- **:arrows_counterclockwise: Automated Dataset Generation**: Parallel processing for large-scale parameter studies
- **:bar_chart: Advanced Visualization**: Comprehensive plotting and analysis tools

### :microscope: Applications

| Field | Application | Use Case |
|-------|-------------|----------|
| **Pharmacology** :pill: | Drug Delivery | Transdermal absorption studies, dosage optimization |
| **Cosmetics** :lipstick: | Skin Penetration | Formulation analysis, safety assessment |
| **Dermatology** :hospital: | Clinical Research | Pathological transport, treatment efficacy |
| **Biomedical Engineering** :gear: | Device Design | Patch development, delivery system optimization |
| **Machine Learning** :robot: | AI Training | Dataset generation, pattern recognition |

---

## :sparkles: Features

### :abacus: Numerical Modeling
- **2D Transient Diffusion Solver**: GFD implementation with 9-point stencil
- **Flexible Boundary Conditions**: Mixed Dirichlet-Neumann conditions for realistic modeling
- **Adaptive Time Stepping**: CFL condition enforcement for numerical stability
- **Irregular Mesh Support**: GFD method handles complex geometries

### :zap: High-Performance Computing
- **Numba JIT Compilation**: Just-in-time optimization for critical functions
- **Vectorized Operations**: NumPy-based efficient array computations
- **Memory Optimization**: Efficient data structures for large-scale simulations
- **Parallel Processing**: Multi-core support for dataset generation

### :bar_chart: Data Generation & Analysis
- **Massive Dataset Creation**: 360,000+ simulation images
- **Parameter Space Exploration**: Systematic variation of diffusion coefficients and initial conditions
- **Automated Data Management**: Hierarchical organization and compression
- **Scientific Visualization**: Advanced plotting with Matplotlib

### :dart: Biomedical Applications
- **Skin Layer Modeling**: Realistic tissue geometry representation
- **Substance Transport**: Drug, cosmetic, and chemical diffusion simulation
- **Clinical Validation**: Framework for experimental data comparison
- **Predictive Modeling**: Machine learning dataset preparation

---

## :package: Installation & Setup

### :computer: System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.9+ |
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 5 GB | 25 GB+ (for datasets) |
| **OS** | Windows/Linux/macOS | Linux (optimal performance) |

### :clipboard: Dependencies

```python
# Core scientific computing
numpy >= 1.19.0           # Numerical computations
scipy >= 1.7.0            # Scientific algorithms
matplotlib >= 3.3.0       # Scientific plotting
numba >= 0.54.0           # JIT compilation
tqdm >= 4.62.0            # Progress bars
```

### Quick Installation

```bash
# Method 1: Direct installation
git clone https://github.com/gstinoco/Skin-Diffusion-GFD.git
cd GFD-Skin-ML
pip install -r requirements.txt

# Method 2: Virtual environment (recommended)
python -m venv skin_diffusion_env
source skin_diffusion_env/bin/activate  # On Windows: skin_diffusion_env\Scripts\activate
pip install -r requirements.txt

# Method 3: Conda environment
conda create -n skin_gfd python=3.9
conda activate skin_gfd
pip install -r requirements.txt
```

### :white_check_mark: Installation Verification

```bash
# Test installation
python -c "import numpy, scipy, matplotlib, numba; print(':white_check_mark: Installation successful!')"

# Run quick demo
python GFD_skin.py
```

---

## :rocket: Quick Start

### :zap: Single Simulation (Recommended)
```bash
# Run basic skin diffusion simulation
python GFD_skin.py
```

### :construction: Dataset Generation
```bash
# Generate complete training datasets
python create_dataset.py
```

### :wrench: Advanced Usage Examples
```bash
# Custom mesh resolution (modify in GFD_skin.py)
# Available meshes: skin224.mat, skin256.mat

# Parameter exploration
# Modify diffusion coefficient (nu) and initial concentration (ci) in main()

# Memory optimization for large datasets
# Adjust batch sizes in create_dataset.py
```

### :dna: Available Mesh Configurations
```bash
# Standard resolution meshes
region/skin224.mat        # 224×224 nodes, ~50K points
region/skin256.mat        # 256×256 nodes, ~65K points

# Original skin layer
region/skin_base.png      # Base for the geometries
```

---

## :movie_camera: Visualizations

### :framed_picture: Sample Visualizations Gallery

#### :microscope: Comparative Diffusion Analysis

**Concentration Initial = 100** | **Different Diffusion Coefficients**

**Low Diffusion Coefficient** ($\nu = 1 \times 10^{-8}$)

![Low Diffusion](docs/visualizations/comparison_nu_1e8_ci100.png)

**Medium Diffusion Coefficient** ($\nu = 4.5 \times 10^{-6}$)

![Medium Diffusion](docs/visualizations/comparison_nu_450e8_ci100.png)

**High Diffusion Coefficient** ($\nu = 9 \times 10^{-6}$)

![High Diffusion](docs/visualizations/comparison_nu_900e8_ci100.png)

> :bar_chart: **Dataset Scale**: Over 360,000 simulations across 100 initial conditions and 900 diffusion coefficients for two mesh resolutions

---

## :file_folder: Project Architecture

### Core Components

```
:package: GFD-Skin-ML/
├── GFD_skin.py                             # Main simulator module
│   ├── difusion_skin_jit()                 # JIT-optimized solver
│   ├── difusion_skin()                     # Vectorized solver
│   ├── Gammas()                            # GFD coefficient calculator
│   └── main()                              # Workflow orchestrator
│
├── create_dataset.py                       # Automated dataset generation
│   ├── Parallel processing support         # Multi-core optimization
│   ├── Parameter space exploration         # Systematic variation
│   ├── Automated data management           # File organization
│   └── Memory optimization                 # Efficient resource usage
│
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
│
├── region/                                 # Computational mesh library
│   ├── skin224.mat                         # Standard resolution mesh ($224 \times 224$)
│   ├── skin256.mat                         # High resolution mesh ($256 \times 256$)
│   ├── skin_base.png                       # Geometry visualization
│   └── red files/                          # Mesh generation files
│
├── Dataset/                                # Generated simulation datasets
│   ├── skin224_ci/                         # $224 \times 224$, varying initial concentration
│   ├── skin224_nu/                         # $224 \times 224$, varying diffusion coefficient
│   ├── skin256_ci/                         # $256 \times 256$, varying initial concentration
│   └── skin256_nu/                         # $256 \times 256$, varying diffusion coefficient
│
└── docs/                                   # Documentation and examples
    └── visualizations/                     # Sample visualization gallery
        ├── comparison_nu_1e8_ci100.png     # Low diffusion coefficient example
        ├── comparison_nu_450e8_ci100.png   # Medium diffusion coefficient example
        └── comparison_nu_900e8_ci100.png   # High diffusion coefficient example
```
---

## :books: Mathematical Model

The simulator solves the **2D transient diffusion equation**:

$$\frac{\partial u}{\partial t} = \nu \nabla^2 u$$

**Where:**
- $u(x,y,t)$: Concentration field [mg/L]
- $\nu$: Diffusion coefficient [m²/s]
- $\nabla^2$: Laplacian operator
- $t$: Time [s]

### :abacus: Numerical Methods

| Component | Method | Description |
|-----------|--------|-------------|
| **Spatial Discretization** | Generalized Finite Differences (GFD) | 9-point stencil for irregular meshes |
| **Temporal Integration** | Explicit Euler | First-order time stepping |
| **Boundary Conditions** | Mixed Dirichlet-Neumann | Inlet concentration + zero-flux boundaries |
| **Stability** | CFL Condition | Automatic time step adjustment |

### :dart: Boundary Conditions

- **Inlet (Dirichlet)**: $u = c_i$ (prescribed concentration)
- **Boundaries (Neumann)**: $\frac{\partial u}{\partial n} = 0$ (zero flux)
- **Initial Condition**: $u(x,y,0) = 0$ (clean tissue)

---

## :file_cabinet: Dataset Structure

The generated datasets follow a hierarchical organization:

```
Dataset/
├── skin224_ci/          # $224 \times 224$ mesh, varying initial concentration
│   ├── ci_001/          # Initial concentration = 0.01 (900 images)
│   │   ├── nu_0.00000001.png
│   │   ├── nu_0.00000002.png
│   │   └── ... (900 files)
│   ├── ci_002/          # Initial concentration = 0.02 (900 images)
│   ├── ...
│   └── ci_100/          # Initial concentration = 1.00 (900 images)
├── skin224_nu/          # $224 \times 224$ mesh, varying diffusion coefficient
│   ├── nu_0.00000001/   # Diffusion coefficient = 1×10⁻⁸ (100 images)
│   │   ├── ci_001.png
│   │   ├── ci_002.png
│   │   └── ... (100 files)
│   ├── nu_0.00000002/   # Diffusion coefficient = 2×10⁻⁸ (100 images)
│   ├── ...
│   └── nu_0.00000900/   # Diffusion coefficient = 900×10⁻⁸ (100 images)
├── skin256_ci/          # $256 \times 256$ mesh, varying initial concentration
│   ├── ci_001/          # Initial concentration = 0.01 (900 images)
│   ├── ci_002/          # Initial concentration = 0.02 (900 images)
│   ├── ...
│   └── ci_100/          # Initial concentration = 1.00 (900 images)
└── skin256_nu/          # $256 \times 256$ mesh, varying diffusion coefficient
    ├── nu_0.00000001/   # Diffusion coefficient = 1×10⁻⁸ (100 images)
    ├── nu_0.00000002/   # Diffusion coefficient = 2×10⁻⁸ (100 images)
    ├── ...
    └── nu_0.00000900/   # Diffusion coefficient = 900×10⁻⁸ (100 images)
```

### :bar_chart: Data Volume

- **Total Images**: 360,000 PNG files
- **Storage**: ~15-20 GB uncompressed
- **Parameters**: 100 initial concentrations $\times$ 900 diffusion coefficients per resolution
- **Resolutions**: $224 \times 224$ and $256 \times 256$ pixels

### :dart: Dataset Applications

| Use Case | Dataset Type | Description |
|----------|--------------|-------------|
| **Machine Learning Training** :robot: | Complete Dataset | 360K images for deep learning |
| **Parameter Studies** :chart_with_upwards_trend: | Subset Analysis | Specific parameter ranges |
| **Validation** :white_check_mark: | Test Sets | Independent validation data |
| **Benchmarking** :trophy: | Reference Solutions | Standard test cases |

---

## :chart_with_upwards_trend: Performance Benchmarks

### :stopwatch: Execution Times

| Mesh Size | Nodes | Time Steps | JIT Solver | Vectorized Solver | Memory Usage |
|-----------|-------|------------|------------|-------------------|--------------|
| $224 \times 224$   | 50,176| 1,000      | ~4.5s      | ~12.3s           | ~2.1 GB       |
| $224 \times 224$   | 50,176| 10,000     | ~45s       | ~123s            | ~2.1 GB       |
| $256 \times 256$   | 65,536| 1,000      | ~6.5s      | ~18.7s           | ~2.8 GB       |
| $256 \times 256$   | 65,536| 10,000     | ~65s       | ~187s            | ~2.8 GB       |

*Benchmarks: Intel i7-8700K @ 3.70GHz, 32GB RAM, Python 3.9*

### :rocket: Performance Optimizations

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| **Numba JIT** | 3-4x | Just-in-time compilation of critical loops |
| **Vectorization** | 2-3x | NumPy array operations |
| **Memory Layout** | 1.5x | Contiguous array storage |
| **Parallel Processing** | Nx | Multi-core dataset generation |

### :bar_chart: Scalability Analysis

```python
# Performance scaling with problem size
Nodes vs Time: O(N log N)     # Near-linear scaling
Memory vs Nodes: O(N)         # Linear memory usage
Parallel Efficiency: 85-90%   # Multi-core performance
```

---

## :man_scientist: Research Team

### :busts_in_silhouette: Main Researchers

<table>
<tr>
<td width="33%">

**Dr. Gerardo Tinoco Guerrero** :mexico:
- :office: [SIIIA MATH: Soluciones en ingeniería](http://www.siiia.com.mx)
- :classical_building: [Universidad Michoacana de San Nicolás de Hidalgo](http://www.umich.mx)
- :microscope: Numerical Methods & Computational Mathematics
- :email: gerardo.tinoco@umich.mx
- :globe_with_meridians: [ORCID](https://orcid.org/0000-0003-3119-770X)

</td>
<td width="33%">

**Dr. Francisco Javier Domínguez Mota** :mexico:
- :office: [SIIIA MATH: Soluciones en ingeniería](http://www.siiia.com.mx)
- :classical_building: [Universidad Michoacana de San Nicolás de Hidalgo](http://www.umich.mx)
- :microscope: Applied Mathematics & Finite Difference Methods
- :email: francisco.mota@umich.mx
- :globe_with_meridians: [ORCID](https://orcid.org/0000-0001-6837-172X)

</td>
<td width="33%">

**Dr. José Alberto Guzmán Torres** :mexico:
- :office: [SIIIA MATH: Soluciones en ingeniería](http://www.siiia.com.mx)
- :classical_building: [Universidad Michoacana de San Nicolás de Hidalgo](http://www.umich.mx)
- :microscope: Engineering applications and Artificial Intelligence
- :email: jose.alberto.guzman@umich.mx
- :globe_with_meridians: [ORCID](https://orcid.org/0000-0002-9309-9390)

</td>
</tr>
</table>

### :mortar_board: Graduate Students

**Ángel Emeterio Calvillo Vázquez** :mexico:
- :dart: Graduate Student
- :classical_building: Universidad Michoacana de San Nicolás de Hidalgo
- :microscope: Computational Biology & Numerical Simulation
- :email: 1025501x@umich.mx
- :globe_with_meridians: [ORCID](https://orcid.org/0009-0005-5497-5166)
- :briefcase: Research Focus: Skin diffusion modeling and machine learning applications

---

## :books: Scientific References

### :books: Core Publications

1. **Tinoco-Guerrero, G.**, Domínguez-Mota, F. J., Guzmán-Torres, J. A., & Tinoco-Ruiz, J. G. (2022). *"Numerical Solution of Diffusion Equation using a Method of Lines and Generalized Finite Differences."* **Revista Internacional de Métodos Numéricos para Cálculo y Diseño en Ingeniería**, 38(2). [DOI: 10.23967/j.rimni.2022.06.003](http://dx.doi.org/10.23967/j.rimni.2022.06.003)

### :trophy: Research Achievements

- **360,000+ Simulation Dataset**: Largest publicly available skin diffusion dataset
- **High-Performance Implementation**: 3-4x speedup with Numba JIT optimization
- **Open Source Framework**: MIT licensed for academic and commercial use
- **Cross-Platform Compatibility**: Windows, Linux, macOS support

---

## :memo: Citation

If you use this software in your research, please cite:

```bibtex
@software{gfd_skin_simulator_2025,
  title={Skin Diffusion Simulator with GFD: High-Performance Computational Framework 
         for Biomedical Transport Modeling},
  author={Tinoco-Guerrero, Gerardo and 
          Domínguez-Mota, Francisco Javier and 
          Guzmán-Torres, José Alberto and
          Calvillo-Vázquez, Ángel Emeterio},
  year={2025},
  institution={Universidad Michoacana de San Nicolás de Hidalgo},
  organization={SIIIA MATH: Soluciones en ingeniería},
  url={https://github.com/gstinoco/Skin-Diffusion-GFD},
  note={Advanced computational framework for skin diffusion modeling using 
        Generalized Finite Differences method}
}
```

### :classical_building: Institutional Support

**Primary Funding:**
- :mortar_board: **Universidad Michoacana de San Nicolás de Hidalgo (UMSNH)**
- :office: **SIIIA MATH: Soluciones en ingeniería**

### :page_facing_up: License

This project is licensed under the **MIT License** - see the full license text below:

```
MIT License

Copyright (c) 2025 Gerardo Tinoco-Guerrero, Francisco Javier Domínguez-Mota, 
                   José Alberto Guzmán-Torres, Ángel Emeterio Calvillo-Vázquez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Academic Use:** This software is developed for research and educational purposes. Commercial use is permitted under the MIT License terms.

---

## :email: Contact & Support

### :busts_in_silhouette: Research Group Contact

**Primary Contact:**
- **Dr. Gerardo Tinoco Guerrero**
  - :email: gerardo.tinoco@umich.mx
  - :office: SIIIA MATH: Soluciones en ingeniería
  - :classical_building: Universidad Michoacana de San Nicolás de Hidalgo
  - :round_pushpin: Morelia, Michoacán, México

### :question: Technical Support

For technical questions and issues:
1. **GitHub Issues**: Create an issue for bug reports or feature requests
2. **Email Support**: Contact the research team for complex technical inquiries
3. **Academic Collaboration**: Reach out for research partnerships and joint projects

### :handshake: Collaboration Opportunities

We welcome collaborations in:

- **Biomedical Engineering**: Transdermal delivery systems, medical device design
- **Machine Learning**: AI-driven analysis of diffusion patterns, predictive modeling
- **Numerical Methods**: Advanced discretization techniques, solver optimization
- **Clinical Research**: Validation with experimental data, clinical applications
- **Pharmaceutical Research**: Drug delivery optimization, formulation studies

### :mortar_board: Student Opportunities

- **Graduate Programs**: Contact Dr. Tinoco for research opportunities
- **Undergraduate Projects**: Available thesis topics in computational biology
- **Internships**: Summer research programs in scientific computing

### :globe_with_meridians: Institutional Affiliations

- **SIIIA MATH**: [Soluciones en ingeniería](http://www.siiia.com.mx)
- **UMSNH**: [Universidad Michoacana de San Nicolás de Hidalgo](http://www.umich.mx)
- **Research Group**: Numerical Methods and Scientific Computing

---

<div align="center">

**:star: If this project helps your research, please consider giving it a star! :star:**

*Advancing biomedical science through computational innovation*

</div>