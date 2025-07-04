# Skin Diffusion Simulator with GFD :petri_dish:

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/) [![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange.svg)](https://numpy.org/) [![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/) [![Numba](https://img.shields.io/badge/Numba-JIT-red.svg)](https://numba.pydata.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Skin Diffusion Simulator using Generalized Finite Differences**

*Advanced computational framework for modeling substance diffusion through biological tissues*

</div>

---

## :star2: Overview

This repository presents a **state-of-the-art computational framework** for simulating diffusion processes in human skin tissue using the **Generalized Finite Differences (GFD)** method. Designed for researchers in pharmacology, dermatology, cosmetics, and biomedical engineering, this simulator combines mathematical rigor with computational efficiency to deliver accurate, high-performance simulations.

### :gear: Key Features

- **:rocket: High-Performance Computing**: Numba JIT compilation for near-C performance
- **:microscope: Scientific Accuracy**: Second-order spatial accuracy with automatic stability verification
- **:abacus: Advanced Numerics**: Generalized Finite Differences for irregular mesh support
- **:bar_chart: Dataset Generation**: Automated large-scale dataset creation for ML applications
- **:art: Scientific Visualization**: Publication-quality contour plots and animations
- **:zap: Parallel Processing**: Multi-core optimization for parameter studies
- **:wrench: Modular Design**: Clean, well-documented, and extensible codebase
- **:chart_with_upwards_trend: Scalable**: Linear scaling from small prototypes to large-scale simulations

### :microscope: Applications

| Field | Application | Use Case |
|-------|-------------|----------|
| **Pharmacology** :pill: | Drug Delivery | Transdermal patch optimization, penetration studies |
| **Cosmetics** :lipstick: | Product Development | Skincare absorption analysis, formulation testing |
| **Dermatology** :hospital: | Clinical Research | Treatment efficacy modeling, skin barrier studies |
| **Bioengineering** :dna: | Device Design | Microneedle systems, iontophoresis optimization |
| **Machine Learning** :robot: | Data Generation | Training datasets for AI-driven drug discovery |

---

## :gear: Technical Specifications

### Mathematical Model

Solves the **2D transient diffusion equation**:

```
∂u/∂t = ν∇²u
```

Where:
- `u(x,y,t)`: Concentration field [mol/m³]
- `ν`: Diffusion coefficient [m²/s]
- `∇²`: Laplacian operator (GFD approximation)

### Numerical Method

- **Spatial Discretization**: Generalized Finite Differences (9-point stencil)
- **Time Integration**: Explicit Euler scheme with CFL stability control
- **Boundary Conditions**: Mixed Dirichlet/Neumann with physiological relevance
- **Mesh Support**: Structured and unstructured grids
- **Accuracy**: Second-order spatial, first-order temporal

### Performance Benchmarks

| Mesh Size | Nodes | Time Steps | Execution Time | Memory Usage |
|-----------|-------|------------|----------------|---------------|
| Small | 50×50 | 1,000 | ~0.1s | ~50 MB |
| Medium | 100×100 | 5,000 | ~2.5s | ~150 MB |
| Large | 200×200 | 10,000 | ~45s | ~800 MB |

*Benchmarks on Intel i7-8700K @ 3.7GHz with 32GB RAM*

## :open_file_folder: Project Architecture

### Core Components

```
📦 Skin-Diffusion-GFD/
├── :dna: GFD_skin.py             # Core diffusion simulator engine
│   ├── difusion_skin_jit()       # JIT-optimized solver (high performance)
│   ├── difusion_skin()           # Vectorized solver (memory efficient)
│   ├── Gammas()                  # GFD coefficient calculator
│   ├── graph_skin()              # Scientific visualization
│   └── main()                    # Complete workflow orchestrator
│
├── :factory: create_dataset.py   # Automated dataset generation framework
│   ├── DatasetGenerator          # Main dataset creation class
│   ├── Parallel processing       # Multi-core parameter sweeps
│   ├── Progress monitoring       # Real-time generation tracking
│   └── Data organization         # Structured output management
│
├── 📋 requirements.txt           # Python dependencies specification
├── 📄 LICENSE                    # MIT License terms
├── 📖 README.md                  # This comprehensive documentation
│
├── :file_folder: region/         # Computational mesh library
│   ├── skin21.mat                # Small (21×21 nodes)
│   ├── skin41.mat                # Medium mesh (41×41 nodes)
│   ├── skin61.mat                # Medium mesh (61×61 nodes)
│   ├── ...                       # Other mesh sizes.
│   └── custom_meshes/            # User-defined geometries
│
└── :bar_chart: Dataset/          # Generated simulation datasets
    ├── ci_001/                   # Initial concentration: 1 unit
    │   ├── nu_1.00e-06.png       # Diffusion coeff: 1×10⁻⁶ m²/s
    │   ├── nu_2.00e-06.png       # Diffusion coeff: 2×10⁻⁶ m²/s
    │   └── ...                   # Parameter sweep results
    ├── ci_002/                   # Initial concentration: 2 units
    └── ...                       # Additional concentration scenarios
```
---

## :package: Installation & Setup

### :computer: System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.7+ | 3.9+ |
| **RAM** | 4 GB | 16 GB+ |
| **CPU** | 2 cores | 8+ cores |
| **Storage** | 1 GB | 10 GB+ (for datasets) |
| **OS** | Windows/Linux/macOS | Linux (optimal performance) |

### :package: Dependencies

```python
# Core scientific computing
numpy >= 1.20.0          # Numerical computations
scipy >= 1.7.0           # Scientific algorithms
matplotlib >= 3.4.0      # Scientific visualization

# Performance optimization
numba >= 0.54.0          # JIT compilation

# Utilities
tqdm >= 4.62.0           # Progress bars
pathlib                  # Path management
multiprocessing          # Parallel processing
```

### Quick Installation

```bash
# Method 1: Direct installation
git clone https://github.com/gstinoco/Skin-Diffusion-GFD.git
cd Skin-Diffusion-GFD
pip install -r requirements.txt

# Method 2: Virtual environment (recommended)
python -m venv skin_diffusion_env
source skin_diffusion_env/bin/activate  # On Windows: skin_diffusion_env\Scripts\activate
pip install -r requirements.txt

# Method 3: Conda environment
conda create -n skin_diffusion python=3.9
conda activate skin_diffusion
pip install -r requirements.txt
```

### :white_check_mark: Installation Verification

```bash
# Test installation
python -c "import GFD_skin; print('✅ Installation successful!')"

# Run quick demo
python GFD_skin.py
```

---

## :abacus: Methodology & Algorithms

### :books: Mathematical Foundation

#### Diffusion Equation

The skin diffusion process is governed by Fick's second law:

```math
∂u/∂t = ν∇²u
```

Where:
- `u(x,y,t)`: Concentration field [mol/m³]
- `ν`: Diffusion coefficient [m²/s]
- `∇²`: Laplacian operator [1/m²]

### :trophy: GFD Advantages

| Feature | Traditional FD | GFD Method |
|---------|----------------|------------|
| **Mesh Type** | Regular grids only | Irregular meshes |
| **Accuracy** | 2nd order | Higher order possible |
| **Geometry** | Simple shapes | Complex biological shapes |
| **Flexibility** | Limited | High adaptability |
| **Stability** | Standard CFL | Enhanced stability |

---

## :scientist: Research Team

### :man_scientist: Principal Researchers

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

---

## :bookmark_tabs: License

```
MIT License

Copyright (c) 2025 Gerardo Tinoco-Guerrero

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

---

## :handshake: Collaboration Opportunities

We're actively seeking collaborations in:
- :dna: **Biomedical Engineering**: Drug delivery applications
- :robot: **Machine Learning**: Neural network training datasets
- :abacus: **Numerical Methods**: Advanced solver development
- :hospital: **Clinical Research**: Validation with experimental data

---

<div align="center">

**:star: If this project helps your research, please consider giving it a star! :star:**

</div>