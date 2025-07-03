# Skin Diffusion Simulator with GFD :petri_dish:

Skin diffusion simulator using the Generalized Finite Differences (GFD) method for modeling and visualizing the diffusion of substances through human skin, with applications in pharmacology, cosmetics, and medicine.

All the codes are distributed under MIT License on GitHub and are free to use, modify, and distribute giving the proper copyright notice.

## Description :memo:

This repository contains a high-precision simulator for skin diffusion processes using the Generalized Finite Differences (GFD) method. The implementation allows working with structured meshes while maintaining high numerical accuracy. The project includes tools for dataset generation, visualization, and is optimized for performance with Numba JIT compilation and multi-core processing.

## Project Structure :open_file_folder:

```
.
├── GFD_skin.py            # Main implementation of the diffusion simulator
├── create_dataset.py      # Dataset generator for neural network training
├── requirements.txt       # Dependencies required for the project
├── LICENSE                # MIT License file
├── region/                # Mesh and geometry files
│   ├── skin21.mat         # Mesh for simulations (21x21 nodes)
│   ├── skin41.mat         # Mesh for simulations (41x41 nodes)
│   ├── skin61.mat         # Mesh for simulations (61x61 nodes)
│   └── ...                # Other meshes and geometry files
└── Dataset/               # Directory where generated datasets are stored
    ├── ci_001/            # Subdirectories organized by initial concentration
    └── ...                # Each subdirectory contains images for different nu values
```

## Requirements :scissors:

- Python 3.7+
- NumPy (≥ 1.20.0)
- SciPy (≥ 1.7.0)
- Matplotlib (≥ 3.4.0)
- Numba (≥ 0.54.0)
- tqdm (≥ 4.62.0, for progress bars)

## Installation :wrench:

```bash
# Clone the repository (Note: Repository URL is placeholder - update with actual URL when available)
# git clone https://github.com/gstinoco/Skin-Diffusion-GFD.git
# cd skin-diffusion-gfd

# Install dependencies
pip install -r requirements.txt
```

## Usage :rocket:

### Run an individual simulation

```bash
python GFD_skin.py
```

This will run a simulation with the default parameters and generate a visualization of the diffusion process.

### Generate a complete dataset

```bash
python create_dataset.py
```

Confirmation will be requested before starting the generation. The dataset will be saved in the `Dataset/` directory with the structure:

```
Dataset/
├── ci_001/
│   ├── nu_1.00e-06.png
│   ├── nu_2.00e-06.png
│   └── ...
├── ci_002/
│   └── ...
└── ...
```

## Simulation Parameters :1234:

### Main physical parameters

- **nu**: Diffusion coefficient (m²/s)
- **u_init**: Initial concentration at the boundary

### Numerical parameters

- **t**: Number of time steps
- **dt**: Time step size

## Methodology :abacus:

The simulator uses the Generalized Finite Differences (GFD) method, which allows working with structured meshes while maintaining high precision. The method calculates Gamma coefficients for each mesh node, representing the contributions of neighboring nodes in the approximation of the differential operator (Laplacian).

The implementation includes:

1. Optimized calculation of GFD coefficients using pseudoinverse matrices
2. Automatic verification of numerical stability (CFL condition)
3. Dirichlet boundary conditions at the inlet and Neumann at the edges
4. Optimization with Numba JIT for intensive calculations
5. Vectorized operations with NumPy for improved performance

The code provides two implementations of the diffusion solver:
- `difusion_skin_jit`: Optimized with Numba JIT for maximum performance
- `difusion_skin`: Vectorized implementation using NumPy's advanced indexing

The dataset generator (`create_dataset.py`) leverages multiprocessing to efficiently generate large datasets for neural network training, with automatic parameter variation and result organization.

## Researchers :scientist:

All the codes presented were developed by:

  - **Dr. Gerardo Tinoco Guerrero** :mexico:<br>
    Universidad Michoacana de San Nicolás de Hidalgo<br>
    Aula CIMNE-Morelia<br>
    gerardo.tinoco@umich.mx<br>
    https://orcid.org/0000-0003-3119-770X

  - **Dr. Francisco Javier Domínguez Mota** :mexico:<br>
    Universidad Michoacana de San Nicolás de Hidalgo<br>
    Aula CIMNE-Morelia<br>
    francisco.mota@umich.mx<br>
    https://orcid.org/0000-0001-6837-172X

  - **Dr. José Alberto Guzmán Torres** :mexico:<br>
    Universidad Michoacana de San Nicolás de Hidalgo<br>
    Aula CIMNE-Morelia<br>
    jose.alberto.guzman@umich.mx<br>
    https://orcid.org/0000-0002-9309-9390

## Students :man_student:

  - **Ángel Emeterio Calvillo Vázquez** :mexico:<br>
    Universidad Michoacana de San Nicolás de Hidalgo<br>
    1025501x@umich.mx

## References :books:

More details on the Methods presented in these codes can be found in the following publications:

Numerical Solution of Diffusion Equation using a Method of Lines and Generalized Finite Differences  
G. Tinoco-Guerrero, F. J. Domínguez-Mota, J. A. Guzmán-Torres, and J. G. Tinoco-Ruiz  
Revista Internacional de Métodos Numéricos para Cálculo y Diseño en Ingeniería, Vol. 38 (2), 2022.  
http://dx.doi.org/10.23967/j.rimni.2022.06.003

## Contributions :handshake:

Contributions are welcome. Please feel free to open an issue or submit a pull request.

## License :bookmark_tabs:

[MIT](LICENSE)

## Contact :email:

For questions or collaborations, please contact [gerardo.tinoco@umich.mx](mailto:gerardo.tinoco@umich.mx).