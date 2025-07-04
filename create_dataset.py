'''
Skin Diffusion Dataset Generator
================================

This module implements a comprehensive dataset generator for skin diffusion simulations
using the Generalized Finite Differences (GFD) method. It creates large-scale datasets
for machine learning applications in dermatological research and pharmaceutical modeling.

Main Features:
-------------
- Parallel Processing: Utilizes multiprocessing for efficient large-scale dataset generation
- Parameter Space Exploration: Systematic sampling of diffusion coefficients and initial concentrations
- Automated Data Management: Organized directory structure with compression capabilities
- Progress Monitoring: Real-time progress tracking with detailed logging
- Memory Optimization: Efficient memory usage for handling large datasets
- Error Handling: Robust error handling and recovery mechanisms

Physical Model:
--------------
The simulations are based on the diffusion equation in biological tissue:

    ∂u/∂t = ν∇²u

Where:
- u: concentration field
- ν: diffusion coefficient (varies by substance and tissue properties)
- t: time
- ∇²: Laplacian operator

Boundary Conditions:
- Dirichlet condition at inlet (constant concentration)
- Neumann conditions at lateral boundaries (no-flux)

Numerical Method:
----------------
The simulations use the Generalized Finite Differences (GFD) method implemented
in the GFD_skin module, which provides:
- High precision on unstructured meshes
- Optimized computation with Numba JIT compilation
- Automatic numerical stability verification (CFL condition)
- Vectorized operations for performance

Dataset Structure:
-----------------
Generated datasets follow this hierarchical organization:

```
dataset_root/
├── ci_001/
│   ├── nu_1.00000000e-08.png
│   ├── nu_2.00000000e-08.png
│   └── ...
├── ci_002/
│   ├── nu_1.00000000e-08.png
│   └── ...
└── compressed/
    ├── ci_001.zip
    └── ci_002.zip
```

Data Files:
----------
- PNG images: High-resolution contour plots of concentration distributions
- Compressed archives: ZIP files for efficient storage and transfer
- Organized by concentration: Each initial concentration has its own folder

Applications:
------------
- Machine learning model training for drug delivery prediction
- Parameter estimation in pharmacokinetic modeling
- Sensitivity analysis of diffusion processes
- Validation datasets for numerical methods
- Educational resources for computational biology

Performance Characteristics:
---------------------------
- Typical simulation time: 0.1-2 seconds per parameter combination
- Memory usage: ~50-100 MB per simulation (depending on mesh size)
- Parallel efficiency: Near-linear scaling with CPU cores
- Storage: ~1-5 MB per simulation (uncompressed)

Development Information:
-----------------------
Developed by: Dr. Gerardo Tinoco Guerrero
Institution: Universidad Michoacana de San Nicolás de Hidalgo
Email: gerardo.tinoco@umich.mx

Funding:
- Secretary of Science, Humanities, Technology and Innovation, SECIHTI, México
- Aula CIMNE-Morelia, México
- SIIIA-MATH: Soluciones de Ingeniería, México

Development Timeline:
- Initial Development: May 2025
- Latest Revision: July 2025
- Version: 1.0

Dependencies:
------------
- numpy: Numerical computations
- scipy: Scientific computing and I/O operations
- tqdm: Progress bar visualization
- pathlib: Modern path handling
- multiprocessing: Parallel processing
- zipfile: Data compression
- GFD_skin: Custom module for diffusion simulations

Notes:
-----
- Ensure sufficient disk space before generating large datasets
- Monitor system resources during parallel execution
- Verify mesh file compatibility before starting generation
- Consider using compression for long-term storage
'''

# Standard library imports
import logging
import multiprocessing as mp
import shutil
import time
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third-party imports
import numpy as np
import scipy.io
from tqdm import tqdm

# Local imports
from GFD_skin import difusion_skin_jit, Gammas, graph

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

DEFAULT_REGION = 'skin61'
DEFAULT_TIME_STEPS = 500
DEFAULT_MAX_WORKERS = None
STABILITY_THRESHOLD = 0.05
COMPRESSION_LEVEL = 6

# ============================================================================
# DATASET GENERATOR CLASS
# ============================================================================

class DatasetGenerator:
    '''
    Dataset generator for neural network training on skin diffusion simulations.
    
    This class handles the parallel generation of skin diffusion simulations using the
    Generalized Finite Differences (GFD) method. It manages directory creation, simulation
    execution, image saving, and final report generation.
    
    The generated dataset includes simulation images with different parameters:
    - Diffusion coefficient (nu): Controls the diffusion speed of the compound
    - Initial concentration (u_init): Initial concentration of the compound at the surface
    
    Key features:
    - Parallel processing to optimize generation times
    - Automatic numerical stability validation (Courant criterion)
    - Automatic file organization by initial concentration
    - Optional image compression to optimize storage
    - Detailed progress reports and final statistics
    
    Attributes:
        base_dir (Path): Base directory where the dataset will be saved
        region (str): Region name (mesh file without .mat extension)
        x, y (numpy.ndarray): Computational mesh coordinates
        m, n (int): Mesh dimensions (rows and columns)
        Gamma (numpy.ndarray): Pre-calculated GFD coefficients for the region
        t_steps (int): Number of time steps for simulations
    '''
    
    def __init__(self, base_dir="Dataset", region=DEFAULT_REGION):
        '''
        Initialize the dataset generator.
        
        Sets up the basic generator parameters, loads the computational mesh
        for the specified region, and pre-calculates the GFD coefficients
        necessary for diffusion simulations.
        
        Args:
            base_dir (str, optional): Base directory to store the dataset.
                                    A subdirectory with the region name will be created.
                                    Default: 'Dataset'
            region (str, optional): Name of the region to simulate. Must correspond
                                  to a .mat file in the 'region/' folder.
                                  Default: 'skin61'
        
        Raises:
            FileNotFoundError: If the region mesh file is not found
            Exception: If there are errors loading the mesh or calculating GFD coefficients
        
        Note:
            The final dataset directory will be: base_dir/region/
            Example: Dataset/skin61/
        '''
        self.base_dir = Path(base_dir) / region
        self.region = region
        self.t_steps = DEFAULT_TIME_STEPS
        
        # Load mesh data and initialize coefficients
        self._load_mesh_data()
    
    # ========================================================================
    # INITIALIZATION AND SETUP METHODS
    # ========================================================================
        
    def _load_mesh_data(self):
        '''
        Load computational mesh data from the region file.
        
        Reads the .mat file corresponding to the specified region and extracts:
        - x, y coordinates of mesh nodes
        - m, n mesh dimensions
        - Pre-calculated Gamma coefficients for the GFD method
        
        The file must be located at: region/{region}.mat
        
        Raises:
            FileNotFoundError: If the region file is not found
            KeyError: If the file does not contain expected variables
            Exception: If there are errors processing the mesh data
        
        Note:
            The Gamma coefficients contain the geometric information necessary
            to apply the Generalized Finite Differences method.
        '''
        try:
            mesh_file = f'region/{self.region}.mat'
            datos = scipy.io.loadmat(mesh_file)
            self.x, self.y = datos["x"], datos["y"]
            self.m, self.n = self.x.shape
            
            # Calculate Gamma coefficients (only once)
            L = np.vstack([[0], [0], [2], [0], [2]])
            self.Gamma = Gammas(self.x, self.y, L)
            
            logger.info(f"✅ Mesh loaded: {self.m}x{self.n} nodes")
            
        except Exception as e:
            logger.error(f"❌ Error loading mesh: {e}")
            raise
    
    def _create_directory_structure(self, u_init_values):
        '''
        Create the directory structure needed to organize the dataset.
        
        Establishes the folder hierarchy that will be used to store
        simulation images organized by initial concentration.
        Each u_init value will have its own directory where images
        corresponding to different nu values will be saved.
        
        Created structure:
        base_dir/
        ├── ci_001/           # Simulations with initial concentration 1
        ├── ci_002/           # Simulations with initial concentration 2
        ├── ci_003/           # Simulations with initial concentration 3
        └── ...
        
        Args:
            u_init_values (numpy.ndarray): Array with initial concentration values
                                         that will be used in simulations
        
        Note:
            - Directory names follow the format "ci_XXX" where XXX is
              the initial concentration value with zero padding
            - All directories are created at once to avoid race conditions
              in parallel processing
        '''
        # Create base directory (including parent directories)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each concentration
        for u_init in u_init_values:
            concentration_dir = self.base_dir / f"ci_{u_init:03d}"
            concentration_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # SIMULATION METHODS
    # ========================================================================
    
    def _generate_single_simulation(self, params):
        '''
        Generate an individual skin diffusion simulation with specified parameters.
        
        Executes a complete diffusion process simulation using the GFD method,
        including numerical stability validation, diffusion equation resolution,
        and final image saving.
        
        Simulation process:
        1. Extract and validate input parameters
        2. Verify numerical stability (Courant criterion)
        3. Configure initial and boundary conditions
        4. Solve diffusion equation step by step
        5. Save final concentration image
        6. Return result with statistics
        
        Args:
            params (tuple): Tuple with simulation parameters:
                          (nu, u_init, simulation_id) where:
                          - nu (float): Diffusion coefficient
                          - u_init (float): Initial concentration at surface
                          - simulation_id (int): Unique simulation identifier
        
        Returns:
            dict: Dictionary with result information:
                - 'success': True if successful, False if failed
                - 'nu': Diffusion coefficient used
                - 'u_init': Initial concentration used
                - 'simulation_id': Simulation identifier
                - 'image_path': Generated image file path (if successful)
                - 'execution_time': Execution time in seconds
                - 'courant_number': Courant number for stability verification
                - 'time_steps': Number of time steps used
                - 'error': Error message (if failed)
        
        Note:
            Simulations that don't meet the numerical stability criterion
            are automatically rejected to avoid incorrect results.
        '''
        nu, u_init, sim_id = params
        
        try:
            start_time = time.time()
            
            # Configure simulation parameters
            t, dt, alpha = self._configure_simulation_parameters(nu)
            
            # Run the diffusion simulation
            u_final = difusion_skin_jit(self.m, self.n, t, dt, nu, u_init, self.Gamma.copy())
            
            # Save simulation result as image
            image_path = self._save_simulation_image(u_final, nu, u_init)
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(nu, u_init, sim_id, image_path, 
                                             execution_time, alpha, t)
            
        except Exception as e:
            return self._create_failure_result(nu, u_init, sim_id, str(e))
    
    def _configure_simulation_parameters(self, nu):
        '''
        Configure simulation parameters ensuring numerical stability.
        
        Calculates the time step and number of steps needed to maintain
        numerical stability according to the Courant criterion. If the Courant
        number exceeds the stability threshold, automatically adjusts
        the number of time steps.
        
        The stability criterion used is:
        α = ν * Δt / Δx² ≤ 0.5
        
        Args:
            nu (float): Diffusion coefficient
            
        Returns:
            tuple: (time_steps, delta_time, courant_number) where:
                  - time_steps (int): Adjusted number of time steps
                  - delta_time (float): Time step in seconds
                  - courant_number (float): Calculated Courant number
        '''
        t = self.t_steps
        T = np.linspace(0, 3600, t)
        dt = T[1] - T[0]
        
        # Calculate minimum grid spacing for stability check
        dx_min = np.min(np.sqrt((self.x[1:, :] - self.x[:-1, :])**2 + 
                              (self.y[1:, :] - self.y[:-1, :])**2))
        alpha = nu * dt / dx_min**2
        
        # Adjust time steps to ensure numerical stability
        while alpha > STABILITY_THRESHOLD:
            t = int(t * 1.1)
            T = np.linspace(0, 3600, t)
            dt = T[1] - T[0]
            alpha = nu * dt / dx_min**2
        
        return t, dt, alpha
    
    def _create_success_result(self, nu, u_init, sim_id, image_path, 
                              execution_time, alpha, t):
        '''
        Create a successful result dictionary with all simulation information.
        
        Args:
            nu (float): Diffusion coefficient
            u_init (float): Initial concentration
            sim_id (int): Simulation identifier
            image_path (Path): Image file path
            execution_time (float): Execution time in seconds
            alpha (float): Courant number
            t (int): Number of time steps
            
        Returns:
            dict: Dictionary with complete successful result information
        '''
        return {
            'success': True,
            'nu': nu,
            'u_init': u_init,
            'simulation_id': sim_id,
            'image_path': str(image_path),
            'execution_time': execution_time,
            'courant_number': alpha,
            'time_steps': t,
            'error': None
        }
    
    def _create_failure_result(self, nu, u_init, sim_id, error_msg):
        '''
        Create a failed result dictionary with error information.
        
        Args:
            nu (float): Diffusion coefficient
            u_init (float): Initial concentration
            sim_id (int): Simulation identifier
            error_msg (str): Descriptive error message
            
        Returns:
            dict: Dictionary with failed result information
        '''
        return {
            'success': False,
            'nu': nu,
            'u_init': u_init,
            'simulation_id': sim_id,
            'image_path': None,
            'execution_time': 0.0,
            'courant_number': None,
            'time_steps': None,
            'error': error_msg
        }
    
    # ========================================================================
    # IMAGE AND OUTPUT METHODS
    # ========================================================================
    
    def _save_simulation_image(self, u_final, nu, u_init):
        '''
        Save the simulation image using the original graph function.
        
        Generates a PNG image with the visualization of the final concentration
        of the compound in the skin mesh. The file is saved in the directory
        corresponding to the initial concentration with a name that includes
        the diffusion coefficient.
        
        Args:
            u_final (np.array): Final simulation result (concentrations)
            nu (float): Diffusion coefficient used
            u_init (float): Initial concentration used
            
        Returns:
            Path: Path of the saved image file
            
        Note:
            The filename format is: nu_{value}.png
            where {value} is the diffusion coefficient with 8 decimals.
        '''
        # Define file path
        concentration_dir = self.base_dir / f"ci_{u_init:03d}"
        filename = f"nu_{nu:.8f}.png"
        image_path = concentration_dir / filename
        
        # Use the original graph function directly with the desired filename
        graph(self.x, self.y, u_final, str(image_path))
        
        return image_path
    
    # ========================================================================
    # MAIN DATASET GENERATION METHOD
    # ========================================================================
    
    def generate_dataset(self, nu_min, nu_max, u_init_min, u_init_max,
                        t_steps=DEFAULT_TIME_STEPS, max_workers=DEFAULT_MAX_WORKERS, 
                        compress_images=True):
        '''
        Generate the complete dataset of skin diffusion simulations.
        
        This is the main method for dataset generation. It coordinates the entire process
        from parameter configuration to final report generation, executing simulations
        in parallel to optimize processing time.
        
        Complete process:
        1. Parameter configuration and value ranges
        2. Creation of organized directory structure
        3. Generation of simulation list to execute
        4. Parallel execution of simulations with progress tracking
        5. Generation of final reports with statistics
        6. Optional image compression for storage optimization
        
        Args:
            nu_min (int): Minimum diffusion coefficient value (coefficient * 1e-8)
                         Example: nu_min=1 → 1e-8 m²/s
            nu_max (int): Maximum diffusion coefficient value (coefficient * 1e-8)
                         Example: nu_max=100 → 100e-8 m²/s
            u_init_min (int): Minimum initial concentration value at surface
            u_init_max (int): Maximum initial concentration value at surface
            t_steps (int, optional): Number of time steps for each simulation.
                                   Default: 1000
            max_workers (int, optional): Maximum number of parallel processes.
                                       None for automatic detection. Default: 4
            compress_images (bool, optional): Whether to compress images by folder to
                                            optimize GitHub uploads. Default: True
            
        Returns:
            tuple: (successful_results, failed_results) where:
                  - successful_results (list): List of successful simulations
                  - failed_results (list): List of failed simulations
        
        Note:
            - For nu: All integer coefficients between min and max are generated
              (e.g.: nu_min=1, nu_max=5 → [1e-8, 2e-8, 3e-8, 4e-8, 5e-8])
            - For u_init: All integer values between min and max are generated
              (e.g.: u_init_min=1, u_init_max=3 → [1, 2, 3])
            - If compress_images=True, ZIP files are created for each concentration folder
            - Execution time depends on number of simulations and available workers
        '''

        start_time = time.time()
        
        # Configure execution parameters
        max_workers = self._configure_workers(max_workers)
        self.t_steps = t_steps
        
        # Generate parameter ranges and simulation list
        u_init_values, nu_values = self._generate_parameter_ranges(nu_min, nu_max, u_init_min, u_init_max)
        simulation_params = self._create_simulation_list(u_init_values, nu_values)
        
        # Setup directory structure
        self._create_directory_structure(u_init_values)
        
        logger.info(f"🚀 Starting generation: {len(simulation_params):,} simulations with {max_workers} workers")
        
        # Execute simulations in parallel
        successful_results, failed_results = self._execute_parallel_simulations(simulation_params, max_workers)
        
        # Generate reports and compress if needed
        total_time = time.time() - start_time
        self._generate_final_report(successful_results, failed_results, total_time)
        
        if len(successful_results) > 0 and compress_images:
            self._compress_images_by_folder()
        
        return successful_results, failed_results
    
    def _configure_workers(self, max_workers):
        '''
        Configure the number of worker processes for parallel execution.
        
        Automatically determines the optimal number of processes if not specified,
        considering the number of available cores and leaving free resources
        for the operating system.
        
        Args:
            max_workers (int or None): Maximum number of workers requested.
                                     If None, calculated automatically.
        
        Returns:
            int: Number of workers configured to use
        
        Note:
            - If max_workers is None: uses min(cpu_count - 1, 8)
            - Always leaves at least one core free for the system
            - Limits to maximum 8 workers to avoid overload
        '''
        if max_workers is None:
            return min(mp.cpu_count() - 1, 8)  # Leave one core free
        return max_workers
    
    def _generate_parameter_ranges(self, nu_min, nu_max, u_init_min, u_init_max):
        '''
        Generate parameter ranges for simulations.
        
        Creates arrays with all diffusion coefficient and initial concentration
        values that will be used in simulations, ensuring complete coverage
        of the parameter space.
        
        Args:
            nu_min (int): Minimum coefficient value (multiplied by 1e-8)
            nu_max (int): Maximum coefficient value (multiplied by 1e-8)
            u_init_min (int): Minimum initial concentration
            u_init_max (int): Maximum initial concentration
        
        Returns:
            tuple: (u_init_values, nu_values) where:
                  - u_init_values (np.ndarray): Array of initial concentrations
                  - nu_values (np.ndarray): Array of diffusion coefficients
        '''
        # For u_init: generate all integer values between min and max
        u_init_values = np.arange(u_init_min, u_init_max + 1, dtype=int)
        
        # For nu: generate values from nu_min to nu_max multiplied by 1e-8
        nu_coefficients = np.arange(nu_min, nu_max + 1)
        nu_values = nu_coefficients * 1e-8
        
        return u_init_values, nu_values
    
    def _create_simulation_list(self, u_init_values, nu_values):
        '''
        Create the complete list of parameters for all simulations.
        
        Generates all possible combinations between initial concentration
        values and diffusion coefficients, assigning a unique ID to each
        simulation to facilitate tracking and result organization.
        
        Args:
            u_init_values (np.ndarray): Array of initial concentration values
            nu_values (np.ndarray): Array of diffusion coefficients
        
        Returns:
            list: List of tuples (nu, u_init, sim_id) where:
                 - nu (float): Diffusion coefficient
                 - u_init (int): Initial concentration
                 - sim_id (int): Unique simulation identifier
        '''
        simulation_params = []
        sim_id = 0
        for u_init in u_init_values:
            for nu in nu_values:
                simulation_params.append((nu, u_init, sim_id))
                sim_id += 1
        return simulation_params
    
    def _execute_parallel_simulations(self, simulation_params, max_workers):
        '''
        Execute simulations in parallel with progress tracking.
        
        Uses a process pool to execute multiple simulations simultaneously,
        showing real-time progress bar and classifying results into
        successful and failed.
        
        Args:
            simulation_params (list): List of simulation parameters
            max_workers (int): Maximum number of parallel processes
        
        Returns:
            tuple: (successful_results, failed_results) where:
                  - successful_results (list): Successfully completed simulations
                  - failed_results (list): Failed simulations
        
        Note:
            - Uses ProcessPoolExecutor for true parallelization
            - Shows real-time progress with tqdm
            - Automatically handles result collection
            - Updates success/failure counters dynamically
        '''
        successful_results = []
        failed_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {executor.submit(self._generate_single_simulation, params): params 
                              for params in simulation_params}
            
            # Process results with progress bar
            with tqdm(total=len(simulation_params), desc="🔄 Generating dataset", 
                     unit="sim", ncols=200) as pbar:
                
                for future in as_completed(future_to_params):
                    result = future.result()
                    
                    if result['success']:
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Successful': len(successful_results),
                        'Failed': len(failed_results)
                    })
        
        return successful_results, failed_results
    
    # ========================================================================
    # REPORTING AND COMPRESSION METHODS
    # ========================================================================
    
    def _generate_final_report(self, successful_results, failed_results, total_time):
        '''
        Generate the final report of the dataset generation process.
        
        Creates a detailed report with complete statistics about the generation
        process, including number of successful and failed simulations, success
        rate, total and average time, dataset location and error details if any.
        
        The report is shown both in the log and console to provide immediate
        feedback to the user.
        
        Args:
            successful_results (list): List of successful simulation results
            failed_results (list): List of failed simulation results
            total_time (float): Total execution time in seconds
        
        Note:
            - Automatically calculates success rate as percentage
            - Converts total time to minutes for better readability
            - Shows absolute path of the generated dataset
            - Includes warnings if there are failed simulations
        '''
        total_sims = len(successful_results) + len(failed_results)
        success_rate = (len(successful_results) / total_sims * 100) if total_sims > 0 else 0
        
        logger.info(f"\n✅ Generation completed: {len(successful_results):,}/{total_sims} successful ({success_rate:.1f}%)")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Dataset saved in: {self.base_dir.absolute()}")
        
        if failed_results:
            logger.warning(f"⚠️  {len(failed_results)} simulations failed")
    
    def _compress_images_by_folder(self):
        '''
        Compress images by folder to optimize GitHub uploads.
        
        Creates ZIP files for each concentration folder (ci_XXX) containing
        all corresponding PNG images. This significantly reduces the number
        of files and makes repository management more efficient.
        
        Compression process:
        1. Identifies all concentration folders (ci_XXX)
        2. Creates a ZIP file for each folder
        3. Compresses all PNG images in the folder
        4. Removes original folder to save space
        5. Reports compression statistics
        
        Benefits:
        - Reduces file count for GitHub (1000 files per push limit)
        - Decreases total repository size
        - Facilitates dataset download and distribution
        - Maintains organization by initial concentration
        
        Note:
            - Only processes folders starting with 'ci_'
            - Uses standard ZIP compression with optimal level
            - Shows progress and compression statistics
            - ZIP files maintain original folder name
        '''
        concentration_dirs = [d for d in self.base_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('ci_')]
        
        if not concentration_dirs:
            logger.warning("⚠️  No directories found to compress")
            return
            
        logger.info(f"📦 Compressing {len(concentration_dirs)} folders...")
        
        total_original_size = 0
        total_compressed_size = 0
        
        for concentration_dir in tqdm(concentration_dirs, desc="🗜️  Compressing folders", unit="folder"):
            original_size, compressed_size = self._compress_single_folder(concentration_dir)
            total_original_size += original_size
            total_compressed_size += compressed_size
        
        # Final compression report
        overall_compression = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
        logger.info(f"✅ Compression completed: {overall_compression:.1f}% reduction, {len(concentration_dirs)} ZIP files created")
    
    def _compress_single_folder(self, concentration_dir):
        '''
        Compress a single concentration folder.
        
        Args:
            concentration_dir (Path): Directory to compress
            
        Returns:
            tuple: (original_size, compressed_size)
        '''
        zip_filename = f"{concentration_dir.name}.zip"
        zip_path = self.base_dir / zip_filename
        
        # Get all PNG files in the directory
        png_files = list(concentration_dir.glob("*.png"))
        
        if not png_files:
            return 0, 0
        
        # Calculate original size
        original_size = sum(f.stat().st_size for f in png_files)
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=COMPRESSION_LEVEL) as zipf:
            for png_file in png_files:
                # Add file to ZIP with relative path
                arcname = f"{concentration_dir.name}/{png_file.name}"
                zipf.write(png_file, arcname)
        
        # Calculate compressed size
        compressed_size = zip_path.stat().st_size
        
        # Delete original folder to save disk space
        shutil.rmtree(concentration_dir)
        
        return original_size, compressed_size


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_regions():
    '''
    Get the list of available regions from the 'region' folder.
    
    Scans the 'region' folder for .mat files containing mesh
    definitions for different anatomical regions.
    
    Returns:
        list: Sorted list of available region names
              (filenames without .mat extension)
    
    Note:
        - Returns empty list if 'region' folder doesn't exist
        - Only considers files with .mat extension
        - Names are sorted alphabetically for consistency
    '''
    region_folder = Path('region')
    if not region_folder.exists():
        return []
    
    mat_files = list(region_folder.glob('*.mat'))
    return sorted([mat_file.stem for mat_file in mat_files])


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    '''
    Main function to generate the simulation dataset.
    
    This function is the entry point of the script when executed directly.
    Coordinates the entire dataset generation process, from initial
    configuration to complete execution, providing an interactive
    interface for the user.
    
    Complete process:
    1. Shows application header
    2. Verifies region availability
    3. Requests user configuration (region, parameters, etc.)
    4. Shows configuration summary for confirmation
    5. Executes dataset generation
    6. Handles errors and provides feedback
    
    Note:
        - Allows user to cancel before starting generation
        - Validates existence of region files before proceeding
        - Provides clear error messages if resources are missing
    '''
    print_header()
    
    # Get and validate available regions
    available_regions = get_available_regions()
    if not available_regions:
        print("❌ Error: No region files found in 'region' folder")
        return
    
    # Get user configuration
    config = get_user_configuration(available_regions)
    
    # Run dataset generation
    run_dataset_generation(config)


def print_header():
    '''
    Print the application header.
    
    Shows an informative banner with the script title and purpose,
    providing visual context to the user about the functionality.
    '''
    print("🧠 DATASET GENERATOR FOR NEURAL NETWORK")
    print("🔬 Skin Diffusion Simulations")
    print("="*50)


def get_user_configuration(available_regions):
    '''
    Get configuration parameters through user input.
    
    Guides the user through an interactive process to configure
    all necessary parameters for dataset generation, including
    region, parameter ranges and processing options.
    
    Args:
        available_regions (list): List of available region names
        
    Returns:
        dict: Configuration dictionary with user preferences:
              - 'region': Selected region
              - 'nu_min', 'nu_max': Diffusion coefficient range
              - 'u_init_min', 'u_init_max': Initial concentration range
              - 't_steps': Number of time steps
              - 'max_workers': Number of parallel processes
              - 'compress_images': Whether to compress images
    
    Note:
        - Validates all user inputs
        - Provides default values when appropriate
        - Shows configuration summary for confirmation
    '''
    config = {}
    
    # Select region
    config['region'] = select_region(available_regions)
    
    # Interactive parameter configuration
    print("\n🔧 Parameter Configuration:")
    
    config.update(get_diffusion_parameters())
    config.update(get_concentration_parameters())
    config['t_steps'] = get_time_steps()
    config['max_workers'] = get_max_workers()
    config['compress_images'] = get_compression_preference()
    
    # Show configuration summary and get confirmation
    show_configuration_summary(config)
    
    return config


def select_region(available_regions):
    '''
    Allow user to select a region from available options.
    
    Shows a numbered list of all available regions and
    prompts the user to select one. Includes a default
    option to facilitate usage.
    
    Args:
        available_regions (list): List of available region names
        
    Returns:
        str: Name of the selected region
    
    Note:
        - Provides 'skin61' as default option
        - Validates that selection is within valid range
        - Allows empty input to use default value
    '''
    print("\n🗺️  AVAILABLE REGIONS:")
    for i, region in enumerate(available_regions, 1):
        print(f"   {i:2d}. {region}")
    
    # Region selection
    while True:
        try:
            choice = input(f"\nSelect region (1-{len(available_regions)}) [default: 3 for skin61]: ").strip()
            if not choice:
                return 'skin61'
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_regions):
                return available_regions[choice_idx]
            else:
                print(f"❌ Please enter a number between 1 and {len(available_regions)}")
        except ValueError:
            print("❌ Please enter a valid number")


def get_diffusion_parameters():
    '''
    Get diffusion coefficient parameters from user.
    
    Prompts the user for minimum and maximum values for the diffusion
    coefficient, which will be multiplied by 1e-8 to obtain the
    actual values in m²/s.
    
    Returns:
        dict: Dictionary with 'nu_min' and 'nu_max'
    
    Note:
        - Values are automatically multiplied by 1e-8
        - Validates that min < max and both are positive
        - Example: input 1-900 → coefficients 1e-8 to 900e-8 m²/s
    '''
    print("\n📊 Diffusion coefficient range (nu):")
    while True:
        try:
            nu_min = int(input("  • Minimum coefficient (will be multiplied by 1e-8, e.g., 1): "))
            nu_max = int(input("  • Maximum coefficient (will be multiplied by 1e-8, e.g., 900): "))
            if nu_min >= nu_max:
                print("  ❌ Error: Minimum value must be less than maximum")
                continue
            if nu_min <= 0 or nu_max <= 0:
                print("  ❌ Error: Values must be positive")
                continue
            return {'nu_min': nu_min, 'nu_max': nu_max}
        except ValueError:
            print("  ❌ Error: Please enter valid integer values")


def get_concentration_parameters():
    '''
    Get initial concentration parameters from user.
    
    Prompts the user for minimum and maximum values for the initial
    concentration of the compound on the skin surface.
    
    Returns:
        dict: Dictionary with 'u_init_min' and 'u_init_max'
    
    Note:
        - Validates that min < max and both are positive
        - Values represent relative concentration (dimensionless)
        - Simulations will be generated for all integers in the range
    '''
    print("\n🧪 Initial concentration range (u_init):")
    while True:
        try:
            u_init_min = int(input("  • Minimum value (e.g., 1): "))
            u_init_max = int(input("  • Maximum value (e.g., 100): "))
            if u_init_min >= u_init_max:
                print("  ❌ Error: Minimum value must be less than maximum")
                continue
            if u_init_min <= 0 or u_init_max <= 0:
                print("  ❌ Error: Values must be positive")
                continue
            return {'u_init_min': u_init_min, 'u_init_max': u_init_max}
        except ValueError:
            print("  ❌ Error: Please enter valid integer values")


def get_time_steps():
    '''
    Get number of time steps from user.
    
    Prompts the user for the number of time steps to be used
    in each simulation. A higher number of steps provides greater
    temporal precision but increases computation time.
    
    Returns:
        int: Number of time steps
    
    Note:
        - Must be a positive value
        - Typical values: 400-1000 steps
        - Affects simulation precision and time
    '''
    print("\n⏱️  Time configuration:")
    while True:
        try:
            t_steps = int(input(f"  • Number of time steps (e.g., {DEFAULT_TIME_STEPS}): "))
            if t_steps <= 0:
                print("  ❌ Error: Number of time steps must be positive")
                continue
            return t_steps
        except ValueError:
            print("  ❌ Error: Please enter a valid integer")


def get_max_workers():
    '''
    Get number of workers from user input.
    
    Returns:
        int or None: Number of workers or None for auto
    '''
    print("\n⚡ Parallel processing:")
    while True:
        try:
            workers_input = input("  • Number of parallel workers [default: auto]: ").strip()
            if not workers_input:
                return None
            max_workers = int(workers_input)
            if max_workers <= 0:
                print("  ❌ Error: Number of workers must be positive")
                continue
            return max_workers
        except ValueError:
            print("  ❌ Error: Please enter a valid integer or leave empty for auto")


def get_compression_preference():
    '''
    Get compression preference from user input.
    
    Returns:
        bool: Whether to compress images
    '''
    compress_choice = input("\nCompress images by folder for GitHub? (Y/n): ").strip().lower()
    return compress_choice in ['', 'y', 'yes', 's', 'si']


def show_configuration_summary(config):
    '''
    Show configuration summary and get user confirmation.
    
    Args:
        config (dict): Configuration dictionary
    '''
    # Calculate actual number of values that will be generated
    total_nu_values = config['nu_max'] - config['nu_min'] + 1
    total_u_init_values = config['u_init_max'] - config['u_init_min'] + 1
    
    print(f"\n📋 Configuration: {config['region']} | {total_nu_values * total_u_init_values:,} simulations | {config['max_workers'] or 'Auto'} workers")
    print(f"   nu: {config['nu_min']}e-8 - {config['nu_max']}e-8 | u_init: {config['u_init_min']} - {config['u_init_max']} | t_steps: {config['t_steps']}")
    
    # Confirm execution
    response = input("\nContinue with generation? (y/N): ")
    if response.lower() not in ['y', 'yes', 's']:
        print("❌ Generation cancelled.")
        exit()


def run_dataset_generation(config):
    '''
    Run the dataset generation with the given configuration.
    
    Args:
        config (dict): Configuration dictionary
    '''
    try:
        # Create generator and execute
        generator = DatasetGenerator(region=config['region'])
        generator.t_steps = config['t_steps']  # Set time steps after initialization
        generator.generate_dataset(
            nu_min=config['nu_min'],
            nu_max=config['nu_max'],
            u_init_min=config['u_init_min'],
            u_init_max=config['u_init_max'],
            t_steps=config['t_steps'],
            max_workers=config['max_workers'],
            compress_images=config['compress_images']
        )
        print("\n🎉 Dataset generation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Dataset generation failed: {e}")
        raise


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()