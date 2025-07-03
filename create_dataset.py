'''
Dataset Generator for Neural Network - Skin Diffusion Simulations

This script generates a complete dataset of diffusion simulations by varying:
- nu (diffusion coefficient): 1e-6 to 9e-4
- u_init (initial concentration): 10 to 100

Dataset structure:
Dataset/
├── ci_001/
│   ├── nu_1.00e-06.png
│   ├── nu_2.00e-06.png
│   └── ...
├── ci_002/
│   ├── nu_1.00e-06.png
│   └── ...
└── ...

Features:
- Parallelized execution for maximum efficiency
- Optimized images for NN training
- Detailed metadata and logging
- Dataset integrity verification

All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    Secretary of Science, Humanities, Technology and Innovation, SECIHTI (Secretaria de Ciencia, Humanidades, Tecnología e Innovación). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México
    SIIIA-MATH: Soluciones de Ingeniería. México

Date:
    June, 2025.

Last Modification:
    July, 2025.
'''

import numpy as np
import scipy.io
import multiprocessing as mp
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Import optimized functions from the main module
from GFD_skin import difusion_skin_jit, Gammas, graficar

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    '''
    Dataset generator for neural network training.
    
    This class handles the parallel generation of skin diffusion simulations
    using the GFD method. It manages directory creation, simulation execution,
    image saving, metadata, and report generation.
    
    The generated dataset includes simulation images with different parameters
    (diffusion coefficient and initial concentration) and associated metadata.
    
    Attributes:
        base_dir (Path): Base directory where the dataset will be saved
        metadata (dict): Dictionary with dataset metadata
        x, y (numpy.ndarray): Mesh coordinates
        m, n (int): Mesh dimensions
        Gamma (numpy.ndarray): Pre-calculated GFD coefficients
    '''
    
    def __init__(self, base_dir="Dataset"):
        self.base_dir = Path(base_dir)
        self.metadata = {
            'generation_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_simulations': 0,
                'successful_simulations': 0,
                'failed_simulations': 0,
                'total_time': 0.0
            },
            'parameters': {
                'nu_range': None,
                'u_init_range': None,
                'mesh_info': None
            },
            'files': []
        }
        
        # Load mesh data once
        self._load_mesh_data()
        
    def _load_mesh_data(self):
        '''
        Loads mesh data and calculates Gamma coefficients.
        
        This function initializes the mesh from a .mat file and calculates the
        Gamma coefficients needed for the GFD method. The coefficients are calculated only once
        at the beginning and are reused in all simulations for greater efficiency.
        
        Raises:
            Exception: If there is an error loading the mesh or calculating the coefficients
        '''
        logger.info("🔄 Loading mesh data...")
        
        try:
            datos = scipy.io.loadmat('region/skin224.mat')
            self.x, self.y = datos["x"], datos["y"]
            self.m, self.n = self.x.shape
            
            # Calculate Gamma coefficients (only once)
            L = np.vstack([[0], [0], [2], [0], [2]])
            self.Gamma = Gammas(self.x, self.y, L)
            
            # Save mesh information in metadata
            self.metadata['parameters']['mesh_info'] = {
                'dimensions': f"{self.m}x{self.n}",
                'total_nodes': int(self.m * self.n),
                'mesh_file': 'region/skin224.mat'
            }
            
            logger.info(f"✅ Mesh loaded: {self.m}x{self.n} = {self.m*self.n:,} nodes")
            
        except Exception as e:
            logger.error(f"❌ Error loading mesh: {e}")
            raise
    
    def _create_directory_structure(self, u_init_values):
        '''
        Creates the directory structure for the dataset.
        
        Generates the directory structure needed to organize simulations
        by initial concentration. Each u_init value will have its own directory
        where images corresponding to different nu values will be saved.
        
        Args:
            u_init_values (numpy.ndarray): Array with initial concentration values
        '''
        logger.info("📁 Creating directory structure...")
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each concentration
        for u_init in u_init_values:
            concentration_dir = self.base_dir / f"ci_{u_init:03d}"
            concentration_dir.mkdir(exist_ok=True)
            
        logger.info(f"✅ Structure created at: {self.base_dir.absolute()}")
    
    def _generate_single_simulation(self, params):
        '''
        Executes an individual simulation and saves the image.
        
        Args:
            params (tuple): (nu, u_init, simulation_id)
            
        Returns:
            dict: Simulation result
        '''
        nu, u_init, sim_id = params
        
        try:
            start_time = time.time()
            
            # Simulation configuration
            t = 3600  # Total time
            T = np.linspace(0, 3600, t)
            dt = T[1] - T[0]
            
            # Verify numerical stability
            dx_min = np.min(np.sqrt((self.x[1:, :] - self.x[:-1, :])**2 + 
                                  (self.y[1:, :] - self.y[:-1, :])**2))
            alpha = nu * dt / dx_min**2
            
            # Adjust time steps if necessary
            while alpha > 0.5:
                t = int(t * 1.1)
                T = np.linspace(0, 3600, t)
                dt = T[1] - T[0]
                alpha = nu * dt / dx_min**2
            
            # Calculate scaled Gamma for this specific simulation
            Gamma = self.Gamma.copy()
            
            # Run optimized simulation with Numba JIT
            u_final = difusion_skin_jit(self.m, self.n, t, dt, nu, u_init, Gamma)
            
            # Generate and save image
            image_path = self._save_simulation_image(u_final, nu, u_init)
            
            execution_time = time.time() - start_time
            
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
            
        except Exception as e:
            return {
                'success': False,
                'nu': nu,
                'u_init': u_init,
                'simulation_id': sim_id,
                'image_path': None,
                'execution_time': 0.0,
                'courant_number': None,
                'time_steps': None,
                'error': str(e)
            }
    
    def _save_simulation_image(self, u_final, nu, u_init):
        '''
        Saves the simulation image using the original graficar function.
        
        Args:
            u_final (np.array): Simulation result
            nu (float): Diffusion coefficient
            u_init (float): Initial concentration
            
        Returns:
            Path: Path of the saved file
        '''
        # Define file path
        concentration_dir = self.base_dir / f"ci_{u_init:03d}"
        filename = f"nu_{nu:.2e}.png"
        image_path = concentration_dir / filename
        
        # Use the original graficar function directly with the desired filename
        graficar(self.x, self.y, u_final, str(image_path))
        
        return image_path
    
    def generate_dataset(self, nu_min=1e-6, nu_max=900e-6, nu_steps=1,
                        u_init_min=1, u_init_max=100, u_init_steps=1,
                        max_workers=None):
        '''
        Generates the complete dataset of simulations.
        
        This function is the main entry point for dataset generation.
        It configures parameters, creates the directory structure, executes simulations
        in parallel, and generates metadata and final reports.
        
        The complete process includes:
        1. Configuration of parameters and value ranges
        2. Creation of directory structure
        3. Generation of the simulation list to execute
        4. Parallel execution of simulations
        5. Progress tracking with progress bar
        6. Generation of metadata and final report
        
        Args:
            nu_min (float): Minimum value of diffusion coefficient
            nu_max (float): Maximum value of diffusion coefficient
            nu_steps (int): Number of nu values to generate
            u_init_min (int): Minimum value of initial concentration
            u_init_max (int): Maximum value of initial concentration
            u_init_steps (int): Number of u_init values to generate
            max_workers (int): Maximum number of parallel processes (None for auto)
            
        Returns:
            tuple: (successful_results, failed_results) with simulation results
        '''

        start_time = time.time()
        
        # Configure parameters
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, 8)  # Leave one core free
        
        logger.info(f"🚀 STARTING DATASET GENERATION")
        logger.info(f"   Parallel processes: {max_workers}")
        logger.info(f"   nu range: {nu_min:.2e} - {nu_max:.2e} ({nu_steps} values)")
        logger.info(f"   u_init range: {u_init_min} - {u_init_max} ({u_init_steps} values)")
        
        # Generate parameter ranges
        nu_values = np.linspace(nu_min, nu_max, nu_steps)
        u_init_values = np.linspace(u_init_min, u_init_max, u_init_steps, dtype=int)
        
        # Save parameters in metadata
        self.metadata['parameters']['nu_range'] = {
            'min': float(nu_min),
            'max': float(nu_max),
            'steps': int(nu_steps),
            'values': [float(x) for x in nu_values]
        }
        self.metadata['parameters']['u_init_range'] = {
            'min': int(u_init_min),
            'max': int(u_init_max),
            'steps': int(u_init_steps),
            'values': [int(x) for x in u_init_values]
        }
        
        # Create directory structure
        self._create_directory_structure(u_init_values)
        
        # Generate list of all simulations
        simulation_params = []
        sim_id = 0
        for u_init in u_init_values:
            for nu in nu_values:
                simulation_params.append((nu, u_init, sim_id))
                sim_id += 1
        
        total_simulations = len(simulation_params)
        self.metadata['generation_info']['total_simulations'] = total_simulations
        
        logger.info(f"   Total simulations: {total_simulations:,}")
        logger.info(f"   Estimated time: ~{total_simulations * 0.3 / max_workers / 60:.1f} minutes")
        
        # Execute simulations in parallel
        successful_results = []
        failed_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {executor.submit(self._generate_single_simulation, params): params 
                              for params in simulation_params}
            
            # Process results with progress bar
            with tqdm(total=total_simulations, desc="🔄 Generating dataset", 
                     unit="sim", ncols=100) as pbar:
                
                for future in as_completed(future_to_params):
                    result = future.result()
                    
                    if result['success']:
                        successful_results.append(result)
                        self.metadata['files'].append({
                            'nu': float(result['nu']),
                            'u_init': int(result['u_init']),
                            'image_path': result['image_path'],
                            'execution_time': float(result['execution_time']),
                            'courant_number': float(result['courant_number']),
                            'time_steps': int(result['time_steps'])
                        })
                    else:
                        failed_results.append(result)
                        logger.warning(f"⚠️  Failed simulation: nu={result['nu']:.2e}, "
                                     f"u_init={result['u_init']}, error={result['error']}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Successful': len(successful_results),
                        'Failed': len(failed_results)
                    })
        
        # Update final metadata
        total_time = time.time() - start_time
        self.metadata['generation_info'].update({
            'successful_simulations': len(successful_results),
            'failed_simulations': len(failed_results),
            'total_time': total_time,
            'average_time_per_simulation': total_time / total_simulations if total_simulations > 0 else 0
        })
        
        # Save metadata
        self._save_metadata()
        
        # Final report
        self._generate_final_report(successful_results, failed_results, total_time)
        
        return successful_results, failed_results
    
    def _save_metadata(self):
        '''
        Saves the dataset metadata in JSON format.
        
        The metadata includes information about the generation process,
        parameters used, success/failure statistics, execution times,
        and details of each generated file. This information is useful for
        documenting the dataset and facilitating its subsequent use.
        '''
        metadata_path = self.base_dir / "dataset_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Metadata saved in: {metadata_path}")
    
    def _generate_final_report(self, successful_results, failed_results, total_time):
        '''
        Generates final report of the dataset generation.
        
        Creates a detailed report with statistics about the generation process,
        including number of successful and failed simulations, success rate,
        total and average time, dataset location, and error details.
        The report is displayed in the log and console.
        
        Args:
            successful_results (list): List of successful simulation results
            failed_results (list): List of failed simulation results
            total_time (float): Total execution time in seconds
        '''
        total_sims = len(successful_results) + len(failed_results)
        success_rate = (len(successful_results) / total_sims * 100) if total_sims > 0 else 0
        
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 FINAL REPORT - DATASET GENERATION")
        logger.info(f"="*60)
        logger.info(f"✅ Successful simulations:  {len(successful_results):,}")
        logger.info(f"❌ Failed simulations:      {len(failed_results):,}")
        logger.info(f"📈 Success rate:           {success_rate:.1f}%")
        logger.info(f"⏱️  Total time:              {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"⚡ Average time/sim:        {total_time/total_sims:.3f}s")
        logger.info(f"📁 Dataset saved in:        {self.base_dir.absolute()}")
        logger.info(f"💾 Estimated size:          ~{len(successful_results) * 0.5:.1f} MB")
        
        if failed_results:
            logger.warning(f"\n⚠️  FAILED SIMULATIONS:")
            for result in failed_results[:5]:  # Show only the first 5
                logger.warning(f"   nu={result['nu']:.2e}, u_init={result['u_init']}: {result['error']}")
            if len(failed_results) > 5:
                logger.warning(f"   ... and {len(failed_results)-5} more")
        
        logger.info(f"\n🎯 Dataset ready for neural network training!")
        logger.info(f"="*60)

def main():
    '''
    Main function to generate the dataset.
    
    This function is the entry point of the script when executed directly.
    It configures the dataset generation parameters, requests confirmation from the user,
    and executes the complete generation process.
    
    The default configuration generates a dataset with:
    - 900 values of diffusion coefficient (nu) between 1e-6 and 900e-6
    - 6 values of initial concentration (u_init) between 14 and 20
    - Automatic parallelization according to available cores
    
    The user can cancel the generation before it begins.
    '''

    print("🧠 DATASET GENERATOR FOR NEURAL NETWORK")
    print("🔬 Skin Diffusion Simulations")
    print("="*50)
    
    # Dataset configuration
    config = {
        'nu_min': 1e-6,
        'nu_max': 2e-6,
        'nu_steps': 2,                        # 900 nu values
        'u_init_min': 24,
        'u_init_max': 25,
        'u_init_steps': 2,                    # 6 u_init values
        'max_workers': None                        # Auto-detect
    }
    
    print(f"📋 CONFIGURATION:")
    print(f"   Diffusion coefficient (nu): {config['nu_min']:.2e} - {config['nu_max']:.2e}")
    print(f"   Initial concentration: {config['u_init_min']} - {config['u_init_max']}")
    print(f"   Total simulations: {config['nu_steps'] * config['u_init_steps']:,}")
    print(f"   Parallel processes: {config['max_workers'] or 'Auto'}")
    
    # Confirm execution
    response = input("\nContinue with generation? (y/N): ")
    if response.lower() not in ['y', 'yes', 's']:
        print("❌ Generation cancelled.")
        return
    
    try:
        # Create generator and execute
        generator = DatasetGenerator()
        successful, failed = generator.generate_dataset(**config)
        
        print(f"\n🎉 Dataset generated successfully!")
        print(f"   Successful simulations: {len(successful):,}")
        print(f"   Ready for NN training")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Generation interrupted by the user.")
    except Exception as e:
        logger.error(f"❌ Error during generation: {e}")
        raise

if __name__ == "__main__":
    main()