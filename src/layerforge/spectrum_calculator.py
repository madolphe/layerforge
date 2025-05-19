import numpy as np
import torch
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Union
from tmm_fast import coh_tmm
from layerforge.material_database import MaterialDatabase, MaterialDatabaseError
import logging
import copy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SpectrumCalculator:
    def __init__(self, config_path: str = "config.yaml", force_refresh: bool = False):
        """Initialize the spectrum calculator.
        
        Args:
            config_path: Path to the configuration file
            force_refresh: If True, force refresh of all material data from refractiveindex.info
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.material_db = MaterialDatabase(config_path, force_refresh=force_refresh)
        self.wavelengths = np.linspace(
            self.config['simulation']['wavelength']['min'],
            self.config['simulation']['wavelength']['max'],
            self.config['simulation']['wavelength']['points']
        )
        
        # Set up angles based on configuration
        if self.config['simulation']['angle']['type'] == 'single':
            self.angles = [self.config['simulation']['angle']['value']]
        else:
            self.angles = np.arange(
                self.config['simulation']['angle']['min'],
                self.config['simulation']['angle']['max'] + self.config['simulation']['angle']['step'],
                self.config['simulation']['angle']['step']
            )
            
        self.polarizations = self._get_polarizations()
    
    def _get_polarizations(self) -> List[str]:
        """Get list of polarizations to compute."""
        pol = self.config['simulation']['polarization']
        if pol == 'both':
            return ['s', 'p']
        return [pol]
    
    def check_materials(self, sequence: List[Tuple[str, float]]) -> bool:
        """Check if all materials in the sequence are available."""
        for material, _ in sequence:
            if not self.material_db.is_material_available(material):
                logger.warning(f"Material {material} is not available and cannot be fetched")
                return False
        return True
    
    def calculate_spectrum(self, sequence: List[Tuple[str, float]]) -> Dict:
        """Calculate optical properties (R, T, A) for a given material sequence.
        
        Args:
            sequence: List of (material, thickness) tuples
            
        Returns:
            Dictionary containing:
            - wavelengths: array of wavelength points
            - angles: array of angle points
            - R, T, A: arrays of shape (n_wavelengths, n_angles) for each polarization
        """
        if not self.check_materials(sequence):
            raise MaterialDatabaseError("Some materials in the sequence are not available")

        # Convert wavelengths and angles to torch tensors
        wavelengths = torch.tensor(self.wavelengths, dtype=torch.float64) * 1e-9  # meters
        angles = torch.tensor(self.angles, dtype=torch.float64) * (np.pi / 180)   # radians

        n_wavelengths = len(self.wavelengths)  # Use length of original array
        n_angles = len(self.angles)  # Use length of original array
        n_layers = len(sequence) + 2  # air + layers + substrate

        # Build refractive index matrix M: shape (1, n_layers, n_wavelengths)
        M = torch.ones((1, n_layers, n_wavelengths), dtype=torch.complex128)
        # Thickness vector T: shape (1, n_layers)
        T = torch.zeros((1, n_layers), dtype=torch.float64)
        # Air
        T[0, 0] = torch.tensor(float('inf'))
        M[0, 0, :] = torch.tensor(1.0 + 0j, dtype=torch.complex128)
        
        # Sequence layers
        for i, (material, thickness) in enumerate(sequence):
            nks = [self.material_db.complex_optical_constants(material, float(w * 1e9)) for w in wavelengths]
            # Check for NaN values in optical constants
            for w_idx, nk in enumerate(nks):
                if np.isnan(nk.real) or np.isnan(nk.imag):
                    raise ValueError(f"Invalid optical constants (NaN) for material {material} "
                                   f"at wavelength {self.wavelengths[w_idx]} nm")
            M[0, i+1, :] = torch.tensor(nks, dtype=torch.complex128)
            T[0, i+1] = thickness
        
        # Substrate (glass)
        M[0, -1, :] = torch.tensor(1.5 + 0j, dtype=torch.complex128)
        T[0, -1] = torch.tensor(float('inf'))

        result = {
                'wavelengths': self.wavelengths,
                'angles': self.angles
            }

        for pol in self.polarizations:
            out = coh_tmm(
                pol,
                M,
                T,
                angles,
                wavelengths,
                device='cpu'
            )

            # Debug tensor shapes
            logger.debug(f"R shape: {out['R'].shape}")
            logger.debug(f"T shape: {out['T'].shape}")
            logger.debug(f"angles shape: {angles.shape}")
            logger.debug(f"wavelengths shape: {wavelengths.shape}")
            logger.debug(f"n_wavelengths: {n_wavelengths}")
            logger.debug(f"n_angles: {n_angles}")

            # Check for NaN values in the output with detailed information
            R_nan = torch.isnan(out['R'])
            T_nan = torch.isnan(out['T'])
            
            if R_nan.any() or T_nan.any():
                # Find problematic combinations
                problematic = []
                # Get the actual shapes from the tensors
                r_shape = out['R'].shape
                t_shape = out['T'].shape
                logger.debug(f"R_nan shape: {R_nan.shape}")
                logger.debug(f"T_nan shape: {T_nan.shape}")
                
                # Ensure we don't exceed array bounds
                max_w_idx = min(r_shape[1], n_wavelengths)
                max_a_idx = min(r_shape[2], n_angles)
                
                # Iterate over the actual dimensions
                for w_idx in range(max_w_idx):
                    for a_idx in range(max_a_idx):
                        if R_nan[0, w_idx, a_idx] or T_nan[0, w_idx, a_idx]:
                            problematic.append(
                                f"wavelength={self.wavelengths[w_idx]}nm, "
                                f"angle={self.angles[a_idx]}°"
                            )
                
                # Get material properties at problematic wavelengths
                material_info = []
                for w_idx in range(max_w_idx):
                    if any(R_nan[0, w_idx]) or any(T_nan[0, w_idx]):
                        material_info.append(f"\nAt wavelength {self.wavelengths[w_idx]}nm:")
                        for i, (material, _) in enumerate(sequence):
                            n, k = self.material_db.get_optical_constants(material, self.wavelengths[w_idx])
                            material_info.append(f"  {material}: n={n:.3f}, k={k:.3f}")
                
                raise ValueError(
                    f"NaN values detected in {pol} polarization calculation.\n"
                    f"Problematic combinations:\n" + "\n".join(problematic) +
                    "\nMaterial properties at problematic wavelengths:" + "".join(material_info)
                )

            # Transpose to match expected shape (wavelengths, angles)
            R_out = copy.deepcopy(out['R'][0].cpu())
            T_out = copy.deepcopy(out['T'][0].cpu())

            result[f'R_{pol}'] = R_out.numpy().T
            result[f'T_{pol}'] = T_out.numpy().T
            result[f'A_{pol}'] = 1 - result[f'R_{pol}'] - result[f'T_{pol}'] 

            # Final check for NaN values in results
            if np.isnan(result[f'R_{pol}']).any() or np.isnan(result[f'T_{pol}']).any() or np.isnan(result[f'A_{pol}']).any():
                raise ValueError(f"NaN values detected in final results for {pol} polarization")

        return result

def main():
    """Example usage of the SpectrumCalculator."""
    from material_generator import MaterialGenerator
    
    # Create generators
    material_gen = MaterialGenerator()
    spectrum_calc = SpectrumCalculator()
    
    # Generate a random sequence
    sequence = material_gen.generate_sequence(3)
    print("Generated sequence:")
    print(material_gen.sequence_to_string(sequence))
    
    # Check materials availability
    if spectrum_calc.check_materials(sequence):
        # Calculate spectrum
        result = spectrum_calc.calculate_spectrum(sequence)
        
        # Print some results
        print("\nResults shape:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.shape}")
    else:
        print("\nSome materials are not available!")

if __name__ == "__main__":
    main() 