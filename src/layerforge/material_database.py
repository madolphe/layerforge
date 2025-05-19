import numpy as np
from pathlib import Path
import yaml
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging
from scipy.interpolate import interp1d
from refractiveindex import RefractiveIndexMaterial
from refractiveindex.refractiveindex import NoExtinctionCoefficient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaterialDatabaseError(Exception):
    """Custom exception for material database errors."""
    pass

class MaterialDatabase:
    def __init__(self, config_path: str = "config.yaml", force_refresh: bool = False):
        """Initialize the material database.
        
        Args:
            config_path: Path to the configuration file
            force_refresh: If True, force refresh of all material data from refractiveindex.info
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.material_dir = Path(self.config['dataset']['material_data_dir'])
        self.material_dir.mkdir(parents=True, exist_ok=True)
        
        self.wavelength_range = self.config['simulation']['wavelength']
        self.materials = {}
        self._load_materials(force_refresh)
    
    def _get_material_mapping(self, material: str) -> str:
        """Get the refractiveindex.info mapping for a material if available."""
        mappings = self.config.get('material_mappings', {})
        if material not in mappings:
            raise MaterialDatabaseError(
                f"No mapping found for material '{material}' in config.yaml and "
                f"no local data file exists at {self.material_dir / f'{material}.txt'}"
            )
        return mappings[material]
    
    def _fetch_from_refractive_index(self, material: str) -> Tuple[List[float], List[float], List[float]]:
        """Fetch optical constants from refractiveindex.info database."""
        mapping = self._get_material_mapping(material)
        shelf, book, page = mapping.split('/')
        
        try:
            # Initialize the material from the database
            material = RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
            
            # Generate wavelength points (in microns for RefractiveIndexMaterial)
            w_min = self.wavelength_range['min']  
            w_max = self.wavelength_range['max']  
            w_points = self.wavelength_range['points']
            wavelengths = np.linspace(w_min, w_max, w_points)
            
            # Get n and k values for each wavelength
            n_values = []
            k_values = []
            for wl in wavelengths:
                try:
                    n = material.get_refractive_index(wl)
                    try:
                        k = material.get_extinction_coefficient(wl)
                        if k is None:
                            logger.warning(f"No extinction coefficient data for {material} at {wl}nm")
                            k = 0.0
                        else:
                            # Handle negative k values
                            if k < 0:
                                logger.warning(f"Negative extinction coefficient ({k:.3f}) for {material} at {wl}nm. Using absolute value.")
                                k = abs(k)
                            
                            # Ensure k is not too large to avoid numerical instability
                            if k > 35:
                                logger.warning(f"Extinction coefficient ({k:.3f}) too large for {material} at {wl}nm. Clamping to 35.")
                                k = 35.0
                    except NoExtinctionCoefficient:
                        # Handle case where material has no extinction coefficient data
                        logger.warning(f"Material {material} has no extinction coefficient data. Using k=0.")
                        k = 0.0
                        
                except Exception as e:
                    logger.error(f"Error getting optical constants for {material} at {wl}nm: {str(e)}")
                    raise MaterialDatabaseError(
                        f"Failed to get optical constants for {material} at {wl}nm: {str(e)}"
                    )
                n_values.append(n)
                k_values.append(k)
            
            # Log the retrieved values for debugging
            logger.debug(f"Retrieved optical constants for {material}:")
            for w, n, k in zip(wavelengths, n_values, k_values):
                logger.debug(f"  {w}nm: n={n:.3f}, k={k:.3f}")
            
            return list(wavelengths), n_values, k_values
            
        except Exception as e:
            print(e)
            raise MaterialDatabaseError(f"Failed to fetch data for {material}: {str(e)}")
    
    def _save_material_data(self, material: str, wavelengths: List[float], 
                          n_values: List[float], k_values: List[float]):
        """Save material data to local file."""
        output_file = self.material_dir / f"{material}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"# Optical constants for {material}\n")
            f.write("# wavelength(nm) n k\n")
            f.write("# Data source: https://refractiveindex.info/\n")
            
            for w, n, k in zip(wavelengths, n_values, k_values):
                f.write(f"{w:.1f} {n:.6f} {k:.6f}\n")
                
        logger.info(f"Saved material data for {material} to {output_file}")
    
    def _load_material_file(self, material_file: Path) -> Tuple[List[float], List[float], List[float]]:
        """Load material data from a local file."""
        wavelengths, n, k = [], [], []
        
        with open(material_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                w, n_val, k_val = map(float, line.strip().split())
                wavelengths.append(w)
                n.append(n_val)
                k.append(k_val)
                
        return wavelengths, n, k
    
    def _load_materials(self, force_refresh: bool = False):
        """Load optical constants for all materials, fetching from web if necessary.
        
        Args:
            force_refresh: If True, force refresh of all material data from refractiveindex.info
        """
        # Get materials from material_mappings instead of materials list
        materials = self.config.get('material_mappings', {}).keys()
        
        for material in materials:
            material_file = self.material_dir / f"{material}.txt"
            
            try:
                if force_refresh or not material_file.exists():
                    logger.info(f"{'Force refreshing' if force_refresh else 'Material data not found locally'} for {material}, fetching from refractiveindex.info...")
                    wavelengths, n, k = self._fetch_from_refractive_index(material)
                    self._save_material_data(material, wavelengths, n, k)
                else:
                    logger.info(f"Loading material data for {material} from local file...")
                    wavelengths, n, k = self._load_material_file(material_file)
                
                # Store the arrays directly
                self.materials[material] = {
                    'wavelengths': np.array(wavelengths),
                    'n': np.array(n),
                    'k': np.array(k)
                }
                
            except Exception as e:
                logger.error(f"Failed to load material {material}: {str(e)}")
                raise
    
    def get_optical_constants(self, material: str, wavelength: float) -> Tuple[float, float]:
        """Get n and k for a material at a specific wavelength."""
        if material not in self.materials:
            raise ValueError(f"Unknown material: {material}")
        
        # Find the index of the closest wavelength
        idx = np.abs(self.materials[material]['wavelengths'] - wavelength).argmin()
        n = float(self.materials[material]['n'][idx])
        k = float(self.materials[material]['k'][idx])
        return n, k
    
    def complex_optical_constants(self, material: str, wavelength: float) -> Tuple[complex, complex]:
        """Get n and k for a material at a specific wavelength."""
        if material not in self.materials:
            raise ValueError(f"Unknown material: {material}")
        n,k = self.get_optical_constants(material, wavelength)        
        return complex(n, k)
    
    def is_material_available(self, material: str) -> bool:
        """Check if material data is available either locally or can be fetched."""
        if (self.material_dir / f"{material}.txt").exists():
            return True
        return material in self.config.get('material_mappings', {}) 