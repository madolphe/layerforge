import numpy as np
import torch
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import json
from layerforge.material_generator import MaterialGenerator
from layerforge.spectrum_calculator import SpectrumCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the dataset generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set random seeds for reproducibility
        np.random.seed(self.config['dataset']['random_seed'])
        torch.manual_seed(self.config['dataset']['random_seed'])
        
        # Initialize generators
        self.material_gen = MaterialGenerator(config_path)
        self.spectrum_calc = SpectrumCalculator(config_path)
        
        # Create output directory
        self.dataset_dir = Path(self.config['dataset']['dataset_dir'])
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_spectrum_tensor(self, spectrum_data: Dict) -> torch.Tensor:
        """Create a numerical encoding of the spectrum data.
        
        Returns:
            torch.Tensor: Shape (num_properties, num_wavelengths, num_angles)
        """
        properties = self.config['dataset']['properties']
        polarizations = ['s', 'p'] if self.config['simulation']['polarization'] == 'both' else [self.config['simulation']['polarization']]
        
        # Initialize output tensor
        num_wavelengths = len(spectrum_data['wavelengths'])
        num_angles = len(spectrum_data['angles'])
        num_properties = len(properties) * len(polarizations)
        
        spectrum_tensor = torch.zeros(num_properties, num_wavelengths, num_angles)
        
        # Fill tensor with data
        idx = 0
        for prop in properties:
            for pol in polarizations:
                key = f"{prop}_{pol}"
                spectrum_tensor[idx] = torch.tensor(spectrum_data[key])
                idx += 1
                
        return spectrum_tensor
    
    def generate_sample(self) -> Tuple[str, torch.Tensor]:
        """Generate a single sample (sequence and its spectrum).
        
        Returns:
            Tuple[str, torch.Tensor]: Material sequence string and spectrum tensor
        """
        # Generate random number of layers
        num_layers = np.random.randint(
            self.config['dataset']['min_layers'],
            self.config['dataset']['max_layers'] + 1
        )
        
        # Generate sequence and calculate spectrum
        sequence = self.material_gen.generate_sequence(num_layers)
        spectrum = self.spectrum_calc.calculate_spectrum(sequence)
        
        # Convert sequence to string format and spectrum to tensor
        sequence_str = self.material_gen.sequence_to_string(sequence)
        spectrum_tensor = self._create_spectrum_tensor(spectrum)
        
        return sequence_str, spectrum_tensor
    
    def generate_dataset(self):
        """Generate the complete dataset and save it."""
        num_samples = self.config['dataset']['num_samples']
        
        # Calculate split sizes
        train_size = int(num_samples * self.config['dataset']['train_split'])
        val_size = int(num_samples * self.config['dataset']['val_split'])
        test_size = num_samples - train_size - val_size
        
        # Generate samples
        logger.info(f"Generating {num_samples} samples...")
        sequences = []
        spectra = []
        
        for _ in tqdm(range(num_samples)):
            seq_str, spec_tensor = self.generate_sample()
            sequences.append(seq_str)
            spectra.append(spec_tensor)
        
        # Stack spectra
        stacked_spectra = torch.stack(spectra)
        
        # Split into train/val/test
        indices = torch.randperm(num_samples)
        
        train_sequences = [sequences[i] for i in indices[:train_size]]
        train_spectra = stacked_spectra[indices[:train_size]]
        
        val_sequences = [sequences[i] for i in indices[train_size:train_size+val_size]]
        val_spectra = stacked_spectra[indices[train_size:train_size+val_size]]
        
        test_sequences = [sequences[i] for i in indices[train_size+val_size:]]
        test_spectra = stacked_spectra[indices[train_size+val_size:]]
        
        # Save datasets
        logger.info("Saving datasets...")
        
        # Save sequences as JSON
        with open(self.dataset_dir / "train_sequences.json", 'w') as f:
            json.dump(train_sequences, f)
        with open(self.dataset_dir / "val_sequences.json", 'w') as f:
            json.dump(val_sequences, f)
        with open(self.dataset_dir / "test_sequences.json", 'w') as f:
            json.dump(test_sequences, f)
            
        # Save spectra as tensors
        ext = self.config['dataset']['save_format']
        torch.save(train_spectra, self.dataset_dir / f"train_spectra.{ext}")
        torch.save(val_spectra, self.dataset_dir / f"val_spectra.{ext}")
        torch.save(test_spectra, self.dataset_dir / f"test_spectra.{ext}")
        
        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "materials": self.config['materials'],
            "properties": self.config['dataset']['properties'],
            "wavelength_range": self.config['simulation']['wavelength'],
            "angle_range": self.config['simulation']['angle'],
            "spectrum_shape": list(stacked_spectra.shape[1:]),
            "sequence_format": "material1_thickness1+material2_thickness2+...",
            "thickness_range": self.config['simulation']['thickness']
        }
        
        with open(self.dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Dataset generated and saved to {self.dataset_dir}")
        logger.info(f"Train samples: {train_size}")
        logger.info(f"Validation samples: {val_size}")
        logger.info(f"Test samples: {test_size}")

def main():
    """Generate the dataset using configuration from config.yaml."""
    generator = DatasetGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main() 