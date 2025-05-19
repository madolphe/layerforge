import random
import yaml
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
from refractiveindex import RefractiveIndexMaterial

class MaterialGenerator:

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the material generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.material_mappings = self.config['material_mappings']
        # Sort materials to maintain consistent order
        self.materials = sorted(self.material_mappings.keys())
        self.max_layers = self.config['simulation']['max_layers']
        self.thickness_range = self.config['simulation']['thickness']
        
        # Initialize material cache
        self._material_cache = {}
        
    def _get_material(self, material_name: str) -> Optional[RefractiveIndexMaterial]:
        """Get material from cache or create new instance."""
        if material_name not in self._material_cache:
            try:
                # Get the mapping for the material
                mapping = self.material_mappings[material_name]
                shelf, book, page = mapping.split('/')
                
                # Create material instance
                material = RefractiveIndexMaterial(
                    shelf=shelf,
                    book=book,
                    page=page
                )
                self._material_cache[material_name] = material
            except Exception as e:
                print(f"Warning: Could not load material {material_name}: {str(e)}")
                return None
                
        return self._material_cache[material_name]
        
    def generate_random_layer(self, exclude_material: str = None) -> Tuple[str, float]:
        """Generate a random material layer with random thickness.
        
        Args:
            exclude_material: Material to exclude from selection (to avoid consecutive same materials)
        """
        # Get available materials (excluding the last used material)
        available_materials = [m for m in self.materials if m != exclude_material]
        
        # Select a random material from available ones
        material = random.choice(available_materials)
        
        # Get material properties and generate thickness
        material_obj = self._get_material(material)
        if material_obj is None:
            # If material can't be loaded, try another one
            return self.generate_random_layer(exclude_material)
            
        thickness = random.randrange(
            self.thickness_range['min'],
            self.thickness_range['max'] + 1,
            self.thickness_range['step']
        )
        
        return material, thickness
    
    def generate_sequence(self, num_layers: int = None) -> List[Tuple[str, float]]:
        """Generate a random sequence of materials with their thicknesses.
        Ensures that consecutive layers are not made of the same material.
        """
        if num_layers is None:
            num_layers = random.randint(1, self.max_layers)
            
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")
            
        if num_layers > self.max_layers:
            raise ValueError(
                f"Number of layers ({num_layers}) exceeds maximum allowed ({self.max_layers})"
            )
        
        sequence = []
        last_material = None
        
        for _ in range(num_layers):
            layer = self.generate_random_layer(exclude_material=last_material)
            sequence.append(layer)
            last_material = layer[0]
        
        return sequence
    
    def get_optical_constants(self, material: str, wavelength: float) -> Tuple[float, float]:
        """Get the optical constants (n, k) for a material at a given wavelength.
        
        Args:
            material: Name of the material
            wavelength: Wavelength in nanometers
            
        Returns:
            Tuple of (refractive index, extinction coefficient)
            If extinction coefficient is not available, returns 0.0 for k
        """
        material_obj = self._get_material(material)
        if material_obj is None:
            raise ValueError(f"Could not load material {material}")
            
        n = float(material_obj.get_refractive_index(wavelength))
        try:
            k = float(material_obj.get_extinction_coefficient(wavelength))
        except Exception:
            # If extinction coefficient is not available, use 0.0
            k = 0.0
        
        return n, k
    
    def sequence_to_string(self, sequence: List[Tuple[str, float]]) -> str:
        """Convert a sequence to a string representation using '+' as separator."""
        return "+".join([f"{mat}_{thick}" for mat, thick in sequence])
    
    def string_to_sequence(self, sequence_str: str) -> List[Tuple[str, float]]:
        """Convert a string representation back to a sequence.
        Expects format: material1_thickness+material2_thickness+...
        """
        layers = sequence_str.split("+")
        sequence = []
        
        for layer in layers:
            material, thickness = layer.split("_")
            sequence.append((material, float(thickness)))
            
        return sequence

def main():
    """Example usage of the MaterialGenerator."""
    generator = MaterialGenerator()
    
    # Generate a random sequence
    sequence = generator.generate_sequence(3)
    print("Generated sequence:")
    for material, thickness in sequence:
        print(f"Material: {material}, Thickness: {thickness}nm")
        
        # Get optical constants at 600nm
        n, k = generator.get_optical_constants(material, 600)
        print(f"  Optical constants at 600nm: n={n:.3f}, k={k:.3f}")
    
    # Convert to string representation
    sequence_str = generator.sequence_to_string(sequence)
    print(f"\nString representation: {sequence_str}")
    
    # Convert back to sequence
    recovered_sequence = generator.string_to_sequence(sequence_str)
    print("\nRecovered sequence:")
    for material, thickness in recovered_sequence:
        print(f"Material: {material}, Thickness: {thickness}nm")

if __name__ == "__main__":
    main() 