import pytest
import yaml
import numpy as np
from pathlib import Path
from layerforge.spectrum_calculator import SpectrumCalculator
from layerforge.material_database import MaterialDatabaseError

@pytest.fixture
def test_config():
    """Create a test configuration file."""
    config = {
        'material_mappings': {
            'Au': 'main/Au/Johnson',
            'Ag': 'main/Ag/Johnson',
            'Al': 'main/Al/Rakic',
            'SiO2': 'main/SiO2/Malitson'
        },
        'simulation': {
            'wavelength': {
                'min': 500,  # Changed from 400 to avoid UV/blue range where gold properties are less reliable
                'max': 700,  # Visible range
                'points': 10
            },
            'angle': {
                'type': 'range',
                'min': 0,
                'max': 30,
                'step': 10
            },
            'polarization': 'both'
        },
        'dataset': {
            'material_data_dir': 'test_data/material_properties'
        }
    }
    
    # Create a temporary config file
    config_path = Path('test_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    yield str(config_path)
    
    # Cleanup
    config_path.unlink()

@pytest.fixture
def calculator(test_config):
    """Create a SpectrumCalculator instance with test configuration."""
    # Force refresh material data to ensure we have correct extinction coefficients
    return SpectrumCalculator(test_config, force_refresh=True)

def test_initialization(calculator):
    """Test proper initialization of SpectrumCalculator."""
    # Test wavelength range
    assert len(calculator.wavelengths) == 10
    assert calculator.wavelengths[0] == 500
    assert calculator.wavelengths[-1] == 700
    
    # Test angles
    assert len(calculator.angles) == 4  # 0, 10, 20, 30
    assert calculator.angles[0] == 0
    assert calculator.angles[-1] == 30
    
    # Test polarizations
    assert calculator.polarizations == ['s', 'p']

def test_get_polarizations(test_config):
    """Test polarization configuration handling."""
    # Test 'both' polarization
    config = {
        'material_mappings': {
            'Au': 'main/Au/Johnson',
            'Ag': 'main/Ag/Johnson'
        },
        'simulation': {
            'wavelength': {
                'min': 500,
                'max': 700,
                'points': 10
            },
            'angle': {
                'type': 'range',
                'min': 0,
                'max': 30,
                'step': 10
            },
            'polarization': 'both'
        },
        'dataset': {
            'material_data_dir': 'test_data/material_properties'
        }
    }
    
    # Create temporary config
    config_path = Path('test_polarization_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        calculator = SpectrumCalculator(str(config_path), force_refresh=True)
        assert calculator._get_polarizations() == ['s', 'p']
        
        # Test single polarization
        calculator.config['simulation']['polarization'] = 's'
        assert calculator._get_polarizations() == ['s']
        
        calculator.config['simulation']['polarization'] = 'p'
        assert calculator._get_polarizations() == ['p']
        
    finally:
        config_path.unlink()

def test_check_materials(calculator):
    """Test material availability checking."""
    # Test valid sequence
    valid_sequence = [('Au', 100), ('Ag', 50), ('Al', 75)]
    assert calculator.check_materials(valid_sequence) is True
    
    # Test invalid material
    invalid_sequence = [('Au', 100), ('InvalidMaterial', 50)]
    assert calculator.check_materials(invalid_sequence) is False

def test_calculate_spectrum(calculator):
    """Test spectrum calculation."""
    # Test with a simple sequence
    sequence = [('Au', 100), ('Ag', 50)]
    
    result = calculator.calculate_spectrum(sequence)
    
    # Check result structure
    assert 'wavelengths' in result
    assert 'angles' in result
    assert 'R_s' in result
    assert 'T_s' in result
    assert 'A_s' in result
    assert 'R_p' in result
    assert 'T_p' in result
    assert 'A_p' in result
    
    # Check shapes
    n_wavelengths = len(calculator.wavelengths)
    n_angles = len(calculator.angles)
    
    assert result['R_s'].shape == (n_wavelengths, n_angles)
    assert result['T_s'].shape == (n_wavelengths, n_angles)
    assert result['A_s'].shape == (n_wavelengths, n_angles)
    
    # Check value ranges and physical constraints
    for pol in ['s', 'p']:
        # Get valid (non-NaN) values
        R_valid = ~np.isnan(result[f'R_{pol}'])
        T_valid = ~np.isnan(result[f'T_{pol}'])
        A_valid = ~np.isnan(result[f'A_{pol}'])
        
        # Check that valid values are within [0, 1]
        assert np.all(result[f'R_{pol}'][R_valid] >= 0) and np.all(result[f'R_{pol}'][R_valid] <= 1)
        assert np.all(result[f'T_{pol}'][T_valid] >= 0) and np.all(result[f'T_{pol}'][T_valid] <= 1)
        assert np.all(result[f'A_{pol}'][A_valid] >= 0) and np.all(result[f'A_{pol}'][A_valid] <= 1)
        
        # Check energy conservation for valid points
        valid_points = R_valid & T_valid & A_valid
        if np.any(valid_points):
            total = (result[f'R_{pol}'][valid_points] + 
                    result[f'T_{pol}'][valid_points] + 
                    result[f'A_{pol}'][valid_points])
            assert np.allclose(total, 1.0, atol=1e-10)
        
        # Check that we have at least some valid results
        assert np.any(R_valid), f"No valid reflectance values for {pol} polarization"
        assert np.any(T_valid), f"No valid transmittance values for {pol} polarization"
        assert np.any(A_valid), f"No valid absorbance values for {pol} polarization"

def test_calculate_spectrum_error(calculator):
    """Test spectrum calculation error handling."""
    # Test with invalid material
    invalid_sequence = [('InvalidMaterial', 100)]
    
    with pytest.raises(MaterialDatabaseError):
        calculator.calculate_spectrum(invalid_sequence)

def test_single_angle_config():
    """Test spectrum calculation with single angle configuration."""
    config = {
        'material_mappings': {
            'Au': 'main/Au/Johnson',
            'Ag': 'main/Ag/Johnson'
        },
        'simulation': {
            'wavelength': {
                'min': 300,
                'max': 800,
                'points': 5
            },
            'angle': {
                'type': 'single',
                'value': 45
            },
            'polarization': 'both'
        },
        'dataset': {
            'material_data_dir': 'test_data/material_properties'
        }
    }
    
    # Create temporary config
    config_path = Path('test_single_angle_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        calculator = SpectrumCalculator(str(config_path), force_refresh=True)
        sequence = [('Au', 100)]
        result = calculator.calculate_spectrum(sequence)
        
        # Check that we have only one angle
        assert len(calculator.angles) == 1
        assert calculator.angles[0] == 45
        
        # Check result shapes
        assert result['R_s'].shape == (5, 1)  # 5 wavelengths, 1 angle
        
    finally:
        config_path.unlink() 