import pytest
import yaml
from pathlib import Path
from layerforge.material_generator import MaterialGenerator

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
            'max_layers': 36,
            'thickness': {
                'min': 5,
                'max': 200,
                'step': 5
            }
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
def generator(test_config):
    """Create a MaterialGenerator instance with test configuration."""
    return MaterialGenerator(test_config)

def test_initialization(generator):
    """Test proper initialization of MaterialGenerator."""
    # Test materials list (order doesn't matter)
    assert set(generator.materials) == {'Au', 'Ag', 'Al', 'SiO2'}
    assert generator.max_layers == 36
    assert generator.thickness_range == {'min': 5, 'max': 200, 'step': 5}

def test_generate_random_layer(generator):
    """Test generation of random layers."""
    # Test basic layer generation
    material, thickness = generator.generate_random_layer()
    assert material in generator.materials
    assert thickness >= generator.thickness_range['min']
    assert thickness <= generator.thickness_range['max']
    assert (thickness - generator.thickness_range['min']) % generator.thickness_range['step'] == 0
    
    # Test exclusion of material
    material, thickness = generator.generate_random_layer(exclude_material='Au')
    assert material != 'Au'
    
    # Test that we can still generate a layer after excluding one material
    # (since we have more than one material available)
    material, thickness = generator.generate_random_layer(exclude_material='Au')
    assert material in ['Ag', 'Al', 'SiO2']

def test_generate_sequence(generator):
    """Test generation of material sequences."""
    # Test with specified number of layers
    sequence = generator.generate_sequence(3)
    assert len(sequence) == 3
    
    # Check no consecutive same materials
    for i in range(len(sequence) - 1):
        assert sequence[i][0] != sequence[i + 1][0]
    
    # Test random number of layers
    sequence = generator.generate_sequence()
    assert 1 <= len(sequence) <= generator.max_layers
    
    # Test that we can generate a sequence longer than the number of materials
    long_sequence = generator.generate_sequence(10)  # We have 4 materials but can generate 10 layers
    assert len(long_sequence) == 10
    
    # Verify no consecutive same materials in long sequence
    for i in range(len(long_sequence) - 1):
        assert long_sequence[i][0] != long_sequence[i + 1][0]
    
    # Test error for invalid number of layers
    with pytest.raises(ValueError):
        generator.generate_sequence(0)  # Less than 1 layer
    
    with pytest.raises(ValueError):
        generator.generate_sequence(generator.max_layers + 1)  # More than max_layers

def test_sequence_conversion(generator):
    """Test conversion between sequence and string representations."""
    # Test sequence to string
    sequence = [('Au', 100), ('Ag', 50), ('Al', 75)]
    sequence_str = generator.sequence_to_string(sequence)
    assert sequence_str == 'Au_100+Ag_50+Al_75'
    
    # Test string to sequence
    recovered_sequence = generator.string_to_sequence(sequence_str)
    assert recovered_sequence == sequence
    
    # Test invalid string format
    with pytest.raises(ValueError):
        generator.string_to_sequence('invalid_format')

def test_optical_constants(generator):
    """Test retrieval of optical constants."""
    # Test valid material with both n and k
    n, k = generator.get_optical_constants('Au', 600)
    assert isinstance(n, float)
    assert isinstance(k, float)
    assert k > 0  # Gold should have non-zero extinction
    
    # Test material that might not have extinction coefficient (like SiO2)
    n, k = generator.get_optical_constants('SiO2', 600)
    assert isinstance(n, float)
    assert isinstance(k, float)
    # k might be 0 for transparent materials
    assert k >= 0
    
    # Test invalid material
    with pytest.raises(ValueError):
        generator.get_optical_constants('InvalidMaterial', 600)

def test_material_cache(generator):
    """Test material caching mechanism."""
    # First call should create cache entry
    material1 = generator._get_material('Au')
    assert 'Au' in generator._material_cache
    
    # Second call should use cached material
    material2 = generator._get_material('Au')
    assert material1 is material2  # Same object reference 