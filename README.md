# Inverse Material Design with Transformers

This project explores the use of transformer architectures to solve inverse problems in material design, specifically focusing on multi-layer materials with specific optical properties.

## Project Structure

```
.
├── config.yaml           # Configuration file for materials and simulation parameters
├── requirements.txt      # Python dependencies
├── src/
│   ├── __init__.py
│   ├── material_generator.py  # Functions for generating random material sequences
│   └── spectrum_calculator.py # Functions for computing optical properties
├── tests/               # Unit tests for the project
├── data/                # Directory for storing generated datasets
└── notebooks/          # Jupyter notebooks for analysis and visualization
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project uses tmm_fast to compute optical properties of multi-layer materials. The workflow consists of:

1. Generating random material sequences with specified thicknesses
2. Computing their optical properties (absorption/reflection spectra)
3. Creating paired datasets for supervised learning

### Configuration

Edit `config.yaml` to modify:
- Available materials
- Layer thickness ranges
- Number of layers
- Wavelength ranges
- Dataset parameters

### Generating Material Sequences

```python
from src.material_generator import MaterialGenerator

generator = MaterialGenerator()
sequence = generator.generate_sequence()
print(generator.sequence_to_string(sequence))  # Output format: "Au_10+Ag_20+SiO2_15"
```

### Generating Datasets

You can generate datasets using the main.py script:

```bash
python main.py
```

This will:
1. Generate random material sequences
2. Calculate their optical properties
3. Save the results in the data/ directory

### Running Tests

The project includes comprehensive unit tests to ensure reliability. Run the tests using pytest:

```bash
pytest tests/
```

The tests cover:
- Material sequence generation
- Optical property calculations
- Configuration handling
- Error cases and edge conditions

## Dataset Format

The generated dataset consists of pairs:
- X: Sequence of materials and thicknesses (e.g., "Au_10+Ag_20+SiO2_15")
- Y: Dictionary of wavelength-value pairs representing optical properties

## License

MIT License 