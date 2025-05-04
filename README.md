# Data Envelopment Analysis (DEA) Dashboard

This project is a web-based dashboard for performing **Data Envelopment Analysis (DEA)** and visualizing the results. The dashboard is built using **Dash** and **Plotly**, and it supports various DEA-related computations, including efficiency analysis, super-efficiency analysis, and visualization of DEA frontiers.

## Features

- **Data Upload**: Upload CSV or Excel files to analyze custom datasets.
- **DEA Analysis**:
  - Compute efficiency scores for Decision-Making Units (DMUs).
  - Perform super-efficiency analysis.
- **Visualization**:
  - 2D DEA frontier graphs.
  - 3D DEA visualizations.
- **Data Management**:
  - Filter, sort, and search datasets.
  - Export selected data and DEA results to Excel.
- **Interactive UI**:
  - Select outputs and inputs for DEA analysis.
  - Randomly select rows by region for analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/MatusMokan/DEA-Analysis-Tool.git>
   cd <DEA-Analysis-Tool>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python dea.py
   ```

4. Open the dashboard in your browser:
   ```
   http://127.0.0.1:8050
   ```

## Reproducing the Python Virtual Environment

To reproduce the Python virtual environment used in this project, follow these steps:

1. Ensure you have Python installed on your system (preferably Python 3.8 or later).
2. Create a new virtual environment:
   ```bash
   python -m venv myenv
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
4. Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

This will install all the necessary libraries and their specific versions to ensure compatibility with the project.

## Dependencies

- **Dash**: For building the web-based dashboard.
- **Plotly**: For creating interactive visualizations.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **dealib**: For DEA and super-efficiency analysis.

## Key Functions

### `calc_dea`
Calculates efficiency scores for DMUs using the `dea` function from `dealib`.

### `compute_super_efficiency`
Performs super-efficiency analysis using the `sdea` function from `dealib`.

### `dea_frontier_plotly`
Generates a 2D DEA frontier graph using Plotly.

### `create_dea_3d_graph`
Generates a 3D DEA visualization for selected inputs and outputs.

## Example Workflow

1. Load a dataset (e.g., `data/mel.xlsx`).
2. Select inputs and outputs for DEA analysis.
3. Generate DEA frontier graphs and analyze efficiency.
4. Export results for further analysis.


## Acknowledgments

- **Dash** and **Plotly** for providing the tools to build interactive dashboards.
- **dealib** for the DEA and super-efficiency analysis functions.

## Contact

For questions or feedback, please contact the project maintainer.