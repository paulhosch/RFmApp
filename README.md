![title_2](https://github.com/user-attachments/assets/2f22c78e-60b9-4e46-9864-06e53de1bbdd)

![Development Status](https://img.shields.io/badge/status-under--development-yellow.svg)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20Deployment-red)](https://rfmapp.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-green.svg)
[![License: GNU GPL v3](https://img.shields.io/badge/License-GNU%20GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Build Status](https://img.shields.io/badge/build-unstable-yellow.svg)
![Issues](https://img.shields.io/github/issues/paulhosch/RFmApp.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Last Commit](https://img.shields.io/github/last-commit/paulhosch/RFmApp.svg)

## 🚧 Under Development 🚧

**Please note**: RFmApp is currently under active development. Some features may not be fully implemented, and the application is subject to changes. 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Access](#access)
- [Local Installation](#local-installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

**RFmApp** is a web application designed for comprehensive evaluation of Random Forests (Breiman 2001) in flood mapping. This tool allows users to investigate sampling and validation strategies, explore remote sensing input features, and perform detailed feature engineering and analysis. The application is tailored for researchers and professionals in environmental sciences and geospatial analysis, providing a robust platform for cross-site evaluation and model validation.

## Features

- **Cross-Site Evaluation**: Assess model performance across multiple geographic sites.
- **Customizable Sampling and Validation Design**:
  - **Stratified Random Sampling**: Ensures proportional representation of each class in your dataset.
  - **Two-Stage Nested Cross Validation**: Provides a more reliable estimation of model performance.
- **Feature Engineering and Analysis**:
  - **Spatial Distribution**: Analyze the spatial characteristics of your input features.
  - **Value Distribution**: Examine the distribution of values within your dataset.
  - **Correlation Analysis**: Identify and analyze correlations between features.
  - **Importance Analysis**: Determine the relative importance of each feature in the model.
- **More to Come**: Future updates will bring additional features and improvements.

## Web Access

The RFmApp has been deployed using Streamlit Cloud and is accessible in all standard Web Browsers at the following URL:

[https://rfmapp.streamlit.app/](https://rfmapp.streamlit.app/)

## Local Installation

### Prerequisites

- Python 3.8+
- pip
- Google Earth Engine API Service Account

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/RFmApp.git
    cd RFmApp
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Authenticate Earth Engine Project**:
    - Follow the instructions [here](https://developers.google.com/earth-engine/guides/service_account) to authenticate your Google Earth Engine project.
4. **Run the application**:
    ```bash
    streamlit run main.py
    ```

## Usage

Detailed usage instructions are implemented within the web application interface itself. Once the application is running, you can upload your datasets, configure sampling strategies, and begin analyzing flood extend with ease.

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your changes align with the project’s goals and style guidelines.

## License

This project is licensed under the **GNU General Public License v3.0**. For more details, see the [LICENSE](https://www.gnu.org/licenses/gpl-3.0) file.

![License: GNU GPL v3](https://img.shields.io/badge/License-GNU%20GPL%20v3-blue.svg)

## Acknowledgments

Special thanks to the developers of the Random Forests algorithm and the contributors to the open-source libraries used in this project. Your tools and knowledge have made this project possible.
