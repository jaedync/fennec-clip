
<h1 align="center">FENNEC Data Clipping Tool</h1>

<p align="center">
  <img src="https://github.com/jaedync/fennec-clip/blob/main/images/Screenshot2023-11-05.png" alt="Sample UI" width="95%">
</p>

## 🚁 Project Overview

The FENNEC Data Clipping Tool is a specialized software designed for the FENNEC Senior Design project at Letourneau University (2023/2024). This tool provides an efficient method for importing and analyzing MAVLink `.bin` files from the Goblin Raw 420 model helicopter.

Our goal is to provide a robust, user-friendly platform for our project team and stakeholders to extract valuable insights and usable dataframes from flight data, enhancing the overall understanding and performance of our aerial systems.

## ✨ Features

- **High-fidelity Data Processing**: Accurately parse and process MAVLink `.bin` files for detailed analysis.
- **Advanced Visualization**: Generate comprehensive visual representations of flight paths and accelerometer data.
- **User-Centric Design**: Intuitive interface and controls ensure ease of use for all user levels.
- **Versatile Data Export**: Export processed data in multiple formats for extended utility and compatibility.

## 🛠 Getting Started

These instructions will guide you through the setup process to get the FENNEC Data Clipping Tool up and running on your local machine.

### Prerequisites

- Node.js (v14.0+)
- Python (v3.8+)

## Required Python Packages

- flask
- flask-cors
- pandas
- werkzeug
- pymavlink
- requests
- openpyxl
- xlsxwriter
- flask_socketio


## 🛠 Installation and Setup

Follow these steps to get the FENNEC Data Clipping Tool up and running:

### Prerequisites

- Ensure you have Python 3.8 or higher installed on your machine.

### Required Python Packages

Install the required Python packages by running the following command:

```sh
pip install flask flask-cors pandas werkzeug pymavlink requests openpyxl xlsxwriter flask_socketio
```

### Running the Server

1. Clone the repository from GitHub:

```sh
git clone https://github.com/yourusername/fennec-data-clipping-tool.git
```

2. Navigate to the cloned directory:

```sh
cd fennec-data-clipping-tool
```

3. Run `backend.py` to start the server:

```sh
python backend.py
```

Your server should now be running at http://localhost:5000 and ready to accept MAVLink `.bin` files for processing.

## 📊 Usage

To utilize the tool, perform the following steps:

1. **Upload Logs**: Drag and drop your `.bin` file into the upload area.
2. **Select Data Range**: Use the timeline to select the start and end points of your data.
3. **Visualize and Analyze**: Observe as the tool charts out the flight path and accelerometer data.
4. **Export**: Choose your desired format and export the data for further analysis.

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact

Should you have any questions, feedback, or want to contribute to the project, please feel free to reach out.

Contact: [me@jaedynchilton.com](mailto:me@jaedynchilton.com)

## 💡 Acknowledgements

- [Chart.js](https://www.chartjs.org/) - For the amazing charting tools.
- [Three.js](https://threejs.org/) - For making 3D rendering a piece of cake.
- The supportive professors at Letourneau University.
