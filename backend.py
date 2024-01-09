from flask import Flask, jsonify, request, render_template, send_from_directory, send_file, Response
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
import os
import re
from bintoexcel import map_mode_number_to_name, calculate_unix_epoch_time_from_timeus, calculate_unix_epoch_time, fetch_weather_data, serialize_obj, match_and_append_weather_data
import simplejson as json
from collections import defaultdict,  Counter
from pymavlink import mavutil
  # Replace "your_module" with the actual module name
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from flask_socketio import SocketIO, emit
import base64
import time
import uuid
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
# Global variable to store the path of the current JSON file
current_json_file = None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize DataFrame
df = pd.DataFrame()
current_excel_file = None

from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2, axis):
    """
    Calculate the north-south (axis='lat') or east-west (axis='lng') 
    great circle distance in meters between two points on the earth.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula components
    a_lat = sin(dlat/2)**2
    a_lon = cos(lat1) * cos(lat2) * sin(dlon/2)**2

    if axis == 'lat':
        a = a_lat
    elif axis == 'lng':
        a = a_lon
    else:
        raise ValueError("Axis must be 'lat' or 'lng'")

    c = 2 * asin(sqrt(a))

    # Radius of Earth in kilometers is 6371. Convert to meters by multiplying by 1000
    distance_in_meters = 6371 * c * 1000
    return distance_in_meters

def normalize_gps_data(gps_df):
    """
    Append distinct lat_m and lng_m columns to the GPS DataFrame.
    """
    ref_lat, ref_lon = gps_df.iloc[0]['Lat'], gps_df.iloc[0]['Lng']
    
    # Calculate north-south distance (latitude difference)
    gps_df['lat_m'] = gps_df.apply(lambda row: haversine_distance(ref_lat, ref_lon, row['Lat'], ref_lon, 'lat'), axis=1)

    # Calculate east-west distance (longitude difference)
    gps_df['lng_m'] = gps_df.apply(lambda row: haversine_distance(ref_lat, ref_lon, ref_lat, row['Lng'], 'lng'), axis=1)
    
    return gps_df

def convert_pressure_to_meters(pressure_in_pascals, temperature_celsius=None):
    """
    Convert pressure from Pascals to altitude in meters using the barometric formula.
    If temperature_celsius is None or not provided, defaults to 15 degrees Celsius (ISA standard conditions at sea level).
    """
    # Default to ISA standard sea level temperature if temperature is None or not provided
    temperature_celsius = 15 if temperature_celsius is None else temperature_celsius

    pressure_in_millibars = pressure_in_pascals / 100
    sea_level_pressure = 1013.25  # ISA standard sea level pressure in millibars
    
    # Convert temperature to Kelvin
    temperature_kelvin = temperature_celsius + 273.15

    # Constants
    R = 8.314462618  # Universal gas constant in J/(mol·K)
    g = 9.80665  # Acceleration due to gravity in m/s^2
    M = 0.0289644  # Molar mass of Earth's air in kg/mol

    # Scale height calculation
    H = R * temperature_kelvin / (g * M)

    # Barometric formula
    altitude = -H * math.log(pressure_in_millibars / sea_level_pressure)
    return altitude

def calculate_average_temperature(weather_data):
    """
    Calculate the average temperature from a list of weather data records.
    
    The function expects each record in the weather_data list to have a 'temp_avg' key with the temperature in Fahrenheit.
    It calculates the average temperature in Celsius.
    
    :param weather_data: List of dictionaries, each containing weather data with a 'temp_avg' key in Fahrenheit.
    :return: The average temperature in Celsius, or None if no valid data is found.
    """
    total_temp_celsius = 0
    count = 0
    for record in weather_data:
        if 'temp_avg' in record:
            # Convert temperature from Fahrenheit to Celsius
            temp_celsius = (record['temp_avg'] - 32) * 5 / 9
            total_temp_celsius += temp_celsius
            count += 1

    return total_temp_celsius / count if count > 0 else None

def convert_bin_to_excel(bin_file_path, excel_file_path, namespace, sid):
    # allowed_packet_types = ["IMU", "MAG", "RCIN", "RCOU", "GPS", "GPA", "BAT", "POWR", "MCU", "BARO", "RATE", "ATT", "VIBE", "HELI", "MODE", "PSCN", "PSCE", "PSCD"]
    allowed_packet_types = ["IMU", "MAG", "RCIN", "RCOU", "GPS", "GPA", "BAT", "POWR", "MCU", "BARO", "RATE", "ATT", "VIBE", "HELI", "MODE", "XKF1", "XKF5"]
    mlog = mavutil.mavlink_connection(bin_file_path)

    # Create a dictionary to hold lists for each allowed packet type and instance
    parsed_data = defaultdict(list)

    emit('status', {'message': 'Starting conversion...', 'progress': 0, 'color': '#FFD700'}, broadcast=True)

    packet_count = 0
    packet_type_counter = Counter()  # For detailed packet type counts
    batch_size = 50000  # Update the progress every 50000 packets

    # Parse the messages and build the data structure
    while True:
        msg = mlog.recv_msg()
        if msg is None:
            break
        msg_dict = msg.to_dict()
        packet_type = msg_dict.get('mavpackettype')
        packet_count = packet_count + 1
        
        if packet_type in allowed_packet_types:
            instance = msg_dict.get('I', 0)
            if packet_type == 'VIBE':
                instance = msg_dict.get('IMU', 0)
            if packet_type == 'MODE':
                mode_name = map_mode_number_to_name(msg_dict.get('Mode', -1))
                msg_dict['ModeName'] = mode_name

            # Create a unique key for each packet type and instance
            key = f"{packet_type}_{instance}" if packet_type in ['IMU', 'VIBE', 'MAG'] else packet_type

            # Append the message dictionary to the list for its packet type and instance
            parsed_data[key].append(msg_dict)

        # Emit progress in batches
        if packet_count % batch_size == 0:
            # Calculate and emit progress
            progress_percentage = "{:.2f}".format(mlog.percent)
            # Check if packet_count is in millions
            if packet_count >= 1000000:
                # Format count in millions with one decimal place
                packet_count_formatted = "{:.2f}m".format(packet_count / 1000000)
            else:
                # Format count in thousands with comma for every three digits
                packet_count_formatted = "{:,}k".format(int(packet_count / 1000))

            emit('status', {
                'message': f'Imported {packet_count_formatted} packets... ({progress_percentage}%)',
                'packet_types': dict(packet_type_counter),
                'progress': 2,
                'color': '#1E90FF'
            }, broadcast=True)
    emit('status', {'message': f'Total MAVLink packets imported: {packet_count:,}!', 'progress': 7, 'color': '#32CD32'}, broadcast=True)

    # Now create DataFrames in one go
    data_frames = {}
    for key, data_list in parsed_data.items():
        # Create the DataFrame for this packet type and instance
        data_frames[key] = pd.DataFrame(data_list).drop(columns=['mavpackettype'], errors='ignore')


    # emit('status', {'message': 'Approximating GPS Time...', 'progress': 10}, broadcast=True)
    # Add closest GPS time to each DataFrame
    gps_df = data_frames.get('GPS', pd.DataFrame())
    if not gps_df.empty:
        gps_df['Unix_Epoch_Time'] = gps_df.apply(calculate_unix_epoch_time, axis=1)
        initial_unix_epoch_time = gps_df['Unix_Epoch_Time'].iloc[0]
        initial_time_us = gps_df['TimeUS'].iloc[0]
        for packet_type, df in data_frames.items():
            df['Unix_Epoch_Time'] = df['TimeUS'].apply(
                lambda x: calculate_unix_epoch_time_from_timeus(x, initial_unix_epoch_time, initial_time_us)
            )

    emit('status', {'message': 'Fetching Weather Data...', 'progress': 15, 'color': '#9370DB'}, broadcast=True)
    # Fetch Weather Data
    min_time = gps_df['Unix_Epoch_Time'].min() if not gps_df.empty else None
    max_time = gps_df['Unix_Epoch_Time'].max() + 900 if not gps_df.empty else None
    # weather_data, _ = fetch_weather_data(min_time, max_time) if min_time and max_time else ([], False)
    
    # Fetching weather data and calculating average temperature
    weather_data, success = fetch_weather_data(min_time, max_time) if min_time and max_time else ([], False)
    avg_temperature_celsius = None
    
    if success and weather_data:
        avg_temperature_celsius = calculate_average_temperature(weather_data)
        if avg_temperature_celsius is not None:
            print(f"Average Temperature: {avg_temperature_celsius} Celsius")
        else:
            print("Failed to calculate average temperature")
    else:
        print("Failed to fetch weather data or data is empty")


    # Normalize GPS Data to latitude and longitude meters if GPS DataFrame exists
    gps_df = data_frames.get('GPS')
    if gps_df is not None:
        gps_df = normalize_gps_data(gps_df)
        data_frames['GPS'] = gps_df  # Update the GPS DataFrame in the dictionary

    # Normalize barometric pressure to altitude meters if BARO DataFrame exists
    baro_df = data_frames.get('BARO')
    if baro_df is not None:
        # Use the average temperature in the altitude calculation
        baro_df['Altitude_Meters_Estimate'] = baro_df['Press'].apply(lambda x: convert_pressure_to_meters(x, avg_temperature_celsius))
        
        # Normalize the altitude to start at zero
        if not baro_df['Altitude_Meters_Estimate'].empty:
            initial_altitude = baro_df['Altitude_Meters_Estimate'].iloc[0]
            baro_df['Altitude_Meters_Estimate'] -= initial_altitude

        data_frames['BARO'] = baro_df  # Update the BARO DataFrame in the dictionary


    # Create the "ALL" table
    all_df = gps_df.copy()
    time_window = 0.2

    # Define the desired order of appending packet types to the "ALL" table
    ordered_packet_types = ["GPS", "RATE", "RCIN", "RCOU", "VIBE_0", "VIBE_1", "HELI",
                            "IMU_0", "IMU_1", "POWR", "MCU", "BAT", "MAG_0", "MAG_1", 
                            # "BARO", "ATT", "GPA", "PSCN", "PSCE", "PSCD"]
                            # "BARO", "ATT", "GPA"]
                            "BARO", "ATT", "GPA", "XKF1", "XKF5"]

    # Rename GPS columns to include "GPS_" prefix
    gps_columns_to_rename = {col: f"GPS_{col}" for col in all_df.columns if col != 'Unix_Epoch_Time'}
    all_df.rename(columns=gps_columns_to_rename, inplace=True)
    emit('status', {'message': 'Processing Packets...', 'progress': 20, 'color': '#FFA500'}, broadcast=True)
    progress = 0
    # Append data in the specified order
    for packet_type in ordered_packet_types:
        if packet_type == 'GPS':
            continue
        if packet_type not in data_frames:
            continue
        progress = progress + 1
        # emit('status', {'message': f'Handling packet type: {packet_type}', 'progress': 22 + progress*4}, broadcast=True)
    
        df = data_frames[packet_type]
        total_packets = len(df)  # Counting the total number of packets for the current packet type
        avg_df = pd.DataFrame()
        
        for gps_time in all_df['Unix_Epoch_Time']:
            min_time = gps_time - time_window
            max_time = gps_time + time_window
            close_rows = df[(df['Unix_Epoch_Time'] >= min_time) & (df['Unix_Epoch_Time'] <= max_time)]
            
            if close_rows.empty:
                avg_row = pd.Series({col: None for col in df.columns})
            else:
                avg_row = close_rows.mean()

            avg_row['Unix_Epoch_Time'] = gps_time
            # Filter out empty or all-NA rows
            if not avg_row.isna().all():
                avg_df = avg_df._append(avg_row, ignore_index=True)

        # Rename columns to include source table
        renamed_columns = {col: f"{packet_type}_{col}" for col in df.columns if col != 'Unix_Epoch_Time'}
        avg_df.rename(columns=renamed_columns, inplace=True)
        
        all_df = pd.merge(all_df, avg_df, on='Unix_Epoch_Time', how='left')

        # Emitting the total packet count for each packet type
        emit('status', {'message': f'Processed {total_packets:,} {packet_type} packets.', 
                        'progress': 22 + progress*4, 'color': '#FFD700'}, broadcast=True)
    # Append weather data to the "ALL" table
    if weather_data:
        all_df = match_and_append_weather_data(all_df, weather_data)

    # Reorder columns to ensure 'Unix_Epoch_Time' is the first column
    reordered_columns = ['Unix_Epoch_Time'] + [col for col in all_df.columns if col != 'Unix_Epoch_Time']
    all_df = all_df[reordered_columns]

    emit('status', {'message': 'Successfully processed all packets! Saving DataFrame...', 'progress': 95, 'color': '#20B2AA'}, broadcast=True)
    
    # Save as Excel
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        all_df.to_excel(writer, sheet_name='ALL', index=False)
        for packet_type, df in data_frames.items():
            reordered_columns = ['Unix_Epoch_Time'] + [col for col in df.columns if col != 'Unix_Epoch_Time']
            df[reordered_columns].to_excel(writer, sheet_name=packet_type, index=False)

def save_df_to_json(df):
    global current_json_file
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.json"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Convert DataFrame to JSON and save to the file
    with open(file_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, ignore_nan=True)

    current_json_file = file_path

# Function to load data from Excel
def load_data_from_excel(file_path):
    global df
    try:
        df = pd.read_excel(file_path, sheet_name='ALL')  # Assuming the sheet name is 'ALL'
        
        required_columns = [
            'GPS_Lat', 'GPS_Lng', 'BARO_Press', 
            'IMU_0_AccX', 'IMU_0_AccY', 'IMU_0_AccZ', 
            'RCOU_C1', 'RCOU_C2', 'RCOU_C3', 'RCOU_C4', 'RCOU_C8'
        ]
        
        if not all(column in df.columns for column in required_columns):
            return "Invalid ArduPilot-ish Data Structure!\nTalk to Jaedyn lol", 400

        df['Datetime_Chicago'] = pd.to_datetime(df['Unix_Epoch_Time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Chicago')
        df['Datetime_Chicago'] = df['Datetime_Chicago'].astype(str)

        save_df_to_json(df)
        
        return "File uploaded and data refreshed", 200
    except Exception as e:
        return str(e), 400
    
# Function to load data
def load_data(file_path):
    global df
    try:
        df = pd.read_csv(file_path)
        
        # Specify the columns you expect
        required_columns = [
            'GPS_Lat', 'GPS_Lng', 'BARO_Press', 
            'IMU_0_AccX', 'IMU_0_AccY', 'IMU_0_AccZ', 
            'RCOU_C1', 'RCOU_C2', 'RCOU_C3', 'RCOU_C4', 'RCOU_C8'
        ]
        
        if not all(column in df.columns for column in required_columns):
            return "Invalid ArduPilot-ish CSV\nTalk to Jaedyn lol", 400

        # Existing code to convert Unix time etc...
        df['Datetime_Chicago'] = pd.to_datetime(df['Unix_Epoch_Time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Chicago')
        df['Datetime_Chicago'] = df['Datetime_Chicago'].astype(str)
        
        save_df_to_json(df)
        
        return "File uploaded and data refreshed", 200
    except Exception as e:
        return str(e), 400

def col_num_to_letter(n):
    """Convert a column number to an Excel-style column letter (e.g., 1 => 'A', 27 => 'AA')."""
    letter = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letter = chr(65 + remainder) + letter
    return letter


from flask_socketio import emit

@socketio.on('upload_and_convert')
def handle_upload_and_convert(json_data):
    global current_excel_file
    filename = json_data.get('filename')
    if not filename:
        emit('status', {'message': 'Filename not provided.'})
        return

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    # Assuming you've already saved the file and filename is valid
    # Perform your existing file conversion and data processing logic here.
    
    file_ext = os.path.splitext(filename)[1]
    
    if file_ext == '.bin':
        # Convert BIN to Excel
        excel_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename.replace('.bin', '.xlsx')))
        convert_bin_to_excel(file_path, excel_file_path, request.namespace, request.sid)  # Perform the conversion
        file_path = excel_file_path  # Update the file_path to the new Excel file
        message, status_code = load_data_from_excel(file_path)
        if status_code == 200:
            current_excel_file = excel_file_path  # Update the variable only if successful
    elif file_ext == '.csv':
        # Load legacy data into DataFrame from CSV
        message, status_code = load_data(file_path)
    elif file_ext == '.xlsx':
        # Load data into DataFrame from Excel
        message, status_code = load_data_from_excel(file_path)
        if status_code == 200:
            current_excel_file = file_path  # Update the variable only if successful
    else:
        emit('status', {'message': 'Unsupported file type', 'progress': 0})
        return
    
    emit('status', {'message': 'Conversion and upload complete', 'progress': 100}, broadcast=True)

@app.route('/upload', methods=['POST'])
def file_upload():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Open the file in a with block to ensure it's closed properly
            with open(file_path, 'wb') as f:
                f.write(file.read())
            return jsonify(message="File uploaded successfully"), 200

        except Exception as e:
            # Log the exception
            print('An error occurred: %s', e)
            return jsonify(error="An error occurred while uploading the file"), 500

    return jsonify(error="Unknown error"), 500

def sanitize_filename(filename):
    # Strip off any file extension first
    filename_without_ext = re.sub(r'\..*$', '', filename)
    
    # Keep only word characters (alphanumeric & underscore)
    safe_str = re.sub(r"[^\w\s-]", '', filename_without_ext)
    
    # Always append .csv
    safe_str += '.csv'
    
    return safe_str

@app.route('/is_data_available', methods=['GET'])
def is_data_available():
    return jsonify({'available': not df.empty})

@app.route('/data', methods=['GET'])
def get_data():
    if current_json_file is None or not os.path.exists(current_json_file):
        return jsonify({'message': 'No data available'}), 404

    return send_file(current_json_file, as_attachment=True, download_name='data.json', mimetype='application/json')
    
@app.route('/speedtest', methods=['GET'])
def speedtest():
    # Generate a fixed size of data, e.g., 10 MB
    data_size_mb = 10
    data = 'A' * data_size_mb * 1024 * 1024  # 1 MB = 1024 * 1024 bytes

    start_time = time.time()
    response = Response(data, mimetype='text/plain')
    end_time = time.time()

    transfer_time = end_time - start_time
    transfer_rate_mbps = (data_size_mb / transfer_time) * 8  # Convert MB/s to Mbps

    # Print transfer time and rate to the console
    print(f"Transfer Time: {transfer_time:.2f} seconds")
    print(f"Transfer Rate: {transfer_rate_mbps:.2f} Mbps")

    # Optionally, include these values in the response headers
    response.headers['X-Transfer-Time'] = str(transfer_time)
    response.headers['X-Transfer-Rate-Mbps'] = str(transfer_rate_mbps)

    return response

# Function to trim sheet data based on time range
def trim_sheet_data(sheet_df, start_time_unix, end_time_unix, time_column_name='Unix_Epoch_Time'):
    return sheet_df[(sheet_df[time_column_name] >= start_time_unix) & (sheet_df[time_column_name] <= end_time_unix)]

def get_mode_table(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        if 'MODE' not in xls.sheet_names:
            return None  # 'MODE' sheet does not exist
        mode_df = pd.read_excel(file_path, sheet_name='MODE')
        return mode_df
    except Exception as e:
        print(f"Error: {e}")
        return None  # Return None in case of any error

@app.route('/get_mode_table', methods=['GET'])
def fetch_mode_table():
    global current_excel_file  # Access the global variable
    if current_excel_file is None:
        return jsonify({'message': 'No Excel file loaded'}), 400
    mode_df = get_mode_table(current_excel_file)
    if mode_df is None:
        return jsonify({'message': 'MODE table not found or an error occurred'}), 404
    return jsonify(mode_df.to_dict(orient='records'))  # Return the MODE table data as JSON

@app.route('/data/csv', methods=['POST'])
def get_data_as_csv():
    if df.empty:
        return jsonify({'message': 'No data available'}), 404

    data = request.json
    start_time = data['start_time']
    end_time = data['end_time']

    raw_filename = data.get('filename', 'filtered_data.csv')
    filename = sanitize_filename(raw_filename)
    
    # Ensure the filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'

    start_time_unix = pd.to_datetime(start_time).timestamp()
    end_time_unix = pd.to_datetime(end_time).timestamp()

    # Filter the DataFrame based on start and end times
    filtered_df = df[(df['Unix_Epoch_Time'] >= start_time_unix) & (df['Unix_Epoch_Time'] <= end_time_unix)]

    if len(filtered_df) == 0:
        return jsonify({'message': 'Filtered data is empty, no CSV generated'}), 404

    csv_str = filtered_df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


@app.route('/export', methods=['POST'])
def export_data():
    global current_excel_file  # Declare the variable as global
    if df.empty:
        return jsonify({'message': 'No data to export'}), 404
    
    data = request.json
    start_time = data['start_time']
    end_time = data['end_time']

    raw_filename = data.get('filename', 'filtered_data.csv')
    filename = sanitize_filename(raw_filename)
    
    # Check if the filename ends with .xlsx, if not, replace it
    if not filename.endswith('.xlsx'):
        filename = filename.replace('.csv', '.xlsx')

    start_time_unix = pd.to_datetime(start_time).timestamp()
    end_time_unix = pd.to_datetime(end_time).timestamp()

    filtered_df = df[(df['Unix_Epoch_Time'] >= start_time_unix) & (df['Unix_Epoch_Time'] <= end_time_unix)]

    if len(filtered_df) == 0:
        return jsonify({'message': 'Filtered data is empty, no Excel generated'})

    file_path = os.path.join(DOWNLOAD_FOLDER, filename)

    # Use xlsxwriter to create a new Excel file and add a worksheet
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Save the main filtered DataFrame
        filtered_df.to_excel(writer, sheet_name='ALL', index=False)

        # Get the xlsxwriter workbook and worksheet objects
        # workbook  = writer.book
        worksheet = writer.sheets['ALL']

        # Define a built-in table style with banded rows for a modern look
        # Choose a style that suits your preference, e.g., 'Table Style Medium 9'
        table_style = 'Table Style Medium 9'

        # Add a table to the worksheet including the banded rows
        worksheet.add_table(0, 0, len(filtered_df), len(filtered_df.columns) - 1, {
            'style': table_style,
            'columns': [{'header': column_name} for column_name in filtered_df.columns]
        })

        # Read other sheets from original Excel and trim data based on start and end times
        if current_excel_file is None:
            return jsonify({'message': 'Original Excel file not found'}), 400

        xls = pd.ExcelFile(current_excel_file)
        for sheet_name in xls.sheet_names:
            if sheet_name == 'ALL':
                continue
            sheet_df = pd.read_excel(current_excel_file, sheet_name=sheet_name)
            trimmed_df = trim_sheet_data(sheet_df, start_time_unix, end_time_unix)
            trimmed_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)



