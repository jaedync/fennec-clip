from flask import Flask, jsonify, request, render_template, send_from_directory, send_file, Response
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
import os
import re
from bintoexcel import map_mode_number_to_name, calculate_unix_epoch_time_from_timeus, calculate_unix_epoch_time, fetch_weather_data, serialize_obj, match_and_append_weather_data
import json
from collections import defaultdict
from pymavlink import mavutil
  # Replace "your_module" with the actual module name
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from flask_socketio import SocketIO, emit
import base64
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize DataFrame
df = pd.DataFrame()
current_excel_file = None

def convert_bin_to_excel(bin_file_path, excel_file_path, namespace, sid):
    allowed_packet_types = ["IMU", "MAG", "RCIN", "RCOU", "GPS", "GPA", "BAT", "POWR", "MCU", "BARO", "RATE", "ATT", "VIBE", "HELI", "MODE"]
    mlog = mavutil.mavlink_connection(bin_file_path)

    # Create a dictionary to hold lists for each allowed packet type and instance
    parsed_data = defaultdict(list)

    emit('status', {'message': 'Starting conversion...', 'progress': 0}, broadcast=True)

    # Parse the messages and build the data structure
    while True:
        msg = mlog.recv_msg()
        if msg is None:
            break
        msg_dict = msg.to_dict()
        packet_type = msg_dict.get('mavpackettype')
        
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

    emit('status', {'message': 'Total MAVLink messages parsed.', 'progress': 5}, broadcast=True)

    # Now create DataFrames in one go
    data_frames = {}
    for key, data_list in parsed_data.items():
        # Create the DataFrame for this packet type and instance
        data_frames[key] = pd.DataFrame(data_list).drop(columns=['mavpackettype'], errors='ignore')


    emit('status', {'message': 'Dataframes complete. Approximating GPS Time...', 'progress': 10}, broadcast=True)
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

    emit('status', {'message': 'Fetching Weather Data...', 'progress': 15}, broadcast=True)
    # Fetch Weather Data
    min_time = gps_df['Unix_Epoch_Time'].min() if not gps_df.empty else None
    max_time = gps_df['Unix_Epoch_Time'].max() + 900 if not gps_df.empty else None
    weather_data, _ = fetch_weather_data(min_time, max_time) if min_time and max_time else ([], False)
    
    # Create the "ALL" table
    all_df = gps_df.copy()
    time_window = 0.2

    # Define the desired order of appending packet types to the "ALL" table
    ordered_packet_types = ["GPS", "RATE", "RCIN", "RCOU", "VIBE_0", "VIBE_1", "HELI",
                            "IMU_0", "IMU_1", "POWR", "MCU", "BAT", "MAG_0", "MAG_1", 
                            "BARO", "ATT", "GPA"]

    # Rename GPS columns to include "GPS_" prefix
    gps_columns_to_rename = {col: f"GPS_{col}" for col in all_df.columns if col != 'Unix_Epoch_Time'}
    all_df.rename(columns=gps_columns_to_rename, inplace=True)
    emit('status', {'message': 'Starting to create the ALL table...', 'progress': 20}, broadcast=True)
    progress = 0
    # Append data in the specified order
    for packet_type in ordered_packet_types:
        if packet_type == 'GPS':
            continue
        if packet_type not in data_frames:
            continue
        progress = progress + 1
        emit('status', {'message': f'Importing packet type: {packet_type}...', 'progress': 22 + progress*4}, broadcast=True)
    
        df = data_frames[packet_type]
        avg_df = pd.DataFrame()
        
        for gps_time in all_df['Unix_Epoch_Time']:
            min_time = gps_time - time_window
            max_time = gps_time + time_window
            close_rows = df[(df['Unix_Epoch_Time'] >= min_time) & (df['Unix_Epoch_Time'] <= max_time)]
            avg_row = close_rows.mean()
            avg_row['Unix_Epoch_Time'] = gps_time
            avg_df = avg_df._append(avg_row, ignore_index=True)

        # Rename columns to include source table
        renamed_columns = {col: f"{packet_type}_{col}" for col in df.columns if col != 'Unix_Epoch_Time'}
        avg_df.rename(columns=renamed_columns, inplace=True)
        
        all_df = pd.merge(all_df, avg_df, on='Unix_Epoch_Time', how='left')

    # Append weather data to the "ALL" table
    if weather_data:
        all_df = match_and_append_weather_data(all_df, weather_data)

    # Reorder columns to ensure 'Unix_Epoch_Time' is the first column
    reordered_columns = ['Unix_Epoch_Time'] + [col for col in all_df.columns if col != 'Unix_Epoch_Time']
    all_df = all_df[reordered_columns]

    emit('status', {'message': 'Successfully created the ALL table. Writing to an Excel File...', 'progress': 90}, broadcast=True)
    # Save as Excel
    with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
        all_df.to_excel(writer, sheet_name='ALL', index=False)
        for packet_type, df in data_frames.items():
            reordered_columns = ['Unix_Epoch_Time'] + [col for col in df.columns if col != 'Unix_Epoch_Time']
            df[reordered_columns].to_excel(writer, sheet_name=packet_type, index=False)

    # Load the workbook and select the 'ALL' sheet
    # book = load_workbook(excel_file_path)
    # sheet = book['ALL']

    # Debugging: Print end_col_letter and len(all_df)
    # end_col_letter = col_num_to_letter(len(all_df.columns))
    # print("End Column Letter:", end_col_letter)
    # print("Row Count:", len(all_df))

    # Create a table
    # table_range = f"A1:{end_col_letter}{len(all_df) + 1}"  # +1 due to the header row
    # print("Table Range:", table_range)

    # table = Table(displayName="DataTable", ref=table_range)

    # emit('status', {'message': 'Styling ALL Table...', 'progress': 95}, broadcast=True)
    # # Add a default style to the table
    # style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
    #                        showLastColumn=False, showRowStripes=True, showColumnStripes=True)
    # table.tableStyleInfo = style

    # Add table to the sheet
    # sheet.add_table(table)

    # emit('status', {'message': 'Conversion Complete! Finishing up...', 'progress': 95}, broadcast=True)
    # Save the changes
    # book.save(excel_file_path)

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
    if df.empty:
        return jsonify({'message': 'No data available'}), 404
    return jsonify(df.to_dict(orient='records'))

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
    raw_filename = data.get('filename', 'ALL.csv')  # Get filename from the client, use "ALL.csv" as default
    filename = sanitize_filename(raw_filename)  # Sanitize the filename

    # Make sure the sanitized filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'

    csv_str = df.to_csv(index=False)
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

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Save the main filtered DataFrame
        filtered_df.to_excel(writer, sheet_name='ALL', index=False)

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
    
    # Add table formatting to the 'ALL' sheet
    book = load_workbook(file_path)
    sheet = book['ALL']
    end_col_letter = col_num_to_letter(sheet.max_column)
    table_range = f"A1:{end_col_letter}{sheet.max_row}"
    table = Table(displayName="DataTable", ref=table_range)
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                           showLastColumn=False, showRowStripes=True, showColumnStripes=True)
    table.tableStyleInfo = style
    sheet.add_table(table)
    book.save(file_path)

    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)



