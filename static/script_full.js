// Initialize variables
let timeLabels, gpsDataLat, gpsDataLng, barometerData;
let timeline, items;
let originalGpsColors; // New variable for original gradient colors
let rcouC1Data, rcouC2Data, rcouC3Data, rcouC4Data, rcouC8Data;

let latestJobId = 0;

let timeoutId = null;
function throttle(func, delay) {
    if (timeoutId) {
        return;
    }
    timeoutId = setTimeout(() => {
        func();
        timeoutId = null;
    }, delay);
}

document.addEventListener('DOMContentLoaded', function() {
    checkDataAvailability();
    // Initialize Web Worker
    const colorWorker = new Worker('/static/worker.js');

    function resetCharts() {
        // Destroy existing timeline if it exists
        if (timeline) {
            timeline.destroy();
            timeline = null;
        }

        // Reset all your chart data variables
        timeLabels = [];
        gpsDataLat = [];
        gpsDataLatMeters = [];
        gpsDataLng = [];
        gpsDataLngMeters = [];
        barometerData = [];
        barometerDataMeters = [];
        imuDataX = [];
        imuDataY = [];
        imuDataZ = [];
        rcouC1Data = [];
        rcouC2Data = [];
        rcouC3Data = [];
        rcouC4Data = [];
        rcouC8Data = [];
        originalGpsColors = [];

        // Reset latestJobId
        latestJobId = 0;

        // Clear the existing charts
        Plotly.purge('dPlot');
        Plotly.purge('xTimeSeries');
        Plotly.purge('yTimeSeries');
        Plotly.purge('zTimeSeries');
        Plotly.purge('rcouC1TimeSeries');
        Plotly.purge('rcouC2TimeSeries');
        Plotly.purge('rcouC3TimeSeries');
        Plotly.purge('rcouC4TimeSeries');
        Plotly.purge('rcouC8TimeSeries');
    }


    function checkDataAvailability() {
        fetch('is_data_available')
        .then(response => response.json())
        .then(data => {
            if (data.available) {
                fetchDataAndInit();  // Fetch data initially
            }
        })
        .catch(error => {
            console.error('Failed to check data availability:', error);
        });
    }
    
    // Listen for messages from worker
    colorWorker.onmessage = function(event) {
        const { jobId, gpsColors, imuTimeSeriesColors, rcouTimeSeriesColors } = event.data;

        // Only update if the job ID matches the latest job
        if (jobId === latestJobId) {
            Plotly.restyle('dPlot', { 'marker.color': [gpsColors] });
            Plotly.restyle('xTimeSeries', { 'marker.color': [imuTimeSeriesColors] });
            Plotly.restyle('yTimeSeries', { 'marker.color': [imuTimeSeriesColors] });
            Plotly.restyle('zTimeSeries', { 'marker.color': [imuTimeSeriesColors] });

            // New lines to update RCOU colors
            Plotly.restyle('rcouC1TimeSeries', { 'marker.color': [rcouTimeSeriesColors] });
            Plotly.restyle('rcouC2TimeSeries', { 'marker.color': [rcouTimeSeriesColors] });
            Plotly.restyle('rcouC3TimeSeries', { 'marker.color': [rcouTimeSeriesColors] });
            Plotly.restyle('rcouC4TimeSeries', { 'marker.color': [rcouTimeSeriesColors] });
            Plotly.restyle('rcouC8TimeSeries', { 'marker.color': [rcouTimeSeriesColors] });
        }
    };

    function getRandomMessage() {
        const messages = [
            "Summoning pixels...",
            "Warming up the hamsters...",
            "Reticulating splines...",
            "Deploying ninja cats...",
            "Unleashing the internet gremlins...",
            "Buffering the buffer...",
            "Distracting you with this message...",
            "Charging the flux capacitor...",
            "Consulting the oracle...",
            "Feeding the code monkeys...",
            "Waking up the caffeine addicts...",
            "Calibrating the pigeon GPS...",
            "Synchronizing the zebra stripes...",
            "Reversing the polarity of the neutron flow...",
            "Downloading more RAM...",
            "Polishing the pixels...",
            "Counting backwards from infinity...",
            "Spinning the hamster wheel...",
            "Assembling the minions...",
            "Rounding up the roundabouts...",
            "Raymond has some 'updates'...",
            "Compiling a compelling experience...",
            "Teaching robots to love...",
            "Convincing the elements to behave...",
            "Setting phasers to fun...",
            "Baking cookies in the server...",
            "Hunting for missing semicolons...",
            "Procrastinating loading sequence...",
            "Googling how to load data faster...",
            "Burning the team's budget...",
            "Feeding hudson 3 more Monster Energy Drinks...",
            "Knitting some pixels...",
            "Blowing digital bubbles...",
            "Constructing additional pylons...",
            "Practicing wizardry on data...",
            "Turning it off and on again...",
            "Unfolding the unfoldable...",
            "Activating party mode...",
            "Remember, don't blink. Blink and you're dead..."
        ];
        
        return messages[Math.floor(Math.random() * messages.length)];
    }
    

    // Fetch data and initialize charts
    function fetchDataAndInit() {
        // Show loading message and spinner
        document.getElementById('dataLoadingMessage').style.display = 'block';
        document.getElementById('randomMessage').innerText = getRandomMessage(); // Set random message
    
        fetch('data')
        .then(response => {
            if (!response.ok) {
                console.error(`HTTP error! status: ${response.status}`);
                return;
            }
            return response.json();
        })
        .then(data => {
            console.log("Fetched Data:", data);  // Log the fetched data
            processData(data);
            initVisualizations();
    
            // Hide loading message and spinner
            document.getElementById('dataLoadingMessage').style.display = 'none';
        })
        .catch(error => {
            // Hide loading message and spinner in case of an error
            document.getElementById('dataLoadingMessage').style.display = 'none';
            console.error("Fetch Error:", error);
        });
    }
    

    // Function to initialize all visualizations
    function initVisualizations() {
        initTimeline();
        initTimeSeriesPlots(imuDataX, imuDataY, imuDataZ);  // Add this line
        // console.log("originalGpsColors:", originalGpsColors);
        init3DChart();
    }    

    // Function to show and hide loading message
    function setLoading(isLoading) {
        const loadingElement = document.getElementById('loadingMessage');
        if (isLoading) {
            loadingElement.style.display = 'block';
        } else {
            loadingElement.style.display = 'none';
        }
    }
    // Initialize socket connection
    const socket = io.connect('https://' + document.domain + ':' + location.port);

// Function to clear all status messages from the queue
function clearStatusMessages() {
    const statusQueue = document.getElementById('statusQueue');
    while (statusQueue.firstChild) {
        statusQueue.removeChild(statusQueue.firstChild);
    }
}

    // Function to append a new status message to the queue
function appendStatusMessage(message, type = 'default', color = '#5cb4b8') {
    const statusQueue = document.getElementById('statusQueue');
    const newStatus = document.createElement('div');
    newStatus.className = `status-message status-${type}`;
    newStatus.innerText = message;

    // Set border-left color based on the provided color
    newStatus.style.borderLeftColor = color;

    // Apply the fade-in animation
    newStatus.style.animation = 'slideFadeIn 0.5s forwards';

    // Insert the new status message at the beginning of the queue
    if (statusQueue.firstChild) {
        statusQueue.insertBefore(newStatus, statusQueue.firstChild);
    } else {
        statusQueue.appendChild(newStatus);
    }

    // Keep only four messages in the queue
    while (statusQueue.childNodes.length > 8) {
        statusQueue.removeChild(statusQueue.lastChild);
    }
}

    // Listen for status messages from the server
    socket.on('status', function(data) {
        console.log(data.message);
        updateProgressBar(data.progress);
        // Check if the color is provided in the data, otherwise default to a standard color
        var messageColor = data.color || '#5cb4b8'; // Default color
        // Append the received status message to the queue with the specified color
        appendStatusMessage(data.message, 'default', messageColor); // Use 'default' or another type as needed
        if (data.message === 'Conversion and upload complete') {
            
            // Hide loading message and spinner
            setLoading(false);
            // Clear the status message queue
            clearStatusMessages();
            // Refresh the charts and data
            resetCharts();
            fetchDataAndInit();
        }
        
        if (data.message === 'Starting conversion...') {
            setLoading(true);
        }
    });


    function updateProgressBar(progress) {
        const progressBar = document.getElementById('progressBar');

        progressBar.style.width = progress + '%';
    }

    document.getElementById('csvFile').addEventListener('change', async function() {
        const file = this.files[0];
        const formData = new FormData();
        formData.append('file', file);

        if (file.name.endsWith('.bin')) {
            setLoading(true);
        }
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            // Emitting upload_and_convert event to server
            socket.emit('upload_and_convert', { filename: file.name });
        }
    });
    
    // Trigger the hidden file input when the "Upload" button is clicked
    document.getElementById('uploadBtn').addEventListener('click', function() {
        document.getElementById('csvFile').click();
    });

// Function to process data
function processData(data) {
    // console.log("Processing Data:", data);  // Log the data being processed
    if (!data || !Array.isArray(data) || data.length === 0) {
        console.error("Data is undefined, null or empty");
        return;
    }
    // Prepare data for the 3D chart
    timeLabels = data.map(row => new Date(row['Datetime_Chicago']));
    gpsDataLat = data.map(row => row['GPS_Lat']);
    gpsDataLatMeters = data.map(row => row['GPS_lat_m']);
    gpsDataLng = data.map(row => row['GPS_Lng']);
    gpsDataLngMeters = data.map(row => row['GPS_lng_m']);
    barometerData = data.map(row => row['BARO_Press']);
    barometerDataMeters = data.map(row => row['BARO_Altitude_Meters_Estimate']);
    
    // Initialize IMU Data
    imuDataX = data.map(row => row['IMU_0_AccX']);
    imuDataY = data.map(row => row['IMU_0_AccY']);
    imuDataZ = data.map(row => row['IMU_0_AccZ']);

    // Initialize RCOU Data
    rcouC1Data = data.map(row => row['RCOU_C1']);
    rcouC2Data = data.map(row => row['RCOU_C2']);
    rcouC3Data = data.map(row => row['RCOU_C3']);
    rcouC4Data = data.map(row => row['RCOU_C4']);
    rcouC8Data = data.map(row => row['RCOU_C8']);

    // Initialize originalGpsColors
    const numPoints = gpsDataLat.length;
    originalGpsColors = Array.from({length: numPoints}, (_, i) => 
        `rgb(${Math.floor(255 * i / numPoints)}, 0, ${Math.floor(255 - 255 * i / numPoints)})`
    );
}

document.addEventListener('DOMContentLoaded', function () {
    // Trigger the hidden file input when the "Upload" button is clicked
    document.getElementById('uploadBtn').addEventListener('click', function () {
        document.getElementById('csvFile').click();
    });

    // Upload data and refresh charts when a file is selected
    document.getElementById('csvFile').addEventListener('change', function () {
        const csvFile = this.files[0];
        const formData = new FormData();
        formData.append('file', csvFile);

        fetch('upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Re-fetch the data and initialize charts
            fetchDataAndInit();
        });
    });

    // Run the check when the page loads
    checkDataAvailability();
});

function convertPressureToFeet(pressureInPascals) {
    // Convert pressure from Pascals to millibars
    var pressureInMillibars = pressureInPascals / 100;
    // Apply the barometric formula
    return (1 - Math.pow(pressureInMillibars / 1013.25, 1/5.255)) * 145366.45;
}
function degreesToRadians(degrees) {
    return degrees * Math.PI / 180;
}

function degreesToFeet(latitudeDegrees, longitudeDegrees, avgLatitude) {
    // Convert latitude to feet (1 degree latitude = ~364,000 feet)
    const latitudeFeet = latitudeDegrees * 364000;
    // Calculate the conversion factor for longitude to feet based on the average latitude
    const longitudeFeet = longitudeDegrees * Math.cos(degreesToRadians(avgLatitude)) * 364000;
    return { latitudeFeet, longitudeFeet };
}

function normalizeGPSData(gpsDataLat, gpsDataLng) {
    // Calculate the average latitude for the longitude conversion
    const avgLatitude = gpsDataLat.reduce((a, b) => a + b, 0) / gpsDataLat.length;
    // Normalize the GPS data to feet
    return gpsDataLat.map((lat, i) => {
        const { latitudeFeet, longitudeFeet } = degreesToFeet(lat - gpsDataLat[0], gpsDataLng[i] - gpsDataLng[0], avgLatitude);
        return { x: longitudeFeet, y: latitudeFeet };
    });
}

function init3DChart() {
    if (!gpsDataLatMeters || !gpsDataLngMeters || !barometerDataMeters) {
        console.error("GPS or Barometer Data is undefined, null or empty");
        return;
    }
    if (typeof originalGpsColors === 'undefined') {
        console.error("originalGpsColors is undefined");
        return;
    }

    // Use the GPS data in meters directly
    const xDistances = gpsDataLatMeters;
    const yDistances = gpsDataLngMeters;

    // Use the barometric altitude data in meters directly
    const altitudeData = barometerDataMeters;
    const minAltitude = Math.min(...altitudeData);
    const normalizedAltitudeData = altitudeData.map(altitude => altitude - minAltitude);

    const trace = {
        x: xDistances,
        y: yDistances,
        z: normalizedAltitudeData,
        mode: 'markers',
        type: 'scatter3d',
        marker: {
            color: originalGpsColors,
            opacity: 0.4
        }
    };

    // Determine the range for the z-axis (altitude) in meters
    const maxAltitude = Math.max(...normalizedAltitudeData);

    const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: '#05060d',
        plot_bgcolor: '#05060d',
        scene: {
            xaxis: {
                title: 'Latitude (m)',
                color: 'white',
                gridcolor: '#888'
            },
            yaxis: {
                title: 'Longitude (m)',
                color: 'white',
                gridcolor: '#888'
            },
            zaxis: {
                title: 'Altitude (m)',
                range: [0, maxAltitude],
                color: 'white',
                gridcolor: '#888'
            },
            aspectratio: { x: 1, y: 1, z: 1 } // This enforces the same scaling factor on all axes
        },
        font: {
            color: 'white'
        }
    };

    Plotly.newPlot('dPlot', [trace], layout, { responsive: true });
}

async function fetchModeTable() {
    try {
        const response = await fetch('/get_mode_table');
        if (!response.ok) {
            console.error('Failed to fetch mode table:', response.statusText);
            return [];
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching mode table:', error);
        return [];
    }
}

async function initTimeline() {
    const container = document.getElementById('visualization');

    const modeTableData = await fetchModeTable();
    const modeItems = modeTableData.map(entry => {
        let className = '';
        switch(entry.ModeName) {
            case 'POSHOLD':
                className = 'mode-poshold';
                break;
            case 'STABILIZE':
                className = 'mode-stabilize';
                break;
            case 'RTL':
                className = 'mode-rtl';
                break;
            default:
                break;
        }
        return {
            start: new Date(entry.Unix_Epoch_Time * 1000),  // Convert to milliseconds
            content: entry.ModeName,
            editable: false,  // Locked marker
            group: 1,  // Assign all mode items to the same group
            className: className
        };
    });
    
    // Create background items
    const backgroundItems = [
        {
            start: new Date(timeLabels[0].getTime() - 60000),
            end: timeLabels[0],
            type: 'background',
            className: 'gray-background'
        },
        {
            start: timeLabels[timeLabels.length - 1],
            end: new Date(timeLabels[timeLabels.length - 1].getTime() + 60000),
            type: 'background',
            className: 'gray-background'
        }
    ];

    const items = new vis.DataSet([...backgroundItems, ...modeItems]);

    const options = {
        min: new Date(timeLabels[0].getTime() - 60000),
        max: new Date(timeLabels[timeLabels.length - 1].getTime() + 60000),
        editable: false,
        zoomMin: 1000,
        zoomMax: 1000 * 60 * 60 * 24,
        zoomable: true,  // Enable zooming
        moveable: true,  // Enable moving the timeline left and right
        snap: function (date, scale, step) {
            return date;
        },
        stack: false,  // Disable stacking
        orientation: 'top'  // Ensure all items are aligned at the top
    };
    
    // Create the timeline
    timeline = new vis.Timeline(container, items, options);

    // Add custom time bars for start and end
    timeline.addCustomTime(timeLabels[0], 'startMarker');
    timeline.addCustomTime(timeLabels[timeLabels.length - 1], 'endMarker');

    // Listen to the `timechange` event to update 3D plot
    timeline.on('timechange', function(event) {
        if (event.id === 'startMarker' || event.id === 'endMarker') {
            updateColors(event.time, event.id);
        }
    });

    styleCustomTimeMarkers();
    
    updateDisplayedTimeRange(timeLabels[0], timeLabels[timeLabels.length - 1]);

}


function updateDisplayedTimeRange(start, end) {
    const startTimeDisplay = document.getElementById('startTimeDisplay');
    const endTimeDisplay = document.getElementById('endTimeDisplay');
    
    const startTimeString = start.toTimeString().split(' ')[0];  // Keep HH:MM:SS
    const endTimeString = end.toTimeString().split(' ')[0];  // Keep HH:MM:SS
    
    startTimeDisplay.innerHTML = `Start: ${startTimeString}`;
    endTimeDisplay.innerHTML = `End: ${endTimeString}`;
}


function styleCustomTimeMarkers() {
    setTimeout(() => {
        const startMarker = document.querySelector('.vis-custom-time.startMarker');
        const endMarker = document.querySelector('.vis-custom-time.endMarker');

        if (startMarker && endMarker) {
            startMarker.style.backgroundColor = 'blue';
            endMarker.style.backgroundColor = 'red';
        }
    }, 0);
}

// Update function with color coding for both 3D and time-series charts
function updateColors(newTime, markerId) {
    throttle(() => {
    let startRange, endRange;

    if (markerId === 'startMarker') {
        startRange = new Date(newTime);
        endRange = new Date(timeline.getCustomTime('endMarker'));
    } else {
        startRange = new Date(timeline.getCustomTime('startMarker'));
        endRange = new Date(newTime);
    }
    
    // Update the displayed time range
    updateDisplayedTimeRange(startRange, endRange);

    // Update the latest job ID and send it to the worker
    latestJobId++;
    colorWorker.postMessage({
        jobId: latestJobId,
        timeLabels,
        originalGpsColors,
        startRange,
        endRange,
        unselectedAlpha: 0.001  // New variable for unselected point opacity
    });
}, 100); // delay of 200 milliseconds
    
}
document.getElementById('downloadCsvBtn').addEventListener('click', function() {
    if (!timeline) {
        applyShakeAnimationIfNoTimeline();
        return;
    }

    // Get custom time markers
    const startMarkerTime = timeline.getCustomTime('startMarker');
    const endMarkerTime = timeline.getCustomTime('endMarker');

    // Check for null or undefined
    if (startMarkerTime && endMarkerTime) {
        const startRange = new Date(startMarkerTime);
        const endRange = new Date(endMarkerTime);

        // Prompt the user for a filename
        let filename = prompt("Enter the filename for the exported data:", "filtered_data.csv");

        // If the user cancels the prompt, filename will be null
        if (filename === null) {
            return;
        }

        // Ensure the filename ends with .csv
        if (!filename.endsWith('.csv')) {
            filename += '.csv';
        }

        fetch('data/csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                start_time: startRange.toISOString(), 
                end_time: endRange.toISOString(), 
                filename: filename 
            }) // Send the start and end times along with the filename
        })
        .then(response => {
            if (response.ok) {
                return response.blob();  // convert to blob
            } else {
                return response.json().then(data => Promise.reject(data));  // reject the Promise with the error message
            }
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;  // Use the filename provided by the user
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error(error);
            alert("Failed to download CSV");
        });
    } else {
        console.error('Start or end marker time not found');
    }
});

// Function to average data points
function averageData(data) {
    if (!data || !data.length) {
        console.error("Data is undefined, null or empty");
        console.error("Stack trace:", new Error().stack); // Log the stack trace
        return [];
    }
    let averagedData = [];
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i];
        if ((i + 1) % 10 === 0) {
            averagedData.push(sum / 10);
            sum = 0;
        }
    }
    return averagedData;
}

// Function to calculate the median of data points
function medianData(data) {
    let medianData = [];
    let chunk = [];
    for (let i = 0; i < data.length; i++) {
        chunk.push(data[i]);
        if ((i + 1) % 10 === 0) {
            chunk.sort((a, b) => a - b);
            const mid = Math.floor(chunk.length / 2);
            const median = chunk.length % 2 === 0 ? (chunk[mid - 1] + chunk[mid]) / 2 : chunk[mid];
            medianData.push(median);
            chunk = [];
        }
    }
    return medianData;
}

// Create a helper function to make a new trace for RCOU data
function createTrace(xData, yData, name) {
    return {
        x: xData,
        y: yData,
        mode: 'lines+markers',
        marker: { color: 'rgb(255,165,0)' },
        line: { color: 'grey' },
        name: name
    };
}

// Initialize the time-series plots with data averaging and color coding
function initTimeSeriesPlots(imuDataX, imuDataY, imuDataZ) {
    // Average every 10 points
    // console.log("imuDataX:", imuDataX);
    // console.log("imuDataY:", imuDataY);
    // console.log("imuDataZ:", imuDataZ);

    const avgImuDataX = averageData(imuDataX);
    const avgImuDataY = averageData(imuDataY);
    const avgImuDataZ = averageData(imuDataZ);

    const avgRcouC1Data = averageData(rcouC1Data);
    const avgRcouC2Data = averageData(rcouC2Data);
    const avgRcouC3Data = averageData(rcouC3Data);
    const avgRcouC4Data = averageData(rcouC4Data);
    const avgRcouC8Data = averageData(rcouC8Data);

    // Calculate new time labels by getting the median of every 10 original points
    const avgTimeLabels = medianData(timeLabels.map(date => date.getTime())).map(time => new Date(time));

    const layout = {
        margin: { l: 50, r: 10, b: 40, t: 40 },
        title: 'Accelerometer Data over Time',
        paper_bgcolor: '#05060d',  // Dark background
        plot_bgcolor: '#05060d',   // Dark background
        xaxis: {
            color: 'white',
            gridcolor: '#888',  // Dark grid lines
            title: 'Time'
        },
        yaxis: {
            color: 'white',
            gridcolor: '#888'  // Dark grid lines
        },
        font: {
            color: 'white'  // Text color
        }
    };

    const traceX = {
        x: avgTimeLabels,
        y: avgImuDataX,
        mode: 'lines+markers',
        marker: { color: 'rgb(0,255,255)' },
        line: { color: 'grey' },
        name: 'X-axis'
    };

    const traceY = {
        x: avgTimeLabels,
        y: avgImuDataY,
        mode: 'lines+markers',
        marker: { color: 'rgb(0,255,255)' },
        line: { color: 'grey' },
        name: 'Y-axis'
    };

    const traceZ = {
        x: avgTimeLabels,
        y: avgImuDataZ,
        mode: 'lines+markers',
        marker: { color: 'rgb(0,255,255)' },
        line: { color: 'grey' },
        name: 'Z-axis'
    };
    
    // Create new time series plots for the RCOU data
    const traceRC1 = createTrace(avgTimeLabels, avgRcouC1Data, 'RCOU_C1');
    const traceRC2 = createTrace(avgTimeLabels, avgRcouC2Data, 'RCOU_C2');
    const traceRC3 = createTrace(avgTimeLabels, avgRcouC3Data, 'RCOU_C3');
    const traceRC4 = createTrace(avgTimeLabels, avgRcouC4Data, 'RCOU_C4');
    const traceRC8 = createTrace(avgTimeLabels, avgRcouC8Data, 'RCOU_C8');

    Plotly.newPlot('xTimeSeries', [traceX], { ...layout, title: 'X-axis Acceleration over Time' });
    Plotly.newPlot('yTimeSeries', [traceY], { ...layout, title: 'Y-axis Acceleration over Time' });
    Plotly.newPlot('zTimeSeries', [traceZ], { ...layout, title: 'Z-axis Acceleration over Time' });
    Plotly.newPlot('rcouC1TimeSeries', [traceRC1], { ...layout, title: 'RC Out Channel 1 (Aileron)' });
    Plotly.newPlot('rcouC2TimeSeries', [traceRC2], { ...layout, title: 'RC Out Channel 2 (Elevator)' });
    Plotly.newPlot('rcouC3TimeSeries', [traceRC3], { ...layout, title: 'RC Out Channel 3 (Throttle)' });
    Plotly.newPlot('rcouC4TimeSeries', [traceRC4], { ...layout, title: 'RC Out Channel 4 (Rudder)' });
    Plotly.newPlot('rcouC8TimeSeries', [traceRC8], { ...layout, title: 'RC Out Channel 8 (RPM Control)' });
}

});

function exportData() {
    if (!timeline) {
        applyShakeAnimationIfNoTimeline();
        return;
    }
    // Show export loading message
    const exportLoadingElement = document.getElementById('exportLoadingMessage');
    exportLoadingElement.style.display = 'block';
    // Get custom time markers
    const startMarkerTime = timeline.getCustomTime('startMarker');
    const endMarkerTime = timeline.getCustomTime('endMarker');

    // Check for null or undefined
    if (startMarkerTime && endMarkerTime) {
        const startRange = new Date(startMarkerTime);
        const endRange = new Date(endMarkerTime);

        // Prompt user for a filename
        const filename = prompt("Enter the filename for the exported data:", "filtered_data.xlsx");

        // If the user cancels the prompt, filename will be null
        if (filename === null) {
            // Hide export loading message
            exportLoadingElement.style.display = 'none';
            return;
        }

        fetch('export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ start_time: startRange.toISOString(), end_time: endRange.toISOString(), filename: filename })
        })
        .then(response => {
            // Hide export loading message
            exportLoadingElement.style.display = 'none';
            if (response.ok) {
                return response.blob();  // convert to blob
            } else {
                return response.json().then(data => Promise.reject(data));  // reject the Promise with the error message
            }
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;  // Use the filename provided by the user
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            // Hide export loading message
            exportLoadingElement.style.display = 'none';
            console.error(error);
            alert("Failed to export data");
        });
    } else {
        exportLoadingElement.style.display = 'none';
        console.error('Start or end marker time not found');
    }
}


function applyShakeAnimationIfNoTimeline() {
    if (!timeline) {
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.classList.add('shake'); // Add shake animation
        setTimeout(() => uploadBtn.classList.remove('shake'), 320); // Remove after animation duration
    }
}