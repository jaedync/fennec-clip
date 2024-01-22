// Initialize variables
let timeLabels, gpsDataLat, gpsDataLng, barometerData;
let timeline, items;
let originalGpsColors;
let rcouC1Data, rcouC2Data, rcouC3Data, rcouC4Data, rcouC8Data;
let imuData;
let xkf1Data;
let modeTableData;
let gpsData;
let RCOUData;

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

const socket = io.connect(document.domain + ':' + location.port);
document.addEventListener('DOMContentLoaded', function() {

    fetchFileList();
    checkDataAvailability();
    // Initialize Web Worker
    const colorWorker = new Worker('/static/worker.js');

    function resetTimeline() {
        if (timeline) {
            timeline.destroy();
            timeline = null;
        }
    }
    
    function reset3DChart() {
        Plotly.purge('dPlot');
    }
    
    function resetTimeSeriesPlots() {
        Plotly.purge('xTimeSeries');
        Plotly.purge('yTimeSeries');
        Plotly.purge('zTimeSeries');
        Plotly.purge('rcouC1TimeSeries');
        Plotly.purge('rcouC2TimeSeries');
        Plotly.purge('rcouC3TimeSeries');
        Plotly.purge('rcouC4TimeSeries');
        Plotly.purge('rcouC8TimeSeries');
    }
    
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

    function fetchFileList() {
        fetch('/get_file_list')
        .then(response => response.json())
        .then(files => {
            const fileListContainer = document.getElementById('fileList');
            fileListContainer.innerHTML = '';  // Clear existing list
            
            files.forEach(file => {
                const fileBox = document.createElement('div');
                fileBox.className = `file-box ${file.type}`;
    
                // Create a container for text elements
                const textContainer = document.createElement('div');
                textContainer.className = 'text-container';
    
                // Create a span for the filename
                const fileNameSpan = document.createElement('span');
                fileNameSpan.innerText = file.filename;
    
                // Create a span for the file size in smaller text
                const fileSizeSpan = document.createElement('span');
                fileSizeSpan.innerText = `${formatBytes(file.size)}`;
                fileSizeSpan.style.fontSize = 'smaller';
                fileSizeSpan.style.display = 'block'; // To make it appear below the filename
    
                // Create a span for the last edited date
                const fileDateSpan = document.createElement('span');
                fileDateSpan.innerText = `${file.mtime}`;
                fileDateSpan.style.fontSize = 'smaller';
                fileDateSpan.style.display = 'block'; // To make it appear below the file size
    
                // Append filename, file size, and last edited date to the text container
                textContainer.appendChild(fileNameSpan);
                textContainer.appendChild(fileSizeSpan);
                textContainer.appendChild(fileDateSpan);
    
                // Append text container to the fileBox
                fileBox.appendChild(textContainer);
    
                // Add a 'Load' button for .bin files
                if (file.type === 'bin') {
                    const loadBtn = document.createElement('button');
                    loadBtn.innerText = 'Load';
                    loadBtn.className = 'load-btn';
                    loadBtn.onclick = (event) => {
                        event.stopPropagation(); // Prevent triggering the file download
                        loadBinFile(file.filename);
                    };
    
                    // Append the Load button after the text container
                    fileBox.appendChild(loadBtn);
                }
    
                // File download functionality
                fileBox.onclick = () => {
                    const downloadLink = document.createElement('a');
                    downloadLink.href = file.url;
                    downloadLink.download = file.filename;
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                };
    
                // Apply styles and add the file box to the container
                fileListContainer.appendChild(fileBox);
            });
        })
        .catch(error => console.error('Error fetching file list:', error));
    }

    
    // Function to load a .bin file using the /load_json endpoint
    function loadBinFile(filename) {
        // Assuming the JSON file has the same name as the BIN file but with a .json extension
        const jsonFilename = filename.replace('.bin', '.json');

        fetch('/load_json', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: jsonFilename })
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to load file');
            }
        })
        .then(data => {
            console.log('File loaded:', data);
            fetchDataAndInit();
        })
        .catch(error => console.error('Error loading file:', error));
    }


    // Helper function to format bytes into a readable format
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
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
    // Listen for messages from worker
    colorWorker.onmessage = function(event) {
        const { jobId, xkf1Colors, imuAccXColors, imuAccYColors, imuAccZColors, RCOUColors } = event.data;

        // Only update if the job ID matches the latest job
        if (jobId === latestJobId) {
            Plotly.restyle('dPlot', { 'marker.color': [xkf1Colors] });
            Plotly.restyle('xTimeSeries', { 'marker.color': [imuAccXColors]});
            Plotly.restyle('yTimeSeries', { 'marker.color': [imuAccYColors]});
            Plotly.restyle('zTimeSeries', { 'marker.color': [imuAccZColors]});

            // New lines to update RCOU colors
            Plotly.restyle('rcouC1TimeSeries', { 'marker.color': [RCOUColors] });
            Plotly.restyle('rcouC2TimeSeries', { 'marker.color': [RCOUColors] });
            Plotly.restyle('rcouC3TimeSeries', { 'marker.color': [RCOUColors] });
            Plotly.restyle('rcouC4TimeSeries', { 'marker.color': [RCOUColors] });
            Plotly.restyle('rcouC8TimeSeries', { 'marker.color': [RCOUColors] });
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
        resetTimeline();
        // Show loading message and spinner
        document.getElementById('dataLoadingMessage').style.display = 'block';
        document.getElementById('randomMessage').innerText = getRandomMessage(); // Set random message

        fetch('data')
        .then(response => response.json())
        .then(jsonData => {
            console.log("Fetched Data:", jsonData);  // Log the fetched data
            
            processData(jsonData);
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
        initTimeSeriesPlots(imuData);
        // console.log("originalGpsColors:", originalGpsColors);
        init3DChart();
        fetchFileList();
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
        
        // Clear any existing messages before appending a new one
        while (statusQueue.firstChild) {
            statusQueue.removeChild(statusQueue.firstChild);
        }

        const newStatus = document.createElement('div');
        newStatus.className = `status-message status-${type}`;
        newStatus.innerText = message;

        // Set border-left color based on the provided color
        newStatus.style.borderLeftColor = color;

        // Apply the fade-in animation
        newStatus.style.animation = 'slideFadeIn 0.5s forwards';

        // Append the new status message
        statusQueue.appendChild(newStatus);
    }


    // Listen for status messages from the server
    socket.on('status', function(data) {
        console.log(data.message);
        if (data.progress) {
            updateProgressBar(data.progress);
        }
        // Check if the color is provided in the data, otherwise default to a standard color
        var messageColor = data.color || '#5cb4b8'; // Default color
        // Append the received status message to the queue with the specified color
        appendStatusMessage(data.message, 'default', messageColor);
        if (data.message === 'Bin File Upload Complete!') {
            // Hide loading message and spinner
            setLoading(false);
            // Clear the status message queue
            clearStatusMessages();
            // Refresh the charts and data
            resetCharts();
            fetchDataAndInit();
            fetchFileList();
        }
        if (data.message === 'File Export Complete!') {
            // Hide loading message and spinner
            setLoading(false);
            // Clear the status message queue
            clearStatusMessages();
            // Refresh file list
            fetchFileList();
        }
        
        if (data.message === 'Starting conversion...') {
            setLoading(true);
        }
    });

    function updateProgressBar(progress) {
        const progressBar = document.getElementById('progressBar');

        progressBar.style.width = progress + '%';
    }
    
    function setProgressBarColor(color) {
        const progressBar = document.getElementById('progressBar');
        progressBar.style.background = color;
    }
    
    function updateLoadingHeaderText(text) {
        const loadingHeader = document.querySelector('.loadingHeaderText');
        if (loadingHeader) {
            loadingHeader.textContent = text;
        }
    }

    document.getElementById('csvFile').addEventListener('change', async function() {
        const file = this.files[0];
        const formData = new FormData();
        formData.append('file', file);

        updateLoadingHeaderText(`Uploading ${file.name}...`);
        setProgressBarColor('#FFA500'); // Orange color for upload

        if (file.name.endsWith('.bin')) {
            setLoading(true);
        }

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);
    
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentage = (e.loaded / e.total) * 100;
                updateProgressBar(percentage);
            }
        };
    
        xhr.onload = function() {
            if (xhr.status === 200) {
                // Emitting upload_and_convert event to server
                socket.emit('upload_and_convert', { filename: file.name });
                // Reset progress bar
                updateProgressBar(0);
                setProgressBarColor('#72dee4'); // Reset to original color
            } else {
                console.error('Upload failed:', xhr.responseText);
            }
        };
    
        xhr.send(formData);
    });
    
    // Trigger the hidden file input when the "Upload" button is clicked
    document.getElementById('uploadBtn').addEventListener('click', function() {
        document.getElementById('csvFile').click();
    });

// Function to process data
function processData(data) {
    if (!data) {
        console.error("Data is undefined or null");
        return;
    }
    
    // Extract XKF1 and IMU data
    xkf1Data = data['XKF1'] || [];
    imuData = data['IMU'] || [];
    modeTableData = data['MODE'] || [];
    gpsData = data['GPS'] || [];
    RCOUData = data['RCOU'] || [];
    console.log("xkf1Data:", xkf1Data);
    console.log("imuData:", imuData);

    // Check if file info is available in the data
    if (data.file_info) {
        const fileInfoEl = document.getElementById('fileInfo');
        fileInfoEl.innerHTML = `
            <p>File: ${data.file_info.file_name}</p>
            <p>Date: ${formatDate(data.file_info.datetime_chicago)}</p>
            <p>Duration: ${formatDuration(data.file_info.flight_duration_seconds)}</p>
        `;
    }
    timeLabels = gpsData.map(row => new Date(row['Datetime_Chicago']));

    // Prepare data for the 3D chart (XKF1)
    gpsDataLatMeters = xkf1Data.map(row => row['PN']);  // North position
    gpsDataLngMeters = xkf1Data.map(row => row['PE']);  // East position
    barometerDataMeters = xkf1Data.map(row => row['PD']);  // Down position (altitude)

    // Initialize originalGpsColors
    const numPoints = gpsDataLatMeters.length;
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

// Helper function to format the date
function formatDate(dateString) {
    const date = new Date(dateString);
    const formattedDate = date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
    return formattedDate;
}

// Helper function to format duration in seconds to a more readable format
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.round(seconds % 60);
    let formattedDuration = '';

    if (hours > 0) {
        formattedDuration += `${hours} ${hours > 1 ? 'hours' : 'hour'} `;
    }
    if (minutes > 0) {
        formattedDuration += `${minutes} ${minutes > 1 ? 'minutes' : 'minute'} `;
    }
    if (secs > 0 || formattedDuration === '') {
        formattedDuration += `${secs} ${secs === 1 ? 'second' : 'seconds'}`;
    }
    return formattedDuration.trim();
}

function init3DChart() {
    if (!gpsDataLatMeters || !gpsDataLngMeters || !barometerDataMeters) {
        console.error("GPS or Barometer Data is undefined, null or empty");
        return;
    }

    // Adjust to use PN, PE, PD for 3D chart
    const trace = {
        x: gpsDataLngMeters,  // East position
        y: gpsDataLatMeters,  // North position
        z: barometerDataMeters.map(alt => -alt),  // Down position (inverted for altitude)
        mode: 'markers',
        type: 'scatter3d',
        marker: {
            color: originalGpsColors,
            opacity: 0.4
        }
    };

    const maxAltitude = Math.max(...barometerDataMeters.map(alt => -alt));

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

    // const modeTableData = await fetchModeTable();
    console.log("mode_table:", modeTableData);
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
        xkf1Data: xkf1Data,
        imuData: imuData,
        RCOUData: RCOUData,
        startRange: startRange,
        endRange: endRange,
        unselectedAlpha: 0.001
    });    
}, 100); // delay of 100 milliseconds
    
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
        let filename = prompt("Enter the filename for the exported data:", "clipped_data.csv");

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

function initTimeSeriesPlots(imuData) {
    if (!imuData || imuData.length === 0) {
        console.error("IMU data is empty or not available");
        return;
    }

    if (!RCOUData || RCOUData.length === 0) {
        console.error("RCOU data is empty or not available");
        return;
    }

    // Extracting time and IMU data
    const timeLabelsIMU = imuData.map(row => new Date(row['Unix_Epoch_Time'] * 1000));
    const imuDataX = imuData.map(row => row['AccX']);
    const imuDataY = imuData.map(row => row['AccY']);
    const imuDataZ = imuData.map(row => row['AccZ']);

    const timeLabelsRCOU = imuData.map(row => new Date(row['Unix_Epoch_Time'] * 1000));
    const rcouC1Data = RCOUData.map(row => row['C1']);
    const rcouC2Data = RCOUData.map(row => row['C2']);
    const rcouC3Data = RCOUData.map(row => row['C3']);
    const rcouC4Data = RCOUData.map(row => row['C4']);
    const rcouC8Data = RCOUData.map(row => row['C8']);

    // Define a layout for the plot
    const layout = {
        margin: { l: 50, r: 10, b: 40, t: 40 },
        title: 'IMU Data over Time',
        paper_bgcolor: '#05060d',  
        plot_bgcolor: '#05060d',   
        xaxis: {
            color: 'white',
            gridcolor: '#888',
            title: 'Time',
            type: 'date' // Specify the x-axis type as date
        },
        yaxis: {
            color: 'white',
            gridcolor: '#888' 
        },
        font: {
            color: 'white'
        }
    };

    // Creating traces for each axis
    const traceX = {
        x: timeLabelsIMU,
        y: imuDataX,
        mode: 'lines+markers',
        line: { color: 'grey' },
        marker: { color: 'rgb(255, 0, 0)' },
        name: 'AccX'
    };

    const traceY = {
        x: timeLabelsIMU,
        y: imuDataY,
        mode: 'lines+markers',
        line: { color: 'grey' },
        marker: { color: 'rgb(0, 255, 0)' },
        name: 'AccY'
    };

    const traceZ = {
        x: timeLabelsIMU,
        y: imuDataZ,
        mode: 'lines+markers',
        line: { color: 'grey' },
        marker: { color: 'rgb(0, 0, 255)' },
        name: 'AccZ'
    };

    const traceRC1 = createTrace(timeLabelsRCOU, rcouC1Data, 'RCOU_C1');
    const traceRC2 = createTrace(timeLabelsRCOU, rcouC2Data, 'RCOU_C2');
    const traceRC3 = createTrace(timeLabelsRCOU, rcouC3Data, 'RCOU_C3');
    const traceRC4 = createTrace(timeLabelsRCOU, rcouC4Data, 'RCOU_C4');
    const traceRC8 = createTrace(timeLabelsRCOU, rcouC8Data, 'RCOU_C8');

    // Plotting the data
    Plotly.newPlot('xTimeSeries', [traceX], layout);
    Plotly.newPlot('yTimeSeries', [traceY], layout);
    Plotly.newPlot('zTimeSeries', [traceZ], layout);
    Plotly.newPlot('rcouC1TimeSeries', [traceRC1], { ...layout, title: 'RC Out Channel 1 (Aileron)' });
    Plotly.newPlot('rcouC2TimeSeries', [traceRC2], { ...layout, title: 'RC Out Channel 2 (Elevator)' });
    Plotly.newPlot('rcouC3TimeSeries', [traceRC3], { ...layout, title: 'RC Out Channel 3 (Throttle)' });
    Plotly.newPlot('rcouC4TimeSeries', [traceRC4], { ...layout, title: 'RC Out Channel 4 (Rudder)' });
    Plotly.newPlot('rcouC8TimeSeries', [traceRC8], { ...layout, title: 'RC Out Channel 8 (RPM Control)' });
}


});

function updateLoadingHeaderText(text) {
    const loadingHeader = document.querySelector('.loadingHeaderText');
    if (loadingHeader) {
        loadingHeader.textContent = text;
    }
}

function exportData(format = 'excel') {
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

        // Prompt user for a filename
        const filename = prompt("Enter the filename for the exported data:", "clip");

        // If the user cancels the prompt, filename will be null
        if (filename === null) {
            return;
        }

        // Update the loading header text for export
        updateLoadingHeaderText(`Exporting ${filename} to ${format.toUpperCase()}...`);

        // Emit an event to the server with the necessary data
        socket.emit('export_data', {
            start_time: startRange.toISOString(), 
            end_time: endRange.toISOString(), 
            filename: filename,
            format: format
        });

    } else {
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