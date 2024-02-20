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
let globalFileName;
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

// Get the modal
var modal = document.getElementById('exportDataModal');

// Get the button that opens the modal
var btn = document.getElementById('showExportModal');

// Get the <span> element that closes the modal
var span = document.getElementsByClassName('close-button')[0];

// When the user clicks the button, open the modal 
btn.onclick = function() {
  modal.style.display = 'block';
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
  modal.style.display = 'none';
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = 'none';
  }
}

const fileTypes = ["Mode", "IMU_0", "IMU_1", "BAT", "MAG_0", "MAG_1", "MAG_2", "BARO", "ATT", "RATE", "XKF1_0", "XKF1_1", "XKF5_0", "RCIN", "RCOU", "VIBE_0", "VIBE_1", "GPS", "GPA", "50HZ"];
const fileTypesSelectionDiv = document.querySelector('.file-types-checkboxes');

// Populate checkboxes
fileTypes.forEach(type => {
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.id = type;
  checkbox.name = 'fileTypes';
  checkbox.value = type;

  const label = document.createElement('label');
  label.htmlFor = type;
  label.appendChild(document.createTextNode(type));

  const div = document.createElement('div');
  div.appendChild(checkbox);
  div.appendChild(label);
  fileTypesSelectionDiv.appendChild(div);
});

// Load selections from local storage
let savedSelections = JSON.parse(localStorage.getItem('selectedFileTypes'));
if (!savedSelections || savedSelections.length === 0) {
  savedSelections = fileTypes;
}
savedSelections.forEach(type => {
  const checkbox = document.getElementById(type);
  if (checkbox) checkbox.checked = true;
});

// Select All and Select None Buttons
document.getElementById('selectAll').addEventListener('click', () => {
  fileTypes.forEach(type => {
    const checkbox = document.getElementById(type);
    checkbox.checked = true;
  });
  updateLocalStorageSelections();
});

document.getElementById('selectNone').addEventListener('click', () => {
  fileTypes.forEach(type => {
    const checkbox = document.getElementById(type);
    checkbox.checked = false;
  });
  updateLocalStorageSelections();
});

// Update local storage on checkbox change
document.querySelectorAll('input[name="fileTypes"]').forEach(checkbox => {
  checkbox.addEventListener('change', () => {
    updateLocalStorageSelections();
  });
});

function updateLocalStorageSelections() {
  const selected = [];
  document.querySelectorAll('input[name="fileTypes"]:checked').forEach(checkbox => {
    selected.push(checkbox.value);
  });
  localStorage.setItem('selectedFileTypes', JSON.stringify(selected));
}



const socket = io.connect(document.domain + ":" + location.port);
document.addEventListener("DOMContentLoaded", function () {
  fetchFileList();
  checkDataAvailability();
  // Initialize Web Worker
  const colorWorker = new Worker("/static/worker.js");

  function resetTimeline() {
    if (timeline) {
      timeline.destroy();
      timeline = null;
    }
  }

  function reset3DChart() {
    Plotly.purge("dPlot");
  }

  function resetTimeSeriesPlots() {
    Plotly.purge("xTimeSeries");
    Plotly.purge("yTimeSeries");
    Plotly.purge("zTimeSeries");
    Plotly.purge("rcouC1TimeSeries");
    Plotly.purge("rcouC2TimeSeries");
    Plotly.purge("rcouC3TimeSeries");
    Plotly.purge("rcouC4TimeSeries");
    Plotly.purge("rcouC8TimeSeries");
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
    Plotly.purge("dPlot");
    Plotly.purge("xTimeSeries");
    Plotly.purge("yTimeSeries");
    Plotly.purge("zTimeSeries");
    Plotly.purge("rcouC1TimeSeries");
    Plotly.purge("rcouC2TimeSeries");
    Plotly.purge("rcouC3TimeSeries");
    Plotly.purge("rcouC4TimeSeries");
    Plotly.purge("rcouC8TimeSeries");
  }

  function fetchFileList() {
    fetch("/get_file_list")
      .then((response) => response.json())
      .then((files) => {
        const fileListContainer = document.getElementById("fileList");
        fileListContainer.innerHTML = ""; // Clear existing list

        files.forEach((file) => {
          const fileBox = document.createElement("div");

          // Check if the file is the currently loaded file
          if (file.filename === globalFileName) {
            fileBox.className = "file-box loaded"; // Set class to 'loaded' only
          } else {
            fileBox.className = `file-box ${file.type}`; // Use file type for class
          }

          // Create a container for text elements
          const textContainer = document.createElement("div");
          textContainer.className = "text-container";

          // Create a span for the filename
          const fileNameSpan = document.createElement("span");
          fileNameSpan.innerText = file.filename;

          // Create a span for the file size in smaller text
          const fileSizeSpan = document.createElement("span");
          fileSizeSpan.innerText = `${formatBytes(file.size)}`;
          fileSizeSpan.style.fontSize = "smaller";
          fileSizeSpan.style.display = "block"; // To make it appear below the filename

          // Create a span for the last edited date
          const fileDateSpan = document.createElement("span");
          fileDateSpan.innerText = `${file.mtime}`;
          fileDateSpan.style.fontSize = "smaller";
          fileDateSpan.style.display = "block"; // To make it appear below the file size

          // Append filename, file size, and last edited date to the text container
          textContainer.appendChild(fileNameSpan);
          textContainer.appendChild(fileSizeSpan);
          textContainer.appendChild(fileDateSpan);

          // Append text container to the fileBox
          fileBox.appendChild(textContainer);

          // Add a 'Load' button for .bin files
          if (file.type === "bin") {
            const loadBtn = document.createElement("button");
            loadBtn.innerText = "Load";
            loadBtn.className = "load-btn";
            if (file.filename === globalFileName) {
              loadBtn.classList.add("disabled");
              loadBtn.disabled = true; // Disable the button
            }
            loadBtn.onclick = (event) => {
              event.stopPropagation(); // Prevent triggering the file download
              loadBinFile(file.filename);
            };

            // Append the Load button after the text container
            fileBox.appendChild(loadBtn);
          }

          // File download functionality
          fileBox.onclick = () => {
            const downloadLink = document.createElement("a");
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
      .catch((error) => console.error("Error fetching file list:", error));
  }

  // Function to load a .bin file using the /load_json endpoint
  function loadBinFile(filename) {
    // Assuming the JSON file has the same name as the BIN file but with a .json extension
    const jsonFilename = filename.replace(".bin", ".json");

    fetch("/load_json", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ filename: jsonFilename }),
    })
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error("Failed to load file");
        }
      })
      .then((data) => {
        console.log("File loaded:", data);
      })
      .catch((error) => console.error("Error loading file:", error));
  }

  // Helper function to format bytes into a readable format
  function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return "0 Bytes";

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  function checkDataAvailability() {
    fetch("is_data_available")
      .then((response) => response.json())
      .then((data) => {
        if (data.available) {
          fetchDataAndInit(); // Fetch data initially
        }
      })
      .catch((error) => {
        console.error("Failed to check data availability:", error);
      });
  }
  // Listen for messages from worker
  // Listen for messages from worker
  colorWorker.onmessage = function (event) {
    const {
      jobId,
      xkf1Colors,
      imuAccXColors,
      imuAccYColors,
      imuAccZColors,
      RCOUColors,
    } = event.data;

    // Only update if the job ID matches the latest job
    if (jobId === latestJobId) {
      Plotly.restyle("dPlot", { "marker.color": [xkf1Colors] });
      Plotly.restyle("xTimeSeries", { "marker.color": [imuAccXColors] });
      Plotly.restyle("yTimeSeries", { "marker.color": [imuAccYColors] });
      Plotly.restyle("zTimeSeries", { "marker.color": [imuAccZColors] });

      // New lines to update RCOU colors
      Plotly.restyle("rcouC1TimeSeries", { "marker.color": [RCOUColors] });
      Plotly.restyle("rcouC2TimeSeries", { "marker.color": [RCOUColors] });
      Plotly.restyle("rcouC3TimeSeries", { "marker.color": [RCOUColors] });
      Plotly.restyle("rcouC4TimeSeries", { "marker.color": [RCOUColors] });
      Plotly.restyle("rcouC8TimeSeries", { "marker.color": [RCOUColors] });
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
      "Remember, don't blink. Blink and you're dead...",
    ];

    return messages[Math.floor(Math.random() * messages.length)];
  }

  // Fetch data and initialize charts
  function fetchDataAndInit() {
    resetTimeline();
    // Show loading message and spinner
    document.getElementById("dataLoadingMessage").style.display = "block";
    document.getElementById("randomMessage").innerText = getRandomMessage(); // Set random message

    fetch("data")
      .then((response) => response.json())
      .then((jsonData) => {
        console.log("Fetched Data:", jsonData); // Log the fetched data

        processData(jsonData);
        initVisualizations();

        // Hide loading message and spinner
        document.getElementById("dataLoadingMessage").style.display = "none";
      })
      .catch((error) => {
        // Hide loading message and spinner in case of an error
        document.getElementById("dataLoadingMessage").style.display = "none";
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
    const loadingElement = document.getElementById("loadingMessage");
    if (isLoading) {
      loadingElement.style.display = "block";
    } else {
      loadingElement.style.display = "none";
    }
  }

  // Function to clear all status messages from the queue
  function clearStatusMessages() {
    const statusQueue = document.getElementById("statusQueue");
    while (statusQueue.firstChild) {
      statusQueue.removeChild(statusQueue.firstChild);
    }
  }

  // Function to append a new status message to the queue
  function appendStatusMessage(message, type = "default", color = "#5cb4b8") {
    const statusQueue = document.getElementById("statusQueue");

    // Clear any existing messages before appending a new one
    while (statusQueue.firstChild) {
      statusQueue.removeChild(statusQueue.firstChild);
    }

    const newStatus = document.createElement("div");
    newStatus.className = `status-message status-${type}`;
    newStatus.innerText = message;

    // Set border-left color based on the provided color
    newStatus.style.borderLeftColor = color;

    // Apply the fade-in animation
    newStatus.style.animation = "slideFadeIn 0.5s forwards";

    // Append the new status message
    statusQueue.appendChild(newStatus);
  }

  // Listen for status messages from the server
  socket.on("status", function (data) {
    console.log(data.message);
    if (data.progress) {
      updateProgressBar(data.progress);
    }
    // Check if the color is provided in the data, otherwise default to a standard color
    var messageColor = data.color || "#5cb4b8"; // Default color
    // Append the received status message to the queue with the specified color
    appendStatusMessage(data.message, "default", messageColor);
    if (data.message === "Bin File Upload Complete!" || data.message === "File selection changed.") {
      // Hide loading message and spinner
      setLoading(false);
      // Clear the status message queue
      clearStatusMessages();
      // Refresh the charts and data
      resetCharts();
      fetchDataAndInit();
      fetchFileList();
    }
    if (data.message === "File Export Complete!") {
      // Hide loading message and spinner
      setLoading(false);
      // Clear the status message queue
      clearStatusMessages();
      // Refresh file list
      fetchFileList();
    }
    if (data.url) {
        // Create a hidden 'a' element
        var downloadLink = document.createElement("a");
        downloadLink.href = data.url;
        downloadLink.style.display = 'none';
        downloadLink.setAttribute('download', '');

        // Append the link to the body and trigger a click to download
        document.body.appendChild(downloadLink);
        downloadLink.click();

        // Remove the link from the DOM
        document.body.removeChild(downloadLink);
    }
    
    if (data.message === "Starting conversion...") {
      setLoading(true);
    }
  });

  function updateProgressBar(progress) {
    const progressBar = document.getElementById("progressBar");

    progressBar.style.width = progress + "%";
  }

  function setProgressBarColor(color) {
    const progressBar = document.getElementById("progressBar");
    progressBar.style.background = color;
  }

  function updateLoadingHeaderText(text) {
    const loadingHeader = document.querySelector(".loadingHeaderText");
    if (loadingHeader) {
      loadingHeader.textContent = text;
    }
  }

  // Show overlay when file is dragged over the window
  window.addEventListener("dragover", function (e) {
    e.preventDefault();
    dragDropOverlay.style.display = "flex";
  });

  // Hide overlay when file is no longer dragged over the window
  window.addEventListener("dragleave", function (e) {
    e.preventDefault();
    if (e.pageX === 0 || e.pageY === 0) {
      // Check if the mouse left the window
      dragDropOverlay.style.display = "none";
    }
  });

  // Handle file drop
  window.addEventListener("drop", function (e) {
    e.preventDefault();
    dragDropOverlay.style.display = "none";

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].name.endsWith(".bin")) {
      uploadFile(files[0]);
    } else {
      alert("Please drop a single .bin file.");
    }
  });

  function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    updateLoadingHeaderText(`Uploading ${file.name}...`);
    setProgressBarColor("#FFA500"); // Orange color for upload

    if (file.name.endsWith(".bin")) {
      setLoading(true);
      appendStatusMessage(
        "Bin file upload in progress...",
        "default",
        "#FFA500" // Orange color
      );
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);

    let uploadStartTime = Date.now(); // Record the start time of the upload
    let lastLoaded = 0; // To track the amount of data uploaded
    let totalSize = file.size; // Total size of the file
    let statusUpdateInterval; // Interval for updating the status message

    // Function to update upload status
    function updateUploadStatus() {
      const elapsedSeconds = (Date.now() - uploadStartTime) / 1000;
      const speed = formatSpeed(lastLoaded / elapsedSeconds);
      const percentage = totalSize
        ? Math.round((lastLoaded / totalSize) * 100)
        : 0;
      appendStatusMessage(
        `Uploading... ${percentage}% (${speed})`,
        "default",
        "#FFA500"
      );
    }

    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) {
        const percentage = Math.round((e.loaded / totalSize) * 100);
        updateProgressBar(percentage); // Continuously update the progress bar
        lastLoaded = e.loaded; // Update the last loaded amount
      }
    };

    xhr.onload = function () {
      clearInterval(statusUpdateInterval); // Clear the interval for status updates
      if (xhr.status === 200) {
        socket.emit("upload_and_convert", { filename: file.name });
        appendStatusMessage("Upload successful!", "success", "#5cb85c"); // Green color for success
      } else {
        appendStatusMessage(
          "Upload failed: " + xhr.responseText,
          "error",
          "#d9534f"
        ); // Red color for error
      }
    };

    xhr.onerror = function () {
      clearInterval(statusUpdateInterval); // Clear the interval for status updates
      appendStatusMessage("Upload error", "error", "#d9534f"); // Red color for error
    };

    xhr.send(formData);

    // Update status immediately and then every 1000 milliseconds
    updateUploadStatus(); // Immediate update
    statusUpdateInterval = setInterval(updateUploadStatus, 1000); // Regular updates
  }

  // Helper function to format speed into Mbps
  function formatSpeed(bitsPerSecond) {
    const sizes = ["Bits/sec", "Kbps", "Mbps", "Gbps", "Tbps"];
    if (bitsPerSecond === 0) return "0 Bits/sec";
    bitsPerSecond *= 8; // Convert bytes to bits
    const i = parseInt(Math.floor(Math.log(bitsPerSecond) / Math.log(1024)));
    return Math.round(bitsPerSecond / Math.pow(1024, i), 2) + " " + sizes[i];
  }

  // Trigger the hidden file input when the "Upload" button is clicked
  document.getElementById("uploadBtn").addEventListener("click", function () {
    // Create a new file input element
    let fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".bin"; // Specify that only .bin files should be accepted
    fileInput.style.display = "none";

    // Append the file input to the document
    document.body.appendChild(fileInput);

    // Trigger file selection when the hidden file input is clicked
    fileInput.click();

    // Handle file selection
    fileInput.onchange = function () {
      if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
      }
      // Remove the file input from the DOM
      document.body.removeChild(fileInput);
    };

    // Remove the file input if the user cancels the file selection dialog
    fileInput.addEventListener("click", function (event) {
      setTimeout(() => {
        if (fileInput.value === "") {
          document.body.removeChild(fileInput);
        }
      }, 0);
    });
  });

  // Function to process data
  function processData(data) {
    if (!data) {
      console.error("Data is undefined or null");
      return;
    }

    // Extract XKF1 and IMU data
    xkf1Data = data["XKF1"] || [];
    imuData = data["IMU"] || [];
    modeTableData = data["MODE"] || [];
    gpsData = data["GPS"] || [];
    RCOUData = data["RCOU"] || [];
    console.log("xkf1Data:", xkf1Data);
    console.log("imuData:", imuData);

    // Check if file info is available in the data
    if (data.file_info) {
      const fileInfoEl = document.getElementById("fileInfo");
      globalFileName = data.file_info.file_name;
      fileInfoEl.innerHTML = `
            <p>File: ${data.file_info.file_name}</p>
            <p>Date: ${formatDate(data.file_info.datetime_chicago)}</p>
            <p>Duration: ${formatDuration(
              data.file_info.flight_duration_seconds
            )}</p>
        `;
    }
    timeLabels = gpsData.map((row) => new Date(row["Datetime_Chicago"]));

    // Prepare data for the 3D chart (XKF1)
    gpsDataLatMeters = xkf1Data.map((row) => row["PN"]); // North position
    gpsDataLngMeters = xkf1Data.map((row) => row["PE"]); // East position
    barometerDataMeters = xkf1Data.map((row) => row["PD"]); // Down position (altitude)

    // Initialize originalGpsColors
    const numPoints = gpsDataLatMeters.length;
    originalGpsColors = Array.from(
      { length: numPoints },
      (_, i) =>
        `rgb(${Math.floor((255 * i) / numPoints)}, 0, ${Math.floor(
          255 - (255 * i) / numPoints
        )})`
    );
  }

  // Helper function to format the date
  function formatDate(dateString) {
    const date = new Date(dateString);
    const formattedDate = date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
    return formattedDate;
  }

  // Helper function to format duration in seconds to a more readable format
  function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.round(seconds % 60);
    let formattedDuration = "";

    if (hours > 0) {
      formattedDuration += `${hours} ${hours > 1 ? "hours" : "hour"} `;
    }
    if (minutes > 0) {
      formattedDuration += `${minutes} ${minutes > 1 ? "minutes" : "minute"} `;
    }
    if (secs > 0 || formattedDuration === "") {
      formattedDuration += `${secs} ${secs === 1 ? "second" : "seconds"}`;
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
      x: gpsDataLngMeters, // East position
      y: gpsDataLatMeters, // North position
      z: barometerDataMeters.map((alt) => -alt), // Down position (inverted for altitude)
      mode: "markers",
      type: "scatter3d",
      marker: {
        color: originalGpsColors,
        opacity: 0.4,
      },
    };

    const maxAltitude = Math.max(...barometerDataMeters.map((alt) => -alt));

    const layout = {
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: "#05060d",
      plot_bgcolor: "#05060d",
      scene: {
        xaxis: {
          title: "Latitude (m)",
          color: "white",
          gridcolor: "#888",
        },
        yaxis: {
          title: "Longitude (m)",
          color: "white",
          gridcolor: "#888",
        },
        zaxis: {
          title: "Altitude (m)",
          range: [0, maxAltitude],
          color: "white",
          gridcolor: "#888",
        },
        aspectratio: { x: 1, y: 1, z: 1 }, // This enforces the same scaling factor on all axes
      },
      font: {
        color: "white",
      },
    };

    Plotly.newPlot("dPlot", [trace], layout, { responsive: true });
  }

  async function fetchModeTable() {
    try {
      const response = await fetch("/get_mode_table");
      if (!response.ok) {
        console.error("Failed to fetch mode table:", response.statusText);
        return [];
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Error fetching mode table:", error);
      return [];
    }
  }

  async function initTimeline() {
    const container = document.getElementById("visualization");

    // const modeTableData = await fetchModeTable();
    console.log("mode_table:", modeTableData);
    const modeItems = modeTableData.map((entry) => {
      let className = "";
      switch (entry.ModeName) {
        case "POSHOLD":
          className = "mode-poshold";
          break;
        case "STABILIZE":
          className = "mode-stabilize";
          break;
        case "RTL":
          className = "mode-rtl";
          break;
        default:
          break;
      }
      return {
        start: new Date(entry.Unix_Epoch_Time * 1000), // Convert to milliseconds
        content: entry.ModeName,
        editable: false, // Locked marker
        group: 1, // Assign all mode items to the same group
        className: className,
      };
    });

    // Create background items
    const backgroundItems = [
      {
        start: new Date(timeLabels[0].getTime() - 60000),
        end: timeLabels[0],
        type: "background",
        className: "gray-background",
      },
      {
        start: timeLabels[timeLabels.length - 1],
        end: new Date(timeLabels[timeLabels.length - 1].getTime() + 60000),
        type: "background",
        className: "gray-background",
      },
    ];

    const items = new vis.DataSet([...backgroundItems, ...modeItems]);

    const options = {
      min: new Date(timeLabels[0].getTime() - 60000),
      max: new Date(timeLabels[timeLabels.length - 1].getTime() + 60000),
      editable: false,
      zoomMin: 1000,
      zoomMax: 1000 * 60 * 60 * 24,
      zoomable: true, // Enable zooming
      moveable: true, // Enable moving the timeline left and right
      snap: function (date, scale, step) {
        return date;
      },
      stack: false, // Disable stacking
      orientation: "top", // Ensure all items are aligned at the top
    };

    // Create the timeline
    timeline = new vis.Timeline(container, items, options);

    // Add custom time bars for start and end
    timeline.addCustomTime(timeLabels[0], "startMarker");
    timeline.addCustomTime(timeLabels[timeLabels.length - 1], "endMarker");

    // Listen to the `timechange` event to update 3D plot
    timeline.on("timechange", function (event) {
      if (event.id === "startMarker" || event.id === "endMarker") {
        updateColors(event.time, event.id);
      }
    });

    styleCustomTimeMarkers();

    updateDisplayedTimeRange(timeLabels[0], timeLabels[timeLabels.length - 1]);
  }

  function updateDisplayedTimeRange(start, end) {
    const startTimeDisplay = document.getElementById("startTimeDisplay");
    const endTimeDisplay = document.getElementById("endTimeDisplay");

    const startTimeString = start.toTimeString().split(" ")[0]; // Keep HH:MM:SS
    const endTimeString = end.toTimeString().split(" ")[0]; // Keep HH:MM:SS

    startTimeDisplay.innerHTML = `Start: ${startTimeString}`;
    endTimeDisplay.innerHTML = `End: ${endTimeString}`;
  }

  function styleCustomTimeMarkers() {
    setTimeout(() => {
      const startMarker = document.querySelector(
        ".vis-custom-time.startMarker"
      );
      const endMarker = document.querySelector(".vis-custom-time.endMarker");

      if (startMarker && endMarker) {
        startMarker.style.backgroundColor = "blue";
        endMarker.style.backgroundColor = "red";
      }
    }, 0);
  }

  // Update function with color coding for both 3D and time-series charts
  function updateColors(newTime, markerId) {
    throttle(() => {
      let startRange, endRange;

      if (markerId === "startMarker") {
        startRange = new Date(newTime);
        endRange = new Date(timeline.getCustomTime("endMarker"));
      } else {
        startRange = new Date(timeline.getCustomTime("startMarker"));
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
        unselectedAlpha: 0.001,
      });
    }, 100); // delay of 100 milliseconds
  }

  // Create a helper function to make a new trace for RCOU data
  function createTrace(xData, yData, name) {
    return {
      x: xData,
      y: yData,
      mode: "lines+markers",
      marker: { color: "rgb(255,165,0)" },
      line: { color: "grey" },
      name: name,
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
    const timeLabelsIMU = imuData.map(
      (row) => new Date(row["Unix_Epoch_Time"] * 1000)
    );
    const imuDataX = imuData.map((row) => row["AccX"]);
    const imuDataY = imuData.map((row) => row["AccY"]);
    const imuDataZ = imuData.map((row) => row["AccZ"]);

    const timeLabelsRCOU = imuData.map(
      (row) => new Date(row["Unix_Epoch_Time"] * 1000)
    );
    const rcouC1Data = RCOUData.map((row) => row["C1"]);
    const rcouC2Data = RCOUData.map((row) => row["C2"]);
    const rcouC3Data = RCOUData.map((row) => row["C3"]);
    const rcouC4Data = RCOUData.map((row) => row["C4"]);
    const rcouC8Data = RCOUData.map((row) => row["C8"]);

    // Define a layout for the plot
    const layout = {
      margin: { l: 50, r: 10, b: 40, t: 40 },
      title: "IMU Data over Time",
      paper_bgcolor: "#05060d",
      plot_bgcolor: "#05060d",
      xaxis: {
        color: "white",
        gridcolor: "#888",
        title: "Time",
        type: "date", // Specify the x-axis type as date
      },
      yaxis: {
        color: "white",
        gridcolor: "#888",
      },
      font: {
        color: "white",
      },
    };

    // Creating traces for each axis
    const traceX = {
      x: timeLabelsIMU,
      y: imuDataX,
      mode: "lines+markers",
      line: { color: "grey" },
      marker: { color: "rgb(255, 0, 0)" },
      name: "AccX",
    };

    const traceY = {
      x: timeLabelsIMU,
      y: imuDataY,
      mode: "lines+markers",
      line: { color: "grey" },
      marker: { color: "rgb(0, 255, 0)" },
      name: "AccY",
    };

    const traceZ = {
      x: timeLabelsIMU,
      y: imuDataZ,
      mode: "lines+markers",
      line: { color: "grey" },
      marker: { color: "rgb(0, 0, 255)" },
      name: "AccZ",
    };

    const traceRC1 = createTrace(timeLabelsRCOU, rcouC1Data, "RCOU_C1");
    const traceRC2 = createTrace(timeLabelsRCOU, rcouC2Data, "RCOU_C2");
    const traceRC3 = createTrace(timeLabelsRCOU, rcouC3Data, "RCOU_C3");
    const traceRC4 = createTrace(timeLabelsRCOU, rcouC4Data, "RCOU_C4");
    const traceRC8 = createTrace(timeLabelsRCOU, rcouC8Data, "RCOU_C8");

    // Plotting the data
    Plotly.newPlot("xTimeSeries", [traceX], layout);
    Plotly.newPlot("yTimeSeries", [traceY], layout);
    Plotly.newPlot("zTimeSeries", [traceZ], layout);
    Plotly.newPlot("rcouC1TimeSeries", [traceRC1], {
      ...layout,
      title: "RC Out Channel 1 (Aileron)",
    });
    Plotly.newPlot("rcouC2TimeSeries", [traceRC2], {
      ...layout,
      title: "RC Out Channel 2 (Elevator)",
    });
    Plotly.newPlot("rcouC3TimeSeries", [traceRC3], {
      ...layout,
      title: "RC Out Channel 3 (Throttle)",
    });
    Plotly.newPlot("rcouC4TimeSeries", [traceRC4], {
      ...layout,
      title: "RC Out Channel 4 (Rudder)",
    });
    Plotly.newPlot("rcouC8TimeSeries", [traceRC8], {
      ...layout,
      title: "RC Out Channel 8 (RPM Control)",
    });
  }
});

function updateLoadingHeaderText(text) {
  const loadingHeader = document.querySelector(".loadingHeaderText");
  if (loadingHeader) {
    loadingHeader.textContent = text;
  }
}
function exportData(format = "excel") {
  if (!timeline) {
    applyShakeAnimationIfNoTimeline();
    return;
  }

  modal.style.display = 'none';
  const selectedFileTypes = JSON.parse(localStorage.getItem('selectedFileTypes')) || [];

  const startMarkerTime = timeline.getCustomTime("startMarker");
  const endMarkerTime = timeline.getCustomTime("endMarker");

  if (startMarkerTime && endMarkerTime) {
    const startRange = new Date(startMarkerTime);
    const endRange = new Date(endMarkerTime);

    const filename = prompt("Enter the filename for the exported data:", "clip");
    if (filename === null) {
      return;
    }

    updateLoadingHeaderText(`Exporting ${filename} to ${format.toUpperCase()}...`);

    const exportData = {
      start_time: startRange.toISOString(),
      end_time: endRange.toISOString(),
      filename: filename,
      format: format,
      file_types: selectedFileTypes
    };

    socket.emit("export_data", exportData);
  } else {
    console.error("Start or end marker time not found");
  }
}


function applyShakeAnimationIfNoTimeline() {
  if (!timeline) {
    const uploadBtn = document.getElementById("uploadBtn");
    uploadBtn.classList.add("shake"); // Add shake animation
    setTimeout(() => uploadBtn.classList.remove("shake"), 320); // Remove after animation duration
  }
}
