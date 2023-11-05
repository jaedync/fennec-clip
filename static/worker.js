// worker.js

// Function to average data points
function averageData(data) {
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

self.onmessage = function(event) {
    const { jobId, timeLabels, originalGpsColors, startRange, endRange, unselectedAlpha } = event.data;
    
    const gpsColors = timeLabels.map((time, index) => {
        return (time >= startRange && time <= endRange) 
            ? originalGpsColors[index] 
            : `rgba(48, 48, 48, ${unselectedAlpha})`;  // Using RGBA
    });
    // console.log(timeLabels)

    const avgTimeLabels = averageData(timeLabels.map(date => date.getTime())).map(time => new Date(time));
    
    const imuTimeSeriesColors = avgTimeLabels.map(time => {
        return (time >= startRange && time <= endRange) 
            ? 'rgb(0,255,255)' 
            : `rgba(48, 48, 48, ${unselectedAlpha})`;  // Using RGBA
    });

    const rcouTimeSeriesColors = avgTimeLabels.map(time => {
        return (time >= startRange && time <= endRange) 
            ? 'rgb(255,165,0)' 
            : `rgba(48, 48, 48, ${unselectedAlpha})`;  // Using RGBA
    });

    // Send back results along with the job ID
    self.postMessage({ 
        jobId, 
        gpsColors, 
        imuTimeSeriesColors,
        rcouTimeSeriesColors
    });
};
