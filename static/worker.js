self.onmessage = function(event) {
    const { jobId, xkf1Data, imuData, RCOUData, startRange, endRange, unselectedAlpha } = event.data;

    // Gradient for xkf1Data
    const xkf1Colors = xkf1Data.map((item, index, array) => {
        let itemTime = new Date(item.Unix_Epoch_Time * 1000);
        const isSelected = itemTime >= startRange && itemTime <= endRange;
        const gradientColor = `rgb(${Math.floor(255 * index / array.length)}, 0, ${Math.floor(255 - 255 * index / array.length)})`;
        return isSelected ? gradientColor : `rgba(24, 24, 24, ${unselectedAlpha})`;
    });

    // Prepare color data for IMU charts
    const createImuColors = (color) => imuData.map(item => {
        let itemTime = new Date(item.Unix_Epoch_Time * 1000);
        return (itemTime >= startRange && itemTime <= endRange) ? color : `rgba(48, 48, 48, ${unselectedAlpha})`;
    });

    // Prepare color data for RCOU charts
    const createRCOUColors = (color) => RCOUData.map(item => {
        let itemTime = new Date(item.Unix_Epoch_Time * 1000);
        return (itemTime >= startRange && itemTime <= endRange) ? color : `rgba(48, 48, 48, ${unselectedAlpha})`;
    });

    self.postMessage({
        jobId, 
        xkf1Colors, 
        imuAccXColors: createImuColors('rgb(255, 0, 0)'), // Red
        imuAccYColors: createImuColors('rgb(0, 255, 0)'), // Green
        imuAccZColors: createImuColors('rgb(0, 0, 255)'), // Blue
        RCOUColors: createRCOUColors('rgb(255,165,0)'), // Orange
    });
};
