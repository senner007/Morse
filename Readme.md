### TODO:

- Model/Algorithm to determine signal presence in incoming audio (100 or 200 pixels ahead)
- Noise generator class to apply realistic noise patterns
- Global class to log model training parameters and results to json file for visualizatoin and db storage 
- Log batch size to results
- Collect global variables for all training and fft-flows. eg. tempo interval, cropping, image size etc
- fix json serialize model config in categorical data log
- add type checking to logging class properties
- random noise on each image in batch
- crop width to json
- fix json on raw data
- experiment with smaller image width on regression prediction
- implement fading patterns before noise calculation and test performance