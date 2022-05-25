### TODO:

- Log batch size to results
- Collect global variables for all training and fft-flows. eg. tempo interval, cropping, image size etc
- add type checking to logging class properties
- crop width to json
- implement fading patterns before noise calculation and test performance
- add custom metrics to replace testing and store to json during training
- save images used for prediction during prediction flow to feature letter verification lookup
- log prediction confidence score during prediction
- train on tempo rescaled images also
- create generator version that trains/tests faster on premade images

## DONE:

- Model/Algorithm to determine signal presence in incoming audio (100 or 200 pixels ahead)
- Noise generator class to apply realistic noise patterns
- Global class to log model training parameters and results to json file for visualizatoin and db storage 
- fix json serialize model config in categorical data log
- random noise on each image in batch
- experiment with smaller image width on regression prediction