
## Abstract 
This report summarizes the analysis of Wind Power Forecasting models to predict how much power a wind farm will produce. Wind Power Forecasting is a particularly challenging task because of the high  variability and the importance of steady and accurate systems to feed the grid. The dataset provided is from a real wind farm located in China by Longyuan Power Group Corp. I created an RNN model using Pytorch to predict the wind power and used the competition's baseline GRU model code. The findings  and performance indicate promising applications of machine learning to advance the accuracy of feeding power grids with sustainable wind energy. The source of the problem description and dataset is from Baidu KDD CUP 2022 [1]. 

## Introduction  
Predicting wind power supply is a difficult and important task to provide clean energy into the grid. The  goal of my project is to build a Wind Power Forecasting model to predict the output of a collection of  wind turbines. This is an important task because the accuracy of forecasting power is critical to ensure  grid stability. Wind power has high variability due to constantly changing wind and environmental  conditions. When feeding a grid with wind power, other sources of power are required when there are  troughs in output from the wind turbines. It is critical for grid systems to know the output ahead of time  so planning can be done to ensure other energy sources are always available. 

The analysis done for this project is to predict the total power output of a wind farm 48 hours in advance. Wind farm data is provided from a utility company for 145 days of power readings from each turbine in a wind farm. I created an RNN (Recurrent Neural Network) to forecast the wind power, the model has superior accuracy in comparison to a baseline moving average forecast.

## Problem Statement  
The project task is to estimate the wind power supply of a wind farm using only historical power  readings (no weather forecasts, etc.). The specific task is to predict the sum of the total power from the  wind farm, $\sum_{i \in Farm} Patv_i$, 48 hours into the future, where $Patv_i$ is the power output of wind turbine i  at a specific timestep. The model estimates the power output 48 hours ahead of time for each turbine every  10 minutes, equating to 288 10-minute interval predictions. For example, at one time point, it is  required to predict a future length-288 wind power supply time-series. The average of RMSE (Root  Mean Square Error) and MAE (Mean Absolute Error) between the actual and predicted power output at  each time point is used to evaluate accuracy. 

## Data Sources 
The source of the dataset is from a wind power producer in China, Longyuan Power Group Corp. Ltd.  The data comes from a wind farm of 134 turbines. The dataset consists of the relative location of each  turbine in the wind farm, wind, temperature, turbine angle and historical wind power.  The time span of the dataset is half a year, where every 10 minutes a Supervisory Control and Data  Acquisition (SCADA) system takes a sample from each wind turbine in the wind farm. There is a total of  245 days of data, consisting of 13 columns with a total of 4.75 million records.  
Below are the detailed columns of the dataset (source: “SDWPF: A Dataset for Spatial Dynamic Wind  Power Forecasting Challenge at KDD Cup 2022”)
<img src="https://github.com/cce21/NNWindForecast/blob/main/img/columns.png" width="500">


## Exploratory Data Analysis 
The power readings in each wind turbine are highly variable, where the power output can go from zero to high values in very short periods of time, then go back to 0 again, Figure 1 below displays the power  output for a specific turbine. The daily mean power plot smooths this pattern, but the daily mean power  output is also highly variable, displayed in Figure 2. 
<img src="https://github.com/cce21/NNWindForecast/blob/main/img/powervstime.png" width="500">

<img src="https://github.com/cce21/NNWindForecast/blob/main/img/meanpowervstime.png" width="500">
The correlation plot below explains the relationship between each feature:
<img src="https://github.com/cce21/NNWindForecast/blob/main/img/correlation.png" width="500">


The correlation matrix shows that the Wind Speed has a high positive correlation with the power output  (‘Patv’), which aligns with the physics of wind turbines. Wind speeds are variable in nature, which is why  the power output has high variability. There is a low correlation for directional features Ndir & Wdir and  temperature features Etmp & Itmp. This indicates the model wouldn’t lose much accuracy and considerations can be made to remove them from the dataset. Also, there is perfect correlation  between the pitch angle features Pab1, Pab2, Pab3. The dataset can further be reduced by using one  variable for the pitch angle. My final model consisted only of the features Wind Speed (Wspd), Nacelle direction (Ndir), and the pitch angle of blade 1 (Pab1). I experimented discarding the other variables because of the low correlation with power generation and the accuracy decreased a trivial amount.


Another EDA discovery was that the power output has a maximum threshold, where after a certain wind  speed is reached, the power output approaches a maximum that cannot be exceeded. This intricacy has  effects on modeling where improvements to my models could be made to take this into account and  ‘trim’ forecasts. The plot below is an example of one turbine’s power output versus time which showcases the behavior from above. 
<img src="https://github.com/cce21/NNWindForecast/blob/main/img/power.png" width="500">



## Proposed Methodology 
First, I developed a simple 30-timestep moving average forecast to use as a baseline model to compare  with more advanced models.  

RNN Model

The reason I chose to use an RNN model is because of the excellent results this deep learning method  exhibits on time series forecasting tasks and the large dataset provided has enough data to effectively  train this model. The RNN model processes memories of sequential data and stores historical data from previous inputs into the model’s internal state, while predicting a target  vector. 

Visual of a simple RNN model architecture:
<img src="https://github.com/cce21/NNWindForecast/blob/main/img/rnn.png" width="500">

I divided the data into training and validation sets and used a testing dataset provided by the  competition for final evaluation. I used the first 153 days of data to train the model, and next 16 days for validation. 
The parameters I tuned to improve model performance were the number of layers, dropout,  number of epochs, batch size, and learning rate. 

## Analysis and Results 
Below is a table summarizing the performance of each best tuned model on the competition test data  set. The LightGBM model performs the best, followed by the GRU model. The moving average model  performs relatively well given its simplicity and quick run time. 
| Model      | MAE   | RMSE  | Score |
|------------|-------|-------|-------|
| Moving Avg | 52.76 | 60.53 | 56.55 |
| RNN        | 44.80 | 55.02 | 49.9  |  

An example of the model performance for 2 48-hour forecast periods of a  
single turbine from the test set is shown on the right for the RNN model. You  
can see that the model is very good at making short-term predictions, but  
then after 24 hours it is not accurate. This illustrates the issues with RNN models in forecasting accurately farther into the future.

<img src="https://github.com/cce21/NNWindForecast/blob/main/img/example_forecast.png" width="500">


## Conclusions 
An important lesson I learned is that training more complex models with large amounts of data is very  difficult. I had a limited amount of experiment time due to very high run times. I think a better strategy  that could have saved more time is using a much smaller subset of data, maybe just 50 days, to try and  identify better parameter configurations for the models. 


Bibliography and Credits 
[1] KDD Cup Competition 2022 - 
https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction 
[2] Jingbo Zhou, Xinjiang Lu, Yixiong Xiao, Jiantao Su, Junfu Lyu, Yanjun Ma, and Dejing Dou. 2022.  SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022




## Data Description
Please refer to KDD Cup 2022 --> Wind Power Forecast --> Task Definition 
(https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)

## Model Training and Testing 

Minimum usage:
```
    python path/to/code/train.py 
    python path/to/code/evaluation.py
```
The trained model will be saved in path/to/working/dir/checkpoints directory. 


    
