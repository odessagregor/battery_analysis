# Battery ML Analysis

Analyzes battery data from the NASA Li-ion battery dataset using Python to create six graphs that together determine the largest contributing factor to battery life-cycle degradation.

## Dataset
NASA Li-ion Battery Dataset

## Output
![Battery ML Analysis](images/battery_ml_analysis.png)

## Methods
Multiple machine learning models are used to predict battery capacity using impedance, cycle number, and ambient temperature. Model performance is evaluated using R² scores and residual analysis.

## Results and Visualizations

**Plot 1: Actual vs. Predicted Capacity**  
The x-axis measures the battery capacity, while the y-axis measures what the capacity was predicted to be by the machine learning model. The red dashed line is x = y, so points that lie on or around it are successfully being captured by the model. The R² score of 0.88 means that 88% of the capacity variation is explained by the model—the capacity is successfully predicted by impedance, cycle number, and temperature. Since the data is fairly linear, there does not appear to be a bias.

**Plot 2: Residual Plot (Actual − Predicted)**  
No clusters or patterns show that the model does not have the wrong functional form or major model errors. Any outliers, such as those on the bottom, may correspond with atypical aging of a cell or a rare cycle.

**Plot 3: Model Performance Comparison by R²**  
Gradient Boosting is the best model, which makes sense since capacity degrades non-linearly with respect to temperature, impedance, and cycle number.

**Plot 4: Battery Capacity Degradation Over Cycles**  
Each point is a discharge cycle from a battery. The points generally trend downward because capacity decreases with repetition as the cycle ages. At temperatures of 43–44 °C, degradation is much faster. This shows that higher temperatures negatively affect batteries. This fits with the idea that batteries perform worse in extreme temperatures, especially extreme heat. This is further confirmed in Plot 6, which highlights that ambient temperature is the most important feature for Gradient Boosting.

**Plot 5: Capacity vs. Total Impedance**  
The x-axis is the total impedance (electrolyte resistance and charge-transfer resistance), and the y-axis is capacity. It shows that higher electrolyte resistance corresponds to lower usable capacity.

**Plot 6: Feature Importance**  
The strongest predictor for aging is ambient temperature, which is much higher than any other feature.

## Running Locally
Clone the repository, install dependencies using `pip install -r requirements.txt`, place `metadata.csv` in `cleaned_dataset/data/`, and run `python battery_ml_analysis.py`.

## Reusability
This code can be used with other datasets with light alterations, allowing it to help classify and improve Li-ion batteries based on testing.

 

