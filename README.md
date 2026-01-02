# Battery ML Analysis

This project analyzes battery data from the NASA Li-ion battery dataset using Python to understand what most strongly contributes to battery life-cycle degradation. The analysis produces six plots that together show how battery capacity changes over time and which variables matter most.

The dataset includes battery capacity, impedance, cycle number, and ambient temperature.

## Output
![Battery ML Analysis Output](images/battery_ml_analysis.png)

**Plot 1: Actual vs. Predicted Capacity**  
The x-axis measures the battery capacity, while the y-axis measures what the capacity was predicted to be by the machine learning model. The red dashed line is x = y, so points that lie on or around it are successfully being captured by the model. The R² score of 0.88 means that 88% of the capacity variation is explained by the model using impedance, cycle number, and temperature. Since the data is fairly linear, there does not appear to be a clear bias.

**Plot 2: Residual Plot (Actual − Predicted)**  
No obvious clusters or patterns appear in the residuals, which suggests the model does not have a major functional form issue. Any outliers may correspond to atypical aging of a cell or a rare discharge cycle.

**Plot 3: Model Performance Comparison by R²**  
Gradient Boosting performs the best. This makes sense since battery capacity degradation is non-linear with respect to temperature, impedance, and cycle number.

**Plot 4: Battery Capacity Degradation Over Cycles**  
Each point represents a discharge cycle from a battery. Capacity generally trends downward as the number of cycles increases. At temperatures of 43–44 °C, degradation occurs much faster, showing that higher temperatures negatively affect battery life. This aligns with known battery behavior under extreme heat and is reinforced by the feature importance results.

**Plot 5: Capacity vs. Total Impedance**  
The x-axis shows total impedance (electrolyte resistance and charge-transfer resistance), and the y-axis shows capacity. Higher impedance corresponds to lower usable capacity, indicating that internal resistance growth plays a key role in degradation.

**Plot 6: Feature Importance**  
Ambient temperature is the strongest predictor of battery aging, significantly outweighing the other features in the Gradient Boosting model.

To run the code locally, clone the repository, install dependencies using `pip install -r requirements.txt`, place `metadata.csv` in `cleaned_dataset/data/`, and run:

With minor changes, this code can be reused on other lithium-ion battery datasets to study degradation behavior and identify dominant aging factors.
