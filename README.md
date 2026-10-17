# Smartphone Activity Data: Visual Exploration (UCI HAPT)

MSc Big Data Analytics coursework (CIS4118, Data Visualisation, Edge Hill University, 2022): interactive visualisation of multivariate smartphone sensor data from 30 people doing everyday activities, and how much a phone can tell about what you're doing and who you are.

## Findings

- **Activity:** after PCA and t-SNE, an untuned LightGBM model (500 estimators) predicted the activity with **93% accuracy** across 12 activities.
- **The person:** within a single activity, the same approach identified *which participant* it was. Walking reached 96%, walking upstairs 92% and walking downstairs 88%. In the t-SNE plot, each person's walking style forms its own cluster.
- **Stairs:** participants walked about 9% faster downstairs than upstairs, assuming equal trips each way. Time spent on stairs was roughly normally distributed (mean 62.95 s, SD 7.90 s).
- **Sensors:** the accelerometer carried slightly more information than the gyroscope, but both mattered.

The `figures/` folder has the plots behind these findings:
- samples per participant
- t-SNE by activity and by participant
- LightGBM feature importance by sensor
- upstairs vs downstairs time
- the stair-time distribution
- total walking time
- label counts

## Data

[Smartphone-Based Recognition of Human Activities and Postural Transitions (HAPT)](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions), UCI Machine Learning Repository (Reyes-Ortiz et al., 2015):
- 30 volunteers aged 19–48, with a Samsung Galaxy S II on the waist.
- Accelerometer and gyroscope sampled at 50 Hz.
- 2.56-second windows with 50% overlap, giving 561 features per window.
- 6 activities plus 6 postural transitions.

The data isn't in this repo; its licence prohibits commercial use. To run the script, download it into a `Dataset/` folder (`features.txt`, `activity_labels.txt`, `Train/` and `Test/`).

## Method

- Combine the train and test sets, scale with `StandardScaler`, then PCA followed by t-SNE.
- LightGBM classifier on a train/test split of the t-SNE embedding, first for activity, then per activity for participant.
- Interactive Plotly charts, plus Matplotlib, Seaborn and SciPy for the statistics.

## Files

| Path | What it is |
|---|---|
| `Coursework 2.py` | Full analysis and visualisation |
| `figures/` | Exported plots |
| `report/Coursework 2.docx` | The coursework report |

**Requirements:** Python 3, NumPy, pandas, Matplotlib, Seaborn, Plotly, SciPy, scikit-learn, LightGBM.

## Known issues

- **The 93% is optimistic.** PCA and t-SNE were fitted on train and test combined before splitting, and the split isn't by volunteer, so the classifier saw test-set structure. The report frames the model as a quick, untuned check that the data separates, not a deployable classifier.
- **Window step.** The report says a data point arrives "every 1.8 secs", but the code correctly uses 1.28 s (2.56 s windows, 50% overlap).
- **Rare activities.** Postural transitions have under 150 samples each, so accuracy on them is poorer, as the report notes.
- **Plotly output.** Running the script writes `temp-plot.html` next to it.
