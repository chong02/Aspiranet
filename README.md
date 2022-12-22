# Aspiranet
This project was conducted as a part of UC Berkeley's Data Science Discovery Program. As a part of the program I was tasked to work alongside a team of other Berkeley Data Science students to help Aspiranet, a social services organization in the state of California analyze their service data and report on any disparities identified as a part of a new standard set by the Joint Commission. As a part of this project, I created an interactive dashboard (`aspiranet_county_level_dashboard.html`) to visualize the data we were provided and an accompanying Python script (`dashboard.py`) to allow them to generate a similar dashboard using new data in the future. A more in-depth discussion of design and reproducability requirements are included below.

## `aspiranet_county_level_dashboard.html`
This dashboard was created via a geopandas explore object saved as an html using both client data and incident data from Aspiranet. More specifically, a snapshot of all of Aspiranet's clients at the start of the project (approximately September 2022) and a snapshot of all incident reports over the past three years from the start of the project were used to create this dashboard. More specifics to recreate this dashboard using further data are included in the `dashboard.py` section (see below).

## `dashboard.py`
