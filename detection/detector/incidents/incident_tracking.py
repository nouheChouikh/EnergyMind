"""Groups flagged hours into incidents, closes them, merges a phase's homes."""

# To be built: the same verdict on the same home or phase over consecutive hours is one incident;
# it closes after 6 normal hours in a row; a phase incident absorbs its homes' rows (one problem, one alert).
