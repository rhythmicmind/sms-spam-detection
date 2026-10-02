# Data card

`data/demo.csv` contains 60 synthetic messages created for this portfolio's offline
smoke tests. Its scores are not evidence of real-world spam detection performance.

For evaluation, run `python download_data.py` to fetch SMS Spam Collection:
Almeida, T. & Hidalgo, J. (2011). UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5CC84
Source: https://archive.ics.uci.edu/dataset/228/sms
UCI lists the dataset under CC BY 4.0; this is separate from the code's MIT license.
Raw UCI data is not bundled here. Do not upload private messages.

Historical English SMS does not represent current scams, email, or Swahili text.
Normalized exact duplicates are removed before splitting. Near duplicates may
remain; compare grouped or temporal splits before claiming deployment readiness.
