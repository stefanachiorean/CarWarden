# How CarWarden works, step by step

## 1. The data
A CAN frame is one broadcast message on the vehicle network: a timestamp, an ID (what kind of
message it is, and its priority: lower ID wins the bus), a length (DLC) and up to 8 payload bytes.
Every ECU hears every frame and nobody authenticates the sender, so an attacker on the bus can
inject frames. The HCRL files mark injected frames with `T` and normal ones with `R`.

## 2. Why windows
One frame says almost nothing. Attacks show up in *behaviour over time*: too many frames, new IDs,
unusual timing. So each capture is cut into 0.1 s windows and each window is summarized by 10 numbers
(src/features.py). That table of windows is what every model sees.

## 3. Detector (unsupervised)
An Isolation Forest is trained on normal windows only. It builds random trees; points that are easy
to isolate with few splits are anomalies. Because it has never seen attacks, it can flag attacks of
a kind it was not trained on. The raw score is rescaled to a 0-100 risk: 50 = threshold chosen so
about 1% of held-out normal windows exceed it.

## 4. Classifier (supervised)
Once a window is flagged, a Random Forest trained on labelled windows names the attack type and gives
a probability (shown as confidence). It only knows the types it was trained on.

## 5. Explanation
Layer 1: for each feature, a robust z-score against normal windows (median and MAD). Features beyond
3 are listed. Layer 2: IDs that never appear in normal traffic, or appear far more than usual.

## 6. Evaluation without cheating
Training uses the first 60% of each capture in time; testing uses the last 40%. Normal windows are
further split into "fit the forest" and "choose the threshold". Reporting precision/recall on windows
the model trained on, or choosing the threshold on test data, would inflate results.

## 7. Evasion test (Phase 2)
We delete a share of the injected frames from the test captures and re-run the detector on the
*same* windows that originally contained the attack. Recall as a function of injection rate shows
where the defence breaks.

## 8. Read the code in this order
data.py -> features.py -> detector.py -> explain.py -> classifier.py -> evaluate.py -> evasion.py -> pipeline.py -> carwarden.py -> app.py.
Each has a matching test in tests/ that shows how it is meant to be used.
