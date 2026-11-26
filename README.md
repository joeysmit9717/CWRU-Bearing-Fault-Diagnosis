CWRU Bearing Fault Classification
=================================

## Dataset & Attribution

The dataset used in this project was obtained from its original source and is not owned or created by me. It is used for educational and portfolio purposes only.

All rights and ownership of the original dataset remain with their respective owners. Please refer to the original source for licensing information and terms of use.

What this folder contains
-------------------------
This portfolio package documents four-class bearing fault classification
on Case Western Reserve University (CWRU) Drive End vibration data,
comparing an unknown-load split with an unknown-defect-severity split
and two feature pipelines.

  - **README.md**          - General overview of the project
  - **BearingFaultReport.pdf**          - Report on the findings
  
Key Findings
------------

The confusion matrix obtained from the BearingFault.py file which holds out a load intensity for testing purposes. 
95.37%  test accuracy.

![Bearing Fault](Report/cm_unknown_load.png)


Python codes
------------
  - **BearingFault.py**                 - train 0/1/3 HP, test 2 HP (FFT+time RF)
  - **BearingFault021unknown.py**       - train 0.007/0.014/0.028", test 0.021"
                                    (FFT+time RF)
  - **BearingFaultDifferentApproach.py** - same 0.021" hold-out with
                                    bandpass-envelope BPFO/BPFI/BSF RF

Dataset
-------

For this project the CWRU Bearing Data Centre .mat NORMAL BASELINE DATA and the 12k DRIVE END BEARING FAULT DATA were used. The total dataset consists of 61 .mat files, consisting of 1 NORMAL BASELINE DATA file and 60 BEARING FAULT DATA files. https://engineering.case.edu/bearingdatacenter/download-data-file

What the analysis does
----------------------
1. Load CWRU Drive End accelerometer .mat files (12 kHz) for Normal,
   Ball, Inner Race (IR), and Outer Race (OR) across defect sizes and
   motor loads. Skip missing OR014 @ 3:00/12:00 and OR028 files.
2. Plot baseline 0.007" @ 0 HP time signals and FFT spectra; print
   time-domain statistics (RMS, kurtosis, crest, etc.).
3. Experiment A (BearingFault.py): non-overlapping 500-sample windows;
   FFT magnitude + RMS/kurtosis/crest/P2P features; Random Forest
   trained on loads {0,1,3} HP and tested on held-out 2 HP.
4. Experiment B (BearingFault021unknown.py): same FFT+time features;
   train on defect sizes {0.007, 0.014, 0.028}", test on held-out
   0.021" (unknown bearing dent severity).
5. Experiment C (BearingFaultDifferentApproach.py): same severity
   split as experiment B; 2048-sample windows; 2-5 kHz bandpass -> Hilbert envelope;
   narrowband energy at BPFO/BPFI/BSF harmonics (+ ratios, envelope
   kurtosis/crest); Random Forest on 15 features.
6. Report test accuracy, classification report, and confusion matrices.

Generated python plots
-----------------
  baseline_stats.png                 RMS / kurtosis / crest @ 0.007" 0 HP
  time_signals_0_2000.png            Time signals, samples 0-2000
  time_signals_10000_10500.png       Time signals, samples 10000-10500
  time_signals_50000_50500.png       Time signals, samples 50000-50500
  fft_magnitude_0_2000.png           FFT magnitude, samples 0-2000
  fft_magnitude_10000_10500.png      FFT magnitude, samples 10000-10500
  fft_magnitude_50000_50500.png      FFT magnitude, samples 50000-50500
  cm_unknown_load.png                Confusion matrix (unknown 2 HP)
  cm_unknown_severity_fft.png        Confusion matrix (0.021", FFT+time)
  cm_unknown_severity_envelope.png   Confusion matrix (0.021", envelope)
  accuracy_comparison.png            Head-to-head test accuracy bar chart
                                   (portfolio summary figure for the report)

How to regenerate the plots
---------------------------

-  BearingFault.py
-  BearingFault021unknown.py
-  BearingFaultDifferentApproach.py


Requirements
------------
Python 3.x with: numpy, matplotlib, scipy, scikit-learn, CWRU NORMAL BASELINE DATA and CWRU 12k DRIVE END BEARING FAULT DATA

Key result snapshot (from last run)
-----------------------------------
  Datasets loaded: 61 (Normal baseline at 0 HP only — no Normal in tests)

  Unknown load (train 0/1/3 HP → test 2 HP, FFT+time RF):
    Test accuracy: 95.37%
    Ball/OR nearly perfect; main errors IR → OR

  Unknown severity (train 0.007/0.014/0.028" → test 0.021", FFT+time RF):
    Test accuracy: 27.93%
    Severe IR/OR confusion; broadband spectra do not transfer by size

  Unknown severity (same split, envelope BPFO/BPFI/BSF RF):
    Test accuracy: 70.42%
    Physics-informed envelope features recover most of the gap

  Takeaway: load shift is easy for FFT features; unknown defect size is
  hard; characteristic-frequency envelope features improve OOD severity
  generalization with far fewer dimensions (15 vs 255).
