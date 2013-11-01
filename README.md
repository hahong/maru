maru (Multi-electrode Array Recording Utilities)
================================================
This project aims to provide various utilities for multi-electrode array
recording setups based on MWorks (http://mworks-project.org/) and
Blackrock/Plxeon systems.


Installation
------------
This project provides various utility scripts and a Python package that the
scripts rely on.

To install:
   `python setup.py install`

To install locally under the home directory:
   `python setup.py install --user`

Note that the package depends on the following Python packages:
  * numpy
  * scipy
  * pytables
  * scikit-learn
  * pywt
  * joblib
  * pymworks (https://github.com/hahong/pymworks.git)
  * pymario (https://github.com/hahong/pymario.git)


Usage
-----
Core utilities and a typical procedure:
```
        +--------+          +------------------+
        | MWorks |          | BlackRock/Plexon |
        +--------+          +------------------+        
             |                       |       
             v                       v
       (Original .mwk)          (.nev/.plx)------------+
             |                       |                 |
             v                       v                 |
   +-----------------------------------------------+   |
   | maru-merge: merge behavior .mwk files and     |   |
   |    neural signal files .nev/.plx to produce   |   |
   |    merged .mwk files                          |   |
   +---------------------+-------------------------+   |
                         |                             |
                         v                             |
       +-------- (Merged .mwk files)                   |
       |                 |                             |
       |                 .                             .
       |                 .                             .
       |                 .  (Optional: spike sorting)  .
       |                 .                             .
       |                 .                             .
       |                 v                             v
       |    +-----------------------------------------------+
       |    | maru-psinfo --wav=<.nev file name>            |
       |    |   collect peristimulus spike information AND  |
       |    |   waveform snippets of the spikes             |
       |    +----------------------+------------------------+
       |                           |
       |                           v
       |      (Peristimulus spiking information w/ waveforms)
       |                           |
       |                           v
       |    +-----------------------------------------------+
       |    | maru-spksort feature                          |
       |    |   compute features for clustering             |
       |    +----------------------+------------------------+
       |                           |
       |                           v
       |              (Feature files.c1.feat.h5)
       |                           |
       |                           v
       |    +-----------------------------------------------+
       |    | maru-spksort cluster                          |
       |    |   sort spikes by clustering                   |
       |    +----------------------+------------------------+
       |                           |
       |                           v
       |            (Sorted spikes files.c2.clu.h5)
       |                           |
       |                           v
       |    +-----------------------------------------------+
       |    | maru-spksort collate                          |
       |    |   merge multiple .c2.clu.h5 files and produce |
       |    |   peristimulus spike information.psf.h5 files |
       |    +---------------------------------+-------------+
       |                                      |
       v                                      |
   +-----------------------------------+      |
   | maru-psinfo: collect peristimulus |      |
   |    spike information without      |      |
   |    spike sorting                  |      |
   +-------------------+---------------+      |
                       |                      |
                       v                      v
               (Peristimulus spiking information.psf.h5)
                                 |
                                 v
          +-----------------------------------------------+
          | maru-util-psinfo2feat: compute feature matrix |
          |    by integrating over time bins              |
          +----------------------+------------------------+
                                 |
                                 v
                         (Features.feat.h5)
                                 |
                                 v
          +-----------------------------------------------+
          | maru-util-featconcat: concatnate multiple     |
          |    feature matrix files into a single file    |
          +----------------------+------------------------+
                                 |
                                 v
                    (Concatenated features.feat.h5)

```

Additional utilities:
  * maru-check-psth: plot PSTHs for .psf.h5 files and provide
    diagnosis about the quality of the channels.
  * maru-check-impedance: compare multiple BlackRock impedance
    log files and visualize changes.
