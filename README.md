maru (Multi-electrode Array Recording Utilities)
================================================
This project aims to provide various utilities for multi-electrode array
recording setups based on MWorks (http://mworks-project.org/) and
Blackrock/Plxeon systems.


Installation
------------
(TBA)
This project provides various utility scripts and a Python package that the
scripts rely on.

To install:
   python setup.py install

To install locally under the home directory:
   python setup.py install --user

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
(TBA)

        +--------+          +------------------+
        | MWorks |          | BlackRock/Plexon |
        +--------+          +------------------+
             |                       |
             v                       v
       (Original .mwk)          (.nev/.plx)
             |                       |
             v                       v
   +-----------------------------------------------+
   | maru-merge: merge behavior .mwk files and     |
   |    neural signal files .nev/.plx to produce   |
   |    merged .mwk files                          |
   +---------------------+-------------------------+
                         |
                         v
       +-------- (Merged .mwk files)
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       |
       v      
   +-----------------------------------+
   | maru-psinfo: collect peristimulus |
   |    firing information             |
   +-----------------------------------+








