===========================================
App Store Data Mining and Analysis Project
===========================================

This project archive contains the source code and data files of the thesis:
"App ecosystem out of balance: 
 An empirical analysis of update interdependence between operating system and application software".

Since the data files are too large for a Github upload, they are provided locally.

-------------------------------------------
1) General Information
-------------------------------------------
In the course of the thesis, five Python modules were developed for data mining and analysis purposes.
An overview of the high-level module architecture can be found in the thesis appendix A.2.
The raw data set of scraped customer reviews along with processed versions and other essential files can be found in the "data_files" folder.

-------------------------------------------
2) Folder directory
-------------------------------------------

Codebase
│
├── README.txt
├── requirements.txt # list with version numbers of all used packages
├── data_files
│   ├── ... # contains review data, model files, app mapping table, and more
└── modules
    ├── appfigures_bot
    │   └── ... # contains module for scraping Appfigures' API
    ├── data_preparator
    │   └── ... # contains module for data preparation
    ├── data_processor
    │   └── ... # contains module for data processing and analysis
    ├── feed_scraper
    │   └── ... # contains module for scraping Apple's RSS feed
    └── main.py # module for controlling the process flow of all other modules

-------------------------------------------
3) Installation
-------------------------------------------
The source code works with Python 3.7, or newer, and implements several open source Python libraries.
The "requirements.txt" contains all Python packages that need to be installed with the pip install command.
The execution of the code is controlled through the "main.py" module.

