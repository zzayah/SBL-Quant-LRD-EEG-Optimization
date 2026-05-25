**Abstract**
The DD-Database contains the physiological signals of 4 EEG channels, 2 EOG channels and 1 ECG channel as well as annotation files corresponding to 10 healthy volunteers between 20 and 50 years old. The signals were collected during the use of a driving simulator under a protocol designed to induce drowsiness. The annotation files have the time marks of subject´s drowsiness events. The experiment was carried out in 2 trials of 2 hours each, so the database contains 40 hours of information. This database is a contribution for the development and evaluation of algorithms and/or systems for detecting drowsiness in drivers. Data acquired during the use of a driving simulator and publicly available, are scarce.

**Background**
Drowsiness causes changes in physiological signals, such as EEG, EOG, and ECG, these changes can be detected through their processing. The detection, based on physiological signals, can anticipate the physical manifestations of drowsiness in people during the execution of a routine task, such as driving a car [1].
Distinctive parameters are extracted from signals, through signal processing techniques. It is known these parameters show modifications in the transition between the awake and sleep state. The analysis of the variation of these parameters allows to differentiate one state from the other [2]. These differences can be used to detect the drowsiness state, early, i.e. before the bodily consequences, which is an advantage.
The drowsiness detection in drivers is an important research area due to the drowsy driving is responsible for many accidents in which people die or get seriously injured. In these accidents, not only human lives are lost, but a huge expense of money is made in insurances and medical care, leading to great economic losses [3].
Having a database of physiological signals acquired in people during the task of driving in a drowsy state is essential for the development and evaluation of algorithms and/or systems for detecting drowsiness in drivers. These type of database, acquired during the use of a driving simulator and publicly available, are scarce.

**Methods**
Data were collected from 10 healthy volunteers between 20 and 50 years old during the use of a driving simulator. All volunteers provided written informed consent. For all subjects, approximately 4 hours of physiological signals were collected.
The experiment itself, consisted on acquiring physiological signals during the use of a driving simulator in a dark and quiet environment while presenting a night driving scenario on a straight route. The day before the test, the subjects were instructed to partially deprived of sleep, attend after lunch at the agreed time and do not take medications, or alcoholic or energy drinks. All these actions pointed to induced drowsiness. The test lasts 2 hours and it was carried out twice for each volunteer in a different day. Additionally, the volunteers were instructed to press an event button when they felt drowsy during the test.
The data are de-identified and free of personal health information (PHI). The volunteer number (01 to 10) and the trial number of the same volunteer (1 or 2) are randomly assigned, both trails always remain together. Then month (11 or 12) and a day from 01 to 30 are randomly assigned. The information of the gender, the year and the channel is kept.
The physiological signals were acquired at a sample frequency of 128 Hz with a 16-bit resolution A/D converter. EEG and EOG were configured as unipolar channels. For ECG a bipolar channel in DIII position was acquired. An AKONIC Neurotrace-Mini PC polysomnography equipment was used for the signals acquisition. The Mini-PC model has a sampling frequency of 128 Hz for all signals, 32 channels plus the system reference and 7 ground inputs filters [4].

**Data description**
The dataset includes recordings of 4 EEG channels (O1, O2, C3, C4 referred to A1 or A2), 2 EOG (LOC and ROC) channels and 1 ECG channel as well as 1 annotation file with the time marks of event button, for each volunteer and for each experiment trial.
The signals and annotations files are available as .edf format, each volunteer completed the test of 2 hours twice so 7 signals files plus an annotation file were generated each time. Totalizing 2 .edf signal files and 2 .edf annotation files for each volunteer.
Each signal file has the raw signal; no hardware or software filter was applied. The annotation files have the time marks (in seconds) of drowsiness. These marks correspond to the volunteer´s drowsiness feeling, registered by the event push button.

**Usage Notes** 
The labeling of files shows the information of #volunteer, gender, #trial and signal type/channel.

**Acknowledgements**
This work was supported by grants from Consejo Nacional de Investigaciones Científicas y Técnicas (CONICET), Universidad Nacional de San Juan (UNSJ) and Secretaría de Estado de Ciencia, Tecnología e Innovación (SECITI): IDEA Project #1400SECITI 0044/2014, all institutions from Argentina.

**References**
**[1]** G. Sikander and S. Anwar, “Driver Fatigue Detection Systems: A Review,” IEEE Trans. Intell. Transp. Syst., vol. 20, no. 6, pp. 2339–2352, 2019.
**[2]** A. Garcés Correa, L. Orosco and E. Laciar (2014). “Technical note: Automatic detection of drowsiness in EEG records based on multimodal analysis”. Medical Engineering & Physics, ISSN: 1350-4533, 36(2): 244–249.
**[3]** National Highway Traffic Safety Administration. “TRAFFIC SAFETY FACTS. Research Note”, 2020. Aviable at https://www.nhtsa.gov/risky-driving/drowsy-driving
**[4]** http://www.akonic.com.ar/Frames-EN/tech-neurotrace.html