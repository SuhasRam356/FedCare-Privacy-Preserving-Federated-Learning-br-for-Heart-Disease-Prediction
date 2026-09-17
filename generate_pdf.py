from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'FedCare: Project Folder Structure Explained', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('helvetica', '', 11)

content = """
This document explains the purpose of each folder in the FedCare project in simple terms.

1. app/
This folder contains the interactive web dashboard for the project (Phase 5).
- dashboard.py: This is the main file that runs the Streamlit web interface you see in your browser. It has all the code for the charts, the network topology, the attack/defense comparisons, and the live risk calculator.

2. data/heart/
This folder holds all the patient data used to train the models.
- combined.csv: All 12,000 patient records together.
- hospital_1.csv to hospital_6.csv: The same 12,000 records split into 6 separate files, representing 6 different hospitals. This simulates how real hospitals keep their data separate and private.

3. fedcare/
This is the core library folder. It contains all the essential logic for federated learning.
- task.py: Contains the AI model design (a neural network) and the code to load data and train the model.
- client_app.py: The code that runs at each hospital to train the model locally on their private data.
- server_app.py: The code that runs on the central server to combine all the hospitals' models into one global model.
- partition.py: Code to split data into different scenarios (like when hospitals have very different patient demographics).
- metrics.py: Code to measure how well the model is performing and how fair it is across hospitals.
- privacy.py: Code that adds 'noise' to the data so that hackers cannot reverse-engineer patient details (Differential Privacy).
- comm_cost.py: Calculates how much internet bandwidth is used to send model updates.

4. fedcare/attacks/
Contains simulations of malicious behavior to test the system's security.
- label_flip.py: Simulates a hospital intentionally giving the wrong answers to confuse the AI.
- model_poison.py: Simulates a hospital directly corrupting the AI model weights before sending them to the server.

5. fedcare/strategy/
Contains different methods for combining the hospitals' models at the central server.
- fedavg_weighted.py: The standard method (averaging).
- fedprox.py: A method that keeps hospitals from diverging too much.
- trimmed_mean.py, median.py, krum.py: 'Byzantine-robust' methods that identify and ignore malicious hospitals.

6. scripts/
Contains helper scripts for plotting graphs and saving model files.
- prepare_data.py, plot_phase2.py, plot_phase4.py: Scripts to generate the graphs and charts.
- save_checkpoint.py: Saves the trained AI model so the risk calculator can use it immediately.

7. tests/
Contains all the automated checks to make sure the code works properly.
- test_baselines.py to test_phase5.py: These files contain 69 tests that automatically run to verify that every part of the project (from basic data loading to the final web dashboard) works without errors.

8. results/
This folder stores all the output data from the experiments.
- It contains CSV files with the results and PNG images of the charts that are displayed on the dashboard.

9. checkpoints/
Stores the finalized, trained AI model (global_model.pt) so it doesn't need to be retrained every time you use the dashboard's risk calculator.

10. docs/screenshots/
Contains the screenshots of the web dashboard that are used to make the README file look nice.

Main Files in the Root Directory:
- run_federated.py, run_phase3_experiments.py, run_phase4_experiments.py: These scripts launch the various experiments.
- demo.py: The master script that verifies everything is working and launches the dashboard for the viva presentation.
- README.md: The comprehensive instruction manual for the project.
"""

pdf.multi_cell(0, 6, content)
pdf.output('Project_Folders_Explained.pdf')
print("PDF successfully generated: Project_Folders_Explained.pdf")
