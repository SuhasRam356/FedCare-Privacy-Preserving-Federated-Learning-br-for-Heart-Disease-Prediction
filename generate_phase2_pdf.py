from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'FedCare: Phase 2 Core Implementation Explained', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('helvetica', '', 11)

content = """
Phase 2 focused on building the core "Federated Learning" system. In Phase 1, we set up the data and the basic AI model. In Phase 2, we actually connected the hospitals together so they can train the model without sharing patient data.

Here is a simple English explanation of the code and files we wrote in Phase 2:

1. fedcare/task.py (The AI Brain)
What it does: This file holds the "recipe" for our AI model and how it learns.
Simple English: We built a Neural Network (a type of AI brain) designed to predict heart disease. It takes 13 medical measurements (like age, cholesterol, and blood pressure) and outputs whether the patient has heart disease or not. It also includes functions to load the data and train the AI on that data.

2. baseline_centralized.py & baseline_local.py (The Baselines)
What they do: These files establish what happens if we DON'T use federated learning.
Simple English: 
- baseline_centralized.py imagines a world where all 6 hospitals put all their 12,000 patient records into one giant central database, ignoring privacy. This gives us the "best possible" AI score to compare against.
- baseline_local.py imagines a world where hospitals refuse to cooperate and each hospital trains an AI only on their own small dataset. This gives us a "worst possible" AI score to compare against.

3. fedcare/client_app.py (The Hospital Side)
What it does: This is the software that runs inside each of the 6 hospitals.
Simple English: Instead of sending data to the server, the central server sends the AI model to the hospital. This file receives the model, trains it using the hospital's private local data for a few rounds, and then sends ONLY the "learned improvements" (called weights) back to the central server. The raw patient data never leaves the building.

4. fedcare/server_app.py (The Server Side)
What it does: This is the software that runs on the central coordinating server.
Simple English: The server's job is to act as a conductor. It sends out the empty AI model to the hospitals. Once the hospitals send back their "learned improvements", this file uses a strategy (like Federated Averaging) to average all the improvements together into one master "Global Model". It then sends the updated Global Model back to the hospitals for the next round.

5. fedcare/strategy/fedavg_weighted.py (The Averaging Rules)
What it does: The mathematical rules for how the server combines the models.
Simple English: When combining the models from the 6 hospitals, the server doesn't just do a simple average. If Hospital 1 has 3,000 patients and Hospital 2 has only 500 patients, the server will trust Hospital 1's improvements more. This "weighted averaging" makes the final AI much smarter.

6. run_federated.py (The Main Engine)
What it does: This is the script that brings everything together and actually runs the simulation.
Simple English: This script creates a virtual server and 6 virtual hospitals on your computer. It tells them to connect to each other using the Flower framework, and run 20 rounds of training. At the end of the 20 rounds, it saves the final metrics (like Accuracy and AUC) into a CSV file so we can see how well the federated learning worked compared to the centralized baseline.
"""

pdf.multi_cell(0, 6, content)
pdf.output('Phase2_Code_Explanation.pdf')
print("PDF successfully generated: Phase2_Code_Explanation.pdf")
