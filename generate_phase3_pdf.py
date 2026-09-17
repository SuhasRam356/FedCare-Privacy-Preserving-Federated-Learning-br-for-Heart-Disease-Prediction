from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'FedCare: Phase 3 Non-IID & FedProx Explained', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('helvetica', '', 11)

content = """
Phase 3 focused on a huge real-world problem in healthcare: Hospitals are very different from each other. 
In Phase 2, we assumed every hospital had roughly the same mix of patients. But in the real world, a hospital in a retirement community might have mostly older patients with high cholesterol, while a sports clinic might have young, fit patients.

In AI terms, this messy, uneven data is called "Non-IID" (Non-Independent and Identically Distributed). Phase 3 proves that our system can survive this real-world messiness.

Here is a deeper, but simple English explanation of the code we wrote in Phase 3:

1. fedcare/partition.py (The Mess Maker)
What it does: It purposely creates uneven, heavily biased datasets for our hospitals.
Simple English: Instead of giving every hospital a fair, even mix of healthy and sick patients, this code purposely groups the data by age and cholesterol levels. It might give Hospital 1 all the old, sick patients, and Hospital 2 all the young, healthy patients. We wrote this code to intentionally break the standard AI we built in Phase 2 so we could learn how to fix it.

2. The Problem with Phase 2 (Client Drift)
What happens: When we run the Phase 2 server (FedAvg) on this new messy data, the AI fails.
Simple English: Because Hospital 1 only sees sick patients, its local AI brain becomes convinced that EVERYONE is sick. Because Hospital 2 only sees healthy patients, its local AI brain becomes convinced that NO ONE is sick. When the central server tries to average these extremely different brains together, the result is complete garbage. The models have "drifted" too far apart.

3. fedcare/strategy/fedprox.py (The Smart Leash)
What it does: This is an advanced mathematical replacement for the basic FedAvg server strategy.
Simple English: FedProx fixes the "Client Drift" problem. It acts like a leash between the central server and the hospitals. It tells the hospitals: "You are allowed to learn from your own local patients, but I am adding a mathematical penalty if your AI brain changes too much from the master global brain." By keeping all the hospitals on a leash, FedProx ensures their models don't drift so far apart that they can't be averaged together. 

4. run_phase3_experiments.py (The Ultimate Test)
What it does: This script runs the simulation twice to prove the fix works.
Simple English: First, this script tells the 6 virtual hospitals to train on the messy, biased data using the old Phase 2 method (FedAvg). It records how badly the AI performs. Then, it resets everything and runs the exact same test, but this time it uses the new FedProx "leash" strategy. Finally, it saves the results to CSV files so we can draw a chart showing that FedProx successfully learned from the messy data while FedAvg failed.
"""

pdf.multi_cell(0, 6, content)
pdf.output('Phase3_Code_Explanation.pdf')
print("PDF successfully generated: Phase3_Code_Explanation.pdf")
