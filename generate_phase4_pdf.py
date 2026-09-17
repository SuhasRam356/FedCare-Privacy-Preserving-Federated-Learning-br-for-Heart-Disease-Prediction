from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'FedCare: Phase 4 Attacks and Defenses Explained', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('helvetica', '', 11)

content = """
Phase 4 focused on Cyber Security. In the real world, a hospital's computer system could get hacked. If a hacker takes control of one of the 6 hospitals in our network, they could try to use that hospital to destroy the main server's Global AI Brain.

Here is a simple English explanation of the code we wrote to simulate these hackers and build defenses against them:

1. fedcare/attacks/label_flip.py (The Liar Hacker)
What it does: Simulates a hacker changing the patient records.
Simple English: Imagine a hacker breaks into Hospital 1. Instead of deleting the data, they quietly flip all the answers. They label all the healthy patients as "sick", and all the sick patients as "healthy". When Hospital 1's AI learns from this flipped data, it learns the exact opposite of the truth. When this lying AI brain is sent to the central server and averaged in, it poisons the whole system.

2. fedcare/attacks/model_poison.py (The Saboteur Hacker)
What it does: Simulates a hacker directly destroying the AI's math.
Simple English: Instead of messing with the patient data, this hacker waits until the hospital's AI brain is finished training. Right before the brain is sent to the central server, the hacker multiplies all the numbers in the brain by 100, scrambling it. When the server tries to average this corrupted brain, it blows up the math for the entire system.

3. The Problem with Phase 2 & 3 (Trusting Everyone)
What happens: The previous servers (FedAvg and FedProx) blindly trust every hospital.
Simple English: If 2 hospitals are hacked and 4 are normal, FedAvg and FedProx don't care. They just average all 6 brains together anyway. This means the 2 hacked brains successfully poison the master AI.

4. fedcare/strategy/trimmed_mean.py, median.py, krum.py (The Security Guards)
What they do: These are advanced math defenses that act like security guards at the central server.
Simple English: Instead of blindly averaging the 6 brains, the server now inspects them first. 
- Trimmed Mean: Looks at all the brains, throws the most extreme ones in the trash, and averages the rest.
- Median: Lines all the brains up in order and just picks the one exactly in the middle, ignoring the crazy ones on the edges.
- Krum: A very complex guard that mathematically compares every brain to every other brain to figure out which ones belong to a "normal" group, and bans any brains that act like loners.

5. run_phase4_experiments.py (The Hacker Simulation)
What it does: Simulates a cyber attack to prove our defenses work.
Simple English: This script starts the 6 hospitals, but intentionally turns 2 of them into malicious hackers. First, it runs the simulation with no security guards (FedAvg), and we watch the hackers completely destroy the AI's accuracy. Then, it resets the simulation and puts our security guards (like Trimmed Mean) at the door. We watch as the security guards successfully identify and block the hackers, keeping the AI's accuracy high!
"""

pdf.multi_cell(0, 6, content)
pdf.output('Phase4_Code_Explanation.pdf')
print("PDF successfully generated: Phase4_Code_Explanation.pdf")
