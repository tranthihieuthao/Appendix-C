import spacy
from spacy.matcher import Matcher
import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================
# 1. MODEL SETUP (TRANSFORMER-BASED)
# ===========================================================
try:
    print("Loading Transformer Model (en_core_web_trf)... Please wait.")
    nlp = spacy.load("en_core_web_trf")
    print("Model loaded successfully!\n")
except OSError:
    print("ERROR: Transformer model not found.")
    print("Please run: python -m spacy download en_core_web_trf")
    exit()

class MT_Grammar_Evaluator:
    def __init__(self):
        self.matcher = Matcher(nlp.vocab)
        self._set_up_rules()
        self.results_log = []

    def _set_up_rules(self):
        # Rule for Passive Voice (Passive auxiliary + Past Participle)
        passive_rule = [{"DEP": "auxpass", "OP": "?"}, {"TAG": "VBN"}]
        # Rule for Relative Clause (Dependency label 'relcl')
        relative_rule = [{"DEP": "relcl"}]
        
        self.matcher.add("PASSIVE_VOICE", [passive_rule])
        self.matcher.add("RELATIVE_CLAUSE", [relative_rule])

    def evaluate(self, en_text, vi_text, category):
        doc = nlp(en_text)
        matches = self.matcher(doc)
        found_structures = [nlp.vocab.strings[m[0]] for m in matches]
        is_correct = False

        # Cross-linguistic logic: Check for Vietnamese functional markers
        if "PASSIVE_VOICE" in found_structures:
            if any(marker in vi_text.lower() for marker in ["được", "bị"]):
                is_correct = True
        elif "RELATIVE_CLAUSE" in found_structures:
            if any(marker in vi_text.lower() for marker in ["mà", "người", "vật", "nơi"]):
                is_correct = True
        else:
            is_correct = True

        self.results_log.append({
            "English Source": en_text,
            "Vietnamese Target": vi_text,
            "Category": category,
            "Result": "Correct" if is_correct else "Error"
        })

# ===========================================================
# 2. DATASET (REPRESENTATIVE SAMPLES)
# ===========================================================
dataset = [
    # Passive Voice Group
    ("The house was painted last week.", "Ngôi nhà được sơn tuần trước.", "Passive"),
    ("A new bridge is being built.", "Một cây cầu mới đang được xây.", "Passive"),
    ("The thief was caught by the police.", "Tên trộm bị cảnh sát bắt.", "Passive"),
    ("Dinner has been prepared.", "Bữa tối đã chuẩn bị xong.", "Passive"), # Error: missing 'được'
    ("The letter was sent by mistake.", "Lá thư đã gửi nhầm.", "Passive"), # Error: missing 'được'
    ("He was punished for being late.", "Anh ấy bị phạt vì đi muộn.", "Passive"),
    ("The car was repaired.", "Chiếc xe đã sửa.", "Passive"), # Error: missing 'được'
    ("The project was completed on time.", "Dự án đã được hoàn thành đúng hạn.", "Passive"),
    
    # Relative Clause Group
    ("The woman who lives next door is a doctor.", "Người phụ nữ sống cạnh nhà là bác sĩ.", "Relative"),
    ("The book that I bought is interesting.", "Cuốn sách mà tôi mua rất thú vị.", "Relative"),
    ("The city where I live is beautiful.", "Thành phố nơi tôi sống rất đẹp.", "Relative"),
    ("The phone, which was expensive, broke.", "Cái điện thoại đã hỏng.", "Relative"), # Error: missing 'mà' or info
    ("This is the house that Jack built.", "Đây là ngôi nhà mà Jack đã xây.", "Relative"),
    ("The man standing there is my uncle.", "Người đàn ông đứng đó là chú tôi.", "Relative"),
    ("I saw the boy who stole the bike.", "Tôi thấy cậu bé người mà đã trộm xe.", "Relative"),
    ("The reasons why he left are unknown.", "Lý do tại sao anh ấy rời đi không ai biết.", "Relative")
]

# ===========================================================
# 3. ANALYSIS EXECUTION
# ===========================================================
evaluator = MT_Grammar_Evaluator()
print("Analyzing linguistic structures...")
for en, vi, cat in dataset:
    evaluator.evaluate(en, vi, cat)

df = pd.DataFrame(evaluator.results_log)
summary = df.groupby(['Category', 'Result']).size().unstack(fill_value=0)

print("\n--- STATISTICAL SUMMARY ---")
print(summary)

# ===========================================================
# 4. VISUALIZATION 1: STACKED BAR CHART
# ===========================================================
summary.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], figsize=(10, 6))
plt.title("MT Grammatical Accuracy by Category (Transformer-based)", fontsize=14)
plt.xlabel("Grammatical Structures", fontsize=12)
plt.ylabel("Number of Sentences", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="MT Outcome")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Figure_1_BarChart.png")
print("\nSuccess! Bar chart saved as 'Figure_1_BarChart.png'.")

# ===========================================================
# 5. VISUALIZATION 2: PIE CHART (OVERALL %)
# ===========================================================
plt.figure(figsize=(8, 8))
error_counts = df['Result'].value_counts()
plt.pie(error_counts, labels=error_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0))
plt.title("Overall Translation Accuracy Percentage", fontsize=14)
plt.savefig("Figure_2_PieChart.png")
print("Success! Pie chart saved as 'Figure_2_PieChart.png'.")

print("\n--- PROCESS COMPLETED ---")
plt.show()