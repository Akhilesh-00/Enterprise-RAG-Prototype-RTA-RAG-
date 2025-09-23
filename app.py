import os
import io
import json
import base64
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

matplotlib.use('Agg')

# --- 1. FLASK SETUP ---
app = Flask(__name__)
CORS(app)  # Allow all origins for frontend access

# --- 2. GEMINI CONFIG & API KEY ---
YOUR_GEMINI_API_KEY = "AIzaSyA8vmZhVmKw2280JRQg9mYkJ9vMMRduOrU"

if not YOUR_GEMINI_API_KEY or YOUR_GEMINI_API_KEY == "PASTE_YOUR_KEY_HERE":
    raise ValueError("Please paste your Gemini API Key into the 'YOUR_GEMINI_API_KEY' variable.")

genai.configure(api_key=YOUR_GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')
print("✅ Gemini 1.5 Pro configured successfully.")

# --- 3. LOAD DATA ---
def load_data():
    paths = [
        "dataset/june_2025_data_full.xlsx",
        "dataset/july_2025_data_full.xlsx",
        "dataset/august_2025_data_full.xlsx"
    ]
    try:
        dfs = [pd.read_excel(p) for p in paths]
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found. Make sure the 'dataset' folder exists and files are named correctly.")
        exit()

    df = pd.concat(dfs, ignore_index=True)
    df.dropna(subset=['trips'], inplace=True)
    df['trips'] = pd.to_numeric(df['trips'], errors='coerce').astype(int)
    print(f"✅ Knowledge base loaded: {len(df)} records")
    return df

df = load_data()

# --- 4. HELPER: CREATE CHART ---
def create_chart(chart_data):
    if not chart_data or not chart_data.get('labels') or not chart_data.get('values'):
        return None
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(chart_data['labels'], chart_data['values'], color=chart_data.get('colors', ['#3B82F6', '#10B981', '#F59E0B']))
    ax.set_ylabel('Trips', fontsize=12, color='white')
    ax.set_title(chart_data.get('title', 'Chart'), fontsize=16, fontweight='bold', pad=20, color='white')
    ax.tick_params(axis='x', colors='white', rotation=0)
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('#1F2937')
    fig.patch.set_facecolor('#111827')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{int(yval):,}', va='bottom', ha='center', fontsize=10, fontweight='bold', color='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# --- 5. FULL LLM HANDLING ---
def generate_answer_full_llm(question, df):
    prompt = f"""
    You are a data analysis expert. Given the dataset below, first decide the operation (SUM, AVERAGE, MAX, MIN, COMPARE),
    the filters to apply (month, transport_mode), then perform the calculation.

    User Question: "{question}"
    Dataset:
    {df.to_dict(orient='records')}

    Instructions:
    1. Decide the operation and filters needed.
    2. Perform the calculation on the 'trips' column.
    3. Provide a one-sentence summary.
    4. If applicable (COMPARE, MAX, MIN), generate chart data with labels and values.
    5. Respond ONLY in JSON format as follows:

    {{
      "operation": "MAX",
      "filters": {{"month": "Aug", "transport_mode": ["Metro"]}},
      "summary": "Station X had the highest trips in August for Metro.",
      "chart_data": {{"title": "Trips by Station", "labels": ["Station1", "Station2"], "values": [1000, 2000]}}
    }}
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(json_str)
    except Exception as e:
        print(f"❌ Error during LLM Generation: {e}")
        return {"summary": "I had trouble summarizing the final result.", "graph": None}

    chart_img = create_chart(result.get('chart_data'))
    return {"summary": result.get('summary'), "graph": chart_img}

# --- 6. FLASK ENDPOINTS ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400
    result = generate_answer_full_llm(user_question, df)
    return jsonify(result)

# --- 7. RUN APP ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render assigned port or default 5000
    app.run(host="0.0.0.0", port=port, debug=True)
