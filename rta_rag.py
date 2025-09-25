import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import json
from flask import Flask
app = Flask(__name__)

# ===============================
# 1. LOAD DATA
# ===============================

DATA_FOLDER = "./dataset"

file_path = os.path.join(DATA_FOLDER, "ridership.csv")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

print(f"CSV loaded successfully. Total rows: {len(df)}")
# ===============================
# 2. EMBEDDINGS + VECTOR DB
# ===============================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("ridership_data")

documents, metadatas, ids = [], [], []
for idx, row in df.iterrows():
    date_str = row['date'].strftime('%A, %B %d, %Y')
    ridership_details = [
        f"{col.replace('_', ' ')} had {int(row[col])} trips"
        for col in df.columns if col != "date" and pd.notna(row[col])
    ]
    atomic_chunk = f"On {date_str}, " + "; ".join(ridership_details) + "."

    documents.append(atomic_chunk)
    metadatas.append({
        "row_id": idx,
        "date": row['date'].isoformat()
    })
    ids.append(str(idx))

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=embedding_model.encode(documents).tolist()
)

print(f"Chroma collection created with {len(documents)} chunks.")

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Gemini setup
genai.configure(api_key="AIzaSyDK0_A0qcZRtWeTY3CMK8Ntn10295dWTas")  # replace with your Gemini API key
synthesis_model = genai.GenerativeModel("gemini-1.5-pro-latest")# ===============================
# 3. RETRIEVAL
# ===============================
def vector_retrieval(query, top_k=10):
    query_emb = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    if not results["documents"]:
        return []
    docs = results["documents"][0]
    scores = results["distances"][0]

    rerank_pairs = [[query, doc] for doc in docs]
    rerank_scores = reranker.predict(rerank_pairs)

    reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]# ===============================
# 4. QUERY CLASSIFIER (UPGRADED)
# ===============================
def classify_query(query):
    """Detect whether query is aggregation (math) or descriptive."""
    agg_keywords = [
        "total", "sum", "average", "mean", "max", "min", "highest", "lowest",
        "busiest", "least", "compare", "vs"
    ]
    if any(word in query.lower() for word in agg_keywords):
        return "aggregation"
    return "descriptive"
# =============================== 
# 5. HYBRID ANSWER ENGINE (UPGRADED)
# ===============================
def answer_query(query):
    query_type = classify_query(query)
    print(f"🔍 Detected query type: {query_type}")

    # Detect transport modes dynamically from CSV columns
    transport_cols = [col for col in df.columns if col != "date" and col.lower() != "week"]  # ignore week column
    matched_cols = [col for col in transport_cols if col.lower() in query.lower()]

    # If query mentions "all transport modes", match all columns
    if "all transport modes" in query.lower():
        matched_cols = transport_cols

    import re

    def _month_number(name):
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        return months.get(name.lower())

    tmap = {c.lower(): c for c in transport_cols}
    ql = query.lower().strip()

    # ----------------------------
    # Function to summarize a slice
    # ----------------------------
    def _slice_summary(slice_df, col):
        sdf = slice_df.copy()
        if not sdf.empty:
            sdf["week_of_month"] = sdf["date"].apply(lambda d: (d.day - 1) // 7 + 1)
        total = sdf[col].sum() if not sdf.empty else 0
        avg = sdf[col].mean() if not sdf.empty else 0
        if not sdf.empty:
            max_row = sdf.loc[sdf[col].idxmax()]
            min_row = sdf.loc[sdf[col].idxmin()]
            busiest_week_series = sdf.groupby("week_of_month")[col].sum()
            busiest_week = busiest_week_series.idxmax() if not busiest_week_series.empty else "N/A"
            busiest_week_trips = busiest_week_series.max() if not busiest_week_series.empty else 0
            max_date = max_row["date"].date()
            min_date = min_row["date"].date()
            max_val = int(max_row[col])
            min_val = int(min_row[col])
        else:
            busiest_week, busiest_week_trips = "N/A", 0
            max_date = min_date = None
            max_val = min_val = 0

        return {
            "total": int(total),
            "avg": float(avg) if avg == avg else 0.0,
            "busiest_day_date": max_date,
            "busiest_day_value": max_val,
            "quietest_day_date": min_date,
            "quietest_day_value": min_val,
            "busiest_week": busiest_week,
            "busiest_week_trips": int(busiest_week_trips),
        }

    # ----------------------------
    # Week-based comparison detection
    # ----------------------------
    week_matches = re.findall(r'(\d+)(?:st|nd|rd|th)? week of (\w+)', ql)
    week_mode = None
    if week_matches:
        week_mode = True
        # Extract transport mode if mentioned
        for col in transport_cols:
            if col.lower() in ql:
                matched_cols = [col]
                break

    # ----------------------------
    # Multi-mode/multi-month comparison detection
    # ----------------------------
    compare_patterns = [
        r"compare\s+([a-z_]+)(?:\s+trips)?\s+(?:of|in)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\s*(?:to|vs\.?|versus)\s*([a-z_]+)(?:\s+trips)?\s+(?:of|in)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?",
        r"compare\s+([a-z_]+)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\s*(?:to|vs\.?|versus)\s*([a-z_]+)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?"
    ]

    compare_match = None
    for pat in compare_patterns:
        m = re.search(pat, ql)
        if m:
            compare_match = m
            break

    # ----------------------------
    # Aggregation branch
    # ----------------------------
    if query_type == "aggregation" and matched_cols:

        # ----------------------------
        # Comparison query branch
        # ----------------------------
        if compare_match:
            mode1, mon1, year1, mode2, mon2, year2 = compare_match.groups()
            mode1 = mode1.strip().lower()
            mode2 = mode2.strip().lower()
            col1 = tmap.get(mode1)
            col2 = tmap.get(mode2)

            if not col1 or not col2:
                pass
            else:
                m1 = _month_number(mon1)
                m2 = _month_number(mon2)
                y1 = int(year1) if year1 else None
                y2 = int(year2) if year2 else None

                df1 = df[df["date"].dt.month == m1].copy()
                df2 = df[df["date"].dt.month == m2].copy()
                if y1:
                    df1 = df1[df1["date"].dt.year == y1]
                if y2:
                    df2 = df2[df2["date"].dt.year == y2]

                # Ensure week_of_month exists
                if 'week_of_month' not in df1.columns:
                    df1["week_of_month"] = df1["date"].apply(lambda d: (d.day - 1) // 7 + 1)
                if 'week_of_month' not in df2.columns:
                    df2["week_of_month"] = df2["date"].apply(lambda d: (d.day - 1) // 7 + 1)

                s1 = _slice_summary(df1, col1)
                s2 = _slice_summary(df2, col2)

                # ----------------------------
                # Prepare plot for comparison (FIXED)
                # ----------------------------
                import matplotlib.pyplot as plt
                import textwrap

                plot_df = pd.DataFrame({
                    "Transport Mode": [f"{col1} {mon1}", f"{col2} {mon2}"],
                    "Total Trips": [s1["total"], s2["total"]],
                    "Average Daily Trips": [s1["avg"], s2["avg"]],
                    "Busiest Week Trips": [s1["busiest_week_trips"], s2["busiest_week_trips"]]
                })

                if not plot_df.empty:
                    plt.style.use('seaborn-v0_8-darkgrid')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    labels = [textwrap.fill(str(label), 15) for label in plot_df["Transport Mode"]]

                    # NEW: metric sync to match generated answer
                    metric_key = "Total Trips"
                    ylabel = "Total Trips"
                    if any(k in ql for k in ["average", "avg", "mean"]):
                        metric_key = "Average Daily Trips"
                        ylabel = "Average Daily Trips"
                    elif "busiest week" in ql:
                        metric_key = "Busiest Week Trips"
                        ylabel = "Busiest Week Trips"

                    values = plot_df[metric_key]
                    bars = ax.bar(labels, values, color=['#3B82F6', '#10B981'])
                    ax.set_ylabel(ylabel, fontsize=14)
                    ax.set_title(f"Ridership Comparison for Query: {query[:50]}...", fontsize=16, fontweight="bold")
                    ymax = max(values) if len(values) else 0
                    for bar, val in zip(bars, values):
                        disp = f"{int(val):,}" if metric_key != "Average Daily Trips" else (f"{val:,.2f}" if val < 1 else f"{val:,.0f}")
                        ax.text(bar.get_x() + bar.get_width()/2, val + (ymax*0.01 if ymax else 0.05),
                                disp, ha="center", va="bottom", fontsize=11, fontweight="bold")
                    plt.show()

                # ----------------------------
                # LLM context
                # ----------------------------
                def _fmt(slice_name, sdict):
                    return (
                        f"{slice_name}: total={sdict['total']:,}, average={sdict['avg']:,.2f}, "
                        f"busiest day={sdict['busiest_day_date']} ({sdict['busiest_day_value']:,} trips), "
                        f"quietest day={sdict['quietest_day_date']} ({sdict['quietest_day_value']:,} trips), "
                        f"busiest week={sdict['busiest_week']} ({sdict['busiest_week_trips']:,} trips)"
                    )

                context_lines = [
                    _fmt(f"{col1} in {mon1}", s1),
                    _fmt(f"{col2} in {mon2}", s2)
                ]
                context = "\n".join(context_lines)

                prompt = f"""
                You are a ridership analysis assistant.
                Based on the following numeric context, generate a full, natural-language comparison for the user query.

                CONTEXT:
                {context}

                USER QUERY:
                {query}

                ANSWER:
                """
                response = synthesis_model.generate_content(prompt)
                return response.text

        # ----------------------------
        # Week-only comparison branch (FIXED)
        # ----------------------------
        if week_mode:
            filtered_dfs = []
            for wk_num, wk_month in week_matches:
                month_num = _month_number(wk_month)
                if month_num:
                    temp_df = df.copy()
                    # Ensure week_of_month exists
                    if 'week_of_month' not in temp_df.columns:
                        temp_df['week_of_month'] = temp_df['date'].apply(lambda d: (d.day - 1) // 7 + 1)
                    temp_df = temp_df[(temp_df['week_of_month'] == int(wk_num)) & (temp_df['date'].dt.month == month_num)]
                    filtered_dfs.append(temp_df)
            if filtered_dfs:
                filtered_df = pd.concat(filtered_dfs)
            else:
                filtered_df = df.copy()
                if 'week_of_month' not in filtered_df.columns:
                    filtered_df['week_of_month'] = filtered_df['date'].apply(lambda d: (d.day - 1) // 7 + 1)

        # ----------------------------
        # Original aggregation logic
        # ----------------------------
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        filtered_dfs = []
        for m, num in months.items():
            if m in ql:
                temp_df = df[df['date'].dt.month == num].copy()
                filtered_dfs.append(temp_df)
        if filtered_dfs:
            filtered_df = pd.concat(filtered_dfs)
        else:
            filtered_df = df.copy()

        for year in range(2020, 2035):
            if str(year) in query:
                filtered_df = filtered_df[filtered_df['date'].dt.year == year]

        if filtered_df.empty:
            return f"❌ No data available for {query}."

        # Ensure week_of_month exists
        if 'week_of_month' not in filtered_df.columns:
            filtered_df['week_of_month'] = filtered_df['date'].apply(lambda d: (d.day - 1) // 7 + 1)

        summary_lines = []
        summary_dicts = []  # store _slice_summary dicts for plotting
        for col in matched_cols:
            s = _slice_summary(filtered_df, col)
            summary_dicts.append(s)
            summary_lines.append(
                f"{col}: total={s['total']:,}, average={s['avg']:,.2f}, busiest day={s['busiest_day_date']} ({int(s['busiest_day_value']):,} trips), "
                f"quietest day={s['quietest_day_date']} ({int(s['quietest_day_value']):,} trips), busiest week={s['busiest_week']} ({int(s['busiest_week_trips']):,} trips)"
            )

        # ----------------------------
        # Plotting (MATCHING TEXT)
        # ----------------------------
        import matplotlib.pyplot as plt
        import textwrap

        plot_df = pd.DataFrame({
            'Transport Mode': matched_cols,
            'Total Trips': [s['total'] for s in summary_dicts],
            'Average Daily Trips': [s['avg'] for s in summary_dicts],
            'Busiest Week Trips': [s['busiest_week_trips'] for s in summary_dicts]
        })

        if not plot_df.empty:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(12, 7))
            raw_labels = plot_df['Transport Mode']
            labels = [textwrap.fill(str(label).replace('_', ' ').title(), 15) for label in raw_labels]

            # NEW: metric sync so plot matches answer text
            metric_key = 'Total Trips'
            ylabel = 'Total Trips'
            if any(k in ql for k in ['average', 'avg', 'mean']):
                metric_key = 'Average Daily Trips'
                ylabel = 'Average Daily Trips'
            elif 'busiest week' in ql:
                metric_key = 'Busiest Week Trips'
                ylabel = 'Busiest Week Trips'

            values = plot_df[metric_key]
            bars = ax.bar(labels, values, color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'])
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(f"Ridership Overview for Query: {query[:50]}...", fontsize=18, fontweight='bold')

            ymax = max(values) if len(values) else 0
            for bar, val in zip(bars, values):
                disp = f"{int(val):,}" if metric_key != 'Average Daily Trips' else (f"{val:,.2f}" if val < 1 else f"{val:,.0f}")
                ax.text(bar.get_x() + bar.get_width()/2, val + (ymax*0.01 if ymax else 0.05),
                        disp, ha="center", va="bottom", fontsize=11, fontweight="bold")
            plt.show()

        context = "\n".join(summary_lines)
        prompt = f"""
        You are a ridership analysis assistant.
        Based on the following numeric context, generate a full, natural-language answer to the user query.

        CONTEXT:
        {context}

        USER QUERY:
        {query}

        ANSWER:
        """
        response = synthesis_model.generate_content(prompt)
        return response.text

    # ----------------------------
    # Descriptive queries
    # ----------------------------
    else:
        docs = vector_retrieval(query, top_k=15)
        if not docs:
            return "❌ No relevant information found."
        context = "\n".join(docs)
        prompt = f"""
        You are a ridership analysis assistant.
        Use ONLY the context below to answer the question in a detailed, natural-language way.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """
        response = synthesis_model.generate_content(prompt)
        return response.text
# ===============================
# 6. TEST QUERIES
# ===============================
q1 = "Average metro trips in March"
print("\n--- Running Query 1 ---")
print(answer_query(q1))

q2 = "compare 2nd week of april with 1st week of March based on tram trips"
print("\n--- Running Query 2 ---")
print(answer_query(q2))

q3 = "compare metro green line  trips of march to average metro trips of march"
print("\n--- Running Query 3 ---")
print(answer_query(q3))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
